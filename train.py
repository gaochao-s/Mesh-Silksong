
import ipdb
import os
import tyro
import math
import time
import shutil
from functools import partial
import traceback
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
# from accelerate.utils import DummyOptim, DummyScheduler
from safetensors.torch import load_file
from model.model import SSMeshTransformer
from config.options import AllConfigs
from silkutils.silksong_tokenization import get_tokenizer_silksong
from silkutils.meshdata.mesh_io import init_logger
import kiui
from model.data_provider import SSDataset, DebugOneDataset, collate_fn, ProgressivelyBalancedSampler

# torch.autograd.set_detect_anomaly(True)



def main():    
    opt = tyro.cli(AllConfigs)
    
    if opt.resume:
        print(f'resuming {opt.resume}')
    else:
        print(f'not resume')
        
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
            mixed_precision=opt.mixed_precision,
            gradient_accumulation_steps=opt.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
    )

    os.makedirs(opt.workspace, exist_ok=True)
    logfile = os.path.join(opt.workspace, 'log.txt')
    logger = init_logger(logfile)

    # print options
    accelerator.print(opt)
    
    # tokenizer
    tokenizer, vocab_size = get_tokenizer_silksong(resolution=opt.discrete_bins, ss_mode=opt.ss_mode)

    print(f'---- engine word table size: {vocab_size}---------')
    # model
    model = SSMeshTransformer(
        dim = opt.model.dim,
        attn_depth = opt.model.depth,
        attn_dim_head = opt.model.attn_dim_head,
        attn_heads = opt.model.attn_heads,
        max_seq_len = opt.max_seq_length,
        dropout = opt.model.dropout,
        mode = opt.mode,
        num_discrete_coors= opt.meto.discrete_bins,
        block_size = opt.meto.block_size,
        offset_size = opt.meto.offset_size,
        conditioned_on_pc = opt.model.conditioned_on_pc,
        encoder_name = opt.model.encoder_name,
        encoder_freeze = opt.model.encoder_freeze,
    )

    # resume
    if opt.resume is not None:
        print(f'resuming {opt.resume}')
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    logger.warning(f'mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
                    print(f'[WARNING] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                logger.warning(f'unexpected param {k}: {v.shape}')
                print(f'[WARNING] unexpected param {k}: {v.shape}')
    
    # count params
    num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_decoder_p = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f'trainable param num: {num_p/1024/1024:.6f} M, GPT param num: {num_decoder_p/1024/1024:.6f} M, total param num: {total_p/1024/1024:.6f}')

    # data
    if opt.data.dataset == 'ss':
        train_dataset = SSDataset(opt, training=True, tokenizer=tokenizer)
        test_dataset = SSDataset(opt, training=False, tokenizer=tokenizer)
        logger.info(f'train dataset size: {len(train_dataset)}')
        logger.info(f'test dataset size: {len(test_dataset)}')

    elif opt.data.dataset == 'debug_one':
        train_dataset = DebugOneDataset(opt, training=True, tokenizer=tokenizer)
        test_dataset = DebugOneDataset(opt, training=False, tokenizer=tokenizer)
        logger.info(f'train dataset size: {len(train_dataset)}')
        logger.info(f'test dataset size: {len(test_dataset)}')
    else:
        raise Exception('not implement dataset')
    
    if opt.data.dataset=='debug_one':
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            collate_fn=partial(collate_fn, opt=opt),
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, opt=opt),
        )
    else:
        if opt.data.resample:
            sampler = ProgressivelyBalancedSampler(
                opt,
                train_dataset,
                face_delta=opt.data.face_delta,
                initial_beta=opt.data.i_beta,
                final_beta=opt.data.e_beta,
                epochs=opt.train.num_epochs
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt.data.batch_size,
                sampler=sampler,
                num_workers=opt.data.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=partial(collate_fn, opt=opt),
            )
        else:
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt.data.batch_size,
                shuffle=True,
                num_workers=opt.data.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=partial(collate_fn, opt=opt),
            )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.data.batch_size,
            shuffle=False,
            num_workers=opt.data.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, opt=opt),
        )


    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.01, betas=(0.9, 0.99))

    total_steps = opt.num_epochs * len(train_dataloader) // opt.gradient_accumulation_steps
    def _lr_lambda(current_step, warmup_ratio=opt.warmup_ratio, num_cycles=0.5, min_ratio=0.5):
        progress = current_step / max(1, total_steps)
        if warmup_ratio > 0 and progress < warmup_ratio:
            return progress / warmup_ratio
        progress = (progress - warmup_ratio) / (1 - warmup_ratio)
        return max(min_ratio, min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    if opt.resume is not None and not opt.ft:
        ckpt_stat_dir=os.path.dirname(opt.resume)
        ckpt_stat_path=os.path.join(ckpt_stat_dir, 'model_state.pth')
        if os.path.exists(ckpt_stat_path):
            logger.info(f'state path exist ! loading optimizer and scheduler')
            checkpoint_stat = torch.load(ckpt_stat_path, map_location='cpu')
            optimizer.load_state_dict(checkpoint_stat['optimizer'])
            scheduler.load_state_dict(checkpoint_stat['scheduler'])
            opt.train.resume_epoch=checkpoint_stat['epoch']+1
            logger.info(f'resume epoch: {opt.train.resume_epoch}')
        else:
            logger.info(f'[WARNING] no state, check you resume epoch {opt.resume_epoch}')
    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # wandb
    if opt.use_wandb and accelerator.is_main_process:
        import wandb # set WAND_API_KEY in env
        wandb.init(entity="you entity", project='MeshSilksong', name=opt.workspace.replace('workspace_', ''), config=opt) 

    # loop
    old_save_dirs = []
    best_loss = 1e9
    for epoch in range(opt.train.resume_epoch, opt.train.num_epochs):
        if epoch%opt.save_epoch==0:
            save_dir = os.path.join(opt.workspace, f'ep{epoch:04d}')
            os.makedirs(save_dir, exist_ok=True)
        if opt.resample:
            sampler.update_epoch(epoch)
            beta_num = sampler.get_distribution_info()['beta']
            logger.info(f'beta {beta_num} for epoch {epoch}')
        # train
        if not opt.debug_eval:
            model.train()
            # if opt.cond_mode == 'point_miche' and 'multi' in opt.workspace:
            #     model._set_static_graph()
            total_loss = 0
            t_start = time.time()
            for i, data in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    
                    optimizer.zero_grad()
                    codes=data['tokens']
                    pc=data['conds']

                    loss = model(codes=codes, pc=pc)

                    accelerator.backward(loss)

                    # gradient clipping
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)


                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.detach()

                if accelerator.is_main_process:
                    # logging
                    if i % 10 == 0:
                        mem_free, mem_total = torch.cuda.mem_get_info()
                        log = f"{epoch:03d}:{i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} loss: {loss.item():.6f}"
                        logger.info(log)
                    
                        
            total_loss = accelerator.gather_for_metrics(total_loss).mean().item()
            torch.cuda.synchronize()
            t_end = time.time()
            if accelerator.is_main_process:
                total_loss /= len(train_dataloader)
                logger.info(f"Train epoch: {epoch} loss: {total_loss:.6f} time: {(t_end - t_start)/60:.2f}min")
            
                # wandb
                if opt.use_wandb:
                    wandb.log({'train_loss': total_loss, 'epoch': epoch})
                    wandb.log({'lr': scheduler.get_last_lr()[0], 'epoch': epoch})
                if opt.use_wandb and opt.resample:
                    wandb.log({'sampler_beta': sampler.get_distribution_info()['beta'], 'epoch': epoch})
            # checkpoint
            if epoch % opt.save_epoch == 0 or epoch == opt.num_epochs - 1:
                accelerator.wait_for_everyone()
                print(f'epoch {epoch} done, saving')
                accelerator.save_model(model, save_dir)
                model_state={
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }
                if accelerator.is_main_process:
                    # save state
                    state_path=os.path.join(save_dir, 'model_state.pth')
                    torch.save(model_state, state_path)
                    # symlink latest checkpoint for linux
                    if os.name == 'posix':
                        os.system(f'ln -sf {os.path.join(f"ep{epoch:04d}", "model.safetensors")} {os.path.join(opt.workspace, "model.safetensors")}')
                    # copy best checkpoint
                    if total_loss < best_loss:
                        best_loss = total_loss
                        shutil.copy(os.path.join(save_dir, 'model.safetensors'), os.path.join(opt.workspace, 'best.safetensors'))
                    old_save_dirs.append(save_dir)
                    if len(old_save_dirs) > 3 and opt.dataset!='debug_one': # save at most 3 ckpts
                        shutil.rmtree(old_save_dirs.pop(0))
        else:
            if accelerator.is_main_process:
                logger.info(f"epoch: {epoch} skip training for debug !!!")

        # eval
        print(f'evaluating, eval_mode {opt.eval_mode}')
        if opt.eval_mode == 'loss' and epoch % opt.save_epoch ==0:
            model.eval()
            with torch.no_grad():
                total_loss = 0
                for i, data in enumerate(test_dataloader):
                    codes=data['tokens']
                    pc=data['conds']

                    loss = model(codes=codes, pc=pc)

                        
                    total_loss += loss.detach()

                total_loss = accelerator.gather_for_metrics(total_loss).mean()
                if accelerator.is_main_process:
                    total_loss /= len(test_dataloader)
                    logger.info(f"Eval epoch: {epoch} loss: {total_loss:.6f}")
                    if opt.use_wandb:
                        wandb.log({'eval_loss': total_loss, 'epoch': epoch})
                    
        # else:
        #     pass
        #     if accelerator.is_main_process:
        #         logger.info(f"Eval epoch: {epoch} skip evaluation.")
            

if __name__ == "__main__":
    main()
