import os
import tyro
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import ipdb
import kiui
import trimesh
from kiui.op import recenter
from kiui.mesh_utils import clean_mesh
import traceback
from config.options import AllConfigs
from silkutils.meshdata.mesh_io import write_obj, trans_write_tokens, trans_compare_write_tokens, trans_write_tokens_direct, write_ply_fix
from model.model import SSMeshTransformer
from silkutils.silksong_tokenization import get_tokenizer_silksong, detokenize_mesh_ss
from datetime import datetime
from model.data_provider_infer import InferDataset, joint_filter, max_filter, collate_fn_infer
from x_transformers.autoregressive_wrapper import top_p, top_k

opt = tyro.cli(AllConfigs)

kiui.seed_everything(opt.seed)
# tokenizer
tokenizer, _ = get_tokenizer_silksong()

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

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')

    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    raise Exception('please set resume path')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().eval().to(device)

num_params = sum([param.nelement() for param in model.decoder.parameters()])
print('Number of parameters: %.2f M' % (num_params / 1e6))

assert opt.test_path_input is not None
if os.path.isdir(opt.test_path_input):
    file_paths = glob.glob(os.path.join(opt.test_path_input, "*"))
else:
    file_paths = [opt.test_path_input]

now=datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H:%M:%S")

ckpt_name=opt.resume

if 'best' not in ckpt_name:
    ep=ckpt_name.split('/')[-2]
    exp_name=ckpt_name.split('/')[-3]
else:
    ep='best'
    exp_name=ckpt_name.split('/')[-2]
fd1=opt.test_path_input.split('/')[-2]
fd2=opt.test_path_input.split('/')[-1]

max_f=None
if opt.infer.max_filter:
    max_f='maxf'
else:
    max_f='nomax'

save_folder=f'{max_f}_{fd1}_{fd2}_{exp_name}_{ep}_{formatted_time}'
# for path in file_paths:
#     process(opt, path, save_folder, tokenizer)
os.makedirs(opt.workspace, exist_ok=True)
target_folder=os.path.join(opt.workspace, save_folder)
os.makedirs(target_folder, exist_ok=True)
method_name='silksong'

infer_dataset=InferDataset(input_type='mesh', input_list=sorted(file_paths))

infer_dataloader=torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=opt.infer.infer_batch,
        drop_last = False,
        shuffle = False,
        collate_fn=collate_fn_infer,
)

with torch.no_grad():
    for it, data in enumerate(infer_dataloader):
        if opt.infer.max_filter:
            codes = model.generate(
                batch_size = opt.infer.infer_batch,
                temperature = opt.infer.temperature,
                pc = data['pc_normal'].cuda().half(),
                filter_logits_fn = max_filter,
                filter_kwargs = dict(k=1),
                return_codes=True,
            )
        else:
            codes = model.generate(
                batch_size = opt.infer.infer_batch,
                temperature = opt.infer.temperature,
                pc = data['pc_normal'].cuda().half(),
                filter_logits_fn = joint_filter,
                filter_kwargs = dict(k=50, p=0.95),
                return_codes=True,
            )

        coords = []
        
        # decoding codes to coordinates
        for i in range(len(codes)):
            code = codes[i]
            full_path = data['full_path'][i]
            code = code[code != model.pad_id].cpu().numpy()
            try:
                verts, faces = detokenize_mesh_ss(tokenizer, code, colorful=True, mani_fix=True)
                coords.append({'v':verts, 'f': faces, 'tokens': code})
            except Exception as e:
                print(f'path generation failed: {full_path}, {str(e)}')
                traceback.print_exc()
                coords.append({'tokens': code})

        # convert coordinates to mesh
        for i in range(opt.infer.infer_batch):
            uid = data['uid'][i]
            pcd = data['pc_normal'][i].cpu().numpy()
            gt_v= data['gt_mesh'][i]['v']
            gt_f= data['gt_mesh'][i]['f']
            if data['gt_mesh'][i]['tokens'] is not None:
                gt_token= data['gt_mesh'][i]['tokens']
            else:
                gt_token=None

            pc_save_name=f'{uid}_{method_name}_POINT.ply'
            gt_save_name=f'{uid}_{method_name}_GT.obj'
            gt_token_save_name=f'{uid}_{method_name}_GT_tokens.txt'

            # save point
            point_cloud = trimesh.points.PointCloud(pcd[..., 0:3])
            point_cloud.export(f'{target_folder}/{pc_save_name}', "ply")

            # save gt
            gt_mesh=trimesh.Trimesh(vertices=gt_v, faces=gt_f)
            gt_mesh.export(os.path.join(target_folder, gt_save_name))
            if gt_token is not None:
                trans_write_tokens_direct(tokens=gt_token, filename=os.path.join(target_folder, gt_token_save_name.replace('.txt', f'_len{len(gt_token):05}.txt')), engine=tokenizer)
            
            # save pred
            pred_dic = coords[i]
            pred_token_save_name=f'{uid}_{method_name}_gen_tokens.txt'
            trans_write_tokens_direct(tokens=pred_dic['tokens'], filename=os.path.join(target_folder, pred_token_save_name.replace('.txt', f'_len{len(pred_dic["tokens"]):05}.txt')), engine=tokenizer)
            np.save(os.path.join(target_folder, pred_token_save_name.replace('.txt', f'_len{len(pred_dic["tokens"]):05}.npy')), pred_dic['tokens'])
            if gt_token is not None:
                trans_compare_write_tokens(tokens=pred_dic['tokens'], tokens_gt=gt_token, filename=os.path.join(target_folder, gt_token_save_name.replace('.txt', f'_len{len(gt_token):05}.txt')), engine=tokenizer)

            if 'v' not in pred_dic:
                continue    
            pred_vert_6_layerColor=pred_dic['v'][0]
            pred_vert_6_ccColor=pred_dic['v'][1]
            
            pred_face=pred_dic['f'][0][:, :3] # decoding from GPT output
            pred_face_F=pred_dic['f'][1][:, :3] # auto fix 
            pred_save_name=f'{uid}_{method_name}_gen.obj'
            

            F_dir=target_folder
            os.makedirs(F_dir, exist_ok=True)
            # the view of connected components
            write_obj(pred_vert_6_ccColor, pred_face_F, os.path.join(F_dir, pred_save_name.replace('.obj','_CCcolor.obj')))
            
            # if you want to visualize layer view (RGB contour lines)
            # write_obj(pred_vert_6_layerColor, pred_face_F, os.path.join(F_dir, pred_save_name.replace('.obj','_layercolor.obj')))
            
            
            

