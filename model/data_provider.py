
import random
import trimesh
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import kiui
from kiui.mesh_utils import clean_mesh, decimate_mesh
from kiui.op import recenter
import pandas as pd
import logging
import traceback
import re
from silkutils import ss_platform
from silkutils.silksong_tokenization import get_tokenizer_silksong,  tokenize_mesh_ss
from silkutils.meshdata.mesh_io import load_mesh_nonorm, normalize_mesh

class ProgressivelyBalancedSampler(Sampler):
    def __init__(self, opt, dataset, face_delta, initial_beta=0.0, final_beta=1.0, epochs=100):
        """
        Args:
            dataset: items is needed
            face_delta: class 
            initial_beta: item-balance
            final_beta: class-balance
            epochs: train
        """
        self.dataset = dataset
        self.face_delta = face_delta
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.epochs = epochs
        self.current_epoch = 0
        
        # calculate labels
        if opt.ss_mode==-1:
            face_nums = [item['face_num'] for item in dataset.items]
        else:
            face_nums = [item['face_num_process'] for item in dataset.items]
        self.labels = [int(face_num // face_delta) for face_num in face_nums]
        
        # original distribution
        unique_labels, label_counts = np.unique(self.labels, return_counts=True)
        self.num_samples = len(dataset.items)
        
        # label information
        self.unique_labels = unique_labels
        self.label_counts = label_counts
        self.label_probs = label_counts / self.num_samples  # original
        
        # class-balance
        self.balanced_probs = np.ones_like(label_counts) / len(label_counts)
        
        # mapping from label-to-item
        self.label_to_indices = {label: [] for label in unique_labels}
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
            
        # init weights
        self.update_weights()
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch
        self.update_weights()
    
    def get_beta(self):
        # calculate current beta
        if self.epochs <= 1:
            return self.final_beta
        
        # linear degrade
        progress = min(self.current_epoch / (self.epochs - 1), 1.0)
        beta = self.initial_beta + (self.final_beta - self.initial_beta) * progress
        return beta
    
    def update_weights(self):
        
        beta = self.get_beta()
        
        # mix two distribution with beta
        mixed_probs = beta * self.balanced_probs + (1 - beta) * self.label_probs
        
        # calculate weights of items
        self.weights = torch.zeros(self.num_samples)
        for label, prob in zip(self.unique_labels, mixed_probs):
            indices = self.label_to_indices[label]
            weight_per_sample = prob / len(indices)
            self.weights[indices] = weight_per_sample
            
        # normalize
        self.weights = self.weights / self.weights.sum()
    
    def __iter__(self):
        # sample with current weight
        indices = np.random.choice(
            len(self.labels),
            size=self.num_samples,
            replace=True,
            p=self.weights.numpy()
        )
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def get_distribution_info(self):
        # for debugging
        beta = self.get_beta()
        return {
            'epoch': self.current_epoch,
            'beta': beta,
            'label_distribution': {
                f"faces_{label*self.face_delta}-{(label+1)*self.face_delta-1}": count
                for label, count in zip(self.unique_labels, self.label_counts)
            },
            'current_sampling_weights': self.weights.numpy()
        }


def sample_pc(mesh, pc_num, with_normal=True, aug=True):
    
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points

    points, face_idx = mesh.sample(50000, return_index=True)
    if aug and random.random() < 0.5:
        points += np.random.randn(*points.shape) * 0.01
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    
    return pc_normal

def get_table_list(subset_name, opt):
    xlsx_dir=opt.xlsx_dir
    testset_xlsx_dir=opt.testset_xlsx_dir
    testset_prefix=opt.testset_prefix
    version=1 if opt.ss_mode==-1 else opt.ss_mode
    xlsx_name=f'meta_all_{subset_name}_res{opt.discrete_bins}_v{version:02}_mergeall_filter{opt.data_filter_cnt:02}.xlsx'
    test_pattern = fr'^{testset_prefix}_meta_all_{subset_name}_res{opt.discrete_bins}_v{version:02}_mergeall_filter\d{{2}}_sample\d{{4}}_sb\d{{2}}\.xlsx$'
    platform_dir=ss_platform.get_base_dir_platform(subset_name)
    df = pd.read_excel(os.path.join(xlsx_dir, xlsx_name))

    test_obj_paths = set()
    if os.path.exists(testset_xlsx_dir):
        for filename in os.listdir(testset_xlsx_dir):
            if re.match(test_pattern, filename):
                test_df = pd.read_excel(os.path.join(testset_xlsx_dir, filename))
                test_obj_paths.update(test_df['id'].tolist())

    print(f'{subset_name}: read {len(df)} data')
    print(f'{subset_name}: testset num {len(test_obj_paths)}')

    if test_obj_paths:
        df = df[~df['id'].isin(test_obj_paths)]

    selected_columns = df[['obj_path', 'token_length','face_num_process','face_num']]
    dict_list = selected_columns.to_dict(orient='records')
    for ele in dict_list:
        ele['obj_path']=os.path.join(platform_dir, ele['obj_path'])
    print(f'{subset_name}: final num {len(dict_list)}')
    return dict_list

def get_table_list_test(subset_name, opt):
    xlsx_dir=opt.xlsx_dir
    version=1 if opt.ss_mode==-1 else opt.ss_mode
    
    test_pattern = fr'^testset_meta_all_{subset_name}_res{opt.discrete_bins}_v{version:02}_mergeall_filter02_sample\d{{4}}_sb\d{{2}}\.xlsx$'
    platform_dir=ss_platform.get_base_dir_platform(subset_name)

    test_dfs=[]
    for filename in os.listdir(xlsx_dir):
        if re.match(test_pattern, filename):
            test_df = pd.read_excel(os.path.join(xlsx_dir, filename))
            test_dfs.append(test_df)
            print(f'find {os.path.join(xlsx_dir, filename)}')

    df=test_dfs[0]

    selected_columns = df[['obj_path', 'token_length','face_num_process']]
    dict_list = selected_columns.to_dict(orient='records')
    for ele in dict_list:
        ele['obj_path']=os.path.join(platform_dir, ele['obj_path'])
    print(f'test set num {len(dict_list)}')
    return dict_list


# global_nonmani_cnt=0
# global_info_list=[]
# global_max_time=0
# global_max_time_path=0
class SSDataset(Dataset):
    def __init__(self, opt, training=True, tokenizer=None):
        # mixed dataset
        self.opt = opt
        self.training = training
        self.data_subsets=[subset for subset in opt.data_subsets.split('*')]

        self.items = []
        for subset_name in self.data_subsets:
            subset_list=get_table_list(subset_name=subset_name, opt=self.opt)
            if opt.ss_mode==-1: # For edgeRunner
                for item in subset_list:
                    if item['face_num'] <= 4000:
                        self.items.append(item)
            else:
                for item in subset_list:
                    if item['token_length'] <= opt.max_seq_length:
                        self.items.append(item)
            print(f'init {subset_name} done, accum num: {len(self.items)}')
  
        if self.training:
            self.items = self.items[:-self.opt.testset_size]
        else:
            self.items = self.items[-self.opt.testset_size:]

        self.vids = list(range(33, 40)) + list(range(12, 24))
        # tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        non_mani_process=True
        results = {}
        this_tokenizer,_ = get_tokenizer_silksong()
        path = self.items[idx]['obj_path']
        # data_start_time=time.time()
        iter_cnt=0
        while True:
            iter_cnt+=1
            if iter_cnt>=2:
                print(f'iter_cnt>=2: {iter_cnt}')
            try:
                
                ### scale augmentation (not for image condition)
                if self.opt.use_scale_aug and self.training and iter_cnt<=2:
                    bound = np.random.uniform(0.75, 0.95)
                    border_ratio = 0.2 + 0.95 - bound
                else:
                    bound = 0.95
                    border_ratio = 0.2

                ### rotation augmentation
                if self.training and self.opt.use_rot_aug and iter_cnt<=2:
                    vid = 36 # no use
                    # azimuth = np.random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], 1)[0]
                    azimuth = np.random.uniform(0, 360)
                else:
                    vid = 36
                    azimuth = 0

                v, f = load_mesh_nonorm(path) # not norm, not clean
                # already cleaned
                v, f = clean_mesh(v, f, min_f=0, min_d=0, remesh=False, verbose=False)

                # face may exceed max_face_length, stats maybe inaccurate...
                if f.shape[0] > self.opt.max_face_length:
                    raise ValueError(f"{f.shape[0]} exceeds face limit: {self.opt.max_face_length}")

                # decimate mesh augmentation
                if self.opt.use_decimate_aug and self.training and iter_cnt<=2:
                    if f.shape[0] >= 200 and random.random() < 0.5:
                        # at most decimate to 25% of original faces.
                        target = np.random.randint(max(100, f.shape[0] // 4), f.shape[0])
                        # print(f'[INFO] decimating {f.shape[0]} to {target} faces...')
                        v, f = decimate_mesh(v, f, target=target, verbose=False)

                # rotate augmentation
                if azimuth != 0 :
                    roty = np.stack([
                        [np.cos(np.radians(-azimuth)), 0, np.sin(np.radians(-azimuth))],
                        [0, 1, 0],
                        [-np.sin(np.radians(-azimuth)), 0, np.cos(np.radians(-azimuth))],
                    ])
                    v = v @ roty.T

                # normalize after rotation in case of oob (augment scale)
                v = normalize_mesh(v, bound=bound)

                # point cloud cond
                
                mesh = trimesh.Trimesh(vertices=v, faces=f)
                cond = sample_pc(mesh, pc_num=4096, with_normal=True, aug=True) # N,6
                
                
                coords, meta_data = tokenize_mesh_ss(this_tokenizer, v, f) # [M]

                # truncate to max length instead of dropping
                if coords.shape[0] >= self.opt.max_seq_length:
                    raise ValueError(f"{coords.shape[0]} exceeds token limit: {self.opt.max_seq_length}.")
                del v
                del f
                del mesh
                break

            except Exception as e:
                
                # raise e # DANGEROUS, may cause infinite loop
                print(f'[dataiter wrong]: {path}, {str(e)}')
                
                idx = np.random.randint(0, len(self.items))
                path = self.items[idx]['obj_path']
                
        # global global_max_time
        # data_delta_time=time.time()-data_start_time
        # if data_delta_time>global_max_time:
        #     global_max_time=data_delta_time
        #     global_max_time_path=path
        # print(f'data {idx}, max edge num: {meta_data.max_edge_num}, time: {data_delta_time}s, {path}')
        # logging.info(f'data {idx} time: {data_delta_time}s, {path}')
        # ss_utils.save_point_cloud_to_ply(cond, 'debug_dataset_iter')
        results['cond'] = cond # [3, H, W] for image, [N, 6] for point
        results['coords'] = coords # [M]
        results['len'] = coords.shape[0] # [1]
        # results['num_faces'] = f.shape[0] # [1] # should be after process?
        results['num_faces'] = meta_data.face_num_p # [1] # should be after process?
        results['num_CC'] = meta_data.CC_num_valid # [1] # should be after process?
        results['azimuth'] = azimuth # [1]
        results['path'] = path

        # a custom collate_fn is needed for padding and masking
        
        return results
    
    
class DebugOneDataset(Dataset):
    def __init__(self, opt, training=True, tokenizer=None):
        # mixed dataset
        self.opt = opt
        self.training = training

        data_one={}
        # data_one['obj_path']='/public/home/group_gaosh/gaochao/3ddatasets/debug_one/boat.obj'
        data_one['obj_path']='/workspace/bpt/debug_one/boat.obj'
        data_one['token_length']=100
        data_one['face_num_process']=2000
        data_one['non_mani_process']=False
        self.items = []
        self.items.append(data_one)
        print(f'init done: {self.items[0]}')
        
        # gobj vid candidates
        self.vids = list(range(33, 40)) + list(range(12, 24))
        # self.vids = list(range(28, 40)) + list(range(0, 24))

        # tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        results = {}
        this_tokenizer,_ = get_tokenizer_silksong()
        path = self.items[idx]['obj_path']
        data_start_time=time.time()
        iter_cnt=0
        while True:
            iter_cnt+=1
            try:
                
                non_mani_process=True
                ### scale augmentation (not for image condition)
                if self.opt.use_scale_aug and self.training:
                # if False:
                    bound = np.random.uniform(0.75, 0.95)
                    border_ratio = 0.2 + 0.95 - bound
                else:
                    bound = 0.95
                    border_ratio = 0.2
             

                ### rotation augmentation
                if self.training and self.opt.use_rot_aug:
                    vid = 36 # no use
                    azimuth = np.random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], 1)[0]
                else:
                    vid = 36
                    azimuth = 0
        
                ### load mesh
                mesh_path = path
                v, f = load_mesh_nonorm(mesh_path) # not norm, not clean
                # already cleaned
                v, f = clean_mesh(v, f, min_f=0, min_d=0, remesh=False, verbose=False)


                # decimate mesh augmentation
                if self.opt.use_decimate_aug and self.training and iter_cnt<4:
                    if f.shape[0] >= 200 and random.random() < 0.5:
                        # at most decimate to 25% of original faces.
                        target = np.random.randint(max(100, f.shape[0] // 4), f.shape[0])
                        # print(f'[INFO] decimating {f.shape[0]} to {target} faces...')
                        v, f = decimate_mesh(v, f, target=target, verbose=False)

                # rotate augmentation
                if azimuth != 0 :
                    roty = np.stack([
                        [np.cos(np.radians(-azimuth)), 0, np.sin(np.radians(-azimuth))],
                        [0, 1, 0],
                        [-np.sin(np.radians(-azimuth)), 0, np.cos(np.radians(-azimuth))],
                    ])
                    v = v @ roty.T

                # normalize after rotation in case of oob (augment scale)
                v = normalize_mesh(v, bound=bound)

                
                mesh = trimesh.Trimesh(vertices=v, faces=f)
                cond = sample_pc(mesh, pc_num=4096, with_normal=True, aug=True) # N,6

                coords, meta_data = tokenize_mesh_ss(this_tokenizer, v, f) # [M]

                # truncate to max length instead of dropping
                if coords.shape[0] >= self.opt.max_seq_length:
                    # print(f'[WARN] {path}: coords.shape[0] > {self.opt.max_seq_length}, truncating...')
                    # coords = coords[:self.opt.max_seq_length]
                    raise ValueError(f"{coords.shape[0]} exceeds token limit.")
                    
                break

            except Exception as e:
                
                # raise e # DANGEROUS, may cause infinite loop
                print(f'[dataiter wrong], {str(e)}')
                traceback.print_exc()
                idx = np.random.randint(0, len(self.items))
                path = self.items[idx]['obj_path']
                
        # print(f'data {idx}, max edge num: {meta_data.max_edge_num}, time: {data_delta_time}s')
        results['cond'] = cond # [3, H, W] for image, [N, 6] for point
        results['coords'] = coords # [M]
        results['len'] = coords.shape[0] # [1]
        # results['num_faces'] = f.shape[0] # [1] # should be after process?
        results['num_faces'] = meta_data.face_num_p # [1] # should be after process?
        results['num_CC'] = meta_data.CC_num_valid # [1] # should be after process?
        results['azimuth'] = azimuth # [1]
        results['path'] = path

        # a custom collate_fn is needed for padding and masking
        
        return results


def collate_fn(batch, opt):

    # conds
    conds = [item['cond'] for item in batch]
    num_faces = [item['num_faces'] for item in batch]
    num_CC = [item['num_CC'] for item in batch]
    azimuths = [item['azimuth'] for item in batch]

    # get max len of this batch
    max_len = max([item['len'] for item in batch])
    max_len = min(max_len, opt.max_seq_length)

    
    # pad or truncate to max_len, and prepare masks
    tokens = []
    labels = []
    masks = []
    num_tokens = []
    for item in batch:
        
        if max_len >= item['len']:
            pad_len = max_len - item['len']

            tokens.append(np.concatenate([
                item['coords'], # mesh tokens
                np.full((pad_len,), opt.pad_token_id), # padding
            ], axis=0)) # [pad to max]

            num_tokens.append(item['len'])
        else:
            raise Exception('?')
            

    results = {}
    results['conds'] = torch.from_numpy(np.stack(conds, axis=0)).float()
    results['num_faces'] = torch.from_numpy(np.stack(num_faces, axis=0)).long()
    results['num_CC'] = torch.from_numpy(np.stack(num_CC, axis=0)).long()
    results['num_tokens'] = torch.from_numpy(np.stack(num_tokens, axis=0)).long()
    results['azimuths'] = torch.from_numpy(np.stack(azimuths, axis=0)).long()
    results['tokens'] = torch.from_numpy(np.stack(tokens, axis=0)).long()
    results['paths'] = [item['path'] for item in batch]

    return results

if __name__ == "__main__":
    import tyro
    from config.options import AllConfigs
    from functools import partial
    
    opt = tyro.cli(AllConfigs)
    kiui.seed_everything(opt.seed)
    
    logging.basicConfig(
    filename='debug_dataset_iter/provider_test_file.log',  # log name
    level=logging.INFO,          # log level
    format='%(asctime)s - %(levelname)s - %(message)s' 
    )
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)  # show WARNING+
    # tokenizer
    tokenizer, _ = get_tokenizer_silksong(opt)

    dataset = SSDataset(opt, training=True, tokenizer=tokenizer)
    # dataset = DebugOneDataset(opt, training=True, tokenizer=tokenizer)
    print(f'len dataset: {len(dataset)}')
    decoding_wrong_time=0
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=partial(collate_fn, opt=opt),
    )

    for i in range(4000):
        if i%100 ==0:
            print(f'--------------------------- iter {i} done ---------------------------')
        results = next(iter(dataloader))

        kiui.lo(results['conds'], results['tokens'], results['azimuths'])
        

        # restore mesh
        for b in range(len(results['masks'])):
            # print(f'iter {i} b {b}, conds: {results['conds'][b].shape}, tokens: {results['tokens'][b].shape}, azimuthis: {results['azimuths'][b].shape}')
            masks = results['masks'][b].numpy()
            tokens = results['labels'][b].numpy()[masks][1+opt.num_cond_tokens:-1]

            # write obj using the original order to check face orientation
            try:
                vertices_ori, faces_ori = detokenize_mesh_ss(tokens, opt.discrete_bins, opt.ss_mode, tokenizer=tokenizer)
            except Exception as e:
                print(f'decoding wrong: {str(e)}')
                logging.info(f'[decoding wrong]: {str(e)}')
                decoding_wrong_time+=1
                continue
            mode_flag= 'silksong' if opt.ss_mode >=0 else 'edgerunner'
            # ss_utils.write_obj(vertices_ori, faces_ori, os.path.join('debug_dataset_iter',f'{mode_flag}_iter{i:02}_b{b:02}.obj'))
            ss_utils.write_ply(vertices_ori, faces_ori, os.path.join('debug_dataset_iter',f'{mode_flag}_iter{i:02}_b{b:02}.obj'))

            # kiui.lo(tokens, faces)
            # print(results['paths'][b])
            # print(f'[INFO] tokens: {tokens.shape[0]}, faces: {faces_ori.shape[0]}, ratio={100 * tokens.shape[0] / (9 * faces_ori.shape[0]):.2f}%')
    # print(f'global nonmanifold cnt: {global_nonmani_cnt}')
    # print(f'decoding wrong time: {decoding_wrong_time}')
    # print(f'global info list {global_info_list}')
    # print(f'max time{global_max_time}, {global_max_time_path}')