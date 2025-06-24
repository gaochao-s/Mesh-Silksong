import trimesh
import numpy as np
from kiui.mesh_utils import clean_mesh
from x_transformers.autoregressive_wrapper import top_p, top_k
from silkutils.silksong_tokenization import get_tokenizer_silksong,  tokenize_mesh_ss
from silkutils.meshdata.mesh_io import load_mesh_nonorm, normalize_mesh
import model.data_provider as data_pro
import torch
import traceback

class Dataset:
    '''
    A toy dataset for inference
    '''
    def __init__(self, input_type, input_list):
        super().__init__()
        self.data = []
        if input_type == 'pc_normal':
            for input_path in input_list:
                # load npy
                cur_data = np.load(input_path)
                # sample 4096
                assert cur_data.shape[0] >= 4096, "input pc_normal should have at least 4096 points"
                idx = np.random.choice(cur_data.shape[0], 4096, replace=False)
                cur_data = cur_data[idx]
                self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0]})

        elif input_type == 'mesh':
            mesh_list, pc_list = [], []
            for input_path in input_list:
                # sample point cloud and normal from mesh
                ####ss
                v, f = load_mesh_nonorm(input_path)
                v, f = clean_mesh(v, f, min_f=0, min_d=0, remesh=False, verbose=False)
                v = normalize_mesh(v, bound=0.95)
                cur_data = trimesh.Trimesh(vertices=v, faces=f)
                #### ss - bpt
                # cur_data = trimesh.load(input_path, force='mesh')
                # cur_data = apply_normalize(cur_data)
                #### bpt
                mesh_list.append(cur_data)
                pc_list.append(sample_pc(cur_data, pc_num=4096, with_normal=True))

            for input_path, cur_data in zip(input_list, pc_list):
                self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0]})
                
        print(f"dataset total data samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = {}
        data_dict['pc_normal'] = self.data[idx]['pc_normal']
        data_dict['uid'] = self.data[idx]['uid']

        return data_dict
    
class InferDataset:
    '''
    A toy dataset for inference
    '''
    def __init__(self, input_type, input_list):
        super().__init__()
        self.data = []
        self.tokenizer, _=get_tokenizer_silksong()
        if input_type == 'mesh':
            pc_list = []
            gt_mesh = []
            for input_path in input_list:
                # sample point cloud and normal from mesh
                v, f = load_mesh_nonorm(input_path)
                v, f = clean_mesh(v, f, min_f=0, min_d=0, remesh=False, verbose=False)
                v = normalize_mesh(v, bound=0.95)
                mesh = trimesh.Trimesh(vertices=v, faces=f)
                points = data_pro.sample_pc(mesh, pc_num=4096, with_normal=True, aug=False)
                try:
                    gt_token, _=tokenize_mesh_ss(tokenizer=self.tokenizer, vertices=v, faces=f)
                except Exception as e:
                    print(f'[DatasetInfer] {input_path}, {str(e)}')
                    traceback.print_exc()
                    gt_token=None
                pc_list.append(points)
                gt_mesh.append({"v":v, "f":f, 'tokens': gt_token})
                print(f'{input_path} Done')

            for input_path, cur_data, gt in zip(input_list, pc_list, gt_mesh):
                self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0], 'full_path': input_path, 'gt_mesh': gt})
        else:
            raise Exception('not implement') 
        
        print(f"infer dataset total data samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = {}
        data_dict['pc_normal'] = self.data[idx]['pc_normal']
        data_dict['uid'] = self.data[idx]['uid']
        data_dict['full_path'] = self.data[idx]['full_path']
        data_dict['gt_mesh'] = self.data[idx]['gt_mesh']

        return data_dict

def joint_filter(logits, k = 50, p=0.95):
    logits = top_k(logits, k = k)
    logits = top_p(logits, thres = p)
    return logits

def max_filter(logits, k = 1):
    logits = top_k(logits, k = k)
    return logits

def apply_normalize(mesh):
    '''
    normalize mesh to [-1, 1]
    '''
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale * 2 * 0.95)

    return mesh



def sample_pc(mesh_path, pc_num, with_normal=False):

    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    mesh = apply_normalize(mesh)
    
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points

    points, face_idx = mesh.sample(50000, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    
    return pc_normal


def collate_fn_infer(batch):

    # conds
    conds = [item['pc_normal'] for item in batch]
            

    results = {}
    results['pc_normal'] = torch.from_numpy(np.stack(conds, axis=0)).float()
    results['uid'] = [item['uid'] for item in batch]
    results['full_path']=[item['full_path'] for item in batch]
    results['gt_mesh']=[item['gt_mesh'] for item in batch]

    return results