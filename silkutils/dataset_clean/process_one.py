import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from silksong_tokenization import get_tokenizer_silksong
from meshdata.mesh_io import load_mesh, load_mesh_modify, write_obj, write_tokens_ori, quick_demo, trans_write_tokens, trans_compare_write_tokens, write_ply_fix
from ss_platform import get_base_dir_platform, get_base_dir_rel
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import kiui
import ipdb
import time
import func_timeout
from func_timeout import func_set_timeout



def get_save_basename(dataset_name, obj_save_name, mode):
    return f'{dataset_name}_{obj_save_name}_v{mode:02}'

# @func_set_timeout(100)
def process_one(**kwargs):
    max_face_after_trimesh=kwargs.get('max_face_num', 16000) # max: 16k
    
    dataset_name=kwargs.get('dataset_name')
    
    obj_path=kwargs.get('obj_path')
    obj_save_name=kwargs.get('obj_save_name')
    save_dir_npy=kwargs.get('save_dir_npy')
    save_dir_meta=kwargs.get('save_dir_meta')
    save_dir_debug=kwargs.get('save_dir_debug')
    resolution=kwargs.get('resolution')
    mode=kwargs.get('mode')
    face_type=kwargs.get('face_type')
    debugging=kwargs.get('debugging')

    structure_limit_kwargs={
        'NM_max_edge_graph': kwargs.get('NM_max_edge_graph', 50),
        'NM_max_nonmani_verts': kwargs.get('NM_max_nonmani_verts', 300),
        'min_CC_face': kwargs.get('min_CC_face', 3),
        'max_face_num_p': kwargs.get('max_face_num_p', 12000)
    }
    
    if debugging is None:
        debugging=False
        # print('debugging false')
    elif debugging == False:
        pass
    else:
        debugging=True
        print('debugging true')
    S_time=time.time()

    save_base_name=get_save_basename(dataset_name=dataset_name, obj_save_name=obj_save_name, mode=mode)

    if debugging:
        save_dir_platform='/workspace/MeshSilksong/silkutils/Debugs/debug_output'
        save_dir_platform=os.path.join(save_dir_platform, save_base_name)
    else:
        save_dir_platform=get_base_dir_platform(dataset_name)

    if face_type=='triangle':
        # support .obj, .glb, .ply ... as well as trimesh support
        obj_path_platform=os.path.join(get_base_dir_platform(dataset_name), obj_path)
        try:
            vertices, faces = load_mesh(obj_path_platform, clean=True)
        except Exception as e:
            raise Exception('[E] loading Failed')
    elif face_type=='multi':
        # support .obj
        obj_path_platform=os.path.join(get_base_dir_platform(dataset_name), obj_path)
        try:
            vertices, faces = load_mesh_modify(obj_path_platform)
        except Exception as e:
            raise Exception('[E] loading Failed')
    elif face_type=='quick_demo':
        vertices, faces = quick_demo(obj_path)
    else:
        raise Exception('[E] wrong face type')
    
    if len(faces)>max_face_after_trimesh:
        # pass
        raise Exception(f'[E] too many faces {len(faces)} > {max_face_after_trimesh}!')


    tokensO_filename=f'tokensO_{save_base_name}.txt' # readable tokens
    tokensE_filename=f'tokensE_{save_base_name}.npy' # encode vertex
    tokensT_filename=f'tokensT_{save_base_name}.txt' # translate from encoded
    meta_filename=f'meta_{save_base_name}.txt'
    decode_filename=f'decode_{save_base_name}.obj'
    
    tokensO_filedir=os.path.join(save_dir_platform, save_dir_debug)
    tokensE_filedir=os.path.join(save_dir_platform, save_dir_npy)
    tokensT_filedir=os.path.join(save_dir_platform, save_dir_debug)
    meta_filedir=os.path.join(save_dir_platform, save_dir_meta)
    decode_filedir=os.path.join(save_dir_platform, save_dir_debug)

    os.makedirs(tokensO_filedir, exist_ok=True)
    os.makedirs(tokensE_filedir, exist_ok=True)
    os.makedirs(tokensT_filedir, exist_ok=True)
    os.makedirs(meta_filedir, exist_ok=True)
    os.makedirs(decode_filedir, exist_ok=True)

    tokensO_path=os.path.join(tokensO_filedir, tokensO_filename)
    tokensE_path=os.path.join(tokensE_filedir, tokensE_filename)
    tokensT_path=os.path.join(tokensT_filedir, tokensT_filename)
    meta_path=os.path.join(meta_filedir, meta_filename)
    # mesh after token decoder
    decode_path=os.path.join(decode_filedir, decode_filename)
    # M1:(GT) read mesh (obj,ply,glb,...) and trimesh process, normalize, clean by kiui
    # M2: mesh processed by silksong: non-mani preprocess, flip fixing, cc classification and coloring
    # M3: mesh processed by vertex layering and sorting, colored by layering
    M1_path=decode_path.replace('decode','M1')
    M2_path=decode_path.replace('decode','M2')
    M3_path=decode_path.replace('decode','M3')
    
    if debugging:
        print(f'Mesh M1 saving to {M1_path}')
        write_obj(vertices, faces, M1_path)


    meta_init_kwargs={
        'version': mode,
        'origin_path': os.path.join(get_base_dir_platform(dataset_name), obj_path),
        'other_path':[None, None],
        'face_type': face_type,
        'resolution': resolution, 
        'M1_path': M1_path,
        'M2_path': M2_path,
        'M3_path': M3_path,
    }

    tokenizer, _ =get_tokenizer_silksong(resolution=resolution, ss_mode=mode, meta_init_data=meta_init_kwargs, structure_limit=structure_limit_kwargs, debugging=debugging)

    tokensO = tokenizer.encode(vertices, faces, non_mani_process=True)
    meta_data_temp=tokenizer.get_metaData()
    if debugging:
        write_tokens_ori(tokensO, tokensO_path)
        meta_data_temp.save_meta(meta_path)


    tokensE, meta_data_temp=tokenizer.token_encode(input_tokens=tokensO, mode=mode)
    E_time=time.time()

    if debugging:
        np.save(tokensE_path, tokensE)  
        meta_data_temp.save_meta(meta_path)
        trans_write_tokens(tokensE, tokensT_path, tokenizer, resolution, mode)
        load_token=np.load(tokensE_path)
    else:
        load_token=tokensE

    # if debugging:
    #     vertices_decode, faces_decode = engine.decode_ori(tokens_ori)

    vertices_decode, faces_decode = tokenizer.decode(load_token, discrete_bins=resolution, mode=mode, colorful=True)

    # save v specified
    meta_data_temp=tokenizer.get_metaData()

    if debugging:
        write_obj(vertices_decode[0], faces_decode, decode_path.replace('.obj', '_layercolor.obj'))
        write_obj(vertices_decode[1], faces_decode, decode_path.replace('.obj', '_CCcolor.obj'))
        meta_data_temp.save_meta(meta_path)

    xlsx_line_new={
        'done': 1,
        'vert_num': meta_data_temp.vert_num,
        'face_num': meta_data_temp.face_num,
        'vert_num_process': meta_data_temp.vert_num_p,
        'face_num_process': meta_data_temp.face_num_p,
        'CC_num': meta_data_temp.CC_num,
        'CC_num_valid': meta_data_temp.CC_num_valid,
        'CC_num_pre': meta_data_temp.CC_num_pre,
        'CC_num_pre_all': meta_data_temp.CC_num_pre_all,
        'max_lv': meta_data_temp.max_lv,
        'max_l': meta_data_temp.max_l,
        'max_edge_num': meta_data_temp.max_edge_num,
        'token_length': meta_data_temp.token_length,
        'compression_rate': meta_data_temp.compression_rate,
        'flipped_face': meta_data_temp.flipped_face_cnt,
        '1v2cc_gen': meta_data_temp.new_generated_verts,
        'nonmani_gen': meta_data_temp.non_manifold_new_gen,
        'nonmani_face': meta_data_temp.non_manifold_vert_cnt,
        'nonmani_process_time': meta_data_temp.non_manifold_process_time,
        'not_success_flip_face': meta_data_temp.not_success_flip_face,
        'merge_repeat_verts': meta_data_temp.merge_repeat_verts_num,
        'replace_facevert_num': meta_data_temp.replace_facevert_num,
        'degraded_face_num': meta_data_temp.degraded_face_num,
        'move_repeat_face_num': meta_data_temp.move_repeat_face_num,
        'CC_invalid_vert_num': meta_data_temp.CC_invalid_verts,
        'CC_invalid_face_num': meta_data_temp.CC_invalid_faces,
        'encode_time': E_time-S_time,
    }
    if debugging:
        for k, v in xlsx_line_new.items():
            print(f'{k}:{v}')
    
    return xlsx_line_new


