import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ss_platform import get_base_dir_platform, get_base_dir_rel, base_dir
from meshdata.mesh_io import load_mesh, load_mesh_modify
import os
import pandas as pd
import trimesh
import numpy as np
from tqdm import tqdm

import subprocess

def load_and_getnum(load, face_type, obj_path_platform, clean):
    have_bug = None
    if load:
        if face_type=='triangle':
            try:
                vertices, faces = load_mesh(obj_path_platform, clean=clean)
            except Exception as e:
                vertices, faces = [], []
                have_bug = str(e)
        elif face_type=='multi':
            try:
                vertices, faces = load_mesh_modify(obj_path_platform)
            except Exception as e:
                vertices, faces = [], []
                have_bug = str(e)
    else:
        vertices, faces= [], []

    vert_num=len(vertices)
    face_num=len(faces)
    return vert_num, face_num, have_bug

def get_xlsx_item():
    item = {  
        'id': None,  # init
        'done': 0,  # after process
        'bug_info': '',  # after process
        'vert_num': None, # init
        'face_num': None, # init
        'resolution': None, # init
        'obj_name': None,  # init  
        'obj_path': None,  # init
        'img_path': None,  # init
        'pc_path': '',  # init
        'npy_path': '',  # after process 
        'token_length': '',  # after process  
        'compression_rate': '',  # after process  
        'meta_path': '',  # after process
        'decode_obj_path': '',  # after process
        'face_type': None # init

    }
    return item

def init_trellis_hssd(type, version, face_type, resolution, load):
    split_number=2000
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir)

    sub_dirs=os.listdir(all_dir_platform)

    all_id=-1
    part_num=-1
    obj_names=[]
    for sub_dir in sub_dirs:
        sub_dir_path_rel=os.path.join(rel_base_dir, sub_dir)
        sub_dir_path_platform=os.path.join(all_dir_platform, sub_dir)
        obj_names+=[os.path.join(sub_dir_path_rel, i) for i in os.listdir(sub_dir_path_platform) if i.endswith('.glb')]

    obj_names_parts=[obj_names[i:i + split_number] for i in range(0, len(obj_names), split_number)]

    for obj_names_part_list in tqdm(obj_names_parts, desc='Processing part list'):  
        part_num+=1
        data_thistype=[]
        # part_dir_platform = os.path.join(all_dir_platform, obj_names_part)  # platform
        # part_dir_rel = os.path.join(rel_base_dir, '3D-FUTURE-model' , obj_name) # rel
        xlsx_name_part=xlsx_name_part_base+f'_p{part_num:04}.xlsx'
        xlsx_path_platform_part=os.path.join(base_dir_platform, xlsx_name_part)
            
        for obj_name_ext_relpath in tqdm(obj_names_part_list, desc='Processing objs'):
            if True:
                # valid obj
                obj_name_ext=os.path.basename(obj_name_ext_relpath)
                obj_path_platform = os.path.join(base_dir_platform, obj_name_ext_relpath)
                obj_name = f'{os.path.splitext(obj_name_ext)[0]}'
                obj_path_rel = obj_name_ext_relpath
                all_id+=1

                vert_num, face_num, have_bug = load_and_getnum(load=load,face_type=face_type, obj_path_platform=obj_path_platform,clean=False)
 

                item=get_xlsx_item()
                item['id']=all_id
                item['bug_info']=have_bug
                item['vert_num']=vert_num
                item['face_num']=face_num
                item['obj_name']=obj_name
                item['obj_path']=obj_path_rel
                item['img_path']=None
                item['face_type']=face_type
                item['resolution']=resolution

                data_thistype.append(item)


        if len(data_thistype)==0:
            part_num-=1
            continue
        df_thistype = pd.DataFrame(data_thistype)  


        df_thistype.to_excel(xlsx_path_platform_part, index=False)
        print(f'part {xlsx_path_platform_part} done!')
        xlsx_path_platform_list.append(xlsx_path_platform_part)

    return xlsx_path_platform_list

def init_trellis_3dfuture(type, version, face_type, resolution, load):
    split_number=2000
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir)

    obj_fix_path=f'raw_model.obj'
    img_fix_path=f'image.jpg'

    all_id=-1
    part_num=-1
    obj_names=os.listdir(all_dir_platform)
    obj_names_parts=[obj_names[i:i + split_number] for i in range(0, len(obj_names), split_number)]

    for obj_names_part_list in tqdm(obj_names_parts, desc='Processing part list'):  
        part_num+=1
        data_thistype=[]
        # part_dir_platform = os.path.join(all_dir_platform, obj_names_part)  # platform
        # part_dir_rel = os.path.join(rel_base_dir, '3D-FUTURE-model' , obj_name) # rel
        xlsx_name_part=xlsx_name_part_base+f'_p{part_num:04}.xlsx'
        xlsx_path_platform_part=os.path.join(base_dir_platform, xlsx_name_part)
            
        for o_name_folder in tqdm(obj_names_part_list, desc='Processing objs'):
            o_name_folder_dir=os.path.join(all_dir_platform, o_name_folder)
            if os.path.isdir(o_name_folder_dir):
                # valid obj
                obj_path_platform = os.path.join(all_dir_platform, o_name_folder, obj_fix_path)
                img_path_platform = os.path.join(all_dir_platform, o_name_folder, img_fix_path)
                obj_name = f'{o_name_folder}'
                obj_path_rel = os.path.join(rel_base_dir, o_name_folder, obj_fix_path)
                all_id+=1

                vert_num, face_num, have_bug = load_and_getnum(load=load,face_type=face_type, obj_path_platform=obj_path_platform,clean=False)

                img_path_rel = os.path.join(rel_base_dir, o_name_folder, img_fix_path)
                if not os.path.exists(img_path_platform):
                    img_path_rel=None

                item=get_xlsx_item()
                item['id']=all_id
                item['bug_info']=have_bug
                item['vert_num']=vert_num
                item['face_num']=face_num
                item['obj_name']=obj_name
                item['obj_path']=obj_path_rel
                item['img_path']=img_path_rel
                item['face_type']=face_type
                item['resolution']=resolution

                data_thistype.append(item)

        if len(data_thistype)==0:
            part_num-=1
            # ipdb.set_trace()
            print('ignore\n')
            continue
        df_thistype = pd.DataFrame(data_thistype)  

        df_thistype.to_excel(xlsx_path_platform_part, index=False)
        print(f'part {xlsx_path_platform_part} done!')
        xlsx_path_platform_list.append(xlsx_path_platform_part)

    return xlsx_path_platform_list

def init_trellis_abo(type, version, face_type, resolution, load):
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir)
    # img_dir_platform=os.path.join(base_dir_platform, img_dir_rel)

    all_id=-1
    part_num=-1
    type_names=os.listdir(all_dir_platform)
    for type_name in tqdm(type_names, desc='Processing types'):  
        part_num+=1
        data_thistype=[]
        type_dir_platform = os.path.join(all_dir_platform, type_name)  # platform
        type_dir_rel = os.path.join(rel_base_dir ,type_name) # rel
        xlsx_name_part=xlsx_name_part_base+f'_p{part_num:04}.xlsx'
        xlsx_path_platform_part=os.path.join(base_dir_platform, xlsx_name_part)
            
        if os.path.isdir(type_dir_platform):
            o_names=os.listdir(type_dir_platform)
            for o_name_ext in tqdm(o_names, desc='Processing objs'):
                obj_path_platform = os.path.join(type_dir_platform, o_name_ext)
                if obj_path_platform.endswith('.glb'):
                    obj_name = f'{type_name}_{os.path.splitext(o_name_ext)[0]}'
                    obj_path_rel = os.path.join(type_dir_rel, o_name_ext)
                    all_id+=1

                    vert_num, face_num, have_bug = load_and_getnum(load=load,face_type=face_type, obj_path_platform=obj_path_platform,clean=False)
 
                    item=get_xlsx_item()
                    item['id']=all_id
                    item['bug_info']=have_bug
                    item['vert_num']=vert_num
                    item['face_num']=face_num
                    item['obj_name']=obj_name
                    item['obj_path']=obj_path_rel
                    item['img_path']=None
                    item['face_type']=face_type
                    item['resolution']=resolution

                    data_thistype.append(item)

        if len(data_thistype)==0:
            part_num-=1
            continue
        df_thistype = pd.DataFrame(data_thistype)  

        df_thistype.to_excel(xlsx_path_platform_part, index=False)
        print(f'part {xlsx_path_platform_part} done!')
        xlsx_path_platform_list.append(xlsx_path_platform_part)

    return xlsx_path_platform_list


def init_trellis_objxl_sketchfab(type, version, face_type, resolution, load):
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir, 'glbs')
    # img_dir_platform=os.path.join(base_dir_platform, img_dir_rel)

    all_id=-1
    part_num=-1
    type_names=os.listdir(all_dir_platform)
    for type_name in tqdm(type_names, desc='Processing types'):  
        part_num+=1
        data_thistype=[]
        type_dir_platform = os.path.join(all_dir_platform, type_name)  # platform
        type_dir_rel = os.path.join(rel_base_dir, 'glbs' ,type_name) # rel
        xlsx_name_part=xlsx_name_part_base+f'_p{part_num:04}.xlsx'
        xlsx_path_platform_part=os.path.join(base_dir_platform, xlsx_name_part)
            
        if os.path.isdir(type_dir_platform):  
            o_names=os.listdir(type_dir_platform)
            for o_name_ext in tqdm(o_names, desc='Processing objs'):
                obj_path_platform = os.path.join(type_dir_platform, o_name_ext)
                if obj_path_platform.endswith('.glb'):
                    obj_name = f'{type_name}_{os.path.splitext(o_name_ext)[0]}'
                    obj_path_rel = os.path.join(type_dir_rel, o_name_ext)
                    all_id+=1

                    vert_num, face_num, have_bug = load_and_getnum(load=load,face_type=face_type, obj_path_platform=obj_path_platform,clean=False)
 
                    item=get_xlsx_item()
                    item['id']=all_id
                    item['bug_info']=have_bug
                    item['vert_num']=vert_num
                    item['face_num']=face_num
                    item['obj_name']=obj_name
                    item['obj_path']=obj_path_rel
                    item['img_path']=None
                    item['face_type']=face_type
                    item['resolution']=resolution

                    data_thistype.append(item)

        if len(data_thistype)==0:
            part_num-=1
            continue
        df_thistype = pd.DataFrame(data_thistype)  

        df_thistype.to_excel(xlsx_path_platform_part, index=False)
        print(f'part {xlsx_path_platform_part} done!')
        xlsx_path_platform_list.append(xlsx_path_platform_part)

    return xlsx_path_platform_list


def init_trellis_objxl_github(type, version, face_type, resolution, load):
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir)
    block_xlsx_path=os.path.join(all_dir_platform, "convert_tables")

    all_id=-1
    part_num=-1
    type_names=sorted(os.listdir(block_xlsx_path))
    table_number=len(type_names)
    print(f'objxl_github: {table_number} blocks') # 3038
    ############ just stat
    # lack_block=[]
    # for index in range(3038):
    #     if f"b{index:06}.xlsx" not in type_names:
    #         lack_block.append(index)
    # print(lack_block)
    # return
    ############
    step=50
    
    for i in tqdm(range(0, table_number, step), desc='Processing types'): 
        
        table_parts=type_names[i: i+step]
        table_parts_all=[]
        for tab in table_parts:
            df=pd.read_excel(os.path.join(block_xlsx_path, tab))
            dic_list=df[['success','info','convert_name','save_path','local_path']].to_dict(orient='records')
            table_parts_all+=dic_list

        part_num+=1
        data_thistype=[]
        
        xlsx_name_part=xlsx_name_part_base+f'_p{part_num:04}.xlsx'
        xlsx_path_platform_part=os.path.join(base_dir_platform, xlsx_name_part)
        
        for item_dic in table_parts_all:
            obj_name=item_dic['convert_name']
            obj_path_rel=item_dic['save_path'].replace(base_dir_platform+'/',"")
            obj_path_platform=item_dic['save_path']

            all_id+=1

            vert_num, face_num, have_bug = load_and_getnum(load=load,face_type=face_type, obj_path_platform=obj_path_platform,clean=False)

            item=get_xlsx_item()
            item['id']=all_id
            item['bug_info']=have_bug
            item['vert_num']=vert_num
            item['face_num']=face_num
            item['obj_name']=obj_name
            item['obj_path']=obj_path_rel
            item['img_path']=None
            item['face_type']=face_type
            item['resolution']=resolution
            item['convert_success']=item_dic['success']
            item['convert_info']=item_dic['info']
            item['raw_path']=item_dic['local_path']

            data_thistype.append(item)

        if len(data_thistype)==0:
            part_num-=1
            continue
        df_thistype = pd.DataFrame(data_thistype)  

        df_thistype.to_excel(xlsx_path_platform_part, index=False)
        print(f'part {xlsx_path_platform_part} done!')
        xlsx_path_platform_list.append(xlsx_path_platform_part)

    return xlsx_path_platform_list





    



if __name__ == "__main__":
    pass