import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ss_platform import get_base_dir_platform, get_base_dir_rel, get_savedirs
from meshdata.mesh_io import load_mesh, load_mesh_modify
import os
import pandas as pd
from tqdm import tqdm
import ipdb
import json
from dataset_clean.clean_trellis import init_trellis_hssd, init_trellis_objxl_github, load_and_getnum, get_xlsx_item, init_trellis_3dfuture, init_trellis_abo, init_trellis_objxl_sketchfab





def init_3dcaricshop(type, version, face_type, resolution, load):
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_path_platform=os.path.join(base_dir_platform, f'meta_all_{type}_res{resolution}_v{version:02}_p0000.xlsx')
    mesh_dir_rel = os.path.join(rel_base_dir, 'tMesh')  
    img_dir_rel = os.path.join(rel_base_dir, 'img')      
    mesh_dir_platform=os.path.join(base_dir_platform, mesh_dir_rel)
    img_dir_platform=os.path.join(base_dir_platform, img_dir_rel)
 
    data = []  

   
    person_names=os.listdir(mesh_dir_platform)
    for person_name in tqdm(person_names, desc='Processing people'):  
        person_path_platform = os.path.join(mesh_dir_platform, person_name)  
        person_path_rel = os.path.join(mesh_dir_rel, person_name)
        if os.path.isdir(person_path_platform): 
            for obj_file in os.listdir(person_path_platform):  
                if obj_file.endswith('.obj'):  
                    obj_name = os.path.splitext(obj_file)[0]
                    obj_path_platform = os.path.join(person_path_platform, obj_file) 
                    obj_path_rel = os.path.join(person_path_rel, obj_file)

                    vert_num, face_num, have_bug = load_and_getnum(load=load,face_type=face_type, obj_path_platform=obj_path_platform,clean=False)
                     
                    img_path_platform = os.path.join(img_dir_platform, person_name, obj_name + '.jpg')

                    img_path_rel = os.path.join(img_dir_rel, person_name, obj_name + '.jpg')
                    if not os.path.exists(img_path_platform):
                        img_path_rel=None
                    
                     
                    item=get_xlsx_item()
                    item['id']=len(data)
                    item['bug_info']=have_bug
                    item['vert_num']=vert_num
                    item['face_num']=face_num
                    item['obj_name']=f"{person_name}_{obj_name}"
                    item['obj_path']=obj_path_rel
                    item['img_path']=img_path_rel
                    item['face_type']=face_type
                    item['resolution']=resolution

                    data.append(item)  

    df = pd.DataFrame(data)  

    df.to_excel(xlsx_path_platform, index=False)
    return xlsx_path_platform

def init_shapenetv2(type, version, face_type, resolution, load):
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
    o_dir_r='models/model_normalized.obj'
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir)
    
    all_id=-1
    part_num=-1
    type_names=os.listdir(all_dir_platform)
    for type_name in tqdm(type_names, desc='Processing types'):  
        part_num+=1
        data_thistype=[]
        type_dir_platform = os.path.join(all_dir_platform, type_name)  # platform
        type_dir_rel = os.path.join(rel_base_dir, type_name) # rel
        xlsx_name_part=xlsx_name_part_base+f'_p{part_num:04}.xlsx'
        xlsx_path_platform_part=os.path.join(base_dir_platform, xlsx_name_part)
            
        if os.path.isdir(type_dir_platform):
            o_names=os.listdir(type_dir_platform)
            for o_name in tqdm(o_names, desc='Processing objs'):
                obj_dir_platform = os.path.join(type_dir_platform, o_name)
                if os.path.isdir(obj_dir_platform):
                    obj_path_platform = os.path.join(obj_dir_platform, o_dir_r)
                    obj_path_rel = os.path.join(type_dir_rel, o_name, o_dir_r)
                    obj_name = f'{type_name}_{o_name}'
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

def init_objaversev1(type, version, face_type, resolution, load):
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

def init_abo(type, version, face_type, resolution, load):
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir, 'original')
    # img_dir_platform=os.path.join(base_dir_platform, img_dir_rel)

    all_id=-1
    part_num=-1
    type_names=os.listdir(all_dir_platform)
    for type_name in tqdm(type_names, desc='Processing types'):  
        part_num+=1
        data_thistype=[]
        type_dir_platform = os.path.join(all_dir_platform, type_name)  # platform
        type_dir_rel = os.path.join(rel_base_dir, 'original' ,type_name) # rel
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


# folder split
def init_3dfuture(type, version, face_type, resolution, load):
    split_number=2000
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir, '3D-FUTURE-model')

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
                obj_path_rel = os.path.join(rel_base_dir, '3D-FUTURE-model', o_name_folder, obj_fix_path)
                all_id+=1

                vert_num, face_num, have_bug = load_and_getnum(load=load,face_type=face_type, obj_path_platform=obj_path_platform,clean=False)

                img_path_rel = os.path.join(rel_base_dir, '3D-FUTURE-model', o_name_folder, img_fix_path)
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


def init_animal3d(type, version, face_type, resolution, load):
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_obj_dir_platform=os.path.join(base_dir_platform, rel_base_dir, 'obj_files', 'obj_files')
    all_img_dir_platform=os.path.join(base_dir_platform, rel_base_dir, 'images', 'images')
    # img_dir_platform=os.path.join(base_dir_platform, img_dir_rel)

    all_id=-1
    part_num=-1
    train_names=os.listdir(all_obj_dir_platform) # train or test
    for train_name in tqdm(train_names, desc='Processing types'):  
        part_num+=1
        data_thistype=[]
        train_dir_platform = os.path.join(all_obj_dir_platform, train_name)  # platform
        img_train_dir_platform = os.path.join(all_img_dir_platform, train_name) 
        train_dir_rel = os.path.join(rel_base_dir, 'obj_files', 'obj_files' , train_name) # rel
        img_train_dir_rel = os.path.join(rel_base_dir, 'images', 'images' , train_name) # rel
        xlsx_name_part=xlsx_name_part_base+f'_p{part_num:04}.xlsx'
        xlsx_path_platform_part=os.path.join(base_dir_platform, xlsx_name_part)
            
        if os.path.isdir(train_dir_platform):  
            type_names=os.listdir(train_dir_platform)
            for type_name in tqdm(type_names, desc='Listing type'):
                type_name_dir_platform=os.path.join(train_dir_platform, type_name)
                type_name_dir_rel = os.path.join(train_dir_rel, type_name)
                for obj_name_ext in os.listdir(type_name_dir_platform):
                    if obj_name_ext.endswith('.obj'):
                        # this is one obj
                        all_id+=1
                        obj_path_platform = os.path.join(type_name_dir_platform, obj_name_ext)
                        obj_path_rel = os.path.join(type_name_dir_rel, obj_name_ext)
                        obj_name = f'{train_name}_{type_name}_{os.path.splitext(obj_name_ext)[0]}'

                        img_name_ext = f'{os.path.splitext(obj_name_ext)[0]}.JPEG'
                        img_name_ext2 = f'{os.path.splitext(obj_name_ext)[0]}.jpg'

                        img_path_platform = os.path.join(img_train_dir_platform, type_name, img_name_ext)
                        img_path_platform2 = os.path.join(img_train_dir_platform, type_name, img_name_ext2)
                        img_path_rel = os.path.join(img_train_dir_rel, type_name, img_name_ext)
                        img_path_rel2 = os.path.join(img_train_dir_rel, type_name, img_name_ext2)

                        vert_num, face_num, have_bug = load_and_getnum(load=load,face_type=face_type, obj_path_platform=obj_path_platform,clean=False)

                        if not os.path.exists(img_path_platform):
                            img_path_platform = img_path_platform2
                            img_path_rel = img_path_rel2
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
            continue
        df_thistype = pd.DataFrame(data_thistype)  


        df_thistype.to_excel(xlsx_path_platform_part, index=False)
        print(f'part {xlsx_path_platform_part} done!')
        xlsx_path_platform_list.append(xlsx_path_platform_part)

    return xlsx_path_platform_list

def init_buildingnet(type, version, face_type, resolution, load):
    split_number=2000
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir, 'OBJ_MODELS')


    all_id=-1
    part_num=-1

    obj_names= [i for i in os.listdir(all_dir_platform) if i.endswith('.obj')]
    obj_names_parts=[obj_names[i:i + split_number] for i in range(0, len(obj_names), split_number)]

    for obj_names_part_list in tqdm(obj_names_parts, desc='Processing part list'):  
        part_num+=1
        data_thistype=[]
        # part_dir_platform = os.path.join(all_dir_platform, obj_names_part)  # platform
        # part_dir_rel = os.path.join(rel_base_dir, '3D-FUTURE-model' , obj_name) # rel
        xlsx_name_part=xlsx_name_part_base+f'_p{part_num:04}.xlsx'
        xlsx_path_platform_part=os.path.join(base_dir_platform, xlsx_name_part)
            
        for obj_name_ext in tqdm(obj_names_part_list, desc='Processing objs'):
            if True:
                # valid obj
                obj_path_platform = os.path.join(all_dir_platform, obj_name_ext)
                obj_name = f'{os.path.splitext(obj_name_ext)[0]}'
                obj_path_rel = os.path.join(rel_base_dir, 'OBJ_MODELS', obj_name_ext)
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


def init_thingi10k(type, version, face_type, resolution, load):
    split_number=2000
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir)

    all_id=-1
    part_num=-1

    obj_names= [i for i in os.listdir(all_dir_platform) if i.endswith('.stl')]
    obj_names_parts=[obj_names[i:i + split_number] for i in range(0, len(obj_names), split_number)]

    for obj_names_part_list in tqdm(obj_names_parts, desc='Processing part list'):  
        part_num+=1
        data_thistype=[]
        # part_dir_platform = os.path.join(all_dir_platform, obj_names_part)  # platform
        # part_dir_rel = os.path.join(rel_base_dir, '3D-FUTURE-model' , obj_name) # rel
        xlsx_name_part=xlsx_name_part_base+f'_p{part_num:04}.xlsx'
        xlsx_path_platform_part=os.path.join(base_dir_platform, xlsx_name_part)
            
        for obj_name_ext in tqdm(obj_names_part_list, desc='Processing objs'):
            if True:
                # valid obj
                obj_path_platform = os.path.join(all_dir_platform, obj_name_ext)
                obj_name = f'{os.path.splitext(obj_name_ext)[0]}'
                obj_path_rel = os.path.join(rel_base_dir, obj_name_ext)
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


def init_toys4k(type, version, face_type, resolution, load):
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_path_platform=os.path.join(base_dir_platform, f'meta_all_{type}_res{resolution}_v{version:02}_p0000.xlsx')
    mesh_dir_rel = os.path.join(rel_base_dir, 'toys4k_obj_files', 'toys4k_obj_files')
    pc_dir_rel = os.path.join(rel_base_dir, "toys4k_point_clouds", 'toys4k_point_clouds')
    
    mesh_dir_platform=os.path.join(base_dir_platform, mesh_dir_rel)
    pc_dir_platform=os.path.join(base_dir_platform, pc_dir_rel)


    data = []  

    type_names=os.listdir(mesh_dir_platform)
    for type_name in tqdm(type_names, desc='Processing type'):
        type_dir_platform = os.path.join(mesh_dir_platform, type_name)
        type_dir_rel = os.path.join(mesh_dir_rel, type_name)
        
        if os.path.isdir(type_dir_platform): 
            obj_names = os.listdir(type_dir_platform)
            for obj_name_f in tqdm(obj_names, desc='Processing objs'):
                obj_name_folder_platform = os.path.join(type_dir_platform, obj_name_f)
                obj_name_folder_rel = os.path.join(type_dir_rel, obj_name_f)
                if os.path.isdir(obj_name_folder_platform):
                    obj_name = f'{type_name}_{obj_name_f}'
                    obj_path_platform = os.path.join(obj_name_folder_platform, 'mesh.obj')  
                    obj_path_rel = os.path.join(obj_name_folder_rel, 'mesh.obj')

                    vert_num, face_num, have_bug = load_and_getnum(load=load,face_type=face_type, obj_path_platform=obj_path_platform,clean=False)
                    pc_path_platform = os.path.join(pc_dir_platform, type_name, obj_name_f, 'pc10K.npz')
                    pc_path_rel = os.path.join(pc_dir_rel, type_name, obj_name_f, 'pc10K.npz')
                    if not os.path.exists(pc_path_platform):
                        pc_path_rel=None
                    

                    item=get_xlsx_item()
                    item['id']=len(data)
                    item['bug_info']=have_bug
                    item['vert_num']=vert_num
                    item['face_num']=face_num
                    item['obj_name']=obj_name
                    item['obj_path']=obj_path_rel
                    item['img_path']=pc_path_rel
                    item['face_type']=face_type
                    item['resolution']=resolution

                    data.append(item)  


    df = pd.DataFrame(data)  
 
    df.to_excel(xlsx_path_platform, index=False)
    return xlsx_path_platform


def init_gso(type, version, face_type, resolution, load):
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    xlsx_path_platform=os.path.join(base_dir_platform, f'meta_all_{type}_res{resolution}_v{version:02}_p0000.xlsx')
    mesh_dir_rel = rel_base_dir
    mesh_dir_platform=os.path.join(base_dir_platform, mesh_dir_rel)
    

    data = []  

    obj_names=os.listdir(mesh_dir_platform)
    for obj_name in tqdm(obj_names, desc='Processing obj'):
        obj_dir_platform = os.path.join(mesh_dir_platform, obj_name)
        obj_dir_rel = os.path.join(mesh_dir_rel, obj_name)
        
        if os.path.isdir(obj_dir_platform): 
            
            obj_name = obj_name
            obj_path_platform = os.path.join(obj_dir_platform, 'meshes', 'model.obj') 
            obj_path_rel = os.path.join(obj_dir_rel, 'meshes', 'model.obj')

            
            vert_num, face_num, have_bug = load_and_getnum(load=load,face_type=face_type, obj_path_platform=obj_path_platform,clean=False)

            item=get_xlsx_item()
            item['id']=len(data)
            item['bug_info']=have_bug
            item['vert_num']=vert_num
            item['face_num']=face_num
            item['obj_name']=obj_name
            item['obj_path']=obj_path_rel
            item['img_path']=None
            item['face_type']=face_type
            item['resolution']=resolution

            data.append(item)  


    df = pd.DataFrame(data)  


    df.to_excel(xlsx_path_platform, index=False)
    return xlsx_path_platform



def init_gobjaversev1(type, version, face_type, resolution, load):
    split_number=5000
    base_dir_platform=get_base_dir_platform(type)
    rel_base_dir=get_base_dir_rel(type)
    json_gobjaverse='silkutils/dataset_clean/gobjaverse_280k_index_to_objaverse.json'
    with open(json_gobjaverse, 'r', encoding='utf-8') as f:
        loaded_gobj = json.load(f)
        
    xlsx_name_part_base=f'meta_all_{type}_res{resolution}_v{version:02}'
    xlsx_path_platform_list=[]
        
    all_dir_platform=os.path.join(base_dir_platform, rel_base_dir, 'glbs')
    all_gobjs=[]
    for index, key in tqdm(enumerate(loaded_gobj.keys()), total=len(loaded_gobj), desc="checking"):
        gobj_full_path=os.path.join(all_dir_platform, loaded_gobj[key])
        rel_path=os.path.join(rel_base_dir, 'glbs', loaded_gobj[key])
        if os.path.exists(gobj_full_path):
            base_name=os.path.basename(gobj_full_path)
            all_gobjs.append({base_name: [key, rel_path]})

    print(f'there are {len(all_gobjs)}/{len(loaded_gobj)} valid objects')
    parts_gobjs=[]
    parts_gobjs=[all_gobjs[i:i + split_number] for i in range(0, len(all_gobjs), split_number)]
    all_id=-1
    part_num=-1
    for gobj_part_list in tqdm(parts_gobjs, desc='Processing part list'):  
        part_num+=1
        data_thispart=[]
        # part_dir_platform = os.path.join(all_dir_platform, obj_names_part)  # platform
        # part_dir_rel = os.path.join(rel_base_dir, '3D-FUTURE-model' , obj_name) # rel
        xlsx_name_part=xlsx_name_part_base+f'_p{part_num:04}.xlsx'
        xlsx_path_platform_part=os.path.join(base_dir_platform, xlsx_name_part)
            
        for gobj in tqdm(gobj_part_list, desc='Processing objs'):
            if True:
                # valid obj
                obj_name_ext= next(iter(gobj))
                info=gobj[obj_name_ext]
                obj_path_rel=info[1]
                gobj_index=info[0]
                obj_path_platform = os.path.join(base_dir_platform, obj_path_rel)
                obj_name = f'{os.path.splitext(obj_name_ext)[0]}'
                
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
                item['gobj_index']=gobj_index

                data_thispart.append(item)

        if len(data_thispart)==0:
            part_num-=1
            continue
        df_thispart = pd.DataFrame(data_thispart)  

        df_thispart.to_excel(xlsx_path_platform_part, index=False)
        print(f'part {xlsx_path_platform_part} done!')
        xlsx_path_platform_list.append(xlsx_path_platform_part)

    return xlsx_path_platform_list




def init_dataset(type, version, face_type, resolution, load):
    
    return init_dataset_dic[type](type, version, face_type, resolution, load=load)



if __name__ == "__main__":
    init_dataset_dic={
    '3dcaricshop': init_3dcaricshop,
    'shapenetv2': init_shapenetv2,
    'objaversev1' : init_objaversev1,
    'abo': init_abo,
    '3dfuture': init_3dfuture,
    'animal3d': init_animal3d,
    'buildingnet': init_buildingnet,
    'thingi10k': init_thingi10k,
    'gso': init_gso,
    'toys4k': init_toys4k,
    'trellis-hssd': init_trellis_hssd,
    'gobjaversev1': init_gobjaversev1,
    'trellis-3dfuture': init_trellis_3dfuture,
    'trellis-abo': init_trellis_abo,
    'trellis-objxl-sketchfab': init_trellis_objxl_sketchfab,
    'trellis-objxl-github': init_trellis_objxl_github,
    
    }
    xlsx_path_platform = init_dataset(type='gobjaversev1', version=4, face_type='triangle', resolution=128, load=False)
