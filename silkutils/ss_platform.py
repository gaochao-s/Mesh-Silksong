import os
###################### local windows or H800 cluster
def get_platform():
    if os.path.exists('/public/SothisAI'):
        platform='H800'
    elif os.path.exists('/workspace/3ddatasets'):
        platform='local_docker'
    else:
        platform='H800_docker'
    return platform

def base_dir():
    if get_platform() == "local_docker":
        return '/workspace/3ddatasets'
    else:
        return '/public/home/group_gaosh/gaochao/3ddatasets'

def get_savedirs():
    save_dirs={
        'npy': 'process_dir/npy_dir',
        'meta': 'process_dir/meta_dir',
        'debug': 'process_dir/debug_dir',
        'spc': 'process_dir/spc_dir',
        'error': 'process_dir/error_log'
    }
    return save_dirs

def get_base_dir_platform(name):
    path_prefix=base_dir()
    if get_platform()=="H800_docker":
        objaverse_fix='_docker'
        trellis_fix='_docker'
    else:
        objaverse_fix=''
        trellis_fix=''
    base_dir_platform_dic_local={
    '3dcaricshop': f"{path_prefix}/3DCaricShop",
    'shapenetv2' : f"{path_prefix}/shapenet/ShapeNetCore.v2",
    'objaversev1' : f"{path_prefix}/Objaverse{objaverse_fix}",
    'gobjaversev1' : f"{path_prefix}/gObjaverse{objaverse_fix}",
    'abo': f"{path_prefix}/ABO",
    '3dfuture': f"{path_prefix}/3DFuture",
    'animal3d': f"{path_prefix}/animal3d",
    'buildingnet' : f"{path_prefix}/BuildingNet",
    'thingi10k' : f"{path_prefix}/Thingi10K/Thingi10K-002/Thingi10K-002/Thingi10K",
    'toys4k' : f"{path_prefix}/TOYS4K",
    'gso' : f"{path_prefix}/GSO",
    }
    base_dir_platform_dic_h800={
    '3dcaricshop': f"{path_prefix}/3DCaricShop",
    'shapenetv2' : f"{path_prefix}/shapenet/shapenet/ShapeNetCore.v2",
    'objaversev1' : f"{path_prefix}/Objaverse{objaverse_fix}",
    'gobjaversev1' : f"{path_prefix}/gObjaverse{objaverse_fix}",
    'abo': f"{path_prefix}/ABO",
    '3dfuture': f"{path_prefix}/3DFuture/3DFuture",
    'animal3d': f"{path_prefix}/animal3d/animal3d",
    'buildingnet' : f"{path_prefix}/BuildingNet/BuildingNet",
    'thingi10k' : f"{path_prefix}/Thingi10K/Thingi10K/Thingi10K",
    'toys4k' : f"{path_prefix}/TOYS4K/TOYS4K",
    'gso' : f"{path_prefix}/GSO/GSO",
    'trellis-hssd': f'{path_prefix}/trellis/HSSD/HSSD/raw',
    'trellis-3dfuture': f'{path_prefix}/trellis/3D-FUTURE/raw',
    'trellis-abo': f'{path_prefix}/trellis/ABO/ABO',
    'trellis-toys4k': '',
    'trellis-objxl-github': f'{path_prefix}/trellis/Objaversexl_github',
    'trellis-objxl-sketchfab': f'{path_prefix}/trellis/Objaversexl_sketchfab/raw/raw',
    }
    if get_platform()=="local_docker":
        return base_dir_platform_dic_local[name]
    else:
        return base_dir_platform_dic_h800[name]

def get_base_dir_rel(name):
    base_dir_rel_dic={
    '3dcaricshop': 'processedData',
    'shapenetv2' : "ShapeNetCore.v2",
    'objaversev1' : 'hf-objaverse-v1',
    'gobjaversev1' : 'hf-objaverse-v1',
    'abo': '3dmodels',
    '3dfuture': '3D-FUTURE-model',
    'animal3d': 'drive-download',
    'buildingnet': 'OBJ_MODELS-001',
    'thingi10k' : 'raw_meshes',
    'toys4k' : 'obj_points',
    'gso' : 'unzipped',
    'trellis-hssd': 'objects',
    'trellis-3dfuture': '3D-FUTURE-model',
    'trellis-abo': 'raw',
    'trellis-toys4k': '',
    'trellis-objxl-github': 'trellis_objxl_github_convert',
    'trellis-objxl-sketchfab': 'hf-objaverse-v1',
    'debug': 'debug_input'
    }
    return base_dir_rel_dic[name]




