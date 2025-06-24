import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm
import logging
from dataset_clean.process_one import process_one
from ss_platform import  get_savedirs, base_dir, get_base_dir_platform

import concurrent.futures
import time
from typing import Dict, Any
import traceback
import re
import random
import argparse
import func_timeout
from func_timeout import func_set_timeout


def extract_xlsx_param(filename):  
 
    pattern = r'meta_all_(?P<type>[^_]+)_res(?P<resolution>\d+)_v(?P<version>\d{2})_p(?P<part_num>\d{4})\.xlsx'  
    
    match = re.search(pattern, filename)  
    
    if match:  
  
        file_info = match.groupdict()  
        file_info['version'] = int(file_info['version'])  
        file_info['part_num'] = int(file_info['part_num'])
        file_info['resolution'] = int(file_info['resolution'])
        return file_info['type'], file_info['resolution'], file_info['version'], file_info['part_num']
    else:  
        raise ValueError("File name does not match the expected pattern.")  


def setup_logging(log_file_path):  
    """  
    配置日志记录  
    :param log_file_path: 日志文件的路径  
    """  
    logging.basicConfig(  
        filename=log_file_path,  #
        level=logging.ERROR,      # 
        format='%(asctime)s - %(levelname)s - %(message)s'  #  
    )  


def init_process_one_params(dataset_name, item, resolution, version, part_num):
    process_one_params={
        'dataset_name': dataset_name,
        'obj_id': item.get('id'),
        'obj_path': item.get('obj_path'),
        'obj_save_name' : item.get('obj_name'),
        'save_dir_npy' : os.path.join(f'p_{part_num:04}_v{version:02}', get_savedirs()['npy']),
        'save_dir_meta' : os.path.join(f'p_{part_num:04}_v{version:02}', get_savedirs()['meta']),
        'save_dir_debug' : os.path.join(f'p_{part_num:04}_v{version:02}', get_savedirs()['debug']),
        'resolution': resolution,
        'mode': version,
        'face_type': item.get('face_type'),
    }
    return process_one_params

def update_item(xlsx_line_new, item):
    for key in xlsx_line_new.keys():
        item[key]=xlsx_line_new[key]
    return item


def setup_logger(obj_id: int, obj_name: str, log_dir: str) -> logging.Logger:  
    #
    logger = logging.getLogger(f'{obj_name}')  
    logger.setLevel(logging.ERROR)  

    # 
    os.makedirs(log_dir, exist_ok=True)  
    
    #   
    log_file_path = os.path.join(log_dir, f"{obj_id:08}_{obj_name}.log")  
    
    file_handler = logging.FileHandler(log_file_path)  
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  
    file_handler.setFormatter(formatter)  

    logger.addHandler(file_handler)  
    return logger

@func_set_timeout(2)
def process_one_tst(**kwargs):
    a=random.randint(0, 4)
    time.sleep(a)
    return {'d':3} 

def process_item(item: Dict[str, Any], log_dir: str, dataset_name: str, resolution: int, version: int, part_num: int) -> Dict[str, Any]:  
    

    vert_num=item.get('vert_num')
    face_num=item.get('face_num')
    if vert_num is not None and face_num is not None:
        if face_num > 16000:
            # logger = setup_logger(obj_id, obj_name, log_dir)  # 
            # logger.error("An error occurred")  
            # logger.error(traceback.format_exc())
            item['bug_info']=f'too many face at first'
            item['done']=0
            
            return item
    
    try:  
        #    
        input_params=init_process_one_params(dataset_name=dataset_name, item=item, resolution=resolution, version=version, part_num=part_num)
        result = process_one(**input_params)
        updated_item=update_item(result, item)
        
        return updated_item
    except func_timeout.exceptions.FunctionTimedOut as ee:
        # logger = setup_logger(obj_id, obj_name, log_dir)  # 
        # logger.error("An error occurred: %s", str(ee))  
        # logger.error(traceback.format_exc())
        item['bug_info']=f'Time out'
        item['done']=0
        return item
    except Exception as e:  
        #  
        # logger = setup_logger(obj_id, obj_name, log_dir)  # 
        # logger.error("An error occurred: %s", str(e))  
        # logger.error(traceback.format_exc())
        item['bug_info']=f'{str(e)}'
        item['done']=0
        return item
    

def process_dataset(xlsx_path_platform, max_workers=24, time_out=30, head=None, batch_ind=0, b_l=1000):
    basic_dir_platform=os.path.dirname(xlsx_path_platform)

    
    dataset_name, resolution, version, part_num=extract_xlsx_param(xlsx_path_platform)
    log_dir=os.path.join(basic_dir_platform, f'p_{part_num:04}_v{version:02}', get_savedirs()['error'])
    with pd.ExcelFile(xlsx_path_platform) as xls:  
        df = pd.read_excel(xls)  
        if head is not None:  
            df = df.head(head) 
    
    total_items=len(df)


    results = []  
    df_batches = []  
    for i in range(0, total_items, b_l):  
        batch = df.iloc[i:i + b_l]  
        df_batches.append(batch)  
    
    print(f'Batch num: {len(df_batches)}, batch size: {len(batch)}')  
     
    for b_i, batch in enumerate(df_batches):
        if b_i != batch_ind:
            continue
        
        done_xlsx_name=os.path.basename(xlsx_path_platform)
        done_xlsx_name=os.path.splitext(done_xlsx_name)[0] + f'_done_b{b_i:03}.xlsx'
        save_path_done_xlsx=os.path.join(basic_dir_platform, done_xlsx_name)
        if os.path.exists(save_path_done_xlsx):
            print(f'find {save_path_done_xlsx} exists, skip')
            continue

        results=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:  
            future_to_item = {executor.submit(process_item, row.to_dict(), log_dir, dataset_name, resolution, version, part_num): row for index, row in batch.iterrows()}  
        
        
            with tqdm(total=len(batch), desc=f"{dataset_name}-p_{part_num:04}-b{b_i}/{len(df_batches)}", unit="item") as pbar:  
                for future in concurrent.futures.as_completed(future_to_item):  
                    item_dic = future_to_item[future].to_dict()
                    try:
                        result = future.result(timeout=time_out)  
                        results.append(result)
                    # except concurrent.futures.TimeoutError:  # 
                    #     item_dic['bug_info'] = 'Timeout occurred'  
                    #     item_dic['done'] = 0  
                    #     results.append(item_dic)  
                    
                    except Exception as e:  
                        item_dic['bug_info']=f'{str(e)}'
                        item_dic['done']=0
                        results.append(item_dic)  
                    
                    # pbar.update(progress_queue.get())
                    pbar.update(1)
    
        
        results_df = pd.DataFrame(results)  
        results_df = results_df.sort_values(by='id')
    
        with pd.ExcelWriter(save_path_done_xlsx) as writer:  
            results_df.to_excel(writer, index=False) 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some data.')  
 
    parser.add_argument('--dataset_name', type=str, required=True,  
                    help='The name of the dataset to process.')  
    parser.add_argument('--part_ind', type=int, required=True,  
                    help='The index of the part to process.')  
    parser.add_argument('--batch_ind', type=int, required=True,  
                    help='The index of the batch to process.')  
    parser.add_argument('--resolution', type=int, required=True,  
                    help='The resolution to process.')  
    parser.add_argument('--version', type=int, required=True,  
                    help='The version to process.')  
    parser.add_argument('--head', type=int, required=True,  
                    help='The head to process.')  
    parser.add_argument('--max_workers', type=int, default=64,  
                    help='The maximum number of workers to use (default: 64).')  
    parser.add_argument('--b_l', type=int, default=1000,  
                    help='batch_length default 1000')  

    args = parser.parse_args()
    dataset_name=args.dataset_name
    part_ind=args.part_ind
    batch_ind=args.batch_ind
    max_workers=args.max_workers
    reso=args.resolution
    version=args.version
    head=args.head
    if head==-1:
        head=None
    b_l=args.b_l

    # xlsx_base_dir=base_dir()

    # data_set_dir={
    #     # "3dfuture": '3DFuture/3DFuture', # H800
    #     # 'toys4k': 'TOYS4K/TOYS4K', # H800
    #     "3dfuture": '3DFuture', # local
    #     'toys4k': 'TOYS4K', # local
    #     'objaversev1': 'Objaverse',
    #     'abo': 'ABO',
    #     'thingi10k': 'Thingi10K/Thingi10K/Thingi10K',
    #     'shapenetv2': 'shapenet/shapenet/ShapeNetCore.v2',
    #     'animal3d': 'animal3d/animal3d',
    #     '3dcaricshop': '3DCaricShop',
    #     'buildingnet': 'BuildingNet/BuildingNet',
    #     'gso': 'GSO/GSO'
    # }

    xlsx_name=f'meta_all_{dataset_name}_res{reso}_v{version:02}_p{part_ind:04}.xlsx'
    # xlsx_full_path=os.path.join(xlsx_base_dir, data_set_dir[dataset_name], xlsx_name)
    xlsx_full_path=os.path.join(get_base_dir_platform(dataset_name), xlsx_name)
    if not os.path.exists(xlsx_full_path):
        print(f'[WARNING] can not find {xlsx_full_path}, file part wrong')
    else:
        process_dataset(xlsx_path_platform=xlsx_full_path, max_workers=max_workers, time_out=30, head=head, batch_ind=batch_ind, b_l=b_l)
    