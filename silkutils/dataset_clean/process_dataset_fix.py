import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm
from dataset_clean import process_one
from ss_platform import  get_savedirs, get_base_dir_platform
from dataset_clean.process_dataset import init_process_one_params, update_item, setup_logger
import concurrent.futures
from typing import Dict, Any
import traceback

import re
import argparse


def process_item_fix(item: Dict[str, Any], log_dir: str, dataset_name: str, resolution: int, version: int, part_num: int) -> Dict[str, Any]:  
    obj_name = item.get('obj_name')  
    obj_id = item.get('id')
    bug_info=item.get('bug_info')
    str_buginfo=str(bug_info)
    key_words=['Time', 'loading', 'Unknown', 'timed', 'process', 'thread', 'pickle', 'open']
    if not any(keyword in str_buginfo for keyword in key_words):
        # print(f'can not handle buginfo {bug_info}')
        return item
    
    try:  
        # 
        print(f'fixing bug: {str_buginfo} of id: {obj_id}')     
        input_params=init_process_one_params(dataset_name=dataset_name, item=item, resolution=resolution, version=version, part_num=part_num)
        result = process_one(**input_params)
        updated_item=update_item(result, item)
        print(f'fixed {updated_item['bug_info']} of id: {obj_id}')
        updated_item['bug_info']=None
        return updated_item
    # except func_timeout.exceptions.FunctionTimedOut as ee:
    #     logger = setup_logger(obj_id, obj_name, log_dir)  # 
    #     logger.error("An error occurred: %s", str(ee))  
    #     logger.error(traceback.format_exc())
    #     item['bug_info']=f'Time out'
    #     item['done']=0
    #     return item
    except Exception as e:  
        
        # logger = setup_logger(obj_id, obj_name, log_dir)  # 为每个obj_name设置一个logger
        # logger.error("An error occurred: %s", str(e))  
        # logger.error(traceback.format_exc())
        item['bug_info']=f'{str(e)}'
        item['done']=0
        return item
    

def process_dataset_fix(xlsx_dic, max_workers=24, head=None):
 
    dataset_name, resolution, version, part_ind, b_i=xlsx_dic['dataset_name'], xlsx_dic['reso'], xlsx_dic['version'], xlsx_dic['part_ind'], xlsx_dic['b_i']

    basic_dir_platform=get_base_dir_platform(dataset_name)
    xlsx_path_platform=os.path.join(get_base_dir_platform(dataset_name), xlsx_dic['file'])
    log_dir=os.path.join(basic_dir_platform, f'p_{part_ind:04}_v{version:02}', get_savedirs()['error'])
    with pd.ExcelFile(xlsx_path_platform) as xls:  
        df = pd.read_excel(xls)  
        if head is not None:  
            df = df.head(head) 

    key_words_reg='Time|loading|Unknown|timed|process|thread|pickle|open'
    
    df_filter = df[df['bug_info'].str.contains(key_words_reg, regex=True, na=False)]

    # print(f'fixing {xlsx_path_platform}')
    print(f'process {len(df_filter)}/{len(df)} items')
    if len(df_filter)==0:
        print('------------- no rows need to process!------------- ')
        return
    save_path_fix_xlsx=xlsx_path_platform
    # save_path_fix_xlsx=os.path.join(basic_dir_platform, os.path.splitext(xlsx_dic['file'])[0]+"_debug.xlsx")  
    if True:

        results=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:  
            future_to_item = {executor.submit(process_item_fix, row.to_dict(), log_dir, dataset_name, resolution, version, part_ind): row for index, row in df.iterrows()}  
        
        
            with tqdm(total=len(df), desc=f"{dataset_name}-p_{part_ind:04}-b{b_i}", unit="item") as pbar:  
                for future in concurrent.futures.as_completed(future_to_item):  
                    item_dic = future_to_item[future].to_dict()
                    try:
                        result = future.result()  
                        results.append(result)
                    # except concurrent.futures.TimeoutError:  
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

        with pd.ExcelWriter(save_path_fix_xlsx) as writer:  
            results_df.to_excel(writer, index=False) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data fixing.')  
 
    parser.add_argument('--dataset_name', type=str, required=True,  
                    help='The name of the dataset to process.')  
    parser.add_argument('--file', type=str, required=True,  
                    help='The name of the dataset to process.')  
    parser.add_argument('--reso', type=int, required=True,  
                    help='The index of the part to process.')  
    parser.add_argument('--version', type=int, required=True,  
                    help='The index of the batch to process.')  
    parser.add_argument('--part_ind', type=int, required=True,  
                    help='The resolution to process.')  
    parser.add_argument('--b_i', type=int, required=True,  
                    help='The resolution to process.')  
    parser.add_argument('--max_workers', type=int, required=True,  
                    help='The resolution to process.')  
     
    args = parser.parse_args()
    
    file_dic={
        'dataset_name': args.dataset_name,
        'reso': args.reso,
        'version': args.version,
        'part_ind': args.part_ind,
        'b_i': args.b_i,
        'file': args.file
    }

    print(f'------------- fixing {file_dic['file']} ------------- ')
    process_dataset_fix(file_dic, max_workers=args.max_workers, head=None)
    print(f'------------- done {file_dic['file']} ------------- ')
    
