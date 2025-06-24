

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  
import re  
import ss_platform

def get_filtered_df(df, filter_version):
    if filter_version == 1: 
        return df[df['bug_info'].isna() & (df['CC_num_valid'] <= 30) & (df['nonmani_process_time'] == 0)] 
    elif filter_version == 2: # success processed file
        return df[df['bug_info'].isna()] 
    elif filter_version == 0:
        return df
    elif filter_version == 3:
        df['compression_rate'] = df['compression_rate'].str.replace('%', '').astype(float)
        # import ipdb;ipdb.set_trace()
        filtered_df = df[
            (df['compression_rate'] <= 65.0) &
            (df['CC_num_valid'] <= 100) &
            (df['token_length'] <= 20000)
        ]
        return filtered_df
    elif filter_version == 4: # num face <= 4k, cc<=5, token_length<40960
        filtered_df = df[
            (df['face_num_process'] <= 4000) &
            (df['CC_num_valid'] <= 5) &
            (df['token_length'] <= 40960)
        ]
        return filtered_df
    elif filter_version == 5: # num face <= 8k, cc<=20, token_length<40960
        filtered_df = df[
            (df['face_num_process'] <= 8000) &
            (df['CC_num_valid'] <= 20) &
            (df['token_length'] <= 40960)
        ]
        return filtered_df
    elif filter_version == 6: # num face <= 12k, cc<=100, token_length<40960
        filtered_df = df[
            (df['face_num_process'] <= 12000) &
            (df['CC_num_valid'] <= 100) &
            (df['token_length'] <= 40960)
        ]
        return filtered_df
    elif filter_version == 7: # 100<=num face <= 4k, cc<=10, token_length<40960
        filtered_df = df[
            (df['face_num_process'] <= 4000) &
            (df['face_num_process'] >= 100) &
            (df['CC_num_valid'] <= 10) &
            (df['token_length'] <= 20480)
        ]
        return filtered_df
    elif filter_version == 8: # num face <= 8k, cc<=20, token_length<20480, max_lv <=200
        filtered_df = df[
            (df['face_num_process'] <= 8000) &
            (df['max_lv'] <= 200) &
            (df['CC_num_valid'] <= 20) &
            (df['token_length'] <= 20480)
        ]
        return filtered_df
    elif filter_version == 9: # num face <= 8k, cc<=20, token_length<10000, max_lv <=200
        filtered_df = df[
            (df['face_num_process'] <= 8000) &
            (df['face_num_process'] >= 80) &
            (df['max_lv'] <= 200) &
            (df['CC_num_valid'] <= 30) &
            (df['token_length'] <= 10000)
        ]
        return filtered_df
    elif filter_version == 10: #  cc<=100, token_length<40960, max_lv <=200
        filtered_df = df[
            (df['max_lv'] <= 200) &
            (df['CC_num_valid'] <= 100) &
            (df['token_length'] <= 40960)
        ]
        return filtered_df
    elif filter_version == 11: #  for trellis more
        filtered_df = df[
            (df['max_lv'] <= 200) &
            (df['face_num_process'] >= 40) &
            (df['CC_num_valid'] <= 50) &
            (df['token_length'] <= 10000)
        ]
        return filtered_df
    elif filter_version == 12: #  for longer, new archi
        filtered_df = df[
            (df['max_lv'] <= 200) &
            (df['face_num_process'] >= 40) &
            (df['CC_num_valid'] <= 100) &
            (df['token_length'] <= 20000)
        ]
        return filtered_df

def merge_part_files(folder_path, version_list=None):  
    
    merged_files = {}  

    pattern = r'meta_all_(.+?)_res(\d+)_v(\d{2})_p(\d{4})_done_b(\d{3})\.xlsx'  

    for filename in os.listdir(folder_path):  
        match = re.match(pattern, filename)  
        if match:  
            dataset_name = match.group(1)  
            reso = int(match.group(2))  
            version = int(match.group(3))  
            part_ind = int(match.group(4))  
            b_i = int(match.group(5))  
            if version_list:
                if version not in version_list:
                    continue

            key = (dataset_name, reso, version, part_ind)  


            file_path = os.path.join(folder_path, filename)  
            print(f'find {file_path}')
            df = pd.read_excel(file_path)  

            if key not in merged_files:  
                merged_files[key] = []  
            merged_files[key].append(df)  

    for key, dfs in merged_files.items():  
        merged_df = pd.concat(dfs, ignore_index=True)  

        dataset_name, reso, version, part_ind = key  
        merged_filename = f'meta_all_{dataset_name}_res{reso}_v{version:02}_p{part_ind:04}_merge.xlsx'  
        
        merged_file_path = os.path.join(folder_path, merged_filename)  

        merged_df.sort_values(by='id').reset_index(drop=True)
        merged_df.to_excel(merged_file_path, index=False)  

        print(f'Merged {len(dfs)} files into {merged_filename}')  

def merge_all_files(directory, version_list=None):  
 
    pattern = re.compile(r'meta_all_(?P<dataset_name>.+?)_res(?P<reso>\d+)_v(?P<version>\d{2})_p(?P<part_ind>\d{4})_merge\.xlsx')  


    files_dict = {}  


    for filename in os.listdir(directory):  
        match = pattern.match(filename)  
        if match:  
            
            dataset_name = match.group('dataset_name')  
            reso = int(match.group('reso'))  
            version = int(match.group('version'))  
            part_ind = int(match.group('part_ind'))  
            if version_list:
                if version not in version_list:
                    continue
            key = (dataset_name, reso, version)  
            file_path = os.path.join(directory, filename)  

             
            if key not in files_dict:  
                files_dict[key] = []  
            files_dict[key].append(file_path)  


    for (dataset_name, reso, version), file_paths in files_dict.items():  
        # List to hold DataFrames  
        dataframes = []  

    
        for file_path in file_paths:  
            df = pd.read_excel(file_path)  
            dataframes.append(df)  

    
        merged_df = pd.concat(dataframes, ignore_index=True)  

      
        output_filename = f'meta_all_{dataset_name}_res{reso}_v{version:02}_mergeall.xlsx'  
        output_path = os.path.join(directory, output_filename)  

     
        merged_df.sort_values(by='id').reset_index(drop=True)
        merged_df.to_excel(output_path, index=False)  

        print(f'Merged {len(file_paths)} files into {output_path}')  


def filter_excel_files(in_directory, out_directory, filter_version, version_list=None):  

    pattern = re.compile(r'meta_all_(?P<dataset_name>.+?)_res(?P<reso>\d+)_v(?P<version>\d{2})_mergeall\.xlsx')  

    for filename in os.listdir(in_directory):  
        match = pattern.match(filename)  
        if match:  
        
            dataset_name = match.group('dataset_name')  
            reso = int(match.group('reso'))  
            version = int(match.group('version'))  
            file_path = os.path.join(in_directory, filename)  
            if version_list:
                if version not in version_list:
                    continue

            df = pd.read_excel(file_path)  

  
            filtered_df = get_filtered_df(df, filter_version)

            output_filename = f'meta_all_{dataset_name}_res{reso}_v{version:02}_mergeall_filter{filter_version:02}.xlsx'  
            output_path = os.path.join(out_directory, output_filename)  

            filtered_df.sort_values(by='id').reset_index(drop=True)
            filtered_df.to_excel(output_path, index=False)  

            print(f'Filtered version {filter_version:02} ---> {filename} and saved to {output_path}') 
            print(f'filter/all {len(filtered_df)}/{len(df)}')

def filter_dataset(dataset_name, filtered_xlsx_save_dir, filter_version, version_list=[4], merge=True):
    work_dir=ss_platform.get_base_dir_platform(dataset_name)
    if merge:
        print(f'Merging {dataset_name}')
        merge_part_files(work_dir, version_list=version_list)
        merge_all_files(work_dir, version_list=version_list)
    print(f'Filtering {dataset_name}, filter verion {filter_version}')
    filter_excel_files(in_directory=work_dir, out_directory=filtered_xlsx_save_dir, filter_version=filter_version, version_list=version_list)


if __name__ == "__main__":
    # dataset_name='objaversev1'
    # datasetnames=['3dcaricshop','3dfuture','abo','animal3d','buildingnet','gso', 'thingi10k','toys4k','shapenetv2','objaversev1']
    # datasetnames=['3dfuture','thingi10k','toys4k','shapenetv2','gobjaversev1']
    filtered_xlsx_save_dir='/public/home/group_gaosh/gaochao/main_workspace/MeshSilksong/datasets/cleaned'
    datasetnames=['gobjaversev1']
    for dataset_name in datasetnames:
        filter_dataset(dataset_name=dataset_name, filtered_xlsx_save_dir=filtered_xlsx_save_dir, filter_version=11, version_list=[4], merge=True)
        