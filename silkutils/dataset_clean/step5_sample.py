
import pandas as pd
import os
import shutil
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ss_platform
from meshdata.mesh_io import load_mesh, write_obj

# sample_save_dir='debug_provider/infer_input_eachdataset'
sample_save_dir='test_set_A'
sample_save_dirC='test_set_C'


table_dir='datasets/cleaned'
sample_test_dir='datasets/sample_test/meshes'
sample_train_dir='datasets/sample_train/meshes'
sample_test_table_dir='datasets/sample_test/tables'
# sample_train_table_dir='datasets/sample_train/tables'


def copy_and_norm(dataset_name, xlsx_dic, save_dir_norm, save_dir_origin):
    # xlsx_dic: sampled_rows[['obj_path','obj_name','id']].to_dict(orient='records')
    base_dir_platform=ss_platform.get_base_dir_platform(dataset_name)
    os.makedirs(save_dir_norm, exist_ok=True)
    os.makedirs(save_dir_origin, exist_ok=True)
    for file_dic in xlsx_dic:
        file_full_path=os.path.join(base_dir_platform, file_dic['obj_path'])

        # copy origin first
        target_file_ext=os.path.splitext(os.path.basename(file_dic['obj_path']))[1]
        target_file_savename=f'ori_{dataset_name}_{file_dic["obj_name"]}{target_file_ext}'
        target_full_path_origin=os.path.join(save_dir_origin, target_file_savename)
        shutil.copy(file_full_path, target_full_path_origin)
        print(f'origin mesh copy to {target_full_path_origin}')

        # try to normalize
        try:
            print('load mesh')
            vertices, faces = load_mesh(file_full_path, clean=True)
            target_full_path_norm=os.path.join(save_dir_norm, f'norm_{dataset_name}_{file_dic["obj_name"]}.obj')
            write_obj(vertices, faces, target_full_path_norm)
            print(f'normed mesh saving to {target_full_path_norm}')
        except Exception as e:
            raise Exception(f'[E] Norm saving Failed, {file_full_path}')

def sample_table_specify(dataset_name, reso, verison, filter_num, num, split_list):

    xlsx_file=f'{table_dir}/meta_all_{dataset_name}_res{reso}_v{verison:02}_mergeall_filter{filter_num:02}.xlsx' 
    df = pd.read_excel(xlsx_file)
    print(f'read num {len(df)}')
    
    for index in range(len(split_list)-1):
        low=split_list[index]
        high=split_list[index+1]
        filtered_df=df[(df['face_num_process'] < high) & (df['face_num_process'] > low)]
        print(f'{low}-{high} filter num {len(filtered_df)}')

        available_samples = len(filtered_df)
        
        if available_samples < num:
            print(f"[WARNING] {low}-{high}: avail: {available_samples} < {num}")
            num = available_samples 
    
        sampled_rows=filtered_df.sample(n=num)
        result=sampled_rows[['obj_path','obj_name','id']].to_dict(orient='records')
        
        save_dir_norm=os.path.join(sample_train_dir, f'train_{dataset_name}_norm', f'face{low:06}-{high:06}')
        save_dir_origin=os.path.join(sample_train_dir, f'train_{dataset_name}_origin', f'face{low:06}-{high:06}')
        
        print(f'sampling {dataset_name}_face{low:06}-{high:06}')
        copy_and_norm(dataset_name, result, save_dir_norm, save_dir_origin)

        
def sample_table_left(dataset_name, xlsx_all, xlsx_exclude, num, sample_batch):

    # xlsx_all = f'{table_dir}/meta_all_{dataset_name}_res{reso}_v{v:02}_mergeall_filter02.xlsx'
    # xlsx_exclude=f'{table_dir}/meta_all_{dataset_name}_res{reso}_v{v:02}_mergeall_filter{filter:02}.xlsx' 

    df_exclude = pd.read_excel(xlsx_exclude) # usually train set
    df_all = pd.read_excel(xlsx_all)

    exclude_id_list=df_exclude['id'].tolist()
    # filtered_df = df_avail[~df_avail['id'].isin(train_id_list)]
    filtered_df = df_all[~df_all['id'].isin(exclude_id_list)]
    sampled_rows=filtered_df.sample(n=num)

    result=sampled_rows[['obj_path','obj_name','id']].to_dict(orient='records')
    
    save_dir_origin=os.path.join(sample_test_dir, f'test_L_{dataset_name}_origin', f'batch_{sample_batch:02}')
    save_dir_norm=os.path.join(sample_test_dir, f'test_L_{dataset_name}_norm', f'batch_{sample_batch:02}')
    
    copy_and_norm(dataset_name=dataset_name, xlsx_dic=result, save_dir_norm=save_dir_norm, save_dir_origin=save_dir_origin)
        

def sample_and_generate_testset_table(dataset_name, reso, verison, filter_num, num, sample_batch):
    os.makedirs(sample_test_table_dir, exist_ok=True)
    xlsx_filter_all=f'{table_dir}/meta_all_{dataset_name}_res{reso}_v{verison:02}_mergeall_filter{filter_num:02}.xlsx' 
    xlsx_filter_sample_save=f'{sample_test_table_dir}/testset_meta_all_{dataset_name}_res{reso}_v{verison:02}_mergeall_filter{filter_num:02}_sample{num:04}_sb{sample_batch:02}.xlsx' 
    df = pd.read_excel(xlsx_filter_all)
    if num>len(df):
        num=len(df)
    sampled_rows=df.sample(n=num)
    # save sampled
    print(f'test set xlsx save to {xlsx_filter_sample_save}')
    sampled_rows.to_excel(xlsx_filter_sample_save, index=False)
    result=sampled_rows[['obj_path','obj_name','id']].to_dict(orient='records')
    
    save_dir_origin=os.path.join(sample_test_dir, f'test_A_{dataset_name}_origin', f'batch_{sample_batch:02}')
    save_dir_norm=os.path.join(sample_test_dir, f'test_A_{dataset_name}_norm', f'batch_{sample_batch:02}')
    copy_and_norm(dataset_name=dataset_name, xlsx_dic=result, save_dir_norm=save_dir_norm, save_dir_origin=save_dir_origin)



if __name__ == "__main__":
    
    all_dataset=['abo','thingi10k']
    # all_dataset=['3dfuture','shapenetv2']
    # all_dataset=['objaversev1']
    all_dataset=['trellis-objxl-github']
    # all_dataset=['gobjaversev1']
    split_list=[0, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    for dataset_name in all_dataset:
        # sample and analyze training set
        sample_table_specify(dataset_name=dataset_name, reso=128, verison=4, filter_num=11, num=50, split_list=split_list)

        # generate test set, excluded in data_provider
        # sample_and_generate_testset_table(dataset_name=dataset_name, reso=128, verison=4, filter_num=11, num=200, sample_batch=0)
        

    