import subprocess
from ss_platform import get_base_dir_platform
import re
import os

def find_satisfied_files(dataset_name_in,reso_in,version_in, part_min, part_max):   
    directory=get_base_dir_platform(dataset_name_in)
    pattern = re.compile(r'meta_all_(?P<dataset_name>\w+)_res(?P<reso>\d+)_v(?P<version>\d{2})_p(?P<part_ind>\d{4})_done_b(?P<b_i>\d{3})\.xlsx')  
    
    satisfied_filesdic=[]

    for filename in os.listdir(directory):  
        match = pattern.match(filename)  
        if match:  
            dataset_name = match.group('dataset_name')  
            reso = int(match.group('reso'))  
            version = int(match.group('version'))  
            part_ind = int(match.group('part_ind'))  
            b_i = int(match.group('b_i'))  
            if dataset_name_in != dataset_name or part_ind<part_min or part_ind>= part_max:
                continue
            if reso_in !=reso or version!=version_in:
                continue
            file_dic={
                'dataset_name': dataset_name,
                'reso': reso,
                'version': version,
                'part_ind': part_ind,
                'b_i': b_i,
                'file': filename,
            }

            satisfied_filesdic.append(file_dic)
    return satisfied_filesdic

def get_command_fix(file_dic, max_workers):
    return f'python dataset_clean/process_dataset_fix.py --dataset_name {file_dic['dataset_name']} --reso {file_dic['reso']} --version {file_dic['version']}\
     --part_ind {file_dic['part_ind']} --b_i {file_dic['b_i']} --file {file_dic['file']} --max_workers {max_workers}'

if __name__ == "__main__":
    dataset_name_in='objaversev1'
    reso_in=128
    version_in=2
    part_min=50
    part_max=100
    max_workers=64
    filesdic_list=find_satisfied_files(dataset_name_in,reso_in,version_in, part_min, part_max)
    for ind, ele in enumerate(filesdic_list):
        print(f'[{ind}] find: {ele['file']}')

    for filedic in filesdic_list:
        command=get_command_fix(filedic, max_workers)
        subprocess.run(command, shell=True)