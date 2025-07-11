import subprocess

def get_command(dataset_name, part_ind, batch_ind, max_workers, resolution, version, head):
    command_template=f'python dataset_clean/process_dataset.py --dataset_name {dataset_name} --part_ind {part_ind} --batch_ind {batch_ind} --max_workers {max_workers} --b_l 1000 --resolution {resolution} --version {version} --head {head}'
    return command_template

def part_batch_mapping(dataset_name):
    # the start and end index of init xlsx file, you may modify it for part processing
    map_part={
        '3dfuture': (0, 9), # for example, xlsx file part index from 0-8, it is based on you situation
        'objaverse': (0, 40), # 0-157
        'gobjaversev1': (40, 53), # 0-52
        'abo': (0, 36),
        'toys4k': (0, 1),
        'thingi10k': (0, 5),
        'shapenetv2': (30, 55), # 0-54
        'animal3d': (0, 2),
        'gso': (0, 1),
        'buildingnet': (0, 1),
        'trellis-hssd': (0, 4),
        'trellis-abo': (0, 36),
        'trellis-3dfuture': (0, 5),
        'trellis-objxl-sketchfab': (120, 160), # 0-159
        'trellis-objxl-github': (50, 61) # 0-60 61 end
    }
    # process files with batch size 1000 because of ulimit on linux. Just keep this > xlsx's mesh number/1000
    map_batch={
        '3dfuture': 10, 
        'objaverse': 8,
        'gobjaversev1': 8,
        'abo': 5,
        'toys4k': 5,
        'thingi10k': 5,
        'shapenetv2': 10,
        'animal3d': 10,
        'gso': 10,
        'buildingnet': 10,
        'trellis-hssd': 5,
        'trellis-abo': 5,
        'trellis-3dfuture': 5,
        'trellis-objxl-sketchfab': 5,
        'trellis-objxl-github': 8,
    }
    return map_part[dataset_name][0], map_part[dataset_name][1], map_batch[dataset_name]


def clean_datasetname(dataset_name, resolution, version, head=-1, max_workers=64):
    part_num0, part_num1, batch_num=part_batch_mapping(dataset_name=dataset_name)
    for i in range(part_num0, part_num1):
        for j in range(batch_num):
            command=get_command(dataset_name=dataset_name, part_ind=i, batch_ind=j, max_workers=max_workers, resolution=resolution, version=version, head=head)
            print(f'command: {command}')
            subprocess.run(command, shell=True)

if __name__ == "__main__":
    datasetname='gobjaversev1'
    resolution=128
    version=4
    head=-1
    max_workers=64
    clean_datasetname(dataset_name=datasetname, resolution=resolution, version=version, head=head, max_workers=max_workers)
    
    