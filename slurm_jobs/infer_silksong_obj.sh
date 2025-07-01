#!/bin/bash
#SBATCH --job-name=infer_silk          # job name
#SBATCH --output=main_workspace/job_logs/infer/infer_%j.log        # logs
#SBATCH --nodes=1                     # nodes applying
#SBATCH --partition=gpu           # partition
#SBATCH --ntasks=1                    # job number
#SBATCH --cpus-per-task=8             # CPU cores per task
#SBATCH --time=48:00:00               # time limit
#SBATCH --mem=64G                     # Memory
#SBATCH --gres=gpu:1                   # GPU number apply


source /public/opt/conda/etc/profile.d/conda.sh
conda activate silk
export PATH="/public/home/group_gaosh/gaochao/.conda/envs/silk/bin:$PATH"
cd /public/home/group_gaosh/gaochao/main_workspace/MeshSilksong

sh scripts/infer_silksong_obj.sh
