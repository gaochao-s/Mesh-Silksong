# set the right conda env path for yours (the following are examples on cluster)
source /public/opt/conda/etc/profile.d/conda.sh
conda activate silk
export PATH="/public/home/group_gaosh/gaochao/.conda/envs/silk/bin:$PATH"
cd /public/home/group_gaosh/gaochao/main_workspace/MeshSilksong
# skip verify of wandb
export CURL_SSL_NO_VERIFY=1
export REQUESTS_CA_BUNDLE=""
# set your wandb api key
export WANDB_API_KEY="you wandb api key"
wandb login --relogin you wandb api key

##########
# mask the above codes if you do not use slurm for cluster. you may activate you conda env and login to wandb manually. 
##########

# specify these params for cluster multinode training, default param is for single cluster node training or single machine training
MACHINE_RANK=${MACHINE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}

# the params for main training script
MSL=10240
RESAMPLE=1
WARMUP=0.05
LR=0.0001
BATCH_SIZE=2
FT=1 
I_BETA=0.0
WORKSPACE="workspace_train/silk_ft_multi16"
RESUME="checkpoints/release-100K/model.safetensors"
NUM_EPOCH=200
SAVE_EPOCH=1
DATASET="ss"
XLSX_DIR="datasets/cleaned"
DATA_SUBSETS="gobjaversev1*3dfuture*toys4k*shapenetv2" # use * to seperate the datasets you want to train
DATA_FILTER=11
EVAL_MODE="loss"

# modify --config_file if you want to use other GPU number
accelerate launch \
    --config_file acc_configs/gpu16.yaml \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MASTER_ADDR \
    train.py \
    --workspace $WORKSPACE \
    --data.resample $RESAMPLE \
    --data.dataset $DATASET \
    --data.i_beta $I_BETA \
    --train.ft $FT \
    --train.warmup_ratio $WARMUP \
    --data.batch_size $BATCH_SIZE \
    --train.lr $LR \
    --train.resume $RESUME \
    --max_seq_length $MSL \
    --data.xlsx_dir $XLSX_DIR \
    --data.data_subsets $DATA_SUBSETS \
    --data.data_filter_cnt $DATA_FILTER \
    --train.num_epochs $NUM_EPOCH \
    --train.eval_mode $EVAL_MODE \
    --train.save_epoch $SAVE_EPOCH
