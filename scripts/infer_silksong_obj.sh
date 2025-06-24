
# source /public/opt/conda/etc/profile.d/conda.sh
# conda activate silk
# export PATH="/public/home/group_gaosh/gaochao/.conda/envs/silk/bin:$PATH"
# cd main_workspace/MeshSilksong

# export CUDA_VISIBLE_DEVICES=4

MSL=10240
REPEAT=1
INFER_BATCH=4 # H800 80G is ok, you may set it smaller if GPU Mem < 80G
TEMPRETURE=0.5
MAX_FILTER=0
WORKSPACE="workspace_infer/silksong_output_test"
TEST_INPUT="datasets/sample_test/meshes/test_mix_origin/batch_00"
RESUME="checkpoints/release-100K/model.safetensors"


python infer.py \
    --workspace $WORKSPACE \
    --train.resume $RESUME \
    --infer.test_path_input $TEST_INPUT \
    --max_seq_length $MSL \
    --infer.test_repeat $REPEAT \
    --infer.infer_batch $INFER_BATCH \
    --infer.temperature $TEMPRETURE \
    --infer.max_filter $MAX_FILTER


