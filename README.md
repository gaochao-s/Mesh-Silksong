# Mesh Silksong: Auto-Regressive Mesh Generation as Weaving Silk

## 1. Environment
#### Conda
```
conda create -n silk python=3.9
conda activate silk
pip install -r requirements.txt
```
#### Docker (Optional)
```
todo
```

## 2. Checkpoints Download

#### Main checkpoint

Currently we released lite version checkpoints trained on 100K public available datasets, refer to training part for reproduction. Checkpoints trained on more datasets will be released soon.

To download the model, use huggingface-cli:

```
python3 -m pip install "huggingface_hub[cli]"
mkdir ./checkpoints
huggingface-cli download gcsong/mesh_silksong --local-dir ./checkpoints
```
Or directly download from [Huggingface](https://huggingface.co/gcsong/mesh_silksong/tree/main), and put checkpoint in this path
```
./checkpoints/release-100K/model.safetensors
```

#### Michelangelo checkpoint

If you want to train the GPT model from scratch, the pretrained [Michelangelo](https://github.com/NeuralCarver/Michelangelo) point-encoder is required for finetune. Just download `shapevae-256.ckpt` from [here](https://huggingface.co/Maikou/Michelangelo/tree/main/checkpoints/aligned_shape_latents) and put it in this path
```
miche/shapevae-256.ckpt
```

## 3. Inference
Run `sh scripts/infer_silksong_obj.sh` for inference, key parameter illustration:
- `INFER_BATCH`: the batch size for inference, you may set it to 1 on limited GPU Memory.
- `WORKSPACE`: the save dir of generated meshes.
- `TEST_INPUT`: the input dir of dense meshes/ground truth meshes. Point cloud will be sampled as GPT condition. We provide some mesh examples sampled from public datasets [here](https://drive.google.com/drive/folders/1zR7UpC1LJPN2mQC_CfR-Dn2lHRWXG5Eb?usp=sharing). Download them and put them in this path
```
datasets/sample_test/meshes/test_mix_origin/batch00/
```
- `RESUME`: main checkpoint path.

If you have a cluster, run the slurm script:
```
sbatch slurm_jobs/infer_silksong_obj.sh
```

# todo: training guidance

# todo: non-manifold process



# Citations
