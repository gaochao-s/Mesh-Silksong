# Dataset Clean Guidance

## 1. 3D Datasets Download
The following 3d datasets are public available:

Main

- [Objaverse(XL)](https://huggingface.co/datasets/allenai/objaverse-xl)
- [3D-FUTURE](https://tianchi.aliyun.com/dataset/98063)
- [Toys4K](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k)
- [gObjaverse](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse)
- [ShapeNet v2](https://shapenet.org/)
- [Trellis](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md)

More for any special need

- [Thingi10K](https://github.com/Thingi10K/Thingi10K)
- [GSO](https://app.gazebosim.org/home)
- [Animal3D](https://xujiacong.github.io/Animal3D/)
- [3DCaricShop](https://github.com/qiuyuda/3DCaricShop)
- [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html#download)
- [BuildingNet](https://github.com/buildingnet/buildingnet_dataset?tab=readme-ov-file)


## 2. Dataset Clean
1. After downloading the 3ddatasets, you may set the right base path in `silkutils/ss_platform.py` based on your own platform. In `ss_platform.py`, the following inited .xlsx file will be stored in the output path of function: `get_base_dir_platform(name)`

If you are processing gObjaverse dataset, download objaversev1 first, and download `gobjaverse_280k_index_to_objaverse.json`  from [here](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse) and put it to 
```
silkutils/dataset_clean/gobjaverse_280k_index_to_objaverse.json
```

2. Run `silkutils/datset_clean/step1_init.py`, modify the input dataset name in the function and several inited .xlsx files will be stored. The following steps will be based on these .xlsx files for data cleaning.

3. Run `silkutils/datset_clean/step2_clean.py` for multi-threaded processing. The data will be processed by silksong tokenization algorithm, and the statistic information will be stored in a new .xlsx file. In the following step, some data will be filtered based on the statistic information (e.g. too many faces). This step may take long time.

4. Run `silkutils/datset_clean/step3_cleanfix.py` if there are any accident when processing some data (e.g. Disk failure or Memory Problems), hence you need not re-run `step2_clean.py` again.

5. Run `silkutils/datset_clean/step4_datafilter.py`.
- `filtered_xlsx_save_dir`: All of processed .xlsx file will be automatically merged to this path. When training, the `model/data_provider.py` should read input .xlsx files from here. 
- `filter_version` : Modify you data filter rule. Refer to function `get_filtered_df(df, filter_version)`. The version `11` is recommended for training with max token length 10240. You may modify a new version if you have any special need.
- The finally .xlsx files should be like `meta_all_{datasetname}_res128_v04_mergeall_filter{filter_version}.xlsx`

## 3. Sample from Dataset for Analysing/Testing/Evaluation
1. Refer to `silkutils/datset_clean/step5_sample.py` for training data sampling and testing data generation.
2. Modify the right path
- `table_dir`: for the dir of input .xlsx files
- `sample_test_dir`: the save dir of test data you sampled
- `sample_test_table_dir`: the save dir of test data's .xlsx file you sampled
- `sample_train_dir`: the save dir of training data you sampled
3. The sampled data will be copyed to a new dir `sample_test_dir` or `sample_train_dir`, for the convenient of preview.
4. The data will be copied twice:
- Directly copying original data for the input of model inference. (with postfix `_origin`)
- Normalize the original data and save it to .obj file, for the convenience of your prereview.  (with postfix `_norm`)

#### 3.1 Sampling training data:

- Run function `sample_table_specify()`.
- View them in `sample_train_dir`.

#### 3.2 Sampling testing data:

- Run function `sample_and_generate_testset_table()`
- The sampled testset's .xlsx file has prefix `testset_`, which will be recognized and excluded when training in `model/data_provider.py`. You may change the prefix in the `DataConfigs.testset_prefix` of `config/options.py` if you have any special need.