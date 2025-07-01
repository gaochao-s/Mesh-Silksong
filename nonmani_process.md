# Nonmanifold Mesh Processing Guidance

## 1. Demo
#### How to Run
Run the code following this templete:
```
python silkutils/silksong_manifold_process.py \
--input_file silkutils/demo_test/shapenetv2_03761084_ee5861.obj \
--quant_resolution 1024 \
--output_dir silkutils/demo_test/manifold_repair \
--verbose
```

#### Output Illustration
Based on the input mesh name, the output files have different prefix:
- `M1_`: Load and normalize the mesh.
- `M2_`: Process the non-manifold mesh to manifold mesh, and color different connected components with distinctive colors.
- `M2f_`: Do the further face orientation consistency fixing. (If could)


## 2. Limitations
1. Currently the code is based on python and has no engineering acceleration for meshes with many faces, so it may be blocked for dense meshes.

2. The nonmanifold processing is binded with mesh vertices quantization to merge redundant faces as many as possible, if you do not want this, you can set a higher quantization resulution for trade-off.

3. The current algorithm may cause face topology like "mobius loop", which may hinder the repair of face orientation consistnecy.