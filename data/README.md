# Data


## Synthetic benchmark (OPMD)

The synthetic simulation and the web demo use the Open Phase-Field Microstructure Dataset (OPMD) v1.0:
https://doi.org/10.5281/zenodo.7702179

Download and extract the dataset so that the paths resolve exactly like this:

```
data/OPMD_1/final_results_allmetrics_inputs_non_spinodal_removed_3000_0.1_10.csv
data/OPMD_1/microstructures_VAE_model/VAE_mics/
```

This dataset provides the underlying phase-field microstructures used by `AutoRASOR_web_demo`.

## DINOv3

DINOv3 (ViT-S+/16) weights are distributed by Meta under their license:
https://github.com/facebookresearch/dinov3

The PyTorch path used by the web demo automatically pulls `facebook/dinov3-vits16plus-pretrain-lvd1689m` through `transformers`.
