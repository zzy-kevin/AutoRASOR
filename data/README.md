# Data

This folder includes the six real-SEM image grids described below. Other
experiment inputs must be downloaded or generated separately.

## Synthetic benchmark (OPMD)

The synthetic dual-magnification generator uses the Open Phase-Field
Microstructure Dataset (OPMD) v1.0:
https://doi.org/10.5281/zenodo.7702179

Download and extract so that the benchmark paths resolve:

```
data/OPMD_1/final_results_allmetrics_inputs_non_spinodal_removed_3000_0.1_10.csv
data/OPMD_1/microstructures_VAE_model/VAE_mics/
```

Used in the manuscript's synthetic benchmarks (Figs. 4, 6, and 8; SI S2 and S3).

## Public real-SEM image grids

The public image dataset is a set of six 14x14 grids in `data/grid1/` through
`data/grid6/`. Each contains one capture-run directory with 196 high-mag PNG
tiles (`high_mag_r0_c0.png` through `high_mag_r13_c13.png`), the unchanged
low-mag overview PNG, and one merged `grid_metadata.json`. Tile coordinates are
zero-based row and column indices from 0 through 13. Each high-mag tile is
448x448 pixels; it was made by cropping the bottom 60 rows from its 896x956
source capture to retain the top 896x896 square, then splitting that square
into four tiles. The low-mag PNG is included without image transformation.

The public folder numbering maps to the original analysis IDs in order:
`grid1` = `grid_7`, `grid2` = `grid_8`, ..., `grid6` = `grid_12`.

## Real-SEM grid embeddings

The exhaustive-grid validation (Fig. 7, SI S5) uses cached DINOv3
embeddings of the six regions, published as `embeddings.npz` files in the
release assets. These cached embeddings retain the original analysis IDs
`grid_7` ... `grid_12`. Place them at:

```
data/grid_embeddings/grid_7/embeddings.npz
...
data/grid_embeddings/grid_12/embeddings.npz
```

Each file contains `new_x_features` (196, 384) low-magnification patch
tokens, `raw_features` (196, 196, 384) per-tile high-magnification patch
tokens, and `grid_indices`. The public image folders `grid1` ... `grid6`
correspond to cached embedding IDs `grid_7` ... `grid_12` in order.

## Capture run directories (Figs. 2 and 5)

Fig. 2 and Fig. 5 are computed from an AutoRASOR capture run - the
unattended three-magnification run of 78 micrographs (500x, 3500x,
24500x) described in Sec. 3.1. A capture run contains micrograph PNGs,
their metadata JSON, and the saved GP model.

## Cosine KernelPCA basis (SI S3 only)

The input-dimensionality ablation projects onto a cosine KernelPCA basis
fitted on the 9907-image SEM corpus of Aversa et al. (2018),
https://doi.org/10.1038/sdata.2018.172. If generated, place the fitted basis at:

```
dinov3_latent_reduction/cosine_pca_layer5.pkl
```

Only SI S3 needs this. Every reported result uses the full 384-D
embedding (`input_dim=0`), which requires no basis.

## Micrograph corpus (SI S1 only)

The positional-bias study averages patch-similarity maps over 100 SEM
micrographs cropped to 1024x1024, read from
`data/processed_images_cropped/`. Any set of micrographs works; the
conclusion (positional bias grows in later layers) is not specific to
these images.

## DINOv3

DINOv3 (ViT-S+/16) weights are distributed by Meta under their license:
https://github.com/facebookresearch/dinov3

The PyTorch path pulls `facebook/dinov3-vits16plus-pretrain-lvd1689m`
through `transformers`. For on-microscope inference we export layer-5
features to ONNX; see `autorasor/feature_extractor_onnx.py`.
