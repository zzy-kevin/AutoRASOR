"""Generator + engine settings for the synthetic-micrograph figures.

This is the configuration the manuscript's synthetic panels (Fig. 3) were
produced with. The benchmarks in ``benchmarks/`` carry their own
``BenchmarkConfig``; the two differ in the arrangement heatmap scale and in
TTA, so they are kept separate rather than silently merged.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    """Configuration for the AL v4 experiment."""
    csv_path: str = "data/OPMD_1/final_results_allmetrics_inputs_non_spinodal_removed_3000_0.1_10.csv"
    image_dir: str = "data/OPMD_1/microstructures_VAE_model/VAE_mics"

    prefit_pca_layer: int = 5
    dino_layer: int = 5

    grid_rows: int = 14
    grid_cols: int = 14
    tile_size: int = 16
    high_mag_size: int = 224

    arrangement_mode: str = 'heatmap_combined'
    heatmap_scale_af: float = 4
    heatmap_scale_mob: float = 4

    enable_degradation: bool = True
    local_blur_max_sigma: float = 1
    global_blur_sigma: float = 0.2
    gaussian_noise_std: float = 0.4
    shot_noise_scale: float = 0.3
    allow_phase_field_duplicates: bool = False

    use_onnx: bool = False
    use_tta: bool = False

    use_bias_subtraction: bool = False
    bias_map_path: str = "DINOv3_bias_removal/bias_map.npy"

    use_mirror_padding: bool = False
    mirror_padding_pct: float = 0.2

    onnx_model_path: str = "SEM_sim_env/dinov3_vits16plus_layer5.onnx"

    input_dim: int = 0
    prefit_cosine_pca_layer: int = 5
    metric: Literal['cosine', 'l1', 'l2'] = 'cosine'

    n_warmup_seeds: int = 3
    neighbors_per_seed: int = 2
    warmup_mode: str = 'lfps'

    n_al_steps: int = 30
    batch_size: int = 3
    gp_training_iters: int = 50
    beta_param: float = 1.0
    normalize_ucb: bool = True
    acq_function: str = 'qlognei'  # 'qucb' or 'qlognei'
    k_neighbors_ambiguity: int = 10

    # Ground-truth ambiguity map: fraction of unsampled patches to use
    gt_ambiguity_sample_pct: float = 1

    base_output_dir: str = "results/al_runs"
    save_low_mag: bool = True
    seed: int = 42
