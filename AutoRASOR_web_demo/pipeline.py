"""
Micrograph + embedding pipeline for the AutoRASOR web demo.
===========================================================

Thin orchestration layer over the public package modules - nothing here
re-implements the science:

  - `autorasor.data_generator`    phase-field low-mag synthesis + high-mag oracle
  - `autorasor.feature_extractor` DINOv3 ViT-S/16+ layer-5 patch embeddings (D4 TTA)
  - `autorasor.ambiguity.compute_gt_ambiguity_from_pool`
                                  ground-truth ambiguity from the full 196-tile pool

One `Micrograph` bundles everything the browser needs for a single 14x14 field:
the degraded low-mag view, the undegraded high-mag montage, per-tile 224px
crops, low/high-mag embeddings, ground-truth ambiguity, and the generator's own
area-fraction / mobility maps.
"""

from __future__ import annotations

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autorasor.data_generator import (  # noqa: E402
    DegradationConfig,
    GeneratorConfig,
    HeatmapConfig,
    PhaseFieldOracle,
    SyntheticLowMagGenerator,
)
from autorasor.feature_extractor import FeatureExtractor  # noqa: E402
from autorasor.ambiguity import compute_gt_ambiguity_from_pool  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

DEFAULT_CSV = PROJECT_ROOT / "data/OPMD_1/final_results_allmetrics_inputs_non_spinodal_removed_3000_0.1_10.csv"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data/OPMD_1/microstructures_VAE_model/VAE_mics"


# ============================================================
# Configuration
# ============================================================

@dataclass
class MicrographParams:
    """Knobs the demo exposes for one synthetic low-mag field."""
    seed: int = 7
    grid_rows: int = 14
    grid_cols: int = 14
    tile_size: int = 16          # low-mag pixels per tile -> 14*16 = 224 = DINOv3 input
    high_mag_size: int = 224     # oracle crop size (also the montage tile size)
    heatmap_scale_af: float = 4.0
    heatmap_scale_mob: float = 4.0
    local_blur_max_sigma: float = 1.0
    global_blur_sigma: float = 0.2
    gaussian_noise_std: float = 0.4
    shot_noise_scale: float = 0.3
    allow_duplicates: bool = False

    def to_generator_config(self) -> GeneratorConfig:
        """Derive every sub-seed from the master seed so a seed reproduces a field."""
        s = int(self.seed)
        return GeneratorConfig(
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            tile_size=self.tile_size,
            mode='heatmap_combined',
            area_fraction_heatmap=HeatmapConfig(scale=self.heatmap_scale_af, seed=3 * s + 1),
            mobility_heatmap=HeatmapConfig(scale=self.heatmap_scale_mob, seed=3 * s + 2),
            degradation=DegradationConfig(
                local_blur_max_sigma=self.local_blur_max_sigma,
                global_blur_sigma=self.global_blur_sigma,
                gaussian_noise_std=self.gaussian_noise_std,
                shot_noise_scale=self.shot_noise_scale,
                degradation_heatmap_config=HeatmapConfig(scale=4.0, seed=3 * s + 3),
            ),
            allow_phase_field_duplicates=self.allow_duplicates,
            seed=s,
        )


# ============================================================
# Micrograph bundle
# ============================================================

@dataclass
class Micrograph:
    params: MicrographParams
    grid_shape: Tuple[int, int]
    low_mag: np.ndarray                 # (H, W) uint8 - degraded, what the "SEM" sees
    low_mag_clean: np.ndarray           # (H, W) uint8 - pre-degradation
    tiles: np.ndarray                   # (N, S, S) uint8 - undegraded high-mag truth
    lm_feats: np.ndarray                # (N, 384) DINOv3 layer-5, L2-normalized
    hm_feats: np.ndarray                # (N, 384)
    gt_ambiguity: np.ndarray            # (N,)
    af_gt: np.ndarray                   # (rows, cols) area fraction actually placed
    mob_gt: np.ndarray                  # (rows, cols) mobility actually placed
    tile_meta: List[dict]
    build_seconds: float
    _png_cache: Dict[str, bytes] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return self.lm_feats.shape[0]

    # -- image encoding -------------------------------------------------
    def png(self, key: str) -> bytes:
        """Encode (and memoize) one of the demo's raster layers as PNG."""
        if key in self._png_cache:
            return self._png_cache[key]
        if key == 'lowmag':
            arr = self.low_mag
        elif key == 'lowmag_clean':
            arr = self.low_mag_clean
        elif key == 'highmag':
            arr = self._montage()
        else:
            raise KeyError("unknown image layer '%s'" % key)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='PNG', compress_level=3)
        data = buf.getvalue()
        self._png_cache[key] = data
        return data

    def tile_png(self, index: int) -> bytes:
        key = "tile:%d" % index
        if key not in self._png_cache:
            buf = io.BytesIO()
            Image.fromarray(self.tiles[index]).save(buf, format='PNG', compress_level=3)
            self._png_cache[key] = buf.getvalue()
        return self._png_cache[key]

    def _montage(self) -> np.ndarray:
        """Full-resolution high-mag montage: every tile at its native oracle size."""
        rows, cols = self.grid_shape
        s = self.tiles.shape[1]
        out = np.zeros((rows * s, cols * s), dtype=np.uint8)
        for i in range(self.n_points):
            r, c = divmod(i, cols)
            out[r * s:(r + 1) * s, c * s:(c + 1) * s] = self.tiles[i]
        return out


# ============================================================
# Pipeline
# ============================================================

class Pipeline:
    """Owns the (expensive) generator and DINOv3 encoder; builds Micrographs."""

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        device: str = 'cuda',
        dino_layer: int = 5,
        use_tta: bool = True,
        k_neighbors_ambiguity: int = 10,
    ):
        self.csv_path = Path(csv_path or DEFAULT_CSV)
        self.image_dir = Path(image_dir or DEFAULT_IMAGE_DIR)
        self.use_tta = use_tta
        self.dino_layer = dino_layer
        self.k_neighbors_ambiguity = k_neighbors_ambiguity

        if not self.csv_path.exists():
            raise FileNotFoundError("OPMD metadata CSV not found: %s" % self.csv_path)
        if not self.image_dir.exists():
            raise FileNotFoundError("Phase-field image directory not found: %s" % self.image_dir)

        self.generator = SyntheticLowMagGenerator(str(self.csv_path), str(self.image_dir))
        self.extractor = FeatureExtractor(use_onnx=False, device=device, dino_layer=dino_layer)
        self.device = self.extractor.device

    # ------------------------------------------------------------------
    def build(
        self,
        params: MicrographParams,
        progress: Optional[Callable[[str, float], None]] = None,
    ) -> Micrograph:
        """Generate a field, embed it at both magnifications, and score GT ambiguity."""
        def note(msg: str, frac: float) -> None:
            if progress is not None:
                progress(msg, frac)

        t_start = time.time()
        cfg = params.to_generator_config()
        rows, cols = cfg.grid_rows, cfg.grid_cols

        note('Assembling phase-field micrograph', 0.05)
        low_mag, af_gt, mob_gt, meta = self.generator.generate_with_ground_truth(cfg)
        low_mag_clean = meta['canvas_clean']
        oracle = PhaseFieldOracle(self.generator, cfg, meta)

        note('Loading high-magnification ground truth', 0.30)
        tiles = self._load_tiles(oracle, rows, cols, params.high_mag_size)

        note('DINOv3 low-magnification patch embeddings', 0.50)
        lm_feats, grid_shape = self.extractor.extract_low_mag_features(low_mag, use_tta=self.use_tta)
        if grid_shape != (rows, cols):
            raise RuntimeError(
                "DINOv3 patch grid %s does not match the tile grid %s. tile_size must be 16 "
                "so that one phase-field tile == one DINOv3 patch." % (grid_shape, (rows, cols))
            )

        note('DINOv3 high-magnification embeddings', 0.65)
        hm_feats = self._embed_tiles(tiles, note)

        note('Ground-truth ambiguity over the full pool', 0.92)
        gt_ambiguity = self._ground_truth_ambiguity(lm_feats, hm_feats)

        note('Ready', 1.0)
        return Micrograph(
            params=params,
            grid_shape=(rows, cols),
            low_mag=low_mag,
            low_mag_clean=low_mag_clean,
            tiles=tiles,
            lm_feats=lm_feats.astype(np.float32),
            hm_feats=hm_feats.astype(np.float32),
            gt_ambiguity=gt_ambiguity.astype(np.float32),
            af_gt=af_gt,
            mob_gt=mob_gt,
            tile_meta=meta['tiles'],
            build_seconds=time.time() - t_start,
        )

    # ------------------------------------------------------------------
    def _load_tiles(self, oracle: PhaseFieldOracle, rows: int, cols: int, size: int) -> np.ndarray:
        """Pull every high-mag crop. PIL releases the GIL, so threads actually help."""
        def crop(i: int) -> np.ndarray:
            r, c = divmod(i, cols)
            img, _ = oracle.get_high_res_crop(r, c, size=size)
            return img

        with ThreadPoolExecutor(max_workers=8) as pool:
            tiles = list(pool.map(crop, range(rows * cols)))
        return np.stack(tiles).astype(np.uint8)

    def _embed_tiles(self, tiles: np.ndarray, note: Callable[[str, float], None]) -> np.ndarray:
        n = tiles.shape[0]
        feats = np.zeros((n, 384), dtype=np.float32)
        for i in range(n):
            feats[i] = self.extractor.extract_high_mag_features(tiles[i], use_tta=self.use_tta)[0]
            if i % 32 == 0:
                note('DINOv3 high-magnification embeddings (%d/%d)' % (i, n), 0.65 + 0.27 * i / n)
        return feats

    def _ground_truth_ambiguity(self, lm_feats: np.ndarray, hm_feats: np.ndarray) -> np.ndarray:
        """Canonical GT ambiguity: every tile's high-mag embedding is in the pool."""
        n = lm_feats.shape[0]
        probe = _AmbiguityProbe(lm_feats, self.k_neighbors_ambiguity)
        return compute_gt_ambiguity_from_pool(
            probe, np.arange(n), hm_feats, probe.x_gp,
            k_neighbors_ambiguity=self.k_neighbors_ambiguity,
        )


class _AmbiguityProbe:
    """Minimal stand-in for UncertaintyEngineV4 in `compute_gt_ambiguity_from_pool`.

    That helper only touches `.x_gp`, `.n_points` and `._sklearn_metric()`. Using
    it directly (rather than a fresh engine) keeps GT ambiguity independent of the
    surrogate's own state, and avoids re-implementing the estimator here.
    """

    def __init__(self, lm_feats: np.ndarray, k: int, metric: str = 'cosine'):
        norms = np.linalg.norm(lm_feats, axis=1, keepdims=True) + 1e-8
        self.x_gp = (lm_feats / norms).astype(np.float32)
        self.n_points = self.x_gp.shape[0]
        self.metric = metric
        self.k_neighbors_ambiguity = k

    def _sklearn_metric(self) -> str:
        return {'cosine': 'cosine', 'l1': 'manhattan', 'l2': 'euclidean'}[self.metric]
