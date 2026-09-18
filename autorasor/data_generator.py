"""
Data Generator and Oracle for AL v2
====================================
Duplicated and simplified from synthetic_lowmag_generator.py and run_al_phase_field.py.

Contains:
- SyntheticDataGenerator: Generates synthetic low-mag images from phase field data
- PhaseFieldOracle: Simulates "zooming in" to provide high-mag crops
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Literal
from scipy.ndimage import gaussian_filter
import random


# ============================================================
# Configuration Dataclasses
# ============================================================

@dataclass
class HeatmapConfig:
    """Configuration for Perlin-like noise heatmap generation."""
    scale: float = 5.0
    octaves: int = 3
    persistence: float = 0.5
    min_val: float = 0.0
    max_val: float = 1.0
    seed: Optional[int] = None


@dataclass
class DegradationConfig:
    """Configuration for image degradation effects."""
    enable_local_blur: bool = True
    local_blur_max_sigma: float = 3.0
    enable_global_blur: bool = True
    global_blur_sigma: float = 1.0
    enable_gaussian_noise: bool = True
    gaussian_noise_std: float = 5.0
    enable_shot_noise: bool = True
    shot_noise_scale: float = 0.02
    degradation_heatmap_config: HeatmapConfig = field(default_factory=HeatmapConfig)


@dataclass
class GeneratorConfig:
    """Main configuration for the synthetic low-mag generator."""
    grid_rows: int = 20
    grid_cols: int = 20
    tile_size: int = 16
    mode: Literal['random', 'heatmap_af', 'heatmap_mobility', 'heatmap_combined'] = 'heatmap_combined'
    area_fraction_heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    mobility_heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    allow_phase_field_duplicates: bool = True
    seed: Optional[int] = None


# ============================================================
# Noise Generator
# ============================================================

class NoiseGenerator:
    """Generates smooth noise heatmaps for arranging images."""

    @staticmethod
    def generate_smooth_noise(shape: Tuple[int, int], config: HeatmapConfig) -> np.ndarray:
        """Generate a smooth noise heatmap using multi-scale Gaussian smoothing."""
        rng = np.random.RandomState(config.seed)
        rows, cols = shape

        combined = np.zeros((rows, cols))
        amplitude = 1.0

        for octave in range(config.octaves):
            noise = rng.randn(rows, cols)
            sigma = config.scale / (2 ** octave)
            smoothed = gaussian_filter(noise, sigma=max(sigma, 0.5))
            combined += amplitude * smoothed
            amplitude *= config.persistence

        # Normalize to [min_val, max_val]
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-12)
        combined = combined * (config.max_val - config.min_val) + config.min_val

        return combined


# ============================================================
# Synthetic Low-Mag Generator
# ============================================================

class SyntheticLowMagGenerator:
    """Generates synthetic low-magnification micrographs from phase field images."""

    def __init__(
        self,
        csv_path: str = "data/OPMD_1/final_results_allmetrics_inputs_non_spinodal_removed_3000_0.1_10.csv",
        image_dir: str = "data/OPMD_1/microstructures_VAE_model/VAE_mics"
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)

        # Load metadata
        self.df = pd.read_csv(self.csv_path)
        self.n_images = len(self.df)
        print(f"[Generator] Loaded {self.n_images} phase field entries")

        # Normalize parameters
        self._normalize_parameters()

    def _normalize_parameters(self):
        """Normalize Area_fraction and mobility to [0, 1]."""
        for col in ['Area_fraction', 'mobility']:
            self.df[f'{col}_norm'] = (self.df[col] - self.df[col].min()) / \
                                     (self.df[col].max() - self.df[col].min() + 1e-12)

    def _center_crop(self, img: Image.Image, size: int) -> Image.Image:
        """Center crop an image to the specified size."""
        w, h = img.size
        left = (w - size) // 2
        top = (h - size) // 2
        return img.crop((left, top, left + size, top + size))

    def _load_image(self, idx: int, tile_size: int) -> Optional[np.ndarray]:
        """Load and preprocess a single phase field image."""
        img_name = f"comp_time_frame3000_{idx}.jpg"
        img_path = self.image_dir / img_name
        
        if not img_path.exists():
            return None
            
        img = Image.open(img_path).convert('L')
        img = self._center_crop(img, 900)
        img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        return np.array(img)

    def _find_closest_image(
        self, target_af: float, target_mob: float,
        used_indices: set, weight_af: float = 0.5, weight_mob: float = 0.5,
        allow_duplicates: bool = True
    ) -> Tuple[int, float, float]:
        """Find the image closest to target AF and mobility values.
        
        Returns (idx, area_fraction, mobility) where idx maps to
        the image filename comp_time_frame3000_{idx}.jpg.
        """
        distances = (
            weight_af * (self.df['Area_fraction_norm'] - target_af) ** 2 +
            weight_mob * (self.df['mobility_norm'] - target_mob) ** 2
        )
        sorted_indices = distances.argsort()

        if not allow_duplicates:
            for i in sorted_indices:
                idx = int(self.df.iloc[i]['idx'])
                if idx not in used_indices:
                    return (
                        idx,
                        float(self.df.iloc[i]['Area_fraction']),
                        float(self.df.iloc[i]['mobility'])
                    )
        
        # If allow_duplicates=True OR all indices are used (fallback)
        i = sorted_indices[0]
        idx = int(self.df.iloc[i]['idx'])
        return idx, float(self.df.iloc[i]['Area_fraction']), float(self.df.iloc[i]['mobility'])

    def _apply_local_blur(self, tile: np.ndarray, blur_intensity: float, max_sigma: float) -> np.ndarray:
        """Apply Gaussian blur to a tile based on degradation intensity."""
        sigma = blur_intensity * max_sigma
        if sigma > 0.1:
            return gaussian_filter(tile.astype(np.float32), sigma=sigma).astype(np.uint8)
        return tile

    def _apply_degradation(self, image: np.ndarray, config: DegradationConfig) -> np.ndarray:
        """Apply global degradation effects to the assembled image."""
        result = image.astype(float)

        if config.enable_global_blur and config.global_blur_sigma > 0:
            result = gaussian_filter(result, sigma=config.global_blur_sigma)

        if config.enable_gaussian_noise and config.gaussian_noise_std > 0:
            noise = np.random.normal(0, config.gaussian_noise_std, result.shape)
            result = result + noise

        if config.enable_shot_noise and config.shot_noise_scale > 0:
            scaled = result * config.shot_noise_scale
            scaled = np.maximum(scaled, 0)
            poisson_noise = np.random.poisson(scaled + 1) - (scaled + 1)
            result = result + poisson_noise / config.shot_noise_scale

        return np.clip(result, 0, 255).astype(np.uint8)

    def generate_with_ground_truth(
        self, config: Optional[GeneratorConfig] = None,
        used_indices: Optional[set] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Generate a synthetic low-mag image with ground truth heatmaps.

        Returns
        -------
        image : np.ndarray, shape (H, W)
        af_gt : np.ndarray, shape (grid_rows, grid_cols) — actual area fraction per tile
        mob_gt : np.ndarray, shape (grid_rows, grid_cols) — actual mobility per tile
        metadata : dict — tile-level metadata
        """
        if config is None:
            config = GeneratorConfig()

        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)

        rows, cols = config.grid_rows, config.grid_cols
        tile_size = config.tile_size

        # Generate arrangement heatmaps
        shape = (rows, cols)
        if config.mode == 'random':
            af_target = np.random.rand(rows, cols)
            mob_target = np.random.rand(rows, cols)
        elif config.mode == 'heatmap_af':
            af_target = NoiseGenerator.generate_smooth_noise(shape, config.area_fraction_heatmap)
            mob_target = np.random.rand(rows, cols)
        elif config.mode == 'heatmap_mobility':
            af_target = np.random.rand(rows, cols)
            mob_target = NoiseGenerator.generate_smooth_noise(shape, config.mobility_heatmap)
        else:  # heatmap_combined
            af_target = NoiseGenerator.generate_smooth_noise(shape, config.area_fraction_heatmap)
            mob_target = NoiseGenerator.generate_smooth_noise(shape, config.mobility_heatmap)

        # Degradation heatmap
        deg_heatmap = NoiseGenerator.generate_smooth_noise(
            shape, config.degradation.degradation_heatmap_config
        )

        # Assemble
        canvas = np.zeros((rows * tile_size, cols * tile_size), dtype=np.uint8)
        af_gt = np.zeros((rows, cols), dtype=np.float32)
        mob_gt = np.zeros((rows, cols), dtype=np.float32)
        if used_indices is None:
            used_indices = set()
        tile_metadata = []

        for r in range(rows):
            for c in range(cols):
                idx, actual_af, actual_mob = self._find_closest_image(
                    af_target[r, c], mob_target[r, c], used_indices,
                    allow_duplicates=config.allow_phase_field_duplicates
                )
                used_indices.add(idx)

                tile = self._load_image(idx, tile_size)
                if tile is None:
                    tile = np.full((tile_size, tile_size), 128, dtype=np.uint8)

                # Apply local blur
                if config.degradation.enable_local_blur:
                    tile = self._apply_local_blur(
                        tile, deg_heatmap[r, c],
                        config.degradation.local_blur_max_sigma
                    )

                canvas[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size] = tile
                af_gt[r, c] = actual_af
                mob_gt[r, c] = actual_mob

                tile_metadata.append({
                    'grid_row': r, 'grid_col': c, 'idx': idx,
                    'area_fraction': actual_af, 'mobility': actual_mob
                })

        # Save undegraded canvas before degradation
        canvas_clean = canvas.copy()

        # Apply global degradation
        canvas = self._apply_degradation(canvas, config.degradation)

        metadata = {
            'config': config,
            'tiles': tile_metadata,
            'af_target': af_target,
            'mob_target': mob_target,
            'canvas_clean': canvas_clean,  # Pre-degradation image
        }

        return canvas, af_gt, mob_gt, metadata


# ============================================================
# Phase Field Oracle
# ============================================================

class PhaseFieldOracle:
    """
    Oracle for "zooming in" on synthetic low-mag phase field images.
    Returns original (clean) phase field tiles at specified positions.
    """

    def __init__(
        self,
        generator: SyntheticLowMagGenerator,
        config: GeneratorConfig,
        metadata: dict
    ):
        self.generator = generator
        self.config = config
        self.metadata = metadata

        # Lookup from grid position to tile info
        self.tile_lookup = {}
        for tile_info in metadata['tiles']:
            key = (tile_info['grid_row'], tile_info['grid_col'])
            self.tile_lookup[key] = tile_info

        self.tile_size = config.tile_size
        self.grid_rows = config.grid_rows
        self.grid_cols = config.grid_cols

    def get_high_res_crop(self, row: int, col: int, size: int = 256) -> Tuple[np.ndarray, dict]:
        """
        Get high-resolution crop for a specific grid cell.

        Parameters
        ----------
        row, col : int
            Grid position.
        size : int
            Output image size.

        Returns
        -------
        image : np.ndarray, shape (size, size)
        tile_info : dict
        """
        row = max(0, min(row, self.grid_rows - 1))
        col = max(0, min(col, self.grid_cols - 1))

        tile_info = self.tile_lookup.get((row, col))
        if tile_info is None:
            return np.full((size, size), 128, dtype=np.uint8), {}

        idx = tile_info['idx']
        image_name = f"comp_time_frame3000_{idx}.jpg"
        image_path = self.generator.image_dir / image_name

        if not image_path.exists():
            return np.full((size, size), 128, dtype=np.uint8), tile_info

        img = Image.open(image_path).convert('L')
        img = self.generator._center_crop(img, 900)
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        return np.array(img), tile_info
