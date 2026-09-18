"""
DINOv3 Feature Extractor for AL v2
===================================
Duplicated and simplified from run_al_phase_field.py for independent modification.

Supports two backends:
- ONNX: For offline/SEM machine compatibility
- PyTorch: Full-image TTA with D4 augmentations
"""

import numpy as np
from PIL import Image
from PIL import Image
import cv2
Image.MAX_IMAGE_PIXELS = None

from typing import Tuple

# --- DINOv3 Configuration ---
MODEL_HF_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_SPECIAL_TOKENS = 5


class FeatureExtractor:
    """
    DINOv3 feature extractor with ONNX or PyTorch backend.

    Parameters
    ----------
    use_onnx : bool
        If True, uses ONNX runtime. If False, uses PyTorch/transformers.
    device : str
        Device for PyTorch backend ('cuda' or 'cpu').
    """

    def __init__(self, use_onnx: bool = True, onnx_model_path: str = None, device: str = 'cuda', dino_layer: int = 12):
        self.use_onnx = use_onnx
        self.dino_layer = dino_layer  # 12 = final layer, 5 = intermediate
        self._use_hidden_states = (dino_layer != 12)  # Need hidden_states for non-final layers

        if self.use_onnx:
            import sys
            from pathlib import Path
            # Add SEM_sim_env to path for ONNX extractor
            sem_env_dir = str(Path(__file__).parent.parent / "SEM_sim_env")
            if sem_env_dir not in sys.path:
                sys.path.insert(0, sem_env_dir)
            from run_onnx_inference import FeatureExtractorONNX
            print(f"[FeatureExtractor] Using ONNX backend (model={onnx_model_path})...")
            self._onnx = FeatureExtractorONNX(onnx_path=onnx_model_path)
            self.device = 'cpu'
            print("[FeatureExtractor] ONNX ready")
        else:
            import torch
            import torchvision.transforms.functional as TF
            from transformers import AutoModel
            self.torch = torch
            self.TF = TF
            self.device = device if torch.cuda.is_available() else 'cpu'
            print(f"[FeatureExtractor] Loading DINOv3 ({MODEL_HF_ID}), layer={dino_layer}...")
            model_kwargs = {}
            if self._use_hidden_states:
                model_kwargs['output_hidden_states'] = True
            self.model = AutoModel.from_pretrained(MODEL_HF_ID, **model_kwargs).to(self.device).eval()
            print(f"[FeatureExtractor] Loaded on {self.device} (layer={dino_layer})")

    # ----------------------------------------------------------
    # PyTorch helpers
    # ----------------------------------------------------------

    def _preprocess_full_image(self, img_array: np.ndarray):
        """Convert numpy array to preprocessed tensor for full image."""
        if len(img_array.shape) == 2:
            img_rgb = np.stack((img_array,) * 3, axis=-1)
        else:
            img_rgb = img_array

        pil_img = Image.fromarray(img_rgb.astype(np.uint8))

        # Ensure dimensions are multiples of 16
        w, h = pil_img.size
        new_w = (w // 16) * 16
        new_h = (h // 16) * 16
        if new_w != w or new_h != h:
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

        img_tensor = self.TF.to_tensor(pil_img)
        img_norm = self.TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return img_norm, new_h, new_w

    def _apply_d4_augmentations(self, img_tensor):
        """Apply D4 dihedral group augmentations (8 variations)."""
        torch = self.torch
        TF = self.TF
        augmented = []

        # Original + 3 rotations
        for k in range(4):
            rotated = torch.rot90(img_tensor, k, dims=[1, 2])
            augmented.append(rotated)

        # Horizontal flip + 3 rotations
        flipped = TF.hflip(img_tensor)
        for k in range(4):
            rotated = torch.rot90(flipped, k, dims=[1, 2])
            augmented.append(rotated)

        return torch.stack(augmented, dim=0)

    # ----------------------------------------------------------
    # Low-mag: full-image feature extraction
    # ----------------------------------------------------------

    def extract_low_mag_features(
        self, img_array: np.ndarray, use_tta: bool = True,
        mirror_padding_pct: float = 0.0
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract per-patch features from full low-mag image.

        Returns
        -------
        features : np.ndarray, shape (N_patches, 384)
        grid_shape : (rows, cols)
        """
        if mirror_padding_pct > 0:
            h, w = img_array.shape[:2]

            # Fix C: Snap padding to whole DINOv3 patches (16px each).
            # Using int(h * pct) can produce non-multiples of 16, causing a
            # fractional-patch misalignment when we crop back the feature grid.
            # By rounding to the nearest whole patch, start_row/col are exact.
            patch_size = 16
            n_pad_h = max(1, round(h * mirror_padding_pct / patch_size))
            n_pad_w = max(1, round(w * mirror_padding_pct / patch_size))
            pad_h = n_pad_h * patch_size  # Always a multiple of 16
            pad_w = n_pad_w * patch_size

            # Pad image with reflection
            img_padded = cv2.copyMakeBorder(
                img_array, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT
            )

            print(f"[FeatureExtractor] Applied mirror padding: {mirror_padding_pct:.0%} "
                  f"(+{n_pad_h} patches H, +{n_pad_w} patches W)")
            print(f"  Original: {h}x{w} -> Padded: {img_padded.shape[:2]} "
                  f"(pad_h={pad_h}px, pad_w={pad_w}px, both multiples of 16)")

            # Extract on larger image
            feats_padded, grid_padded = self._extract_features_impl(img_padded, use_tta)

            grid_h_pad, grid_w_pad = grid_padded
            grid_h_orig = h // 16
            grid_w_orig = w // 16

            # Exact crop: padding added exactly n_pad_h patches on each side,
            # so the original image starts at patch index n_pad_h (no rounding needed).
            start_row = n_pad_h
            start_col = n_pad_w
            end_row = start_row + grid_h_orig
            end_col = start_col + grid_w_orig

            # Reshape to 2D grid and crop
            feats_grid = feats_padded.reshape(grid_h_pad, grid_w_pad, -1)
            feats_cropped = feats_grid[start_row:end_row, start_col:end_col, :]

            return feats_cropped.reshape(-1, feats_cropped.shape[-1]), (grid_h_orig, grid_w_orig)
        
        else:
            return self._extract_features_impl(img_array, use_tta)

    def _extract_features_impl(
        self, img_array: np.ndarray, use_tta: bool
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Internal implementation for low-mag extraction."""
        if self.use_onnx:
            return self._onnx.get_low_mag_features_fullimage(img_array, use_tta=use_tta)

        # PyTorch backend
        torch = self.torch
        img_tensor, new_h, new_w = self._preprocess_full_image(img_array)
        grid_h = new_h // 16
        grid_w = new_w // 16
        grid_shape = (grid_h, grid_w)
        n_patches = grid_h * grid_w

        print(f"[FeatureExtractor] Low-mag: {grid_h}×{grid_w} = {n_patches} patches (layer={self.dino_layer})")

        if use_tta:
            augmented_batch = self._apply_d4_augmentations(img_tensor).to(self.device)

            with torch.inference_mode():
                outputs = self.model(augmented_batch)
                all_patch_tokens = self._get_patch_tokens(outputs)[:, NUM_SPECIAL_TOKENS:, :]

                undone_patches = []
                for aug_idx in range(8):
                    patches = all_patch_tokens[aug_idx]

                    if aug_idx in [1, 3, 5, 7]:
                        patches_grid = patches.reshape(grid_w, grid_h, -1)
                    else:
                        patches_grid = patches.reshape(grid_h, grid_w, -1)

                    if aug_idx == 0:
                        undone = patches_grid
                    elif aug_idx == 1:
                        undone = torch.rot90(patches_grid, -1, dims=[0, 1])
                    elif aug_idx == 2:
                        undone = torch.rot90(patches_grid, 2, dims=[0, 1])
                    elif aug_idx == 3:
                        undone = torch.rot90(patches_grid, 1, dims=[0, 1])
                    elif aug_idx == 4:
                        undone = torch.flip(patches_grid, dims=[1])
                    elif aug_idx == 5:
                        undone = torch.rot90(patches_grid, -1, dims=[0, 1])
                        undone = torch.flip(undone, dims=[1])
                    elif aug_idx == 6:
                        undone = torch.rot90(patches_grid, 2, dims=[0, 1])
                        undone = torch.flip(undone, dims=[1])
                    elif aug_idx == 7:
                        undone = torch.rot90(patches_grid, 1, dims=[0, 1])
                        undone = torch.flip(undone, dims=[1])

                    undone_patches.append(undone.reshape(n_patches, -1))

                stacked = torch.stack(undone_patches, dim=0)
                averaged = torch.mean(stacked, dim=0)
                features = averaged / (torch.norm(averaged, dim=1, keepdim=True) + 1e-8)
        else:
            with torch.inference_mode():
                batch = img_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(batch)
                patches = self._get_patch_tokens(outputs)[:, NUM_SPECIAL_TOKENS:, :]
                features = patches.squeeze(0)
                features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)

        return features.cpu().numpy(), grid_shape

    # ----------------------------------------------------------
    # High-mag: single image embedding
    # ----------------------------------------------------------

    def extract_high_mag_features(
        self, img_array: np.ndarray, use_tta: bool = True
    ) -> np.ndarray:
        """
        Extract features from a single high-mag image (256×256).

        Returns
        -------
        embedding : np.ndarray, shape (1, 384)
        """
        if self.use_onnx:
            return self._onnx.get_high_mag_anchor(img_array, use_tta=use_tta)

        # PyTorch backend
        torch = self.torch
        TF = self.TF

        if len(img_array.shape) == 2:
            img_rgb = np.stack((img_array,) * 3, axis=-1)
        else:
            img_rgb = img_array

        pil_img = Image.fromarray(img_rgb.astype(np.uint8))
        img_tensor = TF.to_tensor(pil_img)
        img_norm = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        if use_tta:
            augmented = self._apply_d4_augmentations(img_norm).to(self.device)
            with torch.inference_mode():
                outputs = self.model(augmented)
                patch_tokens = self._get_patch_tokens(outputs)[:, NUM_SPECIAL_TOKENS:, :]
                embeddings = torch.mean(patch_tokens, dim=1)
                embedding = torch.mean(embeddings, dim=0)
                embedding = embedding / (torch.norm(embedding) + 1e-8)
        else:
            with torch.inference_mode():
                batch = img_norm.unsqueeze(0).to(self.device)
                outputs = self.model(batch)
                patch_tokens = self._get_patch_tokens(outputs)[:, NUM_SPECIAL_TOKENS:, :]
                embedding = torch.mean(patch_tokens, dim=1).squeeze(0)
                embedding = embedding / (torch.norm(embedding) + 1e-8)

        return embedding.cpu().numpy().reshape(1, -1)

    # ----------------------------------------------------------
    # Layer selection helper
    # ----------------------------------------------------------

    def _get_patch_tokens(self, outputs):
        """Extract patch tokens from model output, respecting dino_layer config."""
        if self._use_hidden_states:
            # hidden_states[0] = embedding layer, [1] = layer 1, ..., [L+1] = layer L
            return outputs.hidden_states[self.dino_layer + 1]
        else:
            return outputs.last_hidden_state


