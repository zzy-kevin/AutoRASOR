"""
ONNX-based Feature Extraction for DINOv3 AL v2 Pipeline
=========================================================
Adapted from `feb01/run_onnx_inference.py` and `al_v2/feature_extractor.py`.

Works on CPU with `onnxruntime` and `numpy==1.26.4`.
Extracts exactly Layer 5 features without PyTorch.
"""

import numpy as np
from PIL import Image
import onnxruntime as ort
import os

# Configuration matching al_v2 DINOv3 extractor
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_SPECIAL_TOKENS = 5
HIGH_MAG_SIZE = 224

DEFAULT_ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dinov3_vits16plus_layer5.onnx")

class FeatureExtractorONNX:
    def __init__(self, onnx_path=None):
        if onnx_path is None:
            onnx_path = DEFAULT_ONNX_PATH

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        print(f"[ONNX] Loading model from: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[ONNX] Model loaded. Input: {self.input_name}, Output: {self.output_name}")

    def _preprocess(self, img_array, target_h, target_w, interpolation=Image.BILINEAR):
        """Convert numpy array -> PIL -> resize -> numpy CHW norm."""
        if len(img_array.shape) == 2:
            img_rgb = np.stack((img_array,) * 3, axis=-1)
        else:
            img_rgb = img_array

        pil_img = Image.fromarray(img_rgb.astype(np.uint8))
        if target_h is not None and target_w is not None:
            pil_img = pil_img.resize((target_w, target_h), interpolation)

        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        img_norm = (img_np - IMAGENET_MEAN) / IMAGENET_STD
        img_chw = np.transpose(img_norm, (2, 0, 1)).astype(np.float32)
        return img_chw

    def _apply_d4_augmentations(self, img_chw):
        """Apply D4 dihedral group augmentations."""
        augmented = []
        # Original + 3 rotations
        for k in range(4):
            rotated = np.rot90(img_chw, k, axes=(1, 2))
            augmented.append(rotated)
        # Horizontal flip + 3 rotations
        flipped = np.flip(img_chw, axis=2)
        for k in range(4):
            rotated = np.rot90(flipped, k, axes=(1, 2))
            augmented.append(rotated)
        return np.stack(augmented, axis=0)

    def extract_high_mag_features(self, img_array, use_tta=True):
        """Extract an image embedding from a 224x224 capture."""
        img_chw = self._preprocess(img_array, HIGH_MAG_SIZE, HIGH_MAG_SIZE, Image.LANCZOS)

        if use_tta:
            batch = self._apply_d4_augmentations(img_chw)
        else:
            batch = np.expand_dims(img_chw, axis=0)

        outputs = self.session.run(None, {self.input_name: batch})[0]
        patch_features = outputs[:, NUM_SPECIAL_TOKENS:, :]
        embeddings = np.mean(patch_features, axis=1) # GAP over tokens
        embedding = np.mean(embeddings, axis=0)      # Mean over augs

        norm = np.linalg.norm(embedding) + 1e-8
        embedding_norm = embedding / norm
        return embedding_norm.reshape(1, -1)

    def extract_low_mag_features(self, img_array, use_tta=True):
        """
        Full-image TTA extraction.

        TTA note: D4 augmentations include 90° and 270° rotations which swap H
        and W.  For a non-square image (H ≠ W) these produce tensors of a
        different spatial size than the 0°/180° variants, making batching
        impossible.  We therefore run each of the 8 augmentations individually
        (no batching) and undo the spatial transform on the patch-token grid.
        """
        h, w = img_array.shape[:2]
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16

        img_chw = self._preprocess(img_array, new_h, new_w)
        grid_h, grid_w = new_h // 16, new_w // 16
        grid_shape = (grid_h, grid_w)
        n_patches   = grid_h * grid_w

        print(f"[ONNX] Low-mag extract: {grid_h}x{grid_w} = {n_patches} patches "
              f"({'TTA x8' if use_tta else 'no TTA'})...")

        if use_tta:
            # Build the 8 D4 augmentations, matching _apply_d4_augmentations order:
            #   idx 0-3 : rot90 k=0,1,2,3   (no flip)
            #   idx 4-7 : rot90 k=0,1,2,3   (horizontal flip first)
            flipped_chw = np.flip(img_chw, axis=2)

            undone_patches = []
            for aug_idx in range(8):
                k          = aug_idx % 4
                base_img   = img_chw if aug_idx < 4 else flipped_chw
                aug_img    = np.rot90(base_img, k, axes=(1, 2))

                # --- Forward pass (single image, no batching) ---
                single = np.expand_dims(np.ascontiguousarray(aug_img), axis=0)
                output = self.session.run(None, {self.input_name: single})[0]
                patches = output[0, NUM_SPECIAL_TOKENS:, :]  # (n_aug_patches, feat)

                # 90° and 270° rotations swapped H↔W, so the output grid is also
                # transposed relative to the original grid.
                if k in (1, 3):
                    a_gh, a_gw = grid_w, grid_h   # swapped
                else:
                    a_gh, a_gw = grid_h, grid_w   # original

                patches_grid = patches.reshape(a_gh, a_gw, -1)

                # --- Undo spatial transform on the patch grid ---
                if aug_idx == 0:                         # identity
                    undone = patches_grid
                elif aug_idx == 1:                       # rot90 k=1  → undo k=-1
                    undone = np.rot90(patches_grid, -1, axes=(0, 1))
                elif aug_idx == 2:                       # rot90 k=2  → undo k=2
                    undone = np.rot90(patches_grid,  2, axes=(0, 1))
                elif aug_idx == 3:                       # rot90 k=3  → undo k=1
                    undone = np.rot90(patches_grid,  1, axes=(0, 1))
                elif aug_idx == 4:                       # flip        → undo flip
                    undone = np.flip(patches_grid, axis=1)
                elif aug_idx == 5:                       # flip→rot1   → rot-1→flip
                    undone = np.rot90(patches_grid, -1, axes=(0, 1))
                    undone = np.flip(undone, axis=1)
                elif aug_idx == 6:                       # flip→rot2   → rot2→flip
                    undone = np.rot90(patches_grid,  2, axes=(0, 1))
                    undone = np.flip(undone, axis=1)
                elif aug_idx == 7:                       # flip→rot3   → rot1→flip
                    undone = np.rot90(patches_grid,  1, axes=(0, 1))
                    undone = np.flip(undone, axis=1)

                # undone is always (grid_h, grid_w, feat) after the inverse
                undone_patches.append(undone.reshape(n_patches, -1))

            stacked  = np.stack(undone_patches, axis=0)   # (8, n_patches, feat)
            averaged = np.mean(stacked, axis=0)
            norms    = np.linalg.norm(averaged, axis=1, keepdims=True) + 1e-8
            features = averaged / norms

        else:
            batch   = np.expand_dims(img_chw, axis=0)
            outputs = self.session.run(None, {self.input_name: batch})[0]
            patches = outputs[0, NUM_SPECIAL_TOKENS:, :]
            norms   = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8
            features = patches / norms

        return features, grid_shape
