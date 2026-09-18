"""DINOv3 extraction helpers (PyTorch path used for offline analysis)."""
from typing import Tuple

import numpy as np
import torch
import cv2
from PIL import Image


class PyTorchDINOv3Extractor:
    def __init__(self, device="cuda"):
        from transformers import AutoModel
        import torch
        import torchvision.transforms.functional as TF
        self.device = device
        self.torch = torch
        self.TF = TF
        print(f"[Extractor] Loading PyTorch DINOv3 model 'facebook/dinov3-vits16plus-pretrain-lvd1689m' on {device} (layer=5)...")
        # We enforce output_hidden_states=True to grab exactly layer 5
        self.model = AutoModel.from_pretrained(
            'facebook/dinov3-vits16plus-pretrain-lvd1689m',
            output_hidden_states=True
        ).to(device).eval()

    # ------------------------------------------------------------------
    # D4 dihedral TTA helpers
    # ------------------------------------------------------------------

    def _apply_d4_augmentations(self, img_tensor):
        """Return a (8, C, H, W) batch containing the D4 dihedral group of
        augmentations of *img_tensor* (shape C×H×W):
          indices 0-3 : original + 90°/180°/270° rotations
          indices 4-7 : H-flip  + 90°/180°/270° rotations
        """
        torch = self.torch
        TF    = self.TF
        augmented = []
        for k in range(4):
            augmented.append(torch.rot90(img_tensor, k, dims=[1, 2]))
        flipped = TF.hflip(img_tensor)
        for k in range(4):
            augmented.append(torch.rot90(flipped, k, dims=[1, 2]))
        return torch.stack(augmented, dim=0)  # (8, C, H, W)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_raw_features(self, img_array, use_tta: bool = False):
        """Extract per-token patch features from a 224×224 image.

        Parameters
        ----------
        img_array : np.ndarray  H×W (grayscale) or H×W×3 (RGB), uint8.
        use_tta   : bool
            If True, apply D4 dihedral TTA (8 augmentations), invert each
            transformation back to the canonical grid orientation, then
            mean-average the 8 token grids before returning.
            If False (default), a single forward pass is used (original
            behaviour).

        Returns
        -------
        patch_tokens : np.ndarray, shape (196, 384)
            One row per spatial patch token, in canonical row-major order.
        """
        import numpy as np
        from PIL import Image
        torch = self.torch
        TF    = self.TF

        # ── Pre-process ────────────────────────────────────────────────
        if len(img_array.shape) == 2:
            img_rgb = np.stack([img_array] * 3, axis=-1)
        else:
            img_rgb = img_array

        pil_img  = Image.fromarray(img_rgb.astype(np.uint8))
        img_tensor = TF.to_tensor(pil_img)
        img_norm   = TF.normalize(img_tensor,
                                  mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))  # (C, H, W)

        # For a 224×224 image the patch grid is always 14×14 = 196 tokens.
        grid_h = pil_img.height // 16
        grid_w = pil_img.width  // 16
        n_patches = grid_h * grid_w

        if use_tta:
            # ── TTA path: 8 augmented views in a single batched pass ───
            aug_batch = self._apply_d4_augmentations(img_norm).to(self.device)

            with torch.inference_mode():
                outputs    = self.model(aug_batch)
                layer5_out = outputs.hidden_states[5]          # (8, N_tokens, 384)
                # Drop CLS + register tokens
                all_tokens = layer5_out[:, _DINO_NUM_SPECIAL_TOKENS:, :]  # (8, 196, 384)

            # Invert each D4 transform back to canonical orientation.
            # Augmentation index → inverse operation (same as autorasor/feature_extractor.py):
            #   0: original           → identity
            #   1: rot90 k=1          → rot90 k=-1  (i.e. rot90 k=3)
            #   2: rot90 k=2          → rot90 k=2
            #   3: rot90 k=3          → rot90 k=1
            #   4: hflip              → hflip
            #   5: hflip + rot90 k=1  → rot90 k=-1 then hflip
            #   6: hflip + rot90 k=2  → rot90 k=2  then hflip
            #   7: hflip + rot90 k=3  → rot90 k=1  then hflip
            undone = []
            for aug_idx in range(8):
                # Tokens need to be reshaped into a spatial grid to apply
                # the inverse geometric transform correctly.
                # After rot90 k=odd the spatial axes swap → use (grid_w, grid_h).
                tok = all_tokens[aug_idx].cpu()  # (196, 384)
                if aug_idx in (1, 3, 5, 7):
                    grid = tok.reshape(grid_w, grid_h, -1)
                else:
                    grid = tok.reshape(grid_h, grid_w, -1)

                if aug_idx == 0:
                    inv = grid
                elif aug_idx == 1:
                    inv = torch.rot90(grid, -1, dims=[0, 1])
                elif aug_idx == 2:
                    inv = torch.rot90(grid,  2, dims=[0, 1])
                elif aug_idx == 3:
                    inv = torch.rot90(grid,  1, dims=[0, 1])
                elif aug_idx == 4:
                    inv = torch.flip(grid, dims=[1])
                elif aug_idx == 5:
                    inv = torch.flip(torch.rot90(grid, -1, dims=[0, 1]), dims=[1])
                elif aug_idx == 6:
                    inv = torch.flip(torch.rot90(grid,  2, dims=[0, 1]), dims=[1])
                else:  # aug_idx == 7
                    inv = torch.flip(torch.rot90(grid,  1, dims=[0, 1]), dims=[1])

                undone.append(inv.reshape(n_patches, -1))  # (196, 384)

            # Mean-average the 8 inverse-transformed token grids
            stacked  = torch.stack(undone, dim=0)         # (8, 196, 384)
            averaged = torch.mean(stacked, dim=0)         # (196, 384)
            patch_tokens = averaged.numpy()

        else:
            # ── Single-pass path (original behaviour) ──────────────────
            batch = img_norm.unsqueeze(0).to(self.device)
            with torch.inference_mode():
                outputs    = self.model(batch)
                layer5_out = outputs.hidden_states[5]
                patch_tokens = layer5_out[0, _DINO_NUM_SPECIAL_TOKENS:, :].cpu().numpy()

        return patch_tokens  # (196, 384)


def prepare_for_dino(raw_image: np.ndarray):
    """
    Crop metadata bar, center square, and resize to exactly 224x224 for DINOv3.
    Matches the SEM grid_capture.py layout.
    """
    h, w = raw_image.shape[:2]
    # Content is the top 896 rows; the bottom rows hold the SEM scale/metadata bar.
    # Use min(896, h) so the bar is stripped for the 956-tall capture PNGs too (the old
    # `896 if h >= 960 else h` left the bar in, leaking it into the bottom grid row).
    # Matches the live runtime's fixed CROPPED_HEIGHT crop in sem_runners/run_LFPS_sem.py.
    cropped_height = min(896, h)
    cropped = raw_image[:cropped_height, :]
    
    h_c, w_c = cropped.shape[:2]
    sq_side_raw = min(h_c, w_c)
    x_off_raw = (w_c - sq_side_raw) // 2
    y_off_raw = (h_c - sq_side_raw) // 2
    square = cropped[y_off_raw:y_off_raw + sq_side_raw, x_off_raw:x_off_raw + sq_side_raw]
    dino_input = cv2.resize(square, (224, 224), interpolation=cv2.INTER_LINEAR)
    return dino_input


def pool_embeddings(embeddings: np.ndarray, grid_shape: Tuple[int, int], df: int):
    """
    Pools the low-mag embedding grid if downscale-factor > 1.
    """
    gh, gw = grid_shape
    if df == 1:
        return embeddings
        
    new_gh = gh // df
    new_gw = gw // df
    grid = embeddings.reshape(gh, gw, -1)
    pooled = np.zeros((new_gh, new_gw, embeddings.shape[-1]))
    
    for r in range(new_gh):
        for c in range(new_gw):
            block = grid[r*df:(r+1)*df, c*df:(c+1)*df, :]
            mean_vec = np.mean(block, axis=(0, 1))
            norm = np.linalg.norm(mean_vec) + 1e-8
            pooled[r, c] = mean_vec / norm
            
    return pooled.reshape(-1, embeddings.shape[-1])
