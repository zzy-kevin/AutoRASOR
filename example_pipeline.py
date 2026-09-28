"""One-sample AutoRASOR example: random low fields, then LFPS and ambiguity.

Replace the microscope methods below, then run from the repository root::

    python example_pipeline.py --onnx-model path/to/layer5.onnx --output results/sample

The defaults acquire 3 random low fields. For every low field they capture
3 medium images by LFPS and 3 by ambiguity; for every medium image they
capture 3 high images by LFPS and 3 by ambiguity. DINOv3 ViT-S+/16 layer-5
ONNX weights are not distributed here. Use Python 3.10 or newer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from autorasor.config import AutoRASORConfig
from autorasor.feature_extractor_onnx import FeatureExtractorONNX
from autorasor.io import save_run, write_json
from autorasor.pipeline import AutoRASORPipeline


MAGNIFICATIONS = (500, 3500, 24500)  # Here I use 7x per mag change, can also be 14x14
GRID_SHAPE = (7, 7)  # Pool 2x2 DINOv3 patches per candidate for the 7x transition.


class Microscope:
    """Implement these methods for your microscope; no vendor API is assumed."""

    def sample_bounds(self) -> tuple[float, float, float, float]:
        # Your microscope-specific function to identify the sample's valid
        # low-magnification stage-target box: (x_min, x_max, y_min, y_max).
        # Use one consistent coordinate unit for this box and stage movement.
        raise NotImplementedError("Identify the sample bounds with your microscope")

    def capture_low_mag_at(self, x: float, y: float, magnification: int, capture_id: str) -> np.ndarray:
        # Your SEM stage movement function here: move to the randomly selected
        # (x, y) target inside the sample box, then set magnification.
        # Your SEM autofocus function here. Capture a 224x224 grayscale or RGB
        # uint8 image with the metadata bar removed, and register capture_id
        # so a later patch selection can navigate back to this field.
        raise NotImplementedError("Move and capture a low-magnification field")

    def capture_at_patch(self, parent_id: str, row: int, col: int, grid_shape: tuple[int, int], magnification: int, capture_id: str) -> np.ndarray:
        # Your SEM navigation function here: relocate parent_id, move to its
        # selected patch, and set magnification (7x covers one 7x7 grid cell).
        # Your SEM autofocus function here (and autostigmation if available).
        # Return a 224x224 grayscale or RGB uint8 image and register capture_id.
        raise NotImplementedError("Move and capture the selected patch")


def check_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape[:2] != (224, 224) or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("SEM captures must be 224x224 grayscale or RGB uint8 arrays")
    return image


def checked_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    values = np.asarray(bounds, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or values[0] >= values[1] or values[2] >= values[3]:
        raise ValueError("sample_bounds() must return finite (x_min, x_max, y_min, y_max) with positive width and height")
    return tuple(float(value) for value in values)


def candidate_features(extractor: FeatureExtractorONNX, image: np.ndarray) -> np.ndarray:
    """Pool each 2x2 DINO patch block into a normalized 7x7 candidate grid."""
    patches, shape = extractor.extract_low_mag_features(image, use_tta=False)
    if shape != (14, 14):
        raise RuntimeError(f"expected 14x14 DINOv3 patches, got {shape}")
    blocks = patches.reshape(7, 2, 7, 2, -1).mean(axis=(1, 3)).reshape(49, -1)
    return (blocks / (np.linalg.norm(blocks, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


class SEMBackend:
    """Acquire and embed patches from one parent field."""

    def __init__(self, microscope: Microscope, extractor: FeatureExtractorONNX, parent_id: str, magnification: int, output: Path, label: str):
        self.microscope = microscope
        self.extractor = extractor
        self.parent_id = parent_id
        self.magnification = magnification
        self.output = output
        self.label = label
        self.images: dict[int, np.ndarray] = {}
        self.embeddings: dict[int, np.ndarray] = {}

    def capture_id(self, index: int) -> str:
        return f"{self.label}_{self.parent_id}_{index:02d}"

    def acquire(self, index: int) -> np.ndarray:
        row, col = divmod(index, GRID_SHAPE[1])
        capture_id = self.capture_id(index)
        image = check_image(self.microscope.capture_at_patch(self.parent_id, row, col, GRID_SHAPE, self.magnification, capture_id))
        Image.fromarray(image).save(self.output / f"{capture_id}.png")
        embedding = self.extractor.extract_high_mag_features(image, use_tta=False).reshape(-1)
        self.images[index] = image
        self.embeddings[index] = embedding
        return embedding


@dataclass
class FieldRun:
    parent_id: str
    pipeline: AutoRASORPipeline
    backend: SEMBackend
    output: Path


def run_level(fields: list[tuple[str, np.ndarray]], microscope: Microscope, extractor: FeatureExtractorONNX, magnification: int, lfps_budget: int, al_budget: int, seed: int, output: Path, label: str, model_path: Path) -> list[tuple[str, np.ndarray]]:
    """Collect LFPS pairs across fields first, then run per-field ambiguity AL."""
    runs: list[FieldRun] = []
    for field_number, (parent_id, image) in enumerate(fields):
        field_output = output / parent_id
        field_output.mkdir(parents=True, exist_ok=True)
        backend = SEMBackend(microscope, extractor, parent_id, magnification, field_output, label)
        config = AutoRASORConfig(capture_budget=lfps_budget + al_budget, lfps_warmup_count=lfps_budget, seed=seed + field_number)
        pipeline = AutoRASORPipeline(candidate_features(extractor, image), backend, config, grid_shape=GRID_SHAPE)
        pipeline.run_lfps()
        runs.append(FieldRun(parent_id, pipeline, backend, field_output))

    # Each transition has its own archive. Share pairs from other fields so the
    # first AL selection can use the GP after multiple parents have LFPS data.
    archive: list[tuple[str, np.ndarray, np.ndarray]] = []
    for run in runs:
        for index in run.pipeline.selected_current:
            archive.append((run.parent_id, run.pipeline.x_features[index], run.backend.embeddings[index]))

    captured: list[tuple[str, np.ndarray]] = []
    for run in runs:
        for source_id, x_feature, y_feature in archive:
            if source_id != run.parent_id:
                run.pipeline.add_carried_observation(x_feature, y_feature)
        result = run.pipeline.run_al()
        for record in result["observations"]:
            index = record["index"]
            row, col = divmod(index, GRID_SHAPE[1])
            record.update({"parent_id": run.parent_id, "patch_row": row, "patch_col": col, "image": f"{run.backend.capture_id(index)}.png"})
            captured.append((run.backend.capture_id(index), run.backend.images[index]))
            if record["stage"] == "ambiguity_al":
                archive.append((run.parent_id, run.pipeline.x_features[index], run.backend.embeddings[index]))
        result["config"].update({"parent_id": run.parent_id, "magnification": magnification, "lfps_budget": lfps_budget, "al_budget": al_budget, "onnx_model": str(model_path), "use_tta": False})
        save_run(run.output, result)
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-model", type=Path, required=True, help="Exported DINOv3 ViT-S+/16 layer-5 ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="Directory for images and run records")
    parser.add_argument("--low-count", type=int, default=3, help="Random low-magnification stage targets in the sample box")
    parser.add_argument("--medium-lfps", type=int, default=3, help="LFPS medium captures per low field")
    parser.add_argument("--medium-al", type=int, default=3, help="Ambiguity-selected medium captures per low field")
    parser.add_argument("--high-lfps", type=int, default=3, help="LFPS high captures per medium field")
    parser.add_argument("--high-al", type=int, default=3, help="Ambiguity-selected high captures per medium field")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.low_count < 1 or min(args.medium_lfps, args.medium_al, args.high_lfps, args.high_al) < 1:
        parser.error("all capture counts must be positive")
    if args.medium_lfps + args.medium_al > 49 or args.high_lfps + args.high_al > 49:
        parser.error("each parent field has only 49 selectable patches")
    if args.low_count * args.medium_lfps < 5 or args.low_count * (args.medium_lfps + args.medium_al) * args.high_lfps < 5:
        parser.error("at least five LFPS pairs are needed across fields before each ambiguity stage")

    microscope = Microscope()  # Replace with your microscope adapter.
    bounds = checked_bounds(microscope.sample_bounds())
    extractor = FeatureExtractorONNX(onnx_path=str(args.onnx_model))
    rng = np.random.default_rng(args.seed)
    (args.output / "low").mkdir(parents=True, exist_ok=True)
    low_fields: list[tuple[str, np.ndarray]] = []
    low_targets: list[dict] = []
    for number in range(args.low_count):
        x = float(rng.uniform(bounds[0], bounds[1]))
        y = float(rng.uniform(bounds[2], bounds[3]))
        capture_id = f"low_{number:03d}"
        image = check_image(microscope.capture_low_mag_at(x, y, MAGNIFICATIONS[0], capture_id))
        Image.fromarray(image).save(args.output / "low" / f"{capture_id}.png")
        low_fields.append((capture_id, image))
        low_targets.append({"capture_id": capture_id, "x": x, "y": y})

    medium_fields = run_level(low_fields, microscope, extractor, MAGNIFICATIONS[1], args.medium_lfps, args.medium_al, args.seed, args.output / "medium", "medium", args.onnx_model)
    high_fields = run_level(medium_fields, microscope, extractor, MAGNIFICATIONS[2], args.high_lfps, args.high_al, args.seed + args.low_count, args.output / "high", "high", args.onnx_model)
    write_json(args.output / "run_config.json", {"magnifications": MAGNIFICATIONS, "grid_shape": GRID_SHAPE, "sample_bounds": bounds, "low_targets": low_targets, "onnx_model": str(args.onnx_model), "low_count": args.low_count, "medium_lfps": args.medium_lfps, "medium_al": args.medium_al, "high_lfps": args.high_lfps, "high_al": args.high_al, "seed": args.seed})
    print(f"Saved {len(low_fields)} low, {len(medium_fields)} medium, and {len(high_fields)} high magnification images to {args.output}")


if __name__ == "__main__":
    main()
