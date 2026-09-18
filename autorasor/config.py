"""Validated configuration for the supported public AutoRASOR pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class AutoRASORConfig:
    """Settings shared by simulation, replay, and microscope entry points.

    The public workflow intentionally exposes one scientific configuration. This
    prevents legacy engine defaults or ablation settings from silently changing a
    run launched through a different entry point.
    """

    dino_layer: int = 5
    input_dim: int = 0
    metric: str = "cosine"
    k_neighbors_ambiguity: int = 10
    normalize_features: bool = True
    fit_mode: str = "mll"
    noise_mode: str = "jackknife"
    kernel: str = "matern"
    acquisition: str = "qlognei"
    batch_size: int = 2
    lfps_warmup_count: int = 8
    capture_budget: int = 40
    seed: int = 42
    training_iters: int = 50

    def __post_init__(self) -> None:
        fixed = {
            "dino_layer": (self.dino_layer, 5),
            "input_dim": (self.input_dim, 0),
            "metric": (self.metric, "cosine"),
            "normalize_features": (self.normalize_features, True),
            "fit_mode": (self.fit_mode, "mll"),
            "noise_mode": (self.noise_mode, "jackknife"),
            "kernel": (self.kernel, "matern"),
            "acquisition": (self.acquisition, "qlognei"),
        }
        invalid = [f"{name}={actual!r} (expected {expected!r})" for name, (actual, expected) in fixed.items() if actual != expected]
        if invalid:
            raise ValueError("Unsupported public AutoRASOR configuration: " + ", ".join(invalid))
        if self.k_neighbors_ambiguity < 2:
            raise ValueError("k_neighbors_ambiguity must be at least 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.lfps_warmup_count < 0:
            raise ValueError("lfps_warmup_count cannot be negative")
        if self.capture_budget < 1:
            raise ValueError("capture_budget must be positive")
        if self.lfps_warmup_count > self.capture_budget:
            raise ValueError("lfps_warmup_count cannot exceed capture_budget")
        if self.training_iters < 1:
            raise ValueError("training_iters must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
