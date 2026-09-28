"""Hardware-independent two-stage AutoRASOR acquisition controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .config import AutoRASORConfig
from .engine import UncertaintyEngineV4


class AcquisitionBackend(Protocol):
    """Return the pooled high-magnification feature for a candidate index."""

    def acquire(self, index: int) -> np.ndarray:
        ...


@dataclass
class ObservationRecord:
    step: int
    stage: str
    index: int
    row: Optional[int]
    column: Optional[int]
    y_norm: float


@dataclass
class SelectionRecord:
    step: int
    stage: str
    indices: List[int]
    acquisition_scores: List[Optional[float]]


@dataclass
class MetricRecord:
    step: int
    stage: str
    n_captured: int
    percent_captured: float
    n_carried: int
    predicted_ambiguity_max: Optional[float]
    predicted_uncertainty_max: Optional[float]


EngineFactory = Callable[..., UncertaintyEngineV4]


class AutoRASORPipeline:
    """Run LFPS warmup followed by ambiguity-driven qLogNEI acquisition.

    Carried observations train the surrogate through ``add_observation_direct``.
    They never enter ``selected_current`` and therefore can neither consume the
    current field's budget nor be returned as current-field candidate indices.
    """

    def __init__(
        self,
        x_features: np.ndarray,
        backend: AcquisitionBackend,
        config: Optional[AutoRASORConfig] = None,
        carried_observations: Sequence[Tuple[np.ndarray, np.ndarray]] = (),
        grid_shape: Optional[Tuple[int, int]] = None,
        engine_factory: EngineFactory = UncertaintyEngineV4,
    ) -> None:
        self.config = config or AutoRASORConfig()
        self.x_features = np.asarray(x_features, dtype=np.float32)
        if self.x_features.ndim != 2 or self.x_features.shape[0] == 0:
            raise ValueError("x_features must have shape (n_candidates, feature_dim)")
        if self.config.capture_budget > self.x_features.shape[0]:
            raise ValueError("capture_budget cannot exceed the current candidate pool")
        if grid_shape is not None and int(np.prod(grid_shape)) != self.x_features.shape[0]:
            raise ValueError("grid_shape does not match the candidate pool")
        self.backend = backend
        self.grid_shape = grid_shape
        self.engine = engine_factory(
            x_features=self.x_features,
            input_dim=self.config.input_dim,
            metric=self.config.metric,
            k_neighbors_ambiguity=self.config.k_neighbors_ambiguity,
            acq_function=self.config.acquisition,
            normalize_features=self.config.normalize_features,
            fit_mode=self.config.fit_mode,
            noise_mode=self.config.noise_mode,
            kernel=self.config.kernel,
        )
        self.n_carried = 0
        self._lfps_done = False
        self._al_done = False
        for x_feat, y_feat in carried_observations:
            self.add_carried_observation(x_feat, y_feat)
        self.selected_current: List[int] = []
        self._selected_set: set[int] = set()
        self.observations: List[ObservationRecord] = []
        self.selections: List[SelectionRecord] = []
        self.metrics: List[MetricRecord] = []

    def add_carried_observation(self, x_feat: np.ndarray, y_feat: np.ndarray) -> None:
        """Add a pair from another field without making it selectable here."""
        if self._al_done:
            raise RuntimeError("cannot add observations after active learning has finished")
        if self.engine.add_observation_direct(x_feat, y_feat, dedup_label=None):
            self.n_carried += 1

    def _coordinates(self, index: int) -> Tuple[Optional[int], Optional[int]]:
        if self.grid_shape is None:
            return None, None
        return divmod(index, self.grid_shape[1])

    def _observe(self, index: int, step: int, stage: str) -> None:
        if index in self._selected_set:
            raise RuntimeError(f"candidate {index} was selected more than once")
        y_feat = np.asarray(self.backend.acquire(index), dtype=np.float32).reshape(-1)
        if not self.engine.add_observation(index, y_feat):
            raise RuntimeError(f"engine rejected current-field observation {index}")
        self._selected_set.add(index)
        self.selected_current.append(index)
        row, column = self._coordinates(index)
        self.observations.append(ObservationRecord(step, stage, index, row, column, float(np.linalg.norm(y_feat))))

    def _record_metrics(self, step: int, stage: str, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> None:
        mean_max = None if mean is None or np.asarray(mean).size == 0 else float(np.nanmax(mean))
        std_max = None if std is None or np.asarray(std).size == 0 else float(np.nanmax(std))
        self.metrics.append(MetricRecord(
            step=step,
            stage=stage,
            n_captured=len(self.selected_current),
            percent_captured=100.0 * len(self.selected_current) / self.x_features.shape[0],
            n_carried=self.n_carried,
            predicted_ambiguity_max=mean_max,
            predicted_uncertainty_max=std_max,
        ))

    def run_lfps(self) -> None:
        """Capture LFPS pairs; callers may then share them across fields."""
        if self._lfps_done:
            return
        warmup_count = min(self.config.lfps_warmup_count, self.config.capture_budget)
        warmup = self.engine.warmup_strategy(n_seeds=warmup_count, mode="lfps", rng_seed=self.config.seed) if warmup_count else []
        warmup = [int(i) for i in warmup if int(i) not in self._selected_set][:warmup_count]
        if warmup:
            self.selections.append(SelectionRecord(0, "lfps_warmup", warmup, [None] * len(warmup)))
            for index in warmup:
                self._observe(index, 0, "lfps_warmup")
        self._record_metrics(0, "lfps_warmup")
        self._lfps_done = True

    def run_al(self) -> Dict[str, object]:
        """Complete the remaining budget with ambiguity-driven selection."""
        self.run_lfps()
        if self._al_done:
            return self._result()
        step = 1
        while len(self.selected_current) < self.config.capture_budget:
            remaining = self.config.capture_budget - len(self.selected_current)
            batch_size = min(self.config.batch_size, remaining)
            chosen, scores, mean, std = self.engine.fit_surrogate_and_select(
                batch_size=batch_size,
                training_iters=self.config.training_iters,
            )
            indices: List[int] = []
            for value in np.atleast_1d(chosen):
                index = int(value)
                if 0 <= index < self.x_features.shape[0] and index not in self._selected_set:
                    indices.append(index)
                if len(indices) == batch_size:
                    break
            if not indices:
                raise RuntimeError("acquisition returned no unsampled current-field candidates")
            score_array = np.asarray(scores).reshape(-1) if scores is not None else np.array([])
            selected_scores = [float(score_array[i]) if i < score_array.size else None for i in indices]
            self.selections.append(SelectionRecord(step, "ambiguity_al", indices, selected_scores))
            for index in indices:
                self._observe(index, step, "ambiguity_al")
            self._record_metrics(step, "ambiguity_al", mean, std)
            step += 1

        self._al_done = True
        return self._result()

    def run(self) -> Dict[str, object]:
        """Run both stages for a single field."""
        return self.run_al()

    def _result(self) -> Dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "observations": [asdict(item) for item in self.observations],
            "selections": [asdict(item) for item in self.selections],
            "metrics": [asdict(item) for item in self.metrics],
        }

    def carried_observations(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return all learned pairs for transfer into a later field."""
        return [(np.asarray(x).copy(), np.asarray(y).copy()) for x, y in zip(self.engine.archive_x, self.engine.archive_y)]
