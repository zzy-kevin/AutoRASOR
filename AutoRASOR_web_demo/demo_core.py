"""
Session + run state for the AutoRASOR web demo.
===============================================

Wraps `autorasor.engine.UncertaintyEngineV4` in the canonical configuration
(L2-normalized 384-D DINOv3 features, Matern-5/2 ARD, `fit_gpytorch_mll`,
jackknife heteroskedastic noise) and drives it one acquisition step at a time so
a browser can scrub back and forth through a campaign.

State model
-----------
`state_k` = (archive after k steps, GP posterior fitted on that archive, and the
pick that posterior recommends next). Advancing means: capture `state_k.pending`,
refit, cache `state_k+1`. The engine object always sits at the frontier; the
cursor only scrubs cached history, so stepping backwards is free and exact.

Knowledge carried between micrographs lives in `carried`: raw (x_low, y_high)
pairs replayed into a fresh engine through `add_observation_direct`, which is the
engine's documented cross-dataset path.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
for _p in (str(_HERE), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autorasor.config import AutoRASORConfig  # noqa: E402
from autorasor.engine import UncertaintyEngineV4  # noqa: E402

from pipeline import Micrograph, MicrographParams, Pipeline  # noqa: E402

TOP_K = 20            # canonical benchmark top-k for recall / discovery
MIN_GP_OBS = 5        # engine falls back to random picks below this


# ============================================================
# Configuration
# ============================================================

EngineConfig = AutoRASORConfig


@dataclass
class RunConfig:
    policy: str = 'ambiguity'     # 'ambiguity' | 'lfps' | 'random'
    budget: int = 40              # total high-mag captures on THIS micrograph
    batch_size: int = 2
    acq: str = 'qlognei'
    beta: float = 1.0
    warmup_seeds: int = 8         # LFPS warmup - active learning only
    seed: int = 42

    def sanitized(self, n_points: int) -> "RunConfig":
        policy = self.policy if self.policy in ('ambiguity', 'lfps', 'random') else 'ambiguity'
        acq = 'qlognei'
        budget = int(np.clip(self.budget, 1, n_points))
        warmup = int(np.clip(self.warmup_seeds, 0, budget)) if policy == 'ambiguity' else 0
        return RunConfig(
            policy=policy,
            budget=budget,
            batch_size=int(np.clip(self.batch_size, 1, 16)),
            acq=acq,
            beta=float(np.clip(self.beta, 0.0, 10.0)),
            warmup_seeds=warmup,
            seed=int(self.seed),
        )


@dataclass
class Capture:
    idx: int
    step: int
    kind: str          # 'warmup' | 'policy'


@dataclass
class StepState:
    step: int
    captures: List[Capture]
    pending: List[int]
    gp_mean: Optional[np.ndarray]
    gp_std: Optional[np.ndarray]
    gp_acq: Optional[np.ndarray]
    gp_active: bool
    gp_flat: bool
    obs_ambiguity: Dict[int, float]
    metrics: Dict[str, float]
    n_carried: int
    fit_seconds: float


# ============================================================
# Session
# ============================================================

class DemoSession:
    """One browser session: a micrograph, a surrogate, and a scrubbable campaign."""

    def __init__(self, pipeline: Pipeline, engine_cfg: Optional[EngineConfig] = None):
        self.pipeline = pipeline
        self.engine_cfg = engine_cfg or EngineConfig()
        self.micrograph: Optional[Micrograph] = None
        self.engine: Optional[UncertaintyEngineV4] = None
        self.run_cfg: RunConfig = RunConfig()
        self.history: List[StepState] = []
        self.cursor: int = 0
        self.micrograph_serial: int = 0

        # (x_low, y_high) pairs learned on *previous* micrographs
        self.carried: List[Tuple[np.ndarray, np.ndarray]] = []

        self._pca_cache: Dict[str, dict] = {}
        self._pick_steps: Dict[int, Capture] = {}
        self._lfps_sequence: Optional[List[int]] = None
        self._rng = np.random.RandomState(0)
        self.status: str = 'idle'
        self.progress: Tuple[str, float] = ('', 0.0)

    # -- micrograph ------------------------------------------------------
    def new_micrograph(self, params: MicrographParams, keep_gpr: bool = True) -> Micrograph:
        """Build a new field. `keep_gpr` folds the current archive into `carried`."""
        if keep_gpr and self.engine is not None and self.engine.n_observed > 0:
            self.carried = [
                (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
                for x, y in zip(self.engine.archive_x, self.engine.archive_y)
            ]
        elif not keep_gpr:
            self.carried = []

        def on_progress(msg: str, frac: float) -> None:
            self.progress = (msg, frac)

        # Stay 'building' until the run has been re-seeded too: clients poll this
        # flag and would otherwise read a half-built session.
        self.status = 'building'
        try:
            self.micrograph = self.pipeline.build(params, progress=on_progress)
            self.micrograph_serial += 1
            self._pca_cache = {}
            on_progress('Re-fitting the surrogate on the new field', 0.97)
            self.start_run(self.run_cfg)
        finally:
            self.status = 'idle'
        return self.micrograph

    def reset_gpr(self) -> None:
        """Forget everything the surrogate ever saw, keep the current micrograph."""
        self.carried = []
        self.start_run(self.run_cfg)

    # -- engine ----------------------------------------------------------
    def _rebuild_engine(self) -> None:
        if self.micrograph is None:
            raise RuntimeError('no micrograph loaded')
        cfg = self.engine_cfg
        engine = UncertaintyEngineV4(
            x_features=self.micrograph.lm_feats,
            input_dim=0,                       # 384-D raw features; no PCA in the AL path
            metric=cfg.metric,
            k_neighbors_ambiguity=cfg.k_neighbors_ambiguity,
            acq_function=self.run_cfg.acq,
            normalize_features=cfg.normalize_features,
            fit_mode=cfg.fit_mode,
            noise_mode=cfg.noise_mode,
            kernel=cfg.kernel,
        )
        for x_feat, y_feat in self.carried:
            engine.add_observation_direct(x_feat, y_feat, dedup_label=None)
        self.engine = engine

    # -- run -------------------------------------------------------------
    def start_run(self, cfg: RunConfig) -> None:
        """Reset the campaign on the current micrograph (carried knowledge survives)."""
        if self.micrograph is None:
            raise RuntimeError('no micrograph loaded')
        self.run_cfg = cfg.sanitized(self.micrograph.n_points)
        self.engine_cfg = replace(
            self.engine_cfg,
            batch_size=self.run_cfg.batch_size,
            lfps_warmup_count=self.run_cfg.warmup_seeds,
            capture_budget=self.run_cfg.budget,
            seed=self.run_cfg.seed,
        )
        self._rebuild_engine()
        self._pick_steps = {}
        self.history = []
        self.cursor = 0
        self._rng = np.random.RandomState(self.run_cfg.seed)
        self._lfps_sequence = None

        if self.run_cfg.policy == 'lfps':
            # Canonical LFPS: the whole ordering is known from the low-mag latents
            # alone, before a single capture. Never re-implement it here.
            self._lfps_sequence = list(self.engine.warmup_strategy(
                n_seeds=self.micrograph.n_points, mode='lfps', rng_seed=self.run_cfg.seed
            ))
        elif self.run_cfg.policy == 'ambiguity' and self.run_cfg.warmup_seeds > 0:
            # Warmup applies ONLY to the active-learning policy.
            warmup = self.engine.warmup_strategy(
                n_seeds=self.run_cfg.warmup_seeds, mode='lfps', rng_seed=self.run_cfg.seed
            )
            for idx in warmup[:self.run_cfg.budget]:
                self._observe(int(idx), step=0, kind='warmup')

        self.history.append(self._compute_state(0))

    def step_forward(self) -> bool:
        """Advance the cursor, extending the frontier with a real capture if needed."""
        if not self.history:
            return False
        if self.cursor < len(self.history) - 1:
            self.cursor += 1
            return True
        frontier = self.history[-1]
        if not frontier.pending:
            return False
        for idx in frontier.pending:
            self._observe(int(idx), step=frontier.step + 1, kind='policy')
        self.history.append(self._compute_state(frontier.step + 1))
        self.cursor = len(self.history) - 1
        return True

    def step_back(self) -> bool:
        if self.cursor <= 0:
            return False
        self.cursor -= 1
        return True

    def goto(self, step: int) -> bool:
        step = int(step)
        while step > len(self.history) - 1:
            if not self.step_forward():
                break
        self.cursor = int(np.clip(step, 0, len(self.history) - 1))
        return True

    def run_to_budget(self, max_steps: int = 1000) -> int:
        """Jump to the end of the campaign, extending it to the budget if needed."""
        done = 0
        self.cursor = len(self.history) - 1
        while done < max_steps:
            frontier = self.history[-1]
            if not frontier.pending:
                break
            if not self.step_forward():
                break
            done += 1
        self.cursor = len(self.history) - 1
        return done

    @property
    def state(self) -> StepState:
        return self.history[self.cursor]

    # -- internals -------------------------------------------------------
    def _observe(self, idx: int, step: int, kind: str) -> None:
        assert self.engine is not None and self.micrograph is not None
        if idx in self._pick_steps:
            return
        if self.engine.add_observation(idx, self.micrograph.hm_feats[idx]):
            self._pick_steps[idx] = Capture(idx=idx, step=step, kind=kind)

    def _compute_state(self, step: int) -> StepState:
        assert self.engine is not None and self.micrograph is not None
        cfg = self.run_cfg
        n_captured = len(self._pick_steps)
        want = int(min(cfg.batch_size, max(cfg.budget - n_captured, 0)))
        observed = set(self._pick_steps.keys())

        t0 = time.time()
        pending: List[int] = []
        if cfg.policy == 'ambiguity':
            selected, acq_scores, gp_mean, gp_std = self.engine.fit_surrogate_and_select(
                batch_size=want, beta_param=cfg.beta
            )
            pending = [int(i) for i in np.atleast_1d(selected)][:want]
        else:
            # LFPS / random still get a surrogate fit so the learned-ambiguity
            # panel keeps updating - it just does not drive the selection.
            _, acq_scores, gp_mean, gp_std = self.engine.fit_surrogate_and_select(
                batch_size=0, beta_param=cfg.beta
            )
            if want > 0:
                if cfg.policy == 'lfps':
                    seq = self._lfps_sequence or []
                    pending = [int(i) for i in seq if int(i) not in observed][:want]
                else:
                    candidates = [i for i in range(self.micrograph.n_points) if i not in observed]
                    if candidates:
                        take = min(want, len(candidates))
                        pending = [int(i) for i in self._rng.choice(candidates, size=take, replace=False)]
        fit_seconds = time.time() - t0

        gp_active = self.engine.n_observed >= MIN_GP_OBS
        gp_mean_arr = np.asarray(gp_mean, dtype=np.float64).ravel()
        gp_std_arr = np.asarray(gp_std, dtype=np.float64).ravel()
        gp_acq_arr = np.asarray(acq_scores, dtype=np.float64).ravel()
        gp_flat = bool(gp_active and float(np.ptp(gp_mean_arr)) < 1e-6)

        obs_ambiguity: Dict[int, float] = {}
        if self.engine.n_observed >= 2:
            amb, _, _ = self.engine.compute_ambiguity(np.array(self.engine.archive_indices))
            for pos, arch_idx in enumerate(self.engine.archive_indices):
                if int(arch_idx) in self._pick_steps:
                    obs_ambiguity[int(arch_idx)] = float(amb[pos])

        return StepState(
            step=step,
            captures=sorted(self._pick_steps.values(), key=lambda c: (c.step, c.idx)),
            pending=pending,
            gp_mean=gp_mean_arr if gp_active else None,
            gp_std=gp_std_arr if gp_active else None,
            gp_acq=gp_acq_arr if gp_active else None,
            gp_active=gp_active,
            gp_flat=gp_flat,
            obs_ambiguity=obs_ambiguity,
            metrics=self._metrics(gp_mean_arr if gp_active else None, list(observed)),
            n_carried=len(self.carried),
            fit_seconds=fit_seconds,
        )

    def _metrics(self, gp_mean: Optional[np.ndarray], captured: List[int]) -> Dict[str, float]:
        assert self.engine is not None and self.micrograph is not None
        gt = self.micrograph.gt_ambiguity.astype(np.float64)
        k = min(TOP_K, gt.size)
        top_gt = set(np.argsort(gt)[-k:].tolist())
        out: Dict[str, float] = {
            'n_captured': float(len(captured)),
            'pct_captured': 100.0 * len(captured) / gt.size,
            'discovery': 100.0 * len(top_gt & set(captured)) / k,
            'best_gt_found': float(np.max(gt[captured])) if captured else 0.0,
            'gt_max': float(np.max(gt)),
        }

        if gp_mean is not None and float(np.ptp(gp_mean)) > 1e-9:
            out['spearman'] = float(spearmanr(gp_mean, gt).statistic)
            top_pred = set(np.argsort(gp_mean)[-k:].tolist())
            out['recall'] = 100.0 * len(top_gt & top_pred) / k

        # LFPS's own objective: how far the worst-covered patch sits from the archive.
        if self.engine.archive_x_gp:
            archive = np.asarray(self.engine.archive_x_gp, dtype=np.float64)
            dmin = cosine_distances(self.engine.x_gp.astype(np.float64), archive).min(axis=1)
            out['covering_radius'] = float(dmin.max())
            out['mean_min_dist'] = float(dmin.mean())
        return out

    # -- embeddings for display -----------------------------------------
    def embedding_pca(self, which: str) -> dict:
        """PCA(3) of the patch embeddings; PC1-3 drive the RGB layer, PC1/PC2 the scatter."""
        if which not in ('low', 'high'):
            raise KeyError("embedding_pca expects 'low' or 'high'")
        if which in self._pca_cache:
            return self._pca_cache[which]
        assert self.micrograph is not None
        feats = self.micrograph.lm_feats if which == 'low' else self.micrograph.hm_feats
        pca = PCA(n_components=3, random_state=0)
        scores = pca.fit_transform(feats.astype(np.float64))
        out = {
            'scores': scores.astype(np.float32),
            'explained_variance_ratio': pca.explained_variance_ratio_.astype(np.float32),
        }
        self._pca_cache[which] = out
        return out
