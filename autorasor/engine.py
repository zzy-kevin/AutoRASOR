"""
UncertaintyEngineV4 — BoTorch Active Learning for SEM
=====================================================
Uses BoTorch to run a batch active learning acquisition sequence.
- ARD Matérn 5/2 GP surrogate with FixedNoiseGaussianLikelihood
- qUCB or qLogNoisyExpectedImprovement batch acquisition optimized over the discrete grid
"""

import numpy as np
import torch
import gpytorch
from pathlib import Path
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from typing import List, Tuple, Optional, Literal

import warnings
try:
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import qUpperConfidenceBound
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
    from botorch.optim import optimize_acqf_discrete
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
from gpytorch.mlls import ExactMarginalLogLikelihood

class PureGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, train_Yvar):
        noise = train_Yvar.squeeze(-1)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise, learn_additional_noise=True)
        super().__init__(train_x, train_y.squeeze(-1), likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

import warnings
warnings.filterwarnings("ignore", message=".*Data \\(input features\\) is not contained to the unit cube.*")
warnings.filterwarnings("ignore", message=".*You have passed data through a FixedNoiseGaussianLikelihood that did not match the size of the fixed noise.*")

# ============================================================
# UncertaintyEngineV4
# ============================================================

class UncertaintyEngineV4:
    def __init__(
        self,
        x_features: np.ndarray,
        input_dim: int = 0,
        metric: Literal['cosine', 'l1', 'l2'] = 'cosine',
        prefit_pca_path: Optional[str] = None,
        prefit_cosine_pca_path: Optional[str] = None,
        viz_pca_dim: int = 64,
        k_neighbors_ambiguity: int = 10,
        acq_function: str = 'qucb',  # 'qucb' or 'qlognei'
        normalize_features: bool = False,
        fit_mode: str = 'adam',       # 'adam' (legacy) or 'mll' (fit_gpytorch_mll, BoTorch default priors)
        noise_mode: str = 'heuristic',  # 'heuristic' (base + max*exp(-W)) or 'jackknife'
        kernel: str = 'matern',       # 'matern' (ScaleKernel Matern-5/2 ARD; manuscript's stated kernel — use this) or 'default' (BoTorch default in mll mode: RBF dim-scaled priors; only for ablation, never for reported results)
    ):
        self.normalize_features = normalize_features
        self.fit_mode = fit_mode
        self.noise_mode = noise_mode
        self.kernel = kernel
        if normalize_features:
            norms = np.linalg.norm(x_features, axis=1, keepdims=True) + 1e-8
            x_features = x_features / norms
        self.x_features = x_features.astype(np.float32)  # (N, D) full dim
        self.n_points, self.feat_dim = self.x_features.shape
        self.metric = metric
        self.k_neighbors_ambiguity = k_neighbors_ambiguity
        self.acq_function = acq_function

        # --- Resolve input_dim ---
        if input_dim == 0:
            input_dim = self.feat_dim
        self.input_dim = input_dim
        self.use_cosine_pca = (input_dim < self.feat_dim)

        # --- Cosine Kernel PCA ---
        import pickle
        if self.use_cosine_pca:
            if prefit_cosine_pca_path is not None and Path(prefit_cosine_pca_path).exists():
                print(f"[EngineV4] Loading pre-fitted cosine PCA: {prefit_cosine_pca_path}")
                with open(prefit_cosine_pca_path, 'rb') as f:
                    self.cosine_pca = pickle.load(f)
                print(f"  Cosine PCA components: {self.cosine_pca.n_components}")
            else:
                print(f"[EngineV4] Fitting cosine KernelPCA on grid features (N={self.n_points}, comp={input_dim})")
                self.cosine_pca = KernelPCA(n_components=input_dim, kernel='cosine', random_state=42)
                self.cosine_pca.fit(self.x_features)
            self.x_gp = self.cosine_pca.transform(self.x_features)[:, :input_dim].astype(np.float32)
        else:
            self.cosine_pca = None
            self.x_gp = self.x_features.copy()

        # --- Standard PCA (viz only) ---
        if prefit_pca_path is not None and Path(prefit_pca_path).exists():
            print(f"[EngineV4] Loading standard PCA (viz): {prefit_pca_path}")
            with open(prefit_pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            self.pca_dim = self.pca.n_components
        else:
            print(f"[EngineV4] Fitting standard PCA (viz) on grid features (N={self.n_points})")
            self.pca_dim = min(viz_pca_dim, self.n_points, self.feat_dim)
            self.pca = PCA(n_components=self.pca_dim)
            self.pca.fit(self.x_features)
        self.x_pca = self.pca.transform(self.x_features).astype(np.float32)

        # --- Archive ---
        self.archive_indices: List[int] = []
        self._archive_set: set = set()  # O(1) dedup lookups
        self.archive_x: List[np.ndarray] = []       # raw features (D,)
        self.archive_y: List[np.ndarray] = []       # high-mag y features (D,)
        self.archive_x_pca: List[np.ndarray] = []  # viz PCA projections
        self.archive_x_gp: List[np.ndarray] = []   # GP-space projections (input_dim,)

        self._grid_nn = NearestNeighbors(n_neighbors=20, metric=self._sklearn_metric(), algorithm='auto')
        self._grid_nn.fit(self.x_gp)

        self._last_model = None
        self._last_ard_lengthscales = None

    def _sklearn_metric(self) -> str:
        return {'cosine': 'cosine', 'l1': 'manhattan', 'l2': 'euclidean'}[self.metric]

    def warmup_strategy(self, n_seeds: int = 5, neighbors_per_seed: int = 3, rng_seed: int = 42, mode: str = 'random') -> List[int]:
        rng = np.random.RandomState(rng_seed)
        if mode == 'lfps':
            first_idx = int(rng.choice(self.n_points))
            selected_indices = [first_idx]
            if n_seeds > 1:
                from scipy.spatial.distance import cdist
                metric = self._sklearn_metric()
                cdist_metric = metric if metric != 'manhattan' else 'cityblock'
                distances = cdist(self.x_gp, self.x_gp[[first_idx]], metric=cdist_metric)
                min_distances = distances.flatten()
                
                for _ in range(n_seeds - 1):
                    next_idx = int(np.argmax(min_distances))
                    selected_indices.append(next_idx)
                    new_dist = cdist(self.x_gp, self.x_gp[[next_idx]], metric=cdist_metric).flatten()
                    min_distances = np.minimum(min_distances, new_dist)
            return selected_indices
        else:
            seed_indices = rng.choice(self.n_points, size=n_seeds, replace=False).tolist()
            all_indices = []
            for seed_idx in seed_indices:
                _, neighbor_ids = self._grid_nn.kneighbors(self.x_gp[seed_idx:seed_idx+1], n_neighbors=neighbors_per_seed + 1)
                all_indices.extend([seed_idx] + neighbor_ids[0, 1:].tolist())
            return list(dict.fromkeys(all_indices)) # deduplicate keeping order

    def add_observation(self, grid_idx: int, y_feat: np.ndarray) -> bool:
        """Add observation by grid index into self.x_features. Returns False if duplicate."""
        if grid_idx in self._archive_set:
            return False
        y_feat = y_feat.flatten().astype(np.float32)
        if self.normalize_features:
            y_feat = y_feat / (np.linalg.norm(y_feat) + 1e-8)
        self.archive_indices.append(grid_idx)
        self._archive_set.add(grid_idx)
        self.archive_x.append(self.x_features[grid_idx])
        self.archive_y.append(y_feat)
        self.archive_x_pca.append(self.x_pca[grid_idx])
        self.archive_x_gp.append(self.x_gp[grid_idx])
        return True

    def add_observation_direct(self, x_feat: np.ndarray, y_feat: np.ndarray,
                               dedup_label: Optional[int] = None) -> bool:
        """Add observation with explicit X features — cross-dataset safe.

        Use this when observations come from a DIFFERENT dataset than
        self.x_features (e.g., continual learning on a new image using a
        GP trained on historical data). No index lookup into self.x_features
        is performed.

        Args:
            x_feat:      (D,) raw low-mag feature vector for this observation.
            y_feat:      (D,) high-mag feature vector.
            dedup_label: Optional int used solely for dedup tracking via
                         _archive_set. Use the new-image patch index (0-195).
                         Pass None to skip dedup (allows duplicates).
        Returns:
            True if added, False if dedup_label was already seen.
        """
        if dedup_label is not None and dedup_label in self._archive_set:
            return False

        x_feat = x_feat.flatten().astype(np.float32)
        y_feat = y_feat.flatten().astype(np.float32)
        if self.normalize_features:
            x_feat = x_feat / (np.linalg.norm(x_feat) + 1e-8)
            y_feat = y_feat / (np.linalg.norm(y_feat) + 1e-8)

        # Use a sentinel index well outside any real grid range
        sentinel_idx = self.n_points + len(self.archive_indices)

        self.archive_indices.append(sentinel_idx)
        if dedup_label is not None:
            self._archive_set.add(dedup_label)

        self.archive_x.append(x_feat)
        self.archive_y.append(y_feat)

        # GP-space: raw 384-D (input_dim=0 is the only supported mode going forward)
        self.archive_x_gp.append(x_feat.copy())

        # PCA viz projection (best-effort; zeros if shape mismatch)
        try:
            pca_proj = self.pca.transform(x_feat.reshape(1, -1)).flatten().astype(np.float32)
        except Exception:
            pca_proj = np.zeros(self.pca_dim, dtype=np.float32)
        self.archive_x_pca.append(pca_proj)

        return True

    @property
    def n_observed(self) -> int:
        return len(self.archive_indices)

    def compute_ambiguity(self, indices: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute local ambiguity for a set of query indices.

        `indices` must be indices into self.archive_indices (i.e., the same
        integers stored there, including any sentinel values from
        add_observation_direct).  The method no longer indexes self.x_features
        or self.x_gp with these values, so sentinel indices are safe.
        """
        if k is None: k = self.k_neighbors_ambiguity
        n = len(indices)
        ambiguity = np.zeros(n, dtype=np.float32)
        w_sums    = np.zeros(n, dtype=np.float32)
        centroids = np.zeros((n, self.feat_dim), dtype=np.float32)
        # Leave-one-neighbor-out jackknife variance of the ambiguity estimator
        # (used by noise_mode='jackknife'); aligned with `indices`.
        self._last_jackknife_var = np.zeros(n, dtype=np.float32)

        if self.n_observed < 2: return ambiguity, w_sums, centroids

        # Build index→position map for fast lookup of stored features
        idx_to_pos = {idx: pos for pos, idx in enumerate(self.archive_indices)}

        # Stored per-observation features (always correct, even for sentinels)
        archive_x_gp  = np.array(self.archive_x_gp,  dtype=np.float32)   # (M, D_gp)
        archive_x_full = np.array(self.archive_x,    dtype=np.float32)   # (M, D)
        archive_y      = np.array(self.archive_y,    dtype=np.float32)   # (M, D_y)
        archive_indices_arr = np.array(self.archive_indices)

        # Build query GP features: use stored value if index is in archive,
        # otherwise fall back to self.x_gp (only valid for non-sentinel indices)
        query_x_gp = np.zeros((n, archive_x_gp.shape[1]), dtype=np.float32)
        for i, idx in enumerate(indices):
            if idx in idx_to_pos:
                query_x_gp[i] = archive_x_gp[idx_to_pos[idx]]
                centroids[i]  = archive_x_full[idx_to_pos[idx]]
            elif idx < self.n_points:
                query_x_gp[i] = self.x_gp[idx]
                centroids[i]  = self.x_features[idx]
            # else: sentinel with no stored pos — leave as zeros (edge case)

        k_query = min(k + 1, len(archive_x_gp))
        archive_nn = NearestNeighbors(n_neighbors=k_query, metric=self._sklearn_metric(), algorithm='auto')
        archive_nn.fit(archive_x_gp)
        distances, nn_ids = archive_nn.kneighbors(query_x_gp)

        kth_dists = np.zeros(n, dtype=np.float32)
        for i in range(n):
            mask = archive_indices_arr[nn_ids[i]] != indices[i]
            valid_dists = distances[i][mask]
            if len(valid_dists) > 0: kth_dists[i] = valid_dists[:k][-1]
            else: kth_dists[i] = 1e-6
        l_max = np.median(kth_dists) if len(kth_dists) > 0 else 1e-6

        for i in range(n):
            mask = archive_indices_arr[nn_ids[i]] != indices[i]
            d_i, valid_nn_ids = distances[i][mask][:k], nn_ids[i][mask][:k]
            if len(d_i) == 0: continue

            ell = min(d_i[-1], l_max) + 1e-6
            w_i = np.exp(-d_i**2 / ell**2)
            w_sum = w_i.sum()
            w_sums[i] = w_sum

            if w_sum < 1e-12: continue

            centroids[i] = np.average(archive_x_full[valid_nn_ids], axis=0, weights=w_i)
            y_neighbors  = archive_y[valid_nn_ids]
            y_bar        = np.average(y_neighbors, axis=0, weights=w_i)
            diffs        = np.linalg.norm(y_neighbors - y_bar, axis=1) ** 2
            ambiguity[i] = np.dot(w_i, diffs) / w_sum

            if self.noise_mode == 'jackknife':
                m = len(d_i)
                if m >= 3:
                    loo_vals = []
                    for leave in range(m):
                        keep = np.arange(m) != leave
                        w_k = w_i[keep]
                        ws = w_k.sum()
                        if ws < 1e-12: continue
                        y_k = y_neighbors[keep]
                        yb = np.average(y_k, axis=0, weights=w_k)
                        df = np.linalg.norm(y_k - yb, axis=1) ** 2
                        loo_vals.append(np.dot(w_k, df) / ws)
                    if len(loo_vals) >= 2:
                        loo = np.array(loo_vals)
                        self._last_jackknife_var[i] = (len(loo) - 1) / len(loo) * np.sum((loo - loo.mean()) ** 2)

        return ambiguity, w_sums, centroids

    def _get_random_batch(self, batch_size: int) -> np.ndarray:
        candidates = [i for i in range(self.n_points) if i not in self._archive_set]
        if batch_size <= 0 or len(candidates) == 0:
            return np.array([], dtype=int)
        rng = np.random.RandomState(self.n_observed)
        return rng.choice(candidates, size=min(batch_size, len(candidates)), replace=False)

    def fit_surrogate_and_select(
        self, batch_size: int = 10, beta_param: float = 1.0, training_iters: int = 50,
        normalize_ucb: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits BoTorch SingleTaskGP and runs batch acquisition.

        If batch_size <= 0, runs in predict-only mode (no acquisition).
        Returns: selected_indices, predicted_scores (full grid), mean (full grid), std (full grid)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predict_only = (batch_size <= 0)
        
        if self.n_observed < 5:
            zeros = np.zeros(self.n_points)
            if predict_only:
                return np.array([], dtype=int), zeros, zeros, zeros
            return self._get_random_batch(batch_size), zeros, zeros, zeros

        current_ambiguity, w_sums, _ = self.compute_ambiguity(np.array(self.archive_indices))
        # Use stored GP features (safe for sentinel indices from add_observation_direct)
        valid_x = np.array(self.archive_x_gp, dtype=np.float32)


        if len(valid_x) < 2:
            zeros = np.zeros(self.n_points)
            if predict_only:
                return np.array([], dtype=int), zeros, zeros, zeros
            return self._get_random_batch(batch_size), zeros, zeros, zeros

        train_X = torch.from_numpy(valid_x).double().to(device)
        train_Y = torch.from_numpy(current_ambiguity).double().unsqueeze(-1).to(device)

        y_mean, y_std_val = train_Y.mean(), train_Y.std() + 1e-6
        train_Y_norm = (train_Y - y_mean) / y_std_val

        # Setup Observation Noise
        if self.noise_mode == 'jackknife' and getattr(self, '_last_jackknife_var', None) is not None \
                and len(self._last_jackknife_var) == len(w_sums):
            # Jackknife variance of the ambiguity estimator, expressed in
            # normalized-label units; small floor for numerical stability.
            jack_norm = self._last_jackknife_var / (y_std_val.item() ** 2)
            noise_variance = np.clip(0.01 + jack_norm, 1e-4, 4.0)
        else:
            base_noise, max_penalty = 0.05, 2.0
            noise_variance = base_noise + max_penalty * np.exp(-w_sums)
        train_Yvar = torch.from_numpy(noise_variance).double().unsqueeze(-1).to(device)

        # BoTorch Model configuration
        if BOTORCH_AVAILABLE:
            model = SingleTaskGP(train_X, train_Y_norm, train_Yvar=train_Yvar)
        else:
            model = PureGPR(train_X, train_Y_norm, train_Yvar=train_Yvar)

        if self.fit_mode == 'mll' and BOTORCH_AVAILABLE:
            # 'default' keeps BoTorch's default covariance module (RBF with
            # dimension-scaled lengthscale priors, designed for high-dim
            # inputs); 'matern' swaps in the legacy Matern-5/2 ARD kernel.
            # Either way the MLL is fit properly with L-BFGS instead of a
            # fixed Adam budget.
            if self.kernel == 'matern':
                model.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.input_dim)
                )
            model.double().to(device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        else:
            model.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.input_dim)
            ).to(device)
            model.covar_module.base_kernel.lengthscale = 0.2
            model.double().to(device)

            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # We manually train using Adam to stay faithful to v3's approach
            model.train()
            model.likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            for _ in range(training_iters):
                optimizer.zero_grad()
                output = model(train_X)
                loss = -mll(output, train_Y_norm)
                loss.sum().backward()
                optimizer.step()

        # Predict full grid for plots
        model.eval()
        model.likelihood.eval()
        all_X_t = torch.from_numpy(self.x_gp).double().to(device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            chunk_size = 5000
            pred_means, pred_vars = [], []
            for start in range(0, self.n_points, chunk_size):
                end = min(start + chunk_size, self.n_points)
                pred = model.likelihood(model(all_X_t[start:end]))
                pred_means.append(pred.mean.cpu())
                pred_vars.append(pred.variance.cpu())

            pred_mean_norm = torch.cat(pred_means).numpy().squeeze()
            pred_var_norm = torch.cat(pred_vars).numpy().squeeze()

        predicted_mean = (pred_mean_norm * y_std_val.item()) + y_mean.item()
        predicted_std = np.sqrt(pred_var_norm) * y_std_val.item()

        # Calculate heuristic scores for plotting
        if normalize_ucb:
            m_min, m_max = predicted_mean.min(), predicted_mean.max()
            s_min, s_max = predicted_std.min(), predicted_std.max()
            norm_mean = (predicted_mean - m_min) / max(m_max - m_min, 1e-8)
            norm_std  = (predicted_std  - s_min) / max(s_max - s_min, 1e-8)
            predicted_scores = norm_mean + beta_param * norm_std
        else:
            predicted_scores = predicted_mean + beta_param * predicted_std

        base_kernel = getattr(model.covar_module, 'base_kernel', model.covar_module)
        ls = getattr(base_kernel, 'lengthscale', None)
        self._last_ard_lengthscales = ls.detach().cpu().numpy().flatten() if ls is not None else None

        # --- Predict-only mode: skip acquisition ---
        if predict_only:
            self._last_model = model.cpu()
            # Clean up GPU memory
            del model
            del train_X
            del train_Y
            del train_Y_norm
            del train_Yvar
            del all_X_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc; gc.collect()
            return np.array([], dtype=int), predicted_scores, predicted_mean, predicted_std

        # --- Acquisition ---
        unsampled_indices = [i for i in range(self.n_points) if i not in self._archive_set]
        unsampled_arr = np.array(unsampled_indices)
        unsampled_X = all_X_t[unsampled_indices]
        unsampled_np = self.x_gp[unsampled_arr]

        if self.acq_function in ('ei', 'ucb'):
            # Analytic (sequential) acquisition — pick top-1 per call
            best_f = train_Y_norm.max().item()
            if self.acq_function == 'ei':
                acqf = ExpectedImprovement(model=model, best_f=best_f)
            else:  # ucb
                acqf = UpperConfidenceBound(model=model, beta=beta_param)
            with torch.no_grad():
                acq_vals = acqf(unsampled_X.unsqueeze(1))  # (N_cand,)
            # Pick batch_size points greedily (for sequential: batch_size=1)
            topk = min(batch_size, len(acq_vals))
            _, top_idx = torch.topk(acq_vals, topk)
            selected_grid_indices = unsampled_arr[top_idx.cpu().numpy()].tolist()
            del acqf

        elif self.acq_function == 'qdpp':
            # Greedy DPP-MAP batch selection (Kulesza & Taskar 2012;
            # Chen et al. 2018, "Fast Greedy MAP Inference for DPP").
            # DPP kernel L = diag(q) S diag(q), quality q_i = exp(tau*z_i/2)
            n_cand = len(unsampled_indices)
            Xn = unsampled_np / (np.linalg.norm(unsampled_np, axis=1, keepdims=True) + 1e-8)
            K = Xn @ Xn.T + 1e-6 * np.eye(n_cand)

            # Quality signal: use predicted_mean (unnormalized) as the score
            cand_scores = predicted_mean[unsampled_arr]
            z = (cand_scores - cand_scores.mean()) / (cand_scores.std() + 1e-8)
            tau = 1.0  # quality temperature

            picks = []
            for _ in range(min(batch_size, n_cand)):
                cands = [i for i in range(n_cand) if i not in picks]
                if not cands:
                    break
                if picks:
                    K_SS = K[np.ix_(picks, picks)] + 1e-4 * np.eye(len(picks))
                    L_chol = np.linalg.cholesky(K_SS)
                    k_iS = K[np.ix_(cands, picks)]              # (n_cand_rem, |S|)
                    v = np.linalg.solve(L_chol, k_iS.T)         # (|S|, n_cand_rem)
                    schur = K[cands, cands] - np.sum(v * v, axis=0)
                    dlogdet = np.log(np.clip(schur, 1e-10, None))
                else:
                    dlogdet = np.log(np.clip(K[cands, cands], 1e-10, None))
                eff = tau * z[cands] + dlogdet
                picks.append(int(cands[int(np.argmax(eff))]))
            selected_grid_indices = unsampled_arr[picks].tolist()

        else:
            # Batch acquisition via BoTorch (qlognei, qucb)
            if self.acq_function == 'qlognei':
                acqf = qLogNoisyExpectedImprovement(
                    model=model,
                    X_baseline=train_X,
                    prune_baseline=True,
                )
            else:  # default: qucb
                acqf = qUpperConfidenceBound(model, beta=beta_param)

            candidates, _ = optimize_acqf_discrete(
                acq_function=acqf,
                q=batch_size,
                choices=unsampled_X,
                unique=True,
                max_batch_size=200,
            )

            # Map candidates back to grid indices
            selected_grid_indices = []
            for cand in candidates:
                cand_np = cand.cpu().numpy()
                dists = np.linalg.norm(unsampled_np - cand_np, axis=1)
                idx = int(unsampled_arr[np.argmin(dists)])
                selected_grid_indices.append(idx)
            del acqf

        self._last_model = model.cpu()

        # Clean up GPU memory
        del model
        del train_X
        del train_Y
        del train_Y_norm
        del train_Yvar
        del all_X_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()

        return np.array(selected_grid_indices), predicted_scores, predicted_mean, predicted_std

    def save_gpr(self, path):
        """Save the last trained GP model state dict."""
        if self._last_model is not None:
            torch.save(self._last_model.state_dict(), path)
            print(f"  Saved GPR model to {path}")
        else:
            print("  [Warning] No trained model to save.")
