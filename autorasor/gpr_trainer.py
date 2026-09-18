import numpy as np
import torch
import gpytorch
import sys
import os

from autorasor.engine import UncertaintyEngineV4
try:
    from botorch.acquisition import qUpperConfidenceBound
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.optim import optimize_acqf_discrete
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False

class GPRTrainer:
    def __init__(self, x_features, input_dim=0, metric='cosine', k_neighbors_ambiguity=10, acq_function='qucb',
                 normalize_features=None, fit_mode=None, noise_mode=None, diversity=None, kernel=None):
        """
        x_features: (N_total_patches, D) array of all X features in the dataset.

        Experimental engine variants (defaults preserve legacy behaviour).
        Each can be set as a kwarg or via environment variable so existing
        validation scripts run variants without code changes:
            AL_NORMALIZE_FEATURES=1        -> L2-normalize x/y features (GP on unit sphere)
            AL_FIT_MODE=mll                -> fit_gpytorch_mll + BoTorch default priors
                                              (instead of 50 Adam steps, prior-less ARD Matern)
            AL_NOISE_MODE=jackknife        -> jackknife estimator variance as observation noise
            AL_KERNEL=matern (default)     -> ScaleKernel(Matern-5/2 ARD) when AL_FIT_MODE=mll —
                                              this is the manuscript's stated kernel and should
                                              ALWAYS be used for reported results. AL_KERNEL=default
                                              (BoTorch's RBF, dimension-scaled priors) exists only
                                              for ablation and must never back a reported figure.
            AL_DIVERSITY=penalize          -> local-penalization greedy batch acquisition
                                              (Gonzalez et al., AISTATS 2016 style)
            AL_DIVERSITY=stratified        -> spherical k-means strata over the low-mag latent
                                              space (k = AL_STRATA, default 4); round-robin over
                                              clusters, argmax predicted ambiguity within each
                                              (Cluster-Margin style, Citovsky et al. 2021)
            AL_DIVERSITY=qdpp              -> exact greedy DPP MAP inference over the DPP kernel
                                              L = diag(q) S diag(q), q_i = exp(AL_DPP_LAMBDA*z_i/2)
                                              (z = standardized score), S = cosine kernel; greedy
                                              gain = AL_DPP_LAMBDA*z_i + log schur_i via incremental
                                              Cholesky (Kulesza & Taskar 2012; Chen et al. 2018).
                                              AL_DPP_LAMBDA is now a quality temperature (default 1)
        """
        def _env_flag(name, default):
            v = os.environ.get(name)
            if v is None:
                return default
            return v if not v.isdigit() else bool(int(v))

        if normalize_features is None:
            normalize_features = bool(_env_flag('AL_NORMALIZE_FEATURES', False))
        if fit_mode is None:
            fit_mode = _env_flag('AL_FIT_MODE', 'adam')
        if noise_mode is None:
            noise_mode = _env_flag('AL_NOISE_MODE', 'heuristic')
        if kernel is None:
            kernel = _env_flag('AL_KERNEL', 'matern')
        if diversity is None:
            diversity = _env_flag('AL_DIVERSITY', 'none')
        self.diversity = diversity
        self.n_strata = int(os.environ.get('AL_STRATA', 4))
        self.dpp_lambda = float(os.environ.get('AL_DPP_LAMBDA', 1.0))
        # Stratified-mode state (fit once per image, fixed across steps)
        self._strata_labels = None
        self._strata_queue = None

        if normalize_features or fit_mode != 'adam' or noise_mode != 'heuristic' or diversity != 'none':
            print(f"[GPRTrainer] ENGINE VARIANT ACTIVE: normalize_features={normalize_features}, "
                  f"fit_mode={fit_mode}, noise_mode={noise_mode}, kernel={kernel}, diversity={diversity}")
        if fit_mode == 'mll' and kernel != 'matern':
            print(f"[GPRTrainer] WARNING: kernel='{kernel}' with fit_mode='mll' — the manuscript's "
                  f"stated methodology is Matern-5/2 ARD ('matern'). This run will NOT match reported results.")

        self.engine = UncertaintyEngineV4(
            x_features=x_features,
            input_dim=input_dim,
            metric=metric,
            k_neighbors_ambiguity=k_neighbors_ambiguity,
            acq_function=acq_function,
            normalize_features=normalize_features,
            fit_mode=fit_mode,
            noise_mode=noise_mode,
            kernel=kernel,
        )
        self.is_trained = False
        
    def add_training_data(self, indices, y_features):
        """
        indices: list of ints corresponding to the rows in x_features.
        y_features: (N, D_y) target features.
        """
        for idx, y in zip(indices, y_features):
            self.engine.add_observation(idx, y)

    def add_training_data_direct(self, x_features_list, y_features_list, labels=None):
        """Add training data with explicit X feature vectors — cross-dataset safe.

        Use this when the new observations come from a DIFFERENT image/dataset
        than the one the engine was originally built on (e.g., continual
        learning on a new SEM image using a GP pre-trained on historical data).

        Args:
            x_features_list: list of (D,) low-mag feature arrays.
            y_features_list: list of (D,) high-mag feature arrays.
            labels:          optional list of ints used for dedup tracking
                             (use the new-image patch indices, e.g. 0–195).
        """
        for i, (x, y) in enumerate(zip(x_features_list, y_features_list)):
            label = labels[i] if labels is not None else None
            self.engine.add_observation_direct(x, y, dedup_label=label)
            
    def train(self, training_iters=50):
        print(f"Training GPR with {self.engine.n_observed} observations out of {self.engine.n_points} total patches.")
        if self.engine.n_observed < 5:
            print("Warning: less than 5 observations, GPR training will skip.")
            
        # batch_size <= 0 means predict-only (no acquisition), but it still trains the model!
        self.engine.fit_surrogate_and_select(batch_size=0, training_iters=training_iters)
        self.is_trained = True
        
    def predict_new_image(self, new_x_features, batch_size=3, beta_param=1.0, exclude_indices=None):
        """
        new_x_features: (196, D) features for the new 14x14 micrograph.
        exclude_indices: list of ints to exclude from candidate selection.
        Returns:
            selected_grid_indices: list of indices (0-195) to sample next.
            predicted_scores: (196,) UCB scores.
            predicted_mean: (196,) ambiguity mean.
            predicted_std: (196,) ambiguity std.
        """
        if not self.is_trained or self.engine._last_model is None:
            raise RuntimeError("GPR is not trained yet. Train the model first.")
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # UncertaintyEngineV4.fit_surrogate_and_select parks _last_model on
        # CPU after every fit (to free GPU memory between steps), so it must
        # be moved back to `device` before running inference here — otherwise
        # model(new_X_t) mixes CPU (model) and CUDA (query) tensors.
        model = self.engine._last_model.to(device)
        model.eval()
        model.likelihood.eval()

        if self.engine.normalize_features:
            new_x_features = new_x_features / (np.linalg.norm(new_x_features, axis=1, keepdims=True) + 1e-8)

        # Transform new_x_features to PCA space
        if self.engine.use_cosine_pca:
            new_x_gp = self.engine.cosine_pca.transform(new_x_features)[:, :self.engine.input_dim].astype(np.float32)
        else:
            new_x_gp = new_x_features.astype(np.float32)

        new_X_t = torch.from_numpy(new_x_gp).double().to(device)
        
        # Clear CUDA cache before heavy inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 1. Forward pass to get posterior
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = model.likelihood(model(new_X_t))
            pred_mean_norm = pred.mean.cpu().numpy().squeeze()
            pred_var_norm = pred.variance.cpu().numpy().squeeze()
            
        # 2. Unnormalize (requires recomputing y_mean and y_std_val from archive)
        current_ambiguity, _, _ = self.engine.compute_ambiguity(np.array(self.engine.archive_indices))
        y_mean = current_ambiguity.mean()
        y_std_val = current_ambiguity.std() + 1e-6
        
        predicted_mean = (pred_mean_norm * y_std_val) + y_mean
        predicted_std = np.sqrt(pred_var_norm) * y_std_val
        
        # Calculate heuristic scores
        m_min, m_max = predicted_mean.min(), predicted_mean.max()
        s_min, s_max = predicted_std.min(), predicted_std.max()
        norm_mean = (predicted_mean - m_min) / max(m_max - m_min, 1e-8)
        norm_std  = (predicted_std  - s_min) / max(s_max - s_min, 1e-8)
        predicted_scores = norm_mean + beta_param * norm_std
        
        # 3. Batch Acquisition on the discrete grid
        selected_grid_indices = []
        if batch_size > 0 and self.diversity == 'stratified':
            # Cluster-Margin-style stratified selection (Citovsky et al. 2021):
            # fixed spherical k-means strata over the low-mag latent space;
            # round-robin over clusters, argmax predicted score within each.
            n = len(new_x_features)
            excluded = set(int(i) for i in exclude_indices) if exclude_indices else set()

            if self._strata_labels is None or len(self._strata_labels) != n:
                from sklearn.cluster import KMeans
                Xn = new_x_features / (np.linalg.norm(new_x_features, axis=1, keepdims=True) + 1e-8)
                k = min(self.n_strata, n)
                km = KMeans(n_clusters=k, n_init=10, random_state=0)
                self._strata_labels = km.fit_predict(Xn)
                # Initial queue order: descending predicted-ambiguity mass per cluster
                mass = [predicted_scores[self._strata_labels == c].sum() for c in range(k)]
                self._strata_queue = [c for c in np.argsort(mass)[::-1].tolist()]

            picks = []
            guard = 0
            while len(picks) < batch_size and guard < 4 * len(self._strata_queue) + batch_size:
                guard += 1
                if not self._strata_queue:
                    break
                c = self._strata_queue.pop(0)
                members = [i for i in np.where(self._strata_labels == c)[0]
                           if i not in excluded and i not in picks]
                if not members:
                    continue  # cluster exhausted: drop from queue
                best = members[int(np.argmax(predicted_scores[members]))]
                picks.append(int(best))
                self._strata_queue.append(c)
            # Fallback if clusters exhausted before batch filled
            if len(picks) < batch_size:
                rest = [i for i in np.argsort(predicted_scores)[::-1]
                        if i not in excluded and i not in picks]
                picks.extend(int(i) for i in rest[:batch_size - len(picks)])
            selected_grid_indices = picks
        elif batch_size > 0 and self.diversity == 'qdpp':
            # Exact greedy DPP MAP inference (Kulesza & Taskar 2012;
            # Chen et al. 2018, "Fast Greedy MAP Inference for DPP").
            # DPP kernel L = diag(q) S diag(q), with quality q_i = exp(tau*z_i/2)
            # (z = standardized acquisition score) and diversity S = cosine
            # kernel over normalized x. Because L = diag(q) S diag(q), the
            # log-det gain of adding item i to the conditioning set factorizes:
            #     log det(L_{C+i}) - log det(L_C) = 2 log q_i + log(schur_i^S)
            # where schur_i^S is the Schur complement in the *plain* S kernel.
            # With q_i = exp(tau*z_i/2) the quality term is exactly tau*z_i, so
            # the greedy objective is  tau*z_i + log(schur_i)  -- no free lambda.
            n = len(new_x_features)
            excluded = set(int(i) for i in exclude_indices) if exclude_indices else set()
            Xn = new_x_features / (np.linalg.norm(new_x_features, axis=1, keepdims=True) + 1e-8)
            K = Xn @ Xn.T + 1e-6 * np.eye(n)

            # Standardized quality signal z (acquisition score -> unit scale).
            z = (predicted_scores - predicted_scores.mean()) / (predicted_scores.std() + 1e-8)
            tau = self.dpp_lambda  # repurposed as quality temperature (default 1.0)

            S = sorted(excluded)
            picks = []
            for _ in range(min(batch_size, n - len(excluded))):
                cond = S + picks
                cands = [i for i in range(n) if i not in excluded and i not in picks]
                if not cands:
                    break
                if cond:
                    K_SS = K[np.ix_(cond, cond)] + 1e-4 * np.eye(len(cond))
                    L = np.linalg.cholesky(K_SS)
                    k_iS = K[np.ix_(cands, cond)]              # (n_cand, |S|)
                    v = np.linalg.solve(L, k_iS.T)             # (|S|, n_cand)
                    schur = K[cands, cands] - np.sum(v * v, axis=0)
                    dlogdet = np.log(np.clip(schur, 1e-10, None))
                else:
                    dlogdet = np.log(np.clip(K[cands, cands], 1e-10, None))
                # Exact DPP-MAP greedy gain: 2 log q_i + log schur_i = tau*z_i + dlogdet
                eff = tau * z[cands] + dlogdet
                picks.append(int(cands[int(np.argmax(eff))]))
            selected_grid_indices = picks
        elif batch_size > 0 and self.diversity == 'penalize':
            # Local-penalization greedy batch selection (Gonzalez et al. 2016 style):
            # UCB-style base score, multiplicatively suppressed near every
            # already-sampled patch and every in-batch pick, in cosine geometry.
            n = len(new_x_features)
            Xn = new_x_features / (np.linalg.norm(new_x_features, axis=1, keepdims=True) + 1e-8)
            cos_dist = 1.0 - Xn @ Xn.T  # (n, n)
            k_bw = min(10, n - 1)
            ell = float(np.median(np.sort(cos_dist, axis=1)[:, k_bw])) + 1e-6

            excluded = set(int(i) for i in exclude_indices) if exclude_indices else set()
            picks = []
            for _ in range(min(batch_size, n - len(excluded))):
                eff = predicted_scores.copy()
                for j in excluded | set(picks):
                    eff *= 1.0 - np.exp(-(cos_dist[:, j] ** 2) / ell ** 2)
                for j in excluded | set(picks):
                    eff[j] = -np.inf
                picks.append(int(np.argmax(eff)))
            selected_grid_indices = picks
        elif batch_size > 0:
            if not BOTORCH_AVAILABLE:
                raise RuntimeError("Batch acquisition requested but BoTorch is not installed. Use batch_size=0.")
            if self.engine.acq_function == 'qlognei':
                # Use stored GP features (safe for sentinel indices from add_observation_direct)
                valid_x = np.array(self.engine.archive_x_gp, dtype=np.float32)
                train_X = torch.from_numpy(valid_x).double().to(device)
                acqf = qLogNoisyExpectedImprovement(
                    model=model,
                    X_baseline=train_X,
                    prune_baseline=True,
                )
            else:
                acqf = qUpperConfidenceBound(model, beta=beta_param)
            
            # Filter choices to exclude already sampled/observed indices
            if exclude_indices is not None and len(exclude_indices) > 0:
                exclude_set = set(exclude_indices)
                unsampled_indices = [i for i in range(len(new_x_features)) if i not in exclude_set]
                if len(unsampled_indices) == 0:
                    unsampled_indices = list(range(len(new_x_features)))
            else:
                unsampled_indices = list(range(len(new_x_features)))
            choices_X = new_X_t[unsampled_indices]
                
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                candidates, _ = optimize_acqf_discrete(
                    acq_function=acqf,
                    q=min(batch_size, len(choices_X)),
                    choices=choices_X,
                    unique=True,
                    max_batch_size=32, # reduced from 200 to prevent OOM
                )
            
            # Map candidates back to grid indices (0 to 195), searching only
            # the unsampled subset so an excluded patch can never be returned
            unsampled_arr = np.array(unsampled_indices)
            unsampled_np = new_x_gp[unsampled_arr]
            for cand in candidates:
                cand_np = cand.cpu().numpy()
                dists = np.linalg.norm(unsampled_np - cand_np, axis=1)
                idx = int(unsampled_arr[np.argmin(dists)])
                selected_grid_indices.append(idx)
                
        return selected_grid_indices, predicted_scores, predicted_mean, predicted_std
