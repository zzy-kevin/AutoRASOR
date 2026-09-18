"""Ground-truth localized morphological ambiguity (Sec. 2.4 of the paper)."""
import numpy as np


def compute_gt_ambiguity_from_pool(engine, pool_indices, pool_hm_feats, pool_lm_feats,
                                   k_neighbors_ambiguity=10):
    """
    Compute full-grid GT ambiguity using a given high-mag feature pool.

    Parameters
    ----------
    engine : UncertaintyEngine
        Provides x_features and metric.
    pool_indices : np.ndarray
        Grid indices of pool points.
    pool_hm_feats : np.ndarray, (pool_size, D)
        High-mag features for pool points.
    pool_lm_feats : np.ndarray, (pool_size, D_lm)
        Low-mag features for pool points.
    k_neighbors_ambiguity : int
        Number of neighbors for ambiguity computation.

    Returns
    -------
    gt_ambiguity : np.ndarray, (n_points,)
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(k_neighbors_ambiguity, len(pool_indices) - 1)
    k_query = min(k + 1, len(pool_indices))
    nn_gt = NearestNeighbors(n_neighbors=k_query, metric=engine._sklearn_metric())
    nn_gt.fit(pool_lm_feats)

    all_lm_feats = engine.x_gp
    dists_all, ids_all = nn_gt.kneighbors(all_lm_feats)

    pool_indices_arr = np.array(pool_indices)
    kth_dists = np.zeros(engine.n_points, dtype=np.float32)
    for i in range(engine.n_points):
        mask = pool_indices_arr[ids_all[i]] != i
        valid_dists = dists_all[i][mask][:k]
        kth_dists[i] = valid_dists[-1] if len(valid_dists) > 0 else 0.0
    l_max = np.median(kth_dists)

    gt_ambiguity = np.zeros(engine.n_points, dtype=np.float32)
    for i in range(engine.n_points):
        d_i_raw = dists_all[i]
        ids_i_raw = ids_all[i]

        mask = pool_indices_arr[ids_i_raw] != i
        d_i = d_i_raw[mask][:k]
        ids_i = ids_i_raw[mask][:k]

        if len(d_i) == 0:
            continue

        ell = min(d_i[-1], l_max) + 1e-6
        w_i = np.exp(-d_i**2 / ell**2)
        w_sum = w_i.sum()
        if w_sum < 1e-12:
            continue
        y_neighbors = pool_hm_feats[ids_i]
        y_bar = np.average(y_neighbors, axis=0, weights=w_i)
        diffs = np.linalg.norm(y_neighbors - y_bar, axis=1) ** 2
        gt_ambiguity[i] = np.dot(w_i, diffs) / w_sum

    return gt_ambiguity
