"""Synthetic clustering datasets and similarity matrices for verifying the
exemplar-based clustering self-consistency equations from Appendix C."""

import numpy as np


def generate_gaussian_clusters(n_clusters=4, n_per_cluster=15, dim=2,
                                cluster_std=0.3, spread=2.0,
                                min_separation_factor=4.0,
                                max_resample_attempts=10000,
                                seed=None):
    """Sample points from `n_clusters` isotropic Gaussians.

    Centers are placed by rejection sampling so that every pair of
    centers is at least `min_separation_factor * cluster_std` apart in
    Euclidean distance. With the default factor of 4, neighboring
    cluster Gaussians overlap negligibly (the 2*sigma balls do not
    intersect), so the generative cluster label is essentially the
    same as the similarity-optimal label for every sample.

    Set `min_separation_factor=None` (or 0) to disable the guard band
    and recover the original uniform-center sampling.

    Parameters
    ----------
    n_clusters : int
    n_per_cluster : int
    dim : int
    cluster_std : float
        Per-cluster isotropic standard deviation.
    spread : float
        Centers are drawn uniformly from [-spread, spread]^dim.
    min_separation_factor : float or None
        Pairwise center distances must exceed
        min_separation_factor * cluster_std.
    max_resample_attempts : int
        Cap on the rejection-sampling loop to avoid infinite loops
        when the box is too small to admit `n_clusters` points
        satisfying the guard band.
    seed : int or None

    Returns
    -------
    points : (N, dim) array, where N = n_clusters * n_per_cluster
    labels : (N,) ground-truth cluster index for each point
    centers : (n_clusters, dim) cluster centers
    """
    rng = np.random.default_rng(seed)

    min_dist = (0.0 if min_separation_factor in (None, 0)
                else min_separation_factor * cluster_std)

    centers = np.empty((n_clusters, dim))
    placed = 0
    attempts = 0
    while placed < n_clusters:
        if attempts >= max_resample_attempts:
            raise RuntimeError(
                f"Could not place {n_clusters} centers in "
                f"[-{spread}, {spread}]^{dim} with minimum pairwise "
                f"distance {min_dist:.3f} after "
                f"{max_resample_attempts} attempts. Increase `spread` "
                f"or decrease `min_separation_factor`/`cluster_std`."
            )
        candidate = rng.uniform(-spread, spread, size=dim)
        if placed == 0 or min_dist == 0.0:
            centers[placed] = candidate
            placed += 1
        else:
            d = np.linalg.norm(centers[:placed] - candidate, axis=1).min()
            if d >= min_dist:
                centers[placed] = candidate
                placed += 1
        attempts += 1

    pts, lbl = [], []
    for c in range(n_clusters):
        pts.append(rng.normal(centers[c], cluster_std,
                              size=(n_per_cluster, dim)))
        lbl.extend([c] * n_per_cluster)
    return np.concatenate(pts, axis=0), np.array(lbl), centers


def compute_similarity_matrix(points, preference=None,
                              preference_quantile=0.5):
    """Compute negative-squared-distance similarity matrix and set the
    diagonal preference w_{aa}.

    Following the affinity propagation convention of Frey & Dueck (2007),
    the diagonal "self-similarity" controls how readily a point is chosen
    as an exemplar. Higher preference -> more clusters.

    If `preference` is None, set it to `preference_quantile` of the
    off-diagonal similarity values (default = median, the standard choice).
    """
    diff = points[:, None, :] - points[None, :, :]
    W = -np.sum(diff ** 2, axis=-1).astype(np.float64)

    off_diag = ~np.eye(W.shape[0], dtype=bool)
    if preference is None:
        preference = float(np.quantile(W[off_diag], preference_quantile))
    np.fill_diagonal(W, preference)
    return W, preference
