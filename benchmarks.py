"""Clustering benchmarks for evaluating the self-consistency solver
against on the exemplar-based clustering objective

    S(E) = sum_k max_{a in E} W[k, a] = sum_k W[k, exemplar(k)]

where E is the chosen exemplar set and exemplar(k) = argmax_{a in E} W[k, a].

All benchmarks here return a tuple (assignments, exemplars, S):

    assignments : (N,) int array; assignments[k] is the index of the
                  exemplar to which point k is assigned.
    exemplars   : sorted unique values in `assignments`.
    S           : float, the sum-similarity objective.
"""

from itertools import combinations
from math import comb

import numpy as np
from sklearn.cluster import KMeans


def sum_similarity(W, assignments):
    """S = sum_k W[k, assignments[k]]."""
    idx = np.arange(len(assignments))
    return float(W[idx, assignments].sum())


def assign_to_exemplars(W, exemplars):
    """Build a feasible assignment from an exemplar set:
       - every exemplar is assigned to itself (constraint x_aa >= x_ka),
       - every non-exemplar is assigned to its highest-similarity exemplar.

    Without the self-assignment step the diagonal preference w_{aa} can
    be "gamed" by routing an exemplar to another exemplar, which
    violates the exemplar-clustering constraints."""
    E = np.asarray(sorted(set(int(e) for e in exemplars)), dtype=np.int64)
    N = W.shape[0]
    if E.size == 0:
        return np.zeros(N, dtype=np.int64), E, -np.inf
    is_ex = np.zeros(N, dtype=bool)
    is_ex[E] = True

    assignments = np.empty(N, dtype=np.int64)
    assignments[is_ex] = np.where(is_ex)[0]   # self-assign exemplars
    non_ex = np.where(~is_ex)[0]
    if non_ex.size:
        cols = W[non_ex][:, E]                # (|non_ex|, |E|)
        best = cols.argmax(axis=1)
        assignments[non_ex] = E[best]
    return assignments, E, sum_similarity(W, assignments)


# --------------------------------------------------------------------- #
# Self-consistency wrapper                                              #
# --------------------------------------------------------------------- #
def self_consistency_score(solver, W):
    """Convert a fitted SelfConsistencySolver into the standard tuple."""
    assignments, exemplars = solver.decide()
    return assign_to_exemplars(W, exemplars) if len(exemplars) > 0 else (
        assignments, np.array([]), -np.inf)


# --------------------------------------------------------------------- #
# K-means with exemplar projection                                      #
# --------------------------------------------------------------------- #
def kmeans_baseline(points, W, K, n_init=10, seed=0):
    """Run K-means on `points`, then project each centroid to the
    nearest data point to obtain a feasible exemplar set, then evaluate
    the sum-similarity objective. This restricts K-means to the same
    "centers must be data points" constraint as the exemplar-based
    formulation, so the comparison is apples-to-apples."""
    if K < 1 or K > len(points):
        return None
    km = KMeans(n_clusters=K, n_init=n_init, random_state=seed).fit(points)
    exemplars = []
    for c in range(K):
        d = np.linalg.norm(points - km.cluster_centers_[c], axis=1)
        exemplars.append(int(d.argmin()))
    return assign_to_exemplars(W, exemplars)


def _S_constrained(W, E_arr):
    """Constraint-respecting objective for a given exemplar set:
       sum_{e in E} W[e, e] + sum_{k not in E} max_{a in E} W[k, a]."""
    N = W.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[E_arr] = False
    S_ex = float(np.diag(W)[E_arr].sum())
    S_non_ex = (float(W[mask][:, E_arr].max(axis=1).sum())
                if mask.any() else 0.0)
    return S_ex + S_non_ex


# --------------------------------------------------------------------- #
# Greedy exemplar selection                                             #
# --------------------------------------------------------------------- #
def greedy_baseline(W, K):
    """Greedy growth of the exemplar set under the constraint-respecting
    objective: at each step, add the data point that maximally improves
    S(E) computed with exemplars self-assigned."""
    N = W.shape[0]
    if K < 1 or K > N:
        return None
    chosen = []
    for _ in range(K):
        best_gain = -np.inf
        best_a = -1
        for a in range(N):
            if a in chosen:
                continue
            cand = chosen + [a]
            S = _S_constrained(W, np.asarray(cand, dtype=np.int64))
            if S > best_gain:
                best_gain = S
                best_a = a
        chosen.append(best_a)
    return assign_to_exemplars(W, chosen)


# --------------------------------------------------------------------- #
# Exhaustive optimum (when feasible)                                    #
# --------------------------------------------------------------------- #
def exhaustive_optimum(W, K, max_combinations=5_000_000, chunk_size=50_000):
    """Brute-force search over all C(N, K) exemplar subsets, evaluated
    under the constraint-respecting objective. Returns None when the
    combinatorial size exceeds `max_combinations`.

    Subsets are processed in chunks of `chunk_size` so the inner loop
    is vectorized in NumPy rather than per-combination in Python."""
    N = W.shape[0]
    if K < 1 or K > N:
        return None
    n_combos = comb(N, K)
    if n_combos > max_combinations:
        return None

    diag = np.diag(W)
    best_S = -np.inf
    best_E = None

    combos_iter = combinations(range(N), K)
    while True:
        chunk = []
        for _ in range(chunk_size):
            try:
                chunk.append(next(combos_iter))
            except StopIteration:
                break
        if not chunk:
            break
        E_batch = np.asarray(chunk, dtype=np.int64)               # (B, K)
        B = E_batch.shape[0]

        # Per-batch sum of diagonal (preference) over chosen exemplars
        S_ex = diag[E_batch].sum(axis=1)                          # (B,)

        # max_{a in E[b]} W[k, a] for every (k, b)
        max_sims = W[:, E_batch].max(axis=2)                      # (N, B)

        # Mask for rows that ARE exemplars in each combination
        mask = np.zeros((B, N), dtype=bool)
        rows = np.repeat(np.arange(B), K)
        cols = E_batch.reshape(-1)
        mask[rows, cols] = True
        non_ex = (~mask).T                                        # (N, B)

        S = S_ex + (max_sims * non_ex).sum(axis=0)                # (B,)
        idx = int(S.argmax())
        if S[idx] > best_S:
            best_S = float(S[idx])
            best_E = E_batch[idx]

        if B < chunk_size:
            break

    return assign_to_exemplars(W, best_E)


# --------------------------------------------------------------------- #
# Convenience: gap from optimum                                         #
# --------------------------------------------------------------------- #
def gap_pct(S, S_opt):
    """Relative optimality gap in percent: (S_opt - S) / |S_opt| * 100.
    Smaller is better; 0 means matched the optimum."""
    if S_opt is None or not np.isfinite(S_opt):
        return None
    if S_opt == 0:
        return None
    return float((S_opt - S) / abs(S_opt) * 100.0)
