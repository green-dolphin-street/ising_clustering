"""Verify the exemplar-clustering self-consistency equations of Appendix C.

Pipeline:
  1. Generate well-separated Gaussian clusters and a similarity matrix.
  2. Run the iterative searcher (corrected variant) until convergence
     under a temperature anneal.
  3. Score the resulting clustering against three benchmarks on the
     sum-similarity objective S(E) = sum_k max_{a in E} W[k, a]:
       - K-means + exemplar projection (oracle K and matched K)
       - Greedy submodular maximization (matched K)
       - Exhaustive optimum (when |C(N, K)| is small enough)
"""

import numpy as np

from dataset import generate_gaussian_clusters, compute_similarity_matrix
from solver import SelfConsistencySolver
from benchmarks import (sum_similarity, assign_to_exemplars,
                        self_consistency_score, kmeans_baseline,
                        greedy_baseline, exhaustive_optimum, gap_pct)


def run_one(n_clusters_true=4, n_per_cluster=15, dim=2,
            cluster_std=0.3, spread=2.0, min_separation_factor=4.0,
            seed=0, preference_quantile=0.5,
            T_init=2.0, T_final=0.02, n_anneal_steps=20,
            n_iter_per_T=30, damping=0.5,
            exhaustive_cap=2_000_000, verbose=True):

    points, true_labels, centers = generate_gaussian_clusters(
        n_clusters=n_clusters_true, n_per_cluster=n_per_cluster,
        dim=dim, cluster_std=cluster_std, spread=spread,
        min_separation_factor=min_separation_factor, seed=seed)
    W, preference = compute_similarity_matrix(
        points, preference_quantile=preference_quantile)
    N = W.shape[0]
    print(f"\n=== N={N}, true K={n_clusters_true}, "
          f"sigma={cluster_std}, "
          f"min_sep={min_separation_factor}*sigma, "
          f"preference={preference:.3f} ===")

    # 1. Self-consistency search
    solver = SelfConsistencySolver(
        W, T_init=T_init, T_final=T_final,
        n_anneal_steps=n_anneal_steps, n_iter_per_T=n_iter_per_T,
        damping=damping, variant='corrected', seed=seed)
    solver.run(verbose=False)
    sc_assn, sc_ex, S_sc = self_consistency_score(solver, W)
    K_self = len(sc_ex)
    print(f"\n[Self-consistency]  K_self={K_self}  S={S_sc:.4f}")

    # 2. Optimal at K_self (if combinatorially feasible)
    opt = exhaustive_optimum(W, K_self, max_combinations=exhaustive_cap)
    if opt is None:
        S_opt = None
        print(f"[Optimum @ K_self={K_self}]  skipped "
              f"(C(N, K) > {exhaustive_cap:,})")
    else:
        _, _, S_opt = opt
        print(f"[Optimum @ K_self={K_self}]  S={S_opt:.4f}")

    # 3. Greedy at K_self
    g_assn, g_ex, S_g = greedy_baseline(W, K_self)
    print(f"[Greedy @ K_self={K_self}]   S={S_g:.4f}")

    # 4. K-means at K_self (matched K - fair comparison)
    km_self = kmeans_baseline(points, W, K_self, seed=seed)
    if km_self is not None:
        _, _, S_km_self = km_self
        print(f"[K-means @ K_self={K_self}]  S={S_km_self:.4f}")
    else:
        S_km_self = None

    # 5. K-means at K_true (oracle K - best case for K-means)
    km_true = kmeans_baseline(points, W, n_clusters_true, seed=seed)
    _, _, S_km_true = km_true
    print(f"[K-means @ K_true={n_clusters_true}]  S={S_km_true:.4f}  "
          f"(oracle K)")

    # Gap reporting
    if S_opt is not None:
        print("\n  Optimality gap from S_opt (lower is better):")
        print(f"    self-consistency : {gap_pct(S_sc,       S_opt):6.2f} %")
        print(f"    greedy           : {gap_pct(S_g,        S_opt):6.2f} %")
        if S_km_self is not None:
            print(f"    K-means @ K_self : {gap_pct(S_km_self, S_opt):6.2f} %")

    return dict(
        N=N, K_true=n_clusters_true, K_self=K_self,
        S_self=S_sc, S_opt=S_opt, S_greedy=S_g,
        S_km_self=S_km_self, S_km_true=S_km_true,
    )


def main():
    np.set_printoptions(suppress=True, precision=3)
    configs = [
        dict(n_clusters_true=3, n_per_cluster=10, cluster_std=0.25,
             min_separation_factor=4.0, spread=2.5, seed=0,
             exhaustive_cap=5_000_000),                           # C(30,3)=4K
        dict(n_clusters_true=4, n_per_cluster=10, cluster_std=0.30,
             min_separation_factor=4.0, spread=3.0, seed=1,
             exhaustive_cap=5_000_000),                           # C(40,4)=91K
        dict(n_clusters_true=5, n_per_cluster=10, cluster_std=0.25,
             min_separation_factor=4.0, spread=3.0, seed=2,
             exhaustive_cap=5_000_000),                           # C(50,5)=2.1M
    ]
    summaries = []
    for cfg in configs:
        summaries.append(run_one(**cfg, verbose=False))

    print("\n" + "=" * 70)
    print("SUMMARY (sum-similarity S; smaller |gap| from optimum is better)")
    print("=" * 70)
    print(f"{'N':>4} {'Ktrue':>5} {'Kself':>5} "
          f"{'S_self':>10} {'S_opt':>10} {'gap%':>7} "
          f"{'S_greedy':>10} {'S_kmK':>10}")
    for r in summaries:
        gap = gap_pct(r['S_self'], r['S_opt'])
        gap_str = f"{gap:6.2f}" if gap is not None else "  n/a"
        opt_str = f"{r['S_opt']:10.3f}" if r['S_opt'] is not None else "       n/a"
        kmk_str = (f"{r['S_km_self']:10.3f}"
                   if r['S_km_self'] is not None else "       n/a")
        print(f"{r['N']:>4} {r['K_true']:>5} {r['K_self']:>5} "
              f"{r['S_self']:10.3f} {opt_str} {gap_str:>7} "
              f"{r['S_greedy']:10.3f} {kmk_str}")


if __name__ == "__main__":
    main()
