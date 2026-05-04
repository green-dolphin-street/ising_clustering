"""Render 2D maps of the clustering test cases for the README.

For each test case, three side-by-side panels are produced:
  (i)   ground-truth labels with the true cluster centers,
  (ii)  the self-consistency solver's exemplars and assignments,
  (iii) the exhaustive constrained optimum's exemplars and assignments.

Output files are written to ./figures/cluster_N{N}_K{K}.png.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import (assign_to_exemplars, exhaustive_optimum,
                        gap_pct, medoid_corrected_score,
                        self_consistency_score, sum_similarity)
from dataset import compute_similarity_matrix, generate_gaussian_clusters
from solver import SelfConsistencySolver


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def _color_for(idx, n):
    """Pick a distinct color from a categorical colormap."""
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return cmap(idx % cmap.N)


def _panel(ax, points, assignments, exemplars, title,
           ground_truth_centers=None):
    """Draw one panel: data points colored by cluster, exemplars marked."""
    if exemplars is None or len(exemplars) == 0:
        ax.scatter(points[:, 0], points[:, 1], s=18, c="lightgray",
                   edgecolor="k", linewidth=0.3)
    else:
        # Map each exemplar to a sequential color index for stable colors
        ex_to_idx = {int(e): i for i, e in enumerate(sorted(exemplars))}
        colors = np.array([
            _color_for(ex_to_idx[int(a)], len(ex_to_idx))
            for a in assignments
        ])
        ax.scatter(points[:, 0], points[:, 1], s=22, c=colors,
                   edgecolor="k", linewidth=0.3, zorder=2)
        # Mark exemplars with a hollow square around their data point
        ex_arr = np.asarray(sorted(exemplars), dtype=int)
        ax.scatter(points[ex_arr, 0], points[ex_arr, 1],
                   s=130, marker="s", facecolors="none",
                   edgecolors="black", linewidth=1.5, zorder=3,
                   label="exemplar")

    if ground_truth_centers is not None:
        ax.scatter(ground_truth_centers[:, 0],
                   ground_truth_centers[:, 1],
                   s=70, marker="x", c="black", linewidth=1.4,
                   zorder=4, label="Cluster center for sampling")
    if ground_truth_centers is not None or (
            exemplars is not None and len(exemplars) > 0):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linewidth=0.5)


def render_one(n_clusters_true, n_per_cluster, cluster_std, spread,
               min_separation_factor, seed,
               T_init=2.0, T_final=0.02, n_anneal_steps=20,
               n_iter_per_T=30, damping=0.5,
               exhaustive_cap=5_000_000):
    points, true_labels, centers = generate_gaussian_clusters(
        n_clusters=n_clusters_true, n_per_cluster=n_per_cluster,
        cluster_std=cluster_std, spread=spread,
        min_separation_factor=min_separation_factor, seed=seed)
    W, preference = compute_similarity_matrix(points)
    N = W.shape[0]

    # Ground-truth panel uses the true labels directly (one "exemplar"
    # per true cluster, picked as the data point closest to the true
    # center, so the markers in panel (i) line up with panels ii/iii).
    gt_exemplars = []
    for c in range(n_clusters_true):
        d = np.linalg.norm(points - centers[c], axis=1)
        gt_exemplars.append(int(d.argmin()))
    gt_assn, gt_ex, _ = assign_to_exemplars(W, gt_exemplars)

    # Self-consistency
    solver = SelfConsistencySolver(
        W, T_init=T_init, T_final=T_final,
        n_anneal_steps=n_anneal_steps, n_iter_per_T=n_iter_per_T,
        damping=damping, variant="corrected", seed=seed)
    solver.run(verbose=False)
    sc_assn, sc_ex, S_sc = self_consistency_score(solver, W)

    # Exhaustive optimum at K = K_self
    opt = exhaustive_optimum(W, len(sc_ex), max_combinations=exhaustive_cap)
    if opt is None:
        opt_assn, opt_ex, S_opt = sc_assn, sc_ex, None
        opt_title_extra = "(skipped: too many subsets)"
    else:
        opt_assn, opt_ex, S_opt = opt
        opt_title_extra = f"$S = {S_opt:.2f}$"

    # Medoid-corrected sum-similarity for the self-consistency
    # partition: holds the partition fixed and replaces every
    # exemplar with its cluster medoid, so the resulting score
    # reflects partition quality alone.
    _, _, S_partition_self = medoid_corrected_score(W, sc_assn)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    _panel(axes[0], points, gt_assn, gt_ex,
           f"Ground truth ($K_{{\\mathrm{{true}}}} = {n_clusters_true}$)",
           ground_truth_centers=centers)
    gap_self = gap_pct(S_sc, S_opt)
    gap_partition = gap_pct(S_partition_self, S_opt)
    self_subtitle = (f"$S = {S_sc:.2f}$"
                     + (f", gap $= {gap_self:.2f}\\%$"
                        if gap_self is not None else "")
                     + (f", partition gap $= {gap_partition:.2f}\\%$"
                        if gap_partition is not None else ""))
    _panel(axes[1], points, sc_assn, sc_ex,
           f"Self-consistency ($K_{{\\mathrm{{self}}}} = {len(sc_ex)}$)\n"
           f"{self_subtitle}")
    _panel(axes[2], points, opt_assn, opt_ex,
           f"Exhaustive optimum at $K = {len(sc_ex)}$\n{opt_title_extra}")

    # Lock every panel to the same square viewport so aspect='equal'
    # doesn't silently shrink panels for compact-data test cases.
    # MIN_HALF guarantees a minimum viewport size so the saved panel
    # is the same physical size across test cases regardless of
    # whether the underlying data is more or less spread out.
    MIN_HALF = 3.0
    pad = 0.5
    xmin, ymin = points.min(axis=0) - pad
    xmax, ymax = points.max(axis=0) + pad
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - cx, ymax - cy, MIN_HALF)
    xlim, ylim = (cx - half, cx + half), (cy - half, cy + half)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.suptitle(
        f"$N = {N}$,  $\\sigma = {cluster_std}$,  "
        f"$\\mathrm{{spread}} = {spread}$,  "
        f"guard $= {min_separation_factor}\\sigma$",
        fontsize=12)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"cluster_N{N}_K{n_clusters_true}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  wrote {out}")
    return dict(N=N, K_true=n_clusters_true, K_self=len(sc_ex),
                S_self=S_sc, S_partition=S_partition_self,
                S_opt=S_opt, gap=gap_self, partition_gap=gap_partition)


def main():
    print("Rendering clustering visualizations...")
    configs = [
        dict(n_clusters_true=3, n_per_cluster=10, cluster_std=0.25,
             min_separation_factor=4.0, spread=2.5, seed=0),
        dict(n_clusters_true=4, n_per_cluster=10, cluster_std=0.30,
             min_separation_factor=4.0, spread=3.0, seed=1),
        dict(n_clusters_true=5, n_per_cluster=10, cluster_std=0.25,
             min_separation_factor=4.0, spread=3.0, seed=2),
    ]
    for cfg in configs:
        render_one(**cfg)


if __name__ == "__main__":
    main()
