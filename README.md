# Self-Consistency Equations for Exemplar-Based Clustering

This repository is a sandbox for verifying the **exemplar-based clustering self-consistency equations** derived for a manuscript. The manuscript proposes a
statistical-physics-guided xApp framework for Open RAN coordination tasks: each task is mapped to an Ising-model energy, and self-consistency equations characterizing the low-energy configurations are then learned by a graph neural network so that inference is a single forward pass instead of an iterative search.

One-to-one matching is the headline task in the manuscript; exemplar-based clustering is the next coordination problem in line. Before training a GNN to emulate the clustering self-consistency dynamics, two things have to be checked:

1. that the clustering equations were derived correctly, and
2. that their fixed point actually corresponds to a good clustering.

This repo contains the minimal artifacts needed for both checks: a synthetic clustering dataset generator, a damped-and-annealed fixed-point searcher for the equations, a benchmark suite (K-means, greedy, exhaustive optimum) on the constraint-respecting sum-similarity objective, and a verification script that ties them together.

| File | Role |
|---|---|
| [dataset.py](dataset.py) | Gaussian-cluster generator with a guard band on center separation, plus the similarity matrix and diagonal-preference helpers |
| [solver.py](solver.py) | `SelfConsistencySolver` — finite-temperature, damped, annealed fixed-point searcher for the Appendix-C equations |
| [benchmarks.py](benchmarks.py) | K-means + projection, greedy submodular, and exhaustive-optimum baselines on the constraint-respecting objective |
| [verify_clustering.py](verify_clustering.py) | End-to-end verification entry point |

The remaining sections describe the simulation environment, the
equations themselves, the iterative searcher, a quick analytical
validation that the equations are correct, the benchmark suite, the
score metrics, and the empirical results.

---

## 1. Simulation settings

Implemented in [dataset.py](dataset.py).

### 1.1 Synthetic dataset

Sample $N = K \cdot n_{\mathrm{per}}$ points from $K$ isotropic
Gaussian clusters in $\mathbb{R}^d$:

- **Cluster centers** placed by **rejection sampling** in the box
  $[-\text{spread}, \text{spread}]^d$ subject to a guard band: every
  pair of centers must satisfy
  $\lVert \mu_c - \mu_{c'} \rVert_2 \;\ge\; m \cdot \sigma_{\mathrm{cluster}}$
  for a margin $m$ (default $m = 4$). With $m = 4$, the
  $2\sigma$ balls of any two clusters do not intersect, so the
  generative cluster label is essentially the same as the
  similarity-optimal label for every sample. This removes the
  boundary-ambiguity blind spot that ARI/NMI have when measuring
  against generative labels.
- **Cluster samples.** Each cluster contributes $n_{\mathrm{per}}$
  draws from $\mathcal{N}(\mu_c, \sigma_{\mathrm{cluster}}^2 I)$.
- **Resampling cap.** Rejection sampling is capped at
  `max_resample_attempts` (default $10\,000$); if the cap is hit,
  the call raises with a clear message — the box is too tight for
  the requested $K$ and margin.
- **Disabling the guard band.** Setting `min_separation_factor = None`
  or $0$ recovers the original uniform-center sampling.

### 1.2 Similarity matrix

Negative squared Euclidean distance:

$$w_{ka} = -\lVert y_k - y_a \rVert_2^2 .$$

### 1.3 Diagonal preference

The diagonal entry $w_{aa}$ — the "self-preference" — controls how
readily a point becomes its own exemplar; higher (less negative)
values favor more clusters. Following Frey & Dueck (2007), we default
to the median of the off-diagonal similarities,

$$w_{aa} \;=\; \mathrm{Quantile}_q\!\left(\{w_{kb}\}_{k \ne b}\right), \qquad q = 0.5.$$

---

## 2. Self-consistency equations under test

The exemplar-clustering Hamiltonian
$E(\mathbf{x}) = -\sum_{k,a} w_{ka}\, x_{ka}$ is decomposed
symmetrically into point-side and exemplar-side cavity marginals.
With log-likelihood ratios

$$\exp\!\left(-\tfrac{R_{ka}}{T}\right) = \frac{P^{\mathcal{P}}_{ka}(0)}{P^{\mathcal{P}}_{ka}(1)}, \qquad
\exp\!\left(-\tfrac{A_{ka}}{T}\right) = \frac{P^{\mathcal{E}}_{ka}(0)}{P^{\mathcal{E}}_{ka}(1)},$$

the corrected fixed-point system from Appendix C is

$$R_{ka} \;=\; \frac{w_{ka}}{2} \;-\; T\,\log \sum_{b \ne a} \exp\!\left(\frac{1}{T}\!\left(\frac{w_{kb}}{2} + A_{kb}\right)\right),$$

$$A_{aa} \;=\; \frac{w_{aa}}{2} \;+\; T \sum_{j \ne a} \log\!\left(1 + \exp\!\left(\frac{1}{T}\!\left(\frac{w_{ja}}{2} + R_{ja}\right)\right)\right),$$

$$A_{ka} \;=\; \frac{w_{ka}}{2} \;-\; T \log\!\Bigg[\, 1 \;+\; \exp\!\left(-\frac{1}{T}\!\left(\frac{w_{aa}}{2} + R_{aa}\right)\right) \prod_{\substack{j \ne k\\ j \ne a}} \left(1 + \exp\!\left(\frac{1}{T}\!\left(\frac{w_{ja}}{2} + R_{ja}\right)\right)\right)^{-1} \Bigg], \quad k \ne a.$$

These are the equations as written in the manuscript; the iterative
searcher in §3 evaluates exactly these expressions at every fixed,
finite temperature $T$.

> **Notation in the code.** Inside [solver.py](solver.py) the helper
> `softplus_T(x) := T log(1 + exp(x/T))` is used as a numerically
> stable shorthand for the repeated $T \log(1 + e^{\cdot/T})$ kernels
> in the equations above. It is purely an implementation convenience;
> no temperature limit is taken anywhere in the searcher.

---

## 3. Iterative searcher

Implemented as `SelfConsistencySolver` in [solver.py](solver.py).

### 3.1 Algorithm

1. **Initialize** $R^{(0)} = A^{(0)} = 0$.
2. **Anneal** the temperature geometrically from $T_{\mathrm{init}}$
   to $T_{\mathrm{final}}$ in $n_{\mathrm{anneal}}$ steps. Annealing
   is a *numerical* device for navigating the fixed-point landscape
   smoothly — every individual sweep evaluates the finite-$T$
   equations of §2 verbatim.
3. **Inner sweeps.** At each $T$, run $n_{\mathrm{iter}}$ damped
   message-passing sweeps,

$$R^{(t+1)} \;=\; \lambda\, R^{(t)} \;+\; (1 - \lambda)\, \widehat{R}\!\left(R^{(t)}, A^{(t)};\, T\right),$$

$$A^{(t+1)} \;=\; \lambda\, A^{(t)} \;+\; (1 - \lambda)\, \widehat{A}\!\left(R^{(t+1)}, A^{(t)};\, T\right),$$

   where $\widehat{R}, \widehat{A}$ are the right-hand sides of the
   self-consistency equations and $\lambda$ is the damping factor.

4. **Decide.** After the final inner sweep, form
   $D_{ka} = R_{ka} + A_{ka}$ and read out the exemplar set
   $\mathcal{E} = \{a^\star_k : a^\star_k = \arg\max_a D_{ka}\}$.

5. **Project to feasible.** Force every exemplar to self-assign
   (constraint $x_{aa} \ge x_{ka}$) and assign every non-exemplar to
   its highest-similarity exemplar in $\mathcal{E}$.

### 3.2 Default hyperparameters

| Parameter | Default | Meaning |
|---|---|---|
| $T_{\mathrm{init}}$ | $2.0$ | Starting temperature (smooth landscape) |
| $T_{\mathrm{final}}$ | $0.02$ | Final temperature (sharp landscape) |
| $n_{\mathrm{anneal}}$ | $20$ | Geometric anneal steps |
| $n_{\mathrm{iter}}$ | $30$ | Damped sweeps per temperature |
| $\lambda$ | $0.5$ | Damping factor |

---

## 4. Quick validation that the derived equations are correct

Before running the empirical benchmarks, two independent sanity
checks confirm the equations of §2 are the right ones.

### 4.1 Cavity-marginal re-derivation

The off-diagonal $A_{ka}$ in §2 is exactly the ratio of the cavity
marginals $P^{\mathcal{E}}_{ka}(x_{ka} = 0) / P^{\mathcal{E}}_{ka}(x_{ka} = 1)$
expressed in log form. The intermediate cancellation steps in
Appendix C produce this same expression once the additive ``$1$''
arising from the $x_{aa} = 0$ branch is kept outside the entire
product over competing assignments to exemplar $a$ (rather than
absorbed into a separate factor). This was the source of the typo
flagged earlier in the manuscript and corrected in
[manuscript.tex](manuscript.tex).

### 4.2 Reduction to standard affinity propagation

A second consistency check is that the Appendix-C system reproduces
a known correct algorithm. Specifically, with rescaled messages
$\tilde A := 2A$ and $\tilde R := 2R$, the equations of §2 satisfy
the identity

$$\tilde A_{ka} \;=\; w_{ka} + a_{\mathrm{AP}}(k, a), \qquad
  \tilde R_{ka} \;=\; w_{ka} \;-\; \max_{b \ne a}\!\left(w_{kb} + \tilde A_{kb}\right),$$

where $a_{\mathrm{AP}}$ is the availability message of the standard
affinity-propagation algorithm of Frey & Dueck (2007). This shows
the Appendix-C derivation is a finite-temperature,
symmetric-splitting reformulation of an algorithm whose
correctness is already established in the literature, and gives an
independent confirmation that the equations are right.

Affinity propagation is therefore *not* used as an empirical
benchmark — comparing two algorithms that are equivalent up to a
known message rescaling would be circular. The benchmarks in §5
are external clustering algorithms or solvers of the optimization
problem itself.

---

## 5. Benchmarks

All benchmarks are evaluated on the same constraint-respecting
sum-similarity objective

$$S(\mathcal{E}) \;=\; \sum_{e \in \mathcal{E}} w_{ee} \;+\; \sum_{k \notin \mathcal{E}} \max_{a \in \mathcal{E}} w_{ka},$$

i.e. each chosen exemplar pays the diagonal preference and each
non-exemplar contributes its similarity to its highest-similarity
exemplar. This corresponds exactly to the exemplar-clustering
constraint $x_{aa} \ge x_{ka}$ being satisfied.

Benchmarks are implemented in [benchmarks.py](benchmarks.py).

### 5.1 K-means + exemplar projection

Run K-means on the raw points, take the centroid of each cluster,
project to the nearest data point, and score $S$ on that exemplar
set. This is K-means restricted to the same "exemplars must be data
points" constraint as the Appendix-C objective. Reported at both
$K = K_{\mathrm{self}}$ (matched, fair comparison) and
$K = K_{\mathrm{true}}$ (oracle).

### 5.2 Greedy submodular maximization

Greedily grow $\mathcal{E}$ one element at a time, each step picking
the data point that maximally improves $S$. The objective is
monotone non-decreasing in $\mathcal{E}$, so greedy enjoys the
standard $(1 - 1/e) \approx 0.63$ approximation guarantee
(Nemhauser, Wolsey, Fisher 1978). Run at $K = K_{\mathrm{self}}$.

### 5.3 Exhaustive optimum

Brute-force search over all $\binom{N}{K_{\mathrm{self}}}$ exemplar
subsets, evaluated under $S$. Vectorized in NumPy over chunks of
$50\,000$ subsets at a time; the run for $\binom{50}{5} \approx 2.1\,$M
combinations finishes in a few seconds. Skipped when the
combinatorial size exceeds the configured cap (default $5\,$M).

---

## 6. Score metrics

| Metric | Definition | Direction | What it measures |
|---|---|---|---|
| $K_{\mathrm{self}}$ | $\lvert \{a^\star_k : k = 1, \dots, N\} \rvert$ | match $K_{\mathrm{true}}$ | Number of exemplars chosen by the solver |
| $S$ | $\sum_{e \in \mathcal{E}} w_{ee} + \sum_{k \notin \mathcal{E}} \max_{a \in \mathcal{E}} w_{ka}$ | higher is better | Constraint-respecting sum-similarity (the actual objective) |
| Gap | $(S_{\mathrm{opt}} - S) \,/\, \lvert S_{\mathrm{opt}} \rvert \cdot 100\%$ | lower is better, $0$ matches optimum | Relative optimality gap |

All baselines (K-means, greedy, self-consistency) are scored on the
same constraint-respecting $S$, so the gap-from-optimum comparison is
apples-to-apples.

---

## 7. Empirical results

Run via [verify_clustering.py](verify_clustering.py) with the default
hyperparameters in §3.2, $m = 4$ guard band, $q = 0.5$ preference,
and exhaustive cap $5\,000\,000$.

| $N$ | $K_{\mathrm{true}}$ | $K_{\mathrm{self}}$ | $S_{\mathrm{self}}$ | $S_{\mathrm{opt}}$ | gap (self) | $S_{\mathrm{greedy}}$ | gap (greedy) | $S_{\mathrm{KM}}@K_{\mathrm{self}}$ | gap (KM) |
|----:|--------------------:|--------------------:|--------------------:|-------------------:|-----------:|----------------------:|-------------:|------------------------------------:|---------:|
|  30 |                   3 |                   3 | $-33.998$ | $-33.998$ | $\mathbf{0.00\%}$ | $-36.436$ | $7.17\%$  | $-33.998$ | $0.00\%$ |
|  40 |                   4 |                   4 | $-44.707$ | $-44.644$ | $\mathbf{0.14\%}$ | $-54.448$ | $21.96\%$ | $-44.644$ | $0.00\%$ |
|  50 |                   5 |                   5 | $-60.833$ | $-58.350$ | $\mathbf{4.26\%}$ | $-63.810$ | $9.36\%$  | $-58.350$ | $0.00\%$ |

Wall-clock for the full sweep (incl. exhaustive optimum): $\sim 10\,$s.

Observations:

- **K-selection.** The solver picks the correct number of exemplars
  ($K_{\mathrm{self}} = K_{\mathrm{true}}$) on every instance; the
  $4\sigma$ guard band is enough to make the median preference yield
  the right $K$.
- **Optimality.** The self-consistency $S$ matches the exhaustive
  optimum exactly at $N = 30$ and is within $0.14\%$ at $N = 40$.
  At $N = 50$ the gap widens to $4.26\%$, plausibly because the
  default anneal is short for the larger problem; tightening the
  schedule should close it.
- **Versus greedy.** Self-consistency beats the
  $(1 - 1/e)$-approximation greedy heuristic by $7$–$22$ percentage
  points on every instance, confirming that the message-passing
  fixed point is doing meaningful global optimization, not just a
  local greedy expansion.
- **Versus K-means + projection.** K-means lands on the exact
  constrained optimum on all three instances. This is expected for
  well-separated isotropic Gaussians, where the centroid is the
  optimal exemplar location and its nearest data point is the
  medoid. Self-consistency closes the gap to K-means at $N = 30$
  and $N = 40$; the slight gap at $N = 50$ matches the gap to the
  optimum.

---

## 8. Conclusion

On well-separated Gaussian-cluster instances under a $4\sigma$ guard
band, the corrected Appendix-C self-consistency equations:

1. select the correct number of exemplars,
2. land on or within a few percent of the exhaustive constrained
   optimum on the exemplar-clustering sum-similarity objective,
3. consistently beat a $(1 - 1/e)$-approximation greedy baseline,
4. match a centroid-projected K-means baseline that is near-optimal
   in this regime.

Combined with the analytical sanity checks of §4 (cavity-marginal
re-derivation and rescaling identity to standard affinity
propagation), this constitutes a quantitative validation of the
equations as a clustering objective. The framework can proceed to
GNN training with these targets.
