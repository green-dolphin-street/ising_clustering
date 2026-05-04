"""Iterative searcher for the fixed point of the exemplar-clustering
self-consistency equations from Appendix C.

Two variants of the off-diagonal availability update A_{ka} (k != a) are
provided so the user can compare:

  variant='paper'      -> Eq. (68) as printed in the manuscript:
        A_{ka} = w_{ka}/2
                 - softplus_T( -(w_{aa}/2 + R_{aa}) )
                 + sum_{j != k, j != a} softplus_T( w_{ja}/2 + R_{ja} )

  variant='corrected'  -> derived directly from Eq. (55)-(56):
        A_{ka} = w_{ka}/2
                 - softplus_T( -(w_{aa}/2 + R_{aa})
                                - sum_{j != k, j != a} softplus_T(w_{ja}/2 + R_{ja}) )

  The diagonal Eq. (69) and the responsibility Eq. (70) are unambiguous
  and shared between the two variants:
        A_{aa} = w_{aa}/2 + sum_{j != a} softplus_T( w_{ja}/2 + R_{ja} )
        R_{ka} = w_{ka}/2 - T * log( sum_{b != a} exp((w_{kb}/2 + A_{kb})/T) )

  In the T -> 0 limit, the 'corrected' off-diagonal update reduces to
        A_{ka} = w_{ka}/2 + min(0, (w_{aa}/2 + R_{aa})
                                 + sum_{j != k, j != a} max(0, w_{ja}/2 + R_{ja}))
  which matches the affinity-propagation availability update of
  Frey & Dueck (2007). The 'paper' update does not reduce to that form,
  which is what motivated the discrepancy flag.

  softplus_T(x) := T * log(1 + exp(x / T))
"""

import numpy as np


def softplus_T(x, T):
    """Numerically stable T * log(1 + exp(x / T))."""
    z = x / T
    # log1p(exp(z)) computed stably via the standard softplus identity
    return T * (np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))))


def lse_excluding_self_along_axis(M, T, axis):
    """For each entry (k, a), compute T * log( sum_{b != a} exp(M[k, b]/T) )
    if axis == 1, or T * log( sum_{j != k} exp(M[j, a]/T) ) if axis == 0.

    Implements the standard "subtract the excluded term" trick around the
    row/column max for numerical stability.
    """
    Z = M / T
    if axis == 0:
        Z = Z.T
    # Now exclude along axis=1 of (the possibly transposed) Z.
    z_max = np.max(Z, axis=1, keepdims=True)
    exp_vals = np.exp(Z - z_max)              # >= 0
    total = np.sum(exp_vals, axis=1, keepdims=True)
    excluded = total - exp_vals               # sum_{b != a} exp(z_b - z_max)
    excluded = np.maximum(excluded, 1e-300)   # guard log(0)
    out = T * (np.log(excluded) + z_max)
    if axis == 0:
        out = out.T
    return out


class SelfConsistencySolver:
    """Damped, annealed iteration of the clustering self-consistency
    equations from Appendix C."""

    def __init__(self, W, T_init=2.0, T_final=0.02, n_anneal_steps=20,
                 n_iter_per_T=30, damping=0.5, variant='corrected',
                 init_scale=0.0, seed=None):
        self.W = W.astype(np.float64)
        self.N = W.shape[0]
        self.T_init = T_init
        self.T_final = T_final
        self.n_anneal_steps = n_anneal_steps
        self.n_iter_per_T = n_iter_per_T
        self.damping = damping
        assert variant in ('paper', 'corrected'), variant
        self.variant = variant

        rng = np.random.default_rng(seed)
        self.R = init_scale * rng.standard_normal(W.shape)
        self.A = init_scale * rng.standard_normal(W.shape)

        self.history = []  # list of (T, residual) per outer step

    # ------------------------------------------------------------------ #
    # Updates                                                             #
    # ------------------------------------------------------------------ #
    def _update_R(self, T):
        """R_{ka} = w_{ka}/2 - T * log( sum_{b != a} exp((w_{kb}/2 + A_{kb})/T) )"""
        M = 0.5 * self.W + self.A                       # M[k, b]
        lse = lse_excluding_self_along_axis(M, T, axis=1)  # over b != a
        return 0.5 * self.W - lse

    def _update_A(self, T):
        f = 0.5 * self.W + self.R                       # f[j, a]
        sp_f = softplus_T(f, T)                         # softplus per element
        col_sum = sp_f.sum(axis=0)                      # sum_j sp_f[j, a]
        diag_sp = np.diag(sp_f)                         # sp_f[a, a]
        diag_f = np.diag(f)                             # f[a, a]

        # Diagonal update (Eq. 69): A_{aa} = w_{aa}/2 + sum_{j != a} sp_f[j, a]
        A_diag = 0.5 * np.diag(self.W) + (col_sum - diag_sp)

        # Off-diagonal sum_{j != k, j != a} sp_f[j, a]
        #   = col_sum[a] - sp_f[k, a] - sp_f[a, a]
        sum_excl_off = col_sum[None, :] - sp_f - diag_sp[None, :]

        if self.variant == 'paper':
            # Eq. (68) as printed:
            sp_neg_diag = softplus_T(-diag_f, T)        # softplus_T(-(w_aa/2 + R_aa))
            A_off = 0.5 * self.W - sp_neg_diag[None, :] + sum_excl_off
        else:
            # Corrected form from Eq. (55)/(56):
            inner = -diag_f[None, :] - sum_excl_off     # broadcast over k
            A_off = 0.5 * self.W - softplus_T(inner, T)

        A = A_off.copy()
        np.fill_diagonal(A, A_diag)
        return A

    # ------------------------------------------------------------------ #
    # Iteration                                                           #
    # ------------------------------------------------------------------ #
    def _step(self, T):
        new_R = self._update_R(T)
        self.R = self.damping * self.R + (1 - self.damping) * new_R
        new_A = self._update_A(T)
        delta = np.maximum(np.abs(new_A - self.A).max(),
                           np.abs(new_R - self.R).max())
        self.A = self.damping * self.A + (1 - self.damping) * new_A
        return delta

    def _temperatures(self):
        return np.geomspace(self.T_init, self.T_final, self.n_anneal_steps)

    def run(self, verbose=False):
        for T in self._temperatures():
            last_delta = None
            for _ in range(self.n_iter_per_T):
                last_delta = self._step(T)
            self.history.append((float(T), float(last_delta)))
            if verbose:
                assn, ex = self.decide()
                print(f"  T={T:.4f}  residual={last_delta:.3e}  "
                      f"#exemplars={len(ex)}  "
                      f"feasible={int(self.is_feasible(assn))}")
        return self

    # ------------------------------------------------------------------ #
    # Decision and validation                                             #
    # ------------------------------------------------------------------ #
    def decide(self):
        """Each point picks its highest-scoring exemplar via D = R + A."""
        D = self.R + self.A
        assignments = D.argmax(axis=1)
        exemplars = np.unique(assignments)
        return assignments, exemplars

    def is_feasible(self, assignments):
        """Check the exemplar constraint: every chosen exemplar must
        select itself (i.e. assignments[exemplar] == exemplar)."""
        exemplars = np.unique(assignments)
        return all(assignments[e] == e for e in exemplars)
