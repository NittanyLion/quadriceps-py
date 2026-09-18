"""Exactness check of a rule against the closed-form moments of its weight."""

from __future__ import annotations

import numpy as np

from ._catalog import check_family


def _moments1d(family: str, p: int) -> np.ndarray:
    """One-dimensional moments of the normalized weights: E Z^e for N(0,1), E U^e for U[0,1]."""
    m = np.zeros(p + 1)
    if family == "gh":
        m[0] = 1.0
        for e in range(2, p + 1, 2):
            m[e] = m[e - 2] * (e - 1)        # (e-1)!!
    else:
        m[:] = 1.0 / np.arange(1, p + 2)
    return m


def exactness_error(X, w, p, family) -> float:
    """Largest relative monomial error of the rule ``(X, w)``.

    The maximum, over all monomials ``x^a`` of total degree ``<= p``, of

        ``|sum_i w_i x_i^a - E x^a| / max(sum_i |w_i| |x_i^a|, 1)``

    for the normalized weight of ``family`` (``"gh"``: ``N(0, I_d)``; ``"le"``: uniform on
    ``[0,1]^d``). The denominator is the scale of the sum being computed, so a value near
    machine epsilon means the rule is exact to rounding. Rules returned with
    ``normalize=False`` must be checked in the normalized frame.

    >>> from quadriceps import ghpos
    >>> X, w = ghpos(4, 5)                       # degree 2·5 - 1 = 9
    >>> exactness_error(X, w, 9, "gh") < 1e-14
    True
    """
    check_family(family)
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    n, d = X.shape
    if w.shape != (n,):
        raise ValueError(f"X has {n} rows, w has shape {w.shape}")
    m = _moments1d(family, p)
    e = np.arange(p + 1)
    P = [np.asfortranarray(X[:, k, None] ** e) for k in range(d)]   # P[k][i, e] = x_ik^e; columns contiguous
    A = np.abs(P[d - 1])
    if d == 1:
        s, sa = w @ P[0], np.abs(w) @ A
        return float(np.max(np.abs(s - m) / np.maximum(sa, 1.0)))
    low = e[:, None] + e[None, :]                                    # exponent sums of the last two coordinates

    def descend(k, prev, mom, r):
        # prev: w · x_1^a_1 ⋯ x_k^a_k; mom: its exact moment; r: degree left for x_(k+1), …, x_d
        if k == d - 2:                        # last two coordinates: all their exponent pairs in one product
            B = prev[:, None] * P[k][:, : r + 1]
            S = B.T @ P[k + 1][:, : r + 1]
            SA = np.abs(B).T @ A[:, : r + 1]
            E = np.abs(S - mom * np.outer(m[: r + 1], m[: r + 1])) / np.maximum(SA, 1.0)
            return float(np.max(E[low[: r + 1, : r + 1] <= r]))
        return max(descend(k + 1, prev * P[k][:, a], mom * m[a], r - a) for a in range(r + 1))

    return descend(0, w, 1.0, int(p))
