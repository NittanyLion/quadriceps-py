"""Which rule answers a request (d, p): a stored one, or the cheapest tensor product."""

from __future__ import annotations

import os
from typing import NamedTuple, Optional

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss

from ._catalog import INDEX, RuleInfo, check_family, stored


class NoRuleError(ValueError):
    """No stored rule covers the request, and ``pragmatic`` is ``False``."""


class Atom(NamedTuple):
    """One factor of a plan: a stored rule, or (info is None) the n-node 1-d Gauss rule."""

    d: int
    n: int
    info: Optional[RuleInfo]


def gaussq(p: int) -> int:
    """Nodes of the one-dimensional Gauss rule of degree >= p."""
    return p // 2 + 1


def best_stored(family: str, d: int, p: int) -> Optional[RuleInfo]:
    """The smallest stored rule of dimension d and degree >= p (ties: the lowest degree).

    A rule exact to degree p' >= p is a rule of degree p.
    """
    best = None
    for info in INDEX.values():
        if info.family != family or info.d != d or info.p < p:
            continue
        if best is None or (info.n, info.p) < (best.n, best.p):
            best = info
    return best


def atom(family: str, d: int, p: int) -> Optional[Atom]:
    """The single-rule answer for (d, p): Gauss in one dimension, a stored rule otherwise."""
    if d == 1:
        return Atom(1, gaussq(p), None)
    info = best_stored(family, d, p)
    return None if info is None else Atom(d, info.n, info)


def cheapest(family: str, d: int, p: int):
    """The cheapest product of atoms covering d dimensions at degree p.

    Dynamic programming over the dimension: a product of products is a product, so splitting
    in two is enough. A single stored rule wins ties.
    """
    cost = [0] * (d + 1)
    parts = [None] * (d + 1)
    for k in range(1, d + 1):
        a = atom(family, k, p)
        if a is not None:
            cost[k], parts[k] = a.n, [a]
        for j in range(1, k // 2 + 1):
            c = cost[j] * cost[k - j]
            if parts[k] is None or c < cost[k]:
                cost[k], parts[k] = c, parts[j] + parts[k - j]
    return parts[d], cost[d]


def _norule(family: str, d: int, p: int) -> NoRuleError:
    ps = [info.p for info in INDEX.values() if info.family == family and info.d == d]
    have = (f"no rules are stored for d = {d}" if not ps
            else f"stored rules for d = {d} reach q = {(max(ps) + 1) // 2}, p = {max(ps)}")
    at = f"q = {(p + 1) // 2} (p = {p})" if p % 2 else f"p = {p}"
    name = "GH" if family == "gh" else "Le"
    return NoRuleError(f"no stored positive-weight {name} rule for d = {d}, {at}: {have}; "
                       "pragmatic=True returns the cheapest tensor product of lower-dimensional rules instead")


def plan(family: str, d: int, p: int, pragmatic: bool):
    check_family(family)
    if d < 1:
        raise ValueError(f"the dimension d must be at least 1, got {d}")
    if p < 0:
        raise ValueError(f"the degree p must be nonnegative, got {p}")
    if pragmatic:
        return cheapest(family, d, p)
    a = atom(family, d, p)
    if a is None:
        raise _norule(family, d, p)
    return [a], a.n


def gauss1d(family: str, q: int):
    """The 1-d Gauss rule in the normalized frame: N(0,1) for GH, uniform on [0,1] for Le."""
    if family == "gh":
        x, w = hermegauss(q)                 # weight exp(-x²/2); the weights sum to √(2π)
        return x.reshape(-1, 1), w / np.sqrt(2 * np.pi)
    x, w = leggauss(q)                       # weight 1 on [-1,1]
    return ((x + 1) / 2).reshape(-1, 1), w / 2


def tensor(rules):
    """Tensor product of rules; the first factor varies slowest."""
    n = int(np.prod([len(w) for _, w in rules]))
    d = sum(X.shape[1] for X, _ in rules)
    X = np.empty((n, d))
    w = np.ones(n)
    rep, col = n, 0
    idx = np.arange(n)
    for Xk, wk in rules:
        nk = len(wk)
        rep //= nk
        j = (idx // rep) % nk
        w *= wk[j]
        X[:, col:col + Xk.shape[1]] = Xk[j]
        col += Xk.shape[1]
    return X, w


def _total_memory() -> Optional[int]:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def materialize(family: str, parts, n: int):
    d = sum(a.d for a in parts)
    mem = _total_memory()
    if n * (d + 1) * 8 > (mem if mem is not None else 2**40):
        raise ValueError(f"the cheapest rule for this request has {n} nodes, which does not fit in memory")
    rules = [gauss1d(family, a.n) if a.info is None else stored(a.info) for a in parts]
    if len(rules) == 1:
        return rules[0][0].copy(), rules[0][1].copy()
    return tensor(rules)
