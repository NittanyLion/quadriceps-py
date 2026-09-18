"""The public functions."""

from __future__ import annotations

import operator

import numpy as np

from ._plan import materialize, plan


def _degree(q, p) -> int:
    """The degree a call asks for: p itself, or 2q - 1. Exactly one of q, p must be given."""
    if (q is None) == (p is None):
        raise TypeError("give exactly one of q (positional) and p (keyword)")
    if p is not None:
        return operator.index(p)
    q = operator.index(q)
    if q < 1:
        raise ValueError(f"q must be at least 1, got {q}")
    return 2 * q - 1


def ghpos(d, q=None, *, p=None, normalize=True, pragmatic=False):
    """Positive-weight cubature rule for the Gaussian weight in ``d`` dimensions.

    Returns the smallest such rule the package has.

    ``ghpos(d, q)`` follows ``numpy.polynomial.hermite.hermgauss(q)``: there, ``q`` is the
    number of nodes of the one-dimensional Gauss rule, which is exact to degree ``2q - 1``.
    Here, ``ghpos(d, q)`` returns a rule with the same exactness in ``d`` dimensions, degree
    ``p = 2q - 1``, that replaces the ``q**d``-node product grid; for ``d = 1`` it is the
    ``q``-node Gauss-Hermite rule itself. Alternatively pass the degree as the keyword ``p``:
    ``ghpos(d, q)`` is ``ghpos(d, p=2*q - 1)``.

    Parameters
    ----------
    d : int
        Dimension, ``d >= 1``.
    q : int, optional
        Number of nodes of the one-dimensional Gauss rule whose exactness is wanted, ``q >= 1``.
        Give ``q`` or ``p``, not both.
    p : int, keyword-only, optional
        Degree of exactness, ``p >= 0``. Rules are stored at odd degrees; a request is served
        by the smallest stored rule of degree ``>= p``, so an even ``p`` gets the rule for
        ``p + 1``.
    normalize : bool, default True
        With ``True``, the weight is the standard normal density
        ``(2π)^(-d/2) exp(-|x|²/2)``: the weights sum to 1 and the rule computes ``E f(Z)`` for
        ``Z ~ N(0, I_d)``. With ``False``, the weight is ``exp(-|x|²)``, the convention of
        ``hermgauss``, and the weights sum to ``π^(d/2)``.
    pragmatic : bool, default False
        What to do when no stored rule covers the request. With ``False``, raise
        :class:`quadriceps.NoRuleError` (a ``ValueError``). With ``True``, return the cheapest
        (fewest nodes) tensor product of lower-dimensional rules instead: stored rules and
        one-dimensional Gauss-Hermite rules, combined over the split of ``d`` that minimizes
        the number of nodes. Such a product is a valid positive-weight rule of the requested
        degree; it is just not small. With ``True`` the product is also returned in the rare
        case that it has strictly fewer nodes than the stored rule, so the result is always
        the cheapest the package can build.

    Returns
    -------
    X : numpy.ndarray, shape (n, d)
        The nodes, one per row (also for ``d = 1``).
    w : numpy.ndarray, shape (n,)
        The weights, all strictly positive.

    ``sum(w[i] * f(X[i]))`` equals the integral of ``f`` against the weight for every
    polynomial ``f`` of total degree ``<= p``, up to rounding; :func:`quadriceps.ruleinfo`
    gives the measured error of each stored rule. The arrays are fresh copies. Loaded rules
    are cached, so repeated calls are cheap.

    Examples
    --------
    >>> X, w = ghpos(3, 4)                       # degree 7: 27 nodes instead of 4**3 = 64
    >>> float(np.round(np.sum(w * X[:, 0]**2 * X[:, 1]**4), 12))   # E[Z1² Z2⁴] = 3
    3.0
    >>> X, w = ghpos(3, 21, pragmatic=True)      # beyond the stored degrees: a tensor product
    >>> X, w = ghpos(7, 5, pragmatic=True)       # d = 7 as (d = 2) x (d = 5)
    """
    d = operator.index(d)
    parts, n = plan("gh", d, _degree(q, p), bool(pragmatic))
    X, w = materialize("gh", parts, n)
    if not normalize:                   # ∫ f(x) exp(-|x|²) dx = π^(d/2) E f(Z/√2)
        X /= np.sqrt(2.0)
        w *= np.pi ** (d / 2)
    return X, w


def lepos(d, q=None, *, p=None, normalize=True, pragmatic=False):
    """Positive-weight cubature rule for the uniform weight on a ``d``-dimensional cube.

    Returns the smallest such rule the package has.

    ``lepos(d, q)`` follows ``numpy.polynomial.legendre.leggauss(q)``: ``q`` is the number of
    nodes of the one-dimensional Gauss rule, and ``lepos(d, q)`` returns a rule with the same
    exactness in ``d`` dimensions, degree ``p = 2q - 1``, that replaces the ``q**d``-node
    product grid; for ``d = 1`` it is the ``q``-node Gauss-Legendre rule itself. Alternatively
    pass the degree as the keyword ``p``.

    Parameters
    ----------
    d : int
        Dimension, ``d >= 1``.
    q : int, optional
        Number of nodes of the one-dimensional Gauss rule whose exactness is wanted. Give
        ``q`` or ``p``, not both.
    p : int, keyword-only, optional
        Degree of exactness, ``p >= 0``; an even ``p`` gets the rule for ``p + 1``.
    normalize : bool, default True
        With ``True``, the weight is the uniform density on ``[0,1]^d``: the weights sum to 1
        and the rule computes ``E f(U)`` for ``U`` uniform on the unit cube. With ``False``,
        the rule is for the plain integral over ``[-1,1]^d``, the convention of ``leggauss``,
        and the weights sum to ``2**d``.
    pragmatic : bool, default False
        As for :func:`ghpos`: ``False`` raises :class:`quadriceps.NoRuleError` when no stored
        rule covers the request; ``True`` returns the cheapest tensor product of
        lower-dimensional rules (stored rules and one-dimensional Gauss-Legendre rules).

    Returns
    -------
    X : numpy.ndarray, shape (n, d)
        The nodes, one per row. All nodes of every stored Le rule lie inside the cube.
    w : numpy.ndarray, shape (n,)
        The weights, all strictly positive.

    Examples
    --------
    >>> X, w = lepos(2, 5)                       # degree 9: 17 nodes on [0,1]² instead of 25
    >>> float(np.round(np.sum(w * X[:, 0]**3 * X[:, 1]**2), 12))   # 1/4 · 1/3
    0.083333333333
    >>> X, w = lepos(2, 5, normalize=False)      # the same rule on [-1,1]², weights sum to 4
    >>> X, w = lepos(6, 6, pragmatic=True)       # d = 6 from a product of stored rules
    """
    d = operator.index(d)
    parts, n = plan("le", d, _degree(q, p), bool(pragmatic))
    X, w = materialize("le", parts, n)
    if not normalize:                   # [0,1]^d → [-1,1]^d
        X *= 2.0
        X -= 1.0
        w *= 2.0 ** d
    return X, w


def nnodes(family, d, q=None, *, p=None, pragmatic=False):
    """Number of nodes of the rule :func:`ghpos` / :func:`lepos` returns, without building it.

    ``family`` is ``"gh"`` or ``"le"``; the other arguments are those of :func:`ghpos`. Raises
    the same :class:`quadriceps.NoRuleError` when there is no rule.

    >>> nnodes("gh", 5, 7)
    1135
    """
    _, n = plan(family, operator.index(d), _degree(q, p), bool(pragmatic))
    return n


def ruleinfo(family, d, q=None, *, p=None, pragmatic=False):
    """Describe the rule that :func:`ghpos` / :func:`lepos` returns for the same arguments.

    Returns a list with one entry per tensor factor, in the order of the columns of ``X``. An
    entry is either the :class:`quadriceps.RuleInfo` of a stored rule (dimension, degree, node
    count, measured error, origin) or, for a one-dimensional Gauss factor, the dict
    ``{"family": ..., "d": 1, "p": ..., "n": ..., "origin": "Gauss"}``. A request answered by
    a single stored rule gives a one-element list.

    Use it to find out whom to cite: ``origin`` names the published source of every rule that
    is not the package author's own.
    """
    parts, _ = plan(family, operator.index(d), _degree(q, p), bool(pragmatic))
    return [a.info if a.info is not None
            else {"family": family, "d": 1, "p": 2 * a.n - 1, "n": a.n, "origin": "Gauss"} for a in parts]
