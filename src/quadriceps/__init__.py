"""Positive-weight cubature rules in ``d >= 1`` dimensions for two weight functions.

* **GH** (:func:`ghpos`): the Gaussian weight, by default the standard normal density
  ``N(0, I_d)``;
* **Le** (:func:`lepos`): the uniform weight, by default the uniform density on ``[0,1]^d``.

A rule of degree ``p`` integrates every polynomial of total degree ``<= p`` exactly. All
weights are strictly positive. The rules shipped with the package are the smallest ones known
to its author; ``RULES.md`` lists them and ``NOTICE.md`` says where each one comes from.

The two main functions mirror ``numpy.polynomial.hermite.hermgauss(q)`` and
``numpy.polynomial.legendre.leggauss(q)`` (and ``gausshermite(q)``, ``gausslegendre(q)`` of
Julia's FastGaussQuadrature.jl): ``q`` is the number of nodes of the one-dimensional Gauss
rule, and the rule returned has its degree of exactness, ``p = 2q - 1``.

>>> from quadriceps import ghpos, lepos
>>> X, w = ghpos(3, 4)        # as exact as the 4x4x4 Gauss-Hermite grid (degree 7), 27 nodes
>>> X.shape, w.shape
((27, 3), (27,))
>>> X, w = ghpos(3, p=7)      # the same rule, requested by degree
>>> X, w = lepos(2, 5)        # degree 9 on the unit square: 17 nodes instead of 25

This package is the Python twin of Quadriceps.jl and the R package quadriceps; the three share
their data and their conventions.
"""

from ._api import ghpos, lepos, nnodes, ruleinfo
from ._catalog import RuleInfo, available
from ._plan import NoRuleError
from ._verify import exactness_error

__version__ = "0.1.0"
__all__ = ["NoRuleError", "RuleInfo", "available", "exactness_error", "ghpos", "lepos", "nnodes", "ruleinfo"]
