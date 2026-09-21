<p align="center"><img src="logo.svg" alt="Quadriceps logo" width="200"></p>

# quadriceps (Python)

[![CI](https://github.com/NittanyLion/quadriceps-py/actions/workflows/ci.yml/badge.svg)](https://github.com/NittanyLion/quadriceps-py/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![authored by: JP](authored_by.svg)

> **Paper:** Joris Pinkse, *Positive weight Hermite and Legendre quadrature rules* — arXiv: **[ARXIV-LINK-TBA](https://arxiv.org/abs/ARXIV-LINK-TBA)** (link to be filled in on publication)
>
> **Data deposit:** Zenodo — DOI: **[10.5281/zenodo.22881864](https://doi.org/10.5281/zenodo.22881864)** (reserved; the link resolves once the record is published)

Positive-weight cubature rules in several dimensions, for two weights:

| function | weight (default) | one-dimensional cousin |
|---|---|---|
| `ghpos(d, q)` | standard normal density `N(0, I_d)` on `ℝᵈ` | `numpy.polynomial.hermite.hermgauss(q)` |
| `lepos(d, q)` | uniform density on `[0,1]ᵈ` | `numpy.polynomial.legendre.leggauss(q)` |

A rule of degree `p` is a set of `n` nodes `x_i ∈ ℝᵈ` and weights `w_i > 0` with
`Σ w_i f(x_i) = ∫ f(x) ω(x) dx` for every polynomial `f` of total degree `≤ p`. The product of
`q`-node one-dimensional Gauss rules does this for `p = 2q - 1` with `qᵈ` nodes. The rules
stored here, the smallest positive-weight rules known to the author, do it with far fewer; at
`d = 5` the saving is more than a factor of ten. They cover `2 ≤ d ≤ 5`. [`RULES.md`](RULES.md)
lists every rule with its node count, Möller's lower bound, measured accuracy and origin.

This is the Python twin of [Quadriceps.jl](https://github.com/NittanyLion/Quadriceps.jl) (Julia)
and [quadriceps-r](https://github.com/NittanyLion/quadriceps-r) (R). The three packages share
their data, their conventions and their function names; the data are refreshed from the
Julia package whenever a smaller rule is found.

## Installation

```
pip install quadriceps
```

Python 3.9 or later; the only dependency is numpy.

## Use

```python
import numpy as np
from quadriceps import ghpos, lepos

X, w = ghpos(3, 4)              # d = 3, q = 4 (degree 7): X is 27×3 (one node per row), w has length 27
f = lambda x: x[0]**2 * x[1]**4
sum(wi * f(xi) for xi, wi in zip(X, w))             # E[Z₁² Z₂⁴] = 3.0

X, w = lepos(2, 5)              # q = 5 (degree 9): 17 nodes on the unit square instead of 25
np.sum(w * X[:, 0]**3 * X[:, 1]**2)                 # 1/4 · 1/3

X, w = ghpos(3, p=7)            # the first rule again, requested by its degree
```

Both functions follow `hermgauss(q)` and `leggauss(q)`, with the dimension `d ≥ 1` in front.
There, `q` is the number of nodes of the one-dimensional Gauss rule, which is exact to degree
`2q - 1`. Here, `ghpos(d, q)` returns a `d`-dimensional rule of that same degree `p = 2q - 1`: a
replacement for the `qᵈ`-node product grid, and for `d = 1` the `q`-node Gauss rule itself.
`X` is an `(n, d)` array, `w` an `(n,)` array of positive weights.

To ask for a degree instead, pass the keyword `p`: `ghpos(d, p=7)`, `lepos(d, p=12)`. Any
`p ≥ 0` is accepted. Rules are stored at odd degrees and a request is served by the smallest
stored rule of degree `≥ p`, so an even `p` gets the rule for `p + 1`. Give `q` or `p`, not both.

### `normalize`

numpy's `hermgauss(q)` integrates against `exp(-x²)`. The rules here are made for the standard
normal density, which is what an expectation needs, so `normalize=True` is the default:

| | `normalize=True` (default) | `normalize=False` (numpy's convention) |
|---|---|---|
| `ghpos` | weight `(2π)⁻ᵈᐟ² exp(-‖x‖²/2)`; weights sum to 1 | weight `exp(-‖x‖²)`; weights sum to `πᵈᐟ²` |
| `lepos` | uniform density on `[0,1]ᵈ`; weights sum to 1 | `∫ f(x) dx` over `[-1,1]ᵈ`; weights sum to `2ᵈ` |

So `ghpos(1, q, normalize=False)` is `hermgauss(q)` and `lepos(1, q, normalize=False)` is
`leggauss(q)`, up to the shape of `X`. For `Y ~ N(μ, LLᵀ)` use the nodes `μ + X @ L.T` with the
same weights; for a box, `a + (b - a) * X`.

### `pragmatic`

Rules are stored for `2 ≤ d ≤ 5`, up to a degree that depends on the family and on `d`
(see [`RULES.md`](RULES.md)). For any other request:

* `pragmatic=False` (the default) raises `quadriceps.NoRuleError` (a `ValueError`), whose
  message says how far the stored rules go;
* `pragmatic=True` returns the cheapest tensor product of lower-dimensional rules: the split of
  `d` into stored rules and one-dimensional Gauss rules that needs the fewest nodes. The result
  is a valid positive-weight rule of the requested degree. It is not small, but it is much
  smaller than the plain product grid whenever a stored rule can be a factor.

```python
ghpos(7, 5)                             # NoRuleError: no stored rule in seven dimensions
X, w = ghpos(7, 5, pragmatic=True)      # (d = 2) × (d = 5): a few thousand nodes; the grid has 78125

from quadriceps import nnodes, ruleinfo
nnodes("gh", 10, 3, pragmatic=True)     # the node count, without building the rule
ruleinfo("gh", 7, 5, pragmatic=True)    # the factors, with their origins
```

With `pragmatic=True` a request that a stored rule covers returns that stored rule, as without
it. (If a product were ever strictly cheaper than the stored rule, the product would be
returned; the data contain no such case, and the tests check that.)

### Other functions

* `available(family)` lists the stored rules as `RuleInfo` records (`family`, `d`, `p`, `q`,
  `n`, `moller`, `relerr`, `minweight`, `interior`, `origin`, `source_id`); `family` is `"gh"`
  or `"le"`.
* `nnodes(family, d, q, *, p, pragmatic)` gives a node count without building the rule.
* `ruleinfo(family, d, q, *, p, pragmatic)` describes the rule and its origin.
* `exactness_error(X, w, p, family)` measures how exact a rule is.

Every function has a full docstring (`help(ghpos)`).

All rules are stored in one binary file, `src/quadriceps/data/rules.bin`, with `index.tsv` next
to it as the catalog; [`FORMAT.md`](FORMAT.md) specifies the format, which the Julia and R
packages share byte for byte.

## Accuracy

Rules are stored in double precision. Every stored rule was checked when the data were built:
all weights positive, and the largest relative monomial error over all monomials of degree
`≤ p` below `1e-11`. Most rules sit at `1e-16`–`1e-15`; the largest GH rules at `d = 2, 3` are
the least accurate, at `1e-12`–`1e-11`. The measured value of each rule is in the catalog
(`relerr`) and in [`RULES.md`](RULES.md), and the test suite repeats the check for every rule.
All nodes of the Le rules lie strictly inside the cube.

## Whose rules these are

<!-- BEGIN GENERATED credits -->
120 of the 146 rules were computed from scratch by the author. A further 11 (Legendre) rules were
obtained by node elimination started from Diallo and Worku's published rules. Finally, 15 are
rules from the literature (copied in, or found again by the author's search and recognized).
<!-- END GENERATED credits -->
`ruleinfo` and the `origin` column
of [`RULES.md`](RULES.md) say which is which; cite the source named there when you use such a
rule. [`NOTICE.md`](NOTICE.md) has the details and the license notice that travels with the
derived files.

## Tests

```
pip install -e .[test]
python -m pytest
```

## License

MIT; see [`LICENSE`](LICENSE). The rules that descend from, or coincide with, published rules
carry their sources' notices in [`NOTICE.md`](NOTICE.md).
