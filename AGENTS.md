# AGENTS.md

Guidance for coding agents (and people) working in this repository.

## Paper and deposit

The rules are described in Joris Pinkse, *Positive weight Hermite and Legendre quadrature rules* (2026),
and deposited on Zenodo. All three records are published:

* paper: **[10.5281/zenodo.22904159](https://doi.org/10.5281/zenodo.22904159)** (concept DOI: it always resolves to the latest version)
* rules, to 80 digits: **[10.5281/zenodo.22881864](https://doi.org/10.5281/zenodo.22881864)** (one record for both weight families)
* software snapshot of all five packages, v0.1.0: **[10.5281/zenodo.22883240](https://doi.org/10.5281/zenodo.22883240)**

The paper is not on arXiv; do not add or look for an arXiv link. Keep the block at the top of
`README.md` in step with this one, in all three packages (Quadriceps.jl, quadriceps-py, quadriceps-r).

## What this is

The Python twin of Quadriceps.jl: positive-weight cubature rules for the Gaussian weight
(`ghpos`) and the uniform weight on the cube (`lepos`). The Julia package is the master copy.
Its `build/update.sh` refreshes `src/quadriceps/data/`, `RULES.md`, `FORMAT.md` and `NOTICE.md`
here from the Julia package, runs the tests, commits and pushes. The `GENERATED credits` block of `README.md` is filled in by the same script. **Never edit those by hand**
(and never add per-rule CSV or text files: the rules live in the one binary file), and
make behavior changes in all three packages (Julia, Python, R) together.

## Layout

| path | what it holds |
|---|---|
| `src/quadriceps/_catalog.py` | `RuleInfo`, the catalog (`INDEX`, read from `data/index.tsv` at import), reader of `data/rules.bin` (format `QUADRICEPS1`, see `FORMAT.md`), cache, `available` |
| `src/quadriceps/_plan.py` | which rule answers a request: `best_stored`, `atom`, `cheapest` (dynamic program over splits of `d`), `NoRuleError`, `gauss1d`, `tensor`, `materialize` |
| `src/quadriceps/_api.py` | `ghpos`, `lepos`, `nnodes`, `ruleinfo`, and the `normalize=False` transforms |
| `src/quadriceps/_verify.py` | `exactness_error` |
| `src/quadriceps/data/` | `rules.bin` (all rules, one binary file) and `index.tsv` (catalog) — generated |
| `tests/` | every stored rule, both conventions, the fallback, the error paths; docstring examples run as doctests |

## Commands

```
python3 -m pytest          # about 10 s; conftest.py pins BLAS to one thread
```

## Conventions that must hold

* **`q` and `p`.** The positional argument `q` is the number of nodes of the one-dimensional
  Gauss rule, as in `hermgauss(q)`; the rule returned has degree `p = 2q - 1`, and for `d = 1`
  it is the `q`-node Gauss rule. The degree is passed as the keyword `p`. Exactly one of the
  two. Internally everything works in `p`.
* **Normalized frame inside.** Files, cache, tensor products and `exactness_error` use
  `N(0, I_d)` for GH and the uniform density on `[0,1]ᵈ` for Le, weights summing to 1.
  `normalize=False` is applied once, at the end, in `_api.py`. One-dimensional GH factors come
  from `hermegauss` divided by `√(2π)`.
* **`normalize=True` is the default**, unlike numpy's `hermgauss`. This is deliberate.
* **`pragmatic`.** `False`: `NoRuleError` when no stored rule covers the request. `True`: the
  cheapest tensor product of stored rules and Gauss rules; a stored rule wins ties.
* **Positive weights only**, relative monomial error below `1e-11`; the tests enforce both.
* **Returned arrays are copies.** `X` has shape `(n, d)`, also for `d = 1`.
* numpy is the only dependency. American spelling.

## Repository

`authored_by.svg` is the author's shield (the same file as in MemoryLayouts.jl); the README shows
it after the other badges. Do not replace it with a generated shields.io badge.

Quality gate, the analog of Aqua.jl: `ruff check .` must pass (configuration in
`pyproject.toml`; CI runs it), and `test_public_api` checks that everything in `__all__` exists
and is documented. The logo is `logo.svg`, a copy of the one in Quadriceps.jl.

GitHub Actions (`.github/workflows/ci.yml`) runs the tests on every push to `main` and on pull
requests; the unattended data updates trigger it too. Check `gh run list` after pushing.

Private, `github.com/NittanyLion/quadriceps-py`, branch `main`. MIT license (`LICENSE`, and `license`
in `pyproject.toml`; author's choice 2026-09-19); `NOTICE.md` carries the notices of the rules
that descend from published ones.
