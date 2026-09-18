"""The catalog of stored rules (data/index.tsv) and the loader for the rule files."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from importlib import resources

import numpy as np

FAMILIES = ("gh", "le")


@dataclass(frozen=True)
class RuleInfo:
    """Catalog entry of one stored rule.

    Attributes
    ----------
    family : str
        ``"gh"`` or ``"le"``.
    d, p, n : int
        Dimension, degree of exactness, number of nodes.
    moller : int
        Möller's lower bound on ``n`` for this ``(d, p)``, or ``-1`` where it is not tabulated.
        A rule with ``n == moller`` is proven minimal.
    relerr : float
        Largest relative monomial error of the stored double-precision rule over all monomials
        of total degree ``<= p``, measured when the data were built (see
        :func:`quadriceps.exactness_error`).
    minweight : float
        Smallest weight (always ``> 0``).
    interior : bool
        ``True`` when every node lies inside the integration domain (always ``True`` for GH).
    origin : str
        Who the rule belongs to: ``"own"``, or ``"derived: ..."``, ``"same-rule: ..."``,
        ``"transcribed: ..."`` followed by the published source.
    file : str
        File name under ``data/<family>/``.
    """

    family: str
    d: int
    p: int
    n: int
    moller: int
    relerr: float
    minweight: float
    interior: bool
    origin: str
    file: str

    @property
    def q(self) -> int:
        """The ``q`` that requests this rule: ``(p + 1) // 2``."""
        return (self.p + 1) // 2


def check_family(family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; use 'gh' or 'le'")
    return family


def _datadir():
    return resources.files("quadriceps") / "data"


def _read_index() -> dict:
    index = {}
    with (_datadir() / "index.tsv").open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("family"):
                continue
            r = line.split("\t")
            info = RuleInfo(r[0], int(r[1]), int(r[2]), int(r[3]), int(r[4]), float(r[5]), float(r[6]),
                            r[7] == "yes", r[8], r[9])
            index[(info.family, info.d, info.p)] = info
    return index


INDEX = _read_index()
_CACHE: dict = {}
_LOCK = threading.Lock()


def stored(info: RuleInfo):
    """The stored rule behind a catalog entry, in the normalized frame.

    The cached arrays are shared; callers hand out copies.
    """
    key = (info.family, info.d, info.p)
    with _LOCK:
        if key not in _CACHE:
            with (_datadir() / info.family / info.file).open(encoding="utf-8") as f:
                a = np.loadtxt(f, delimiter=",", comments="#", ndmin=2)
            if a.shape != (info.n, info.d + 1):
                raise RuntimeError(f"{info.file}: shape {a.shape} does not match the catalog")
            _CACHE[key] = (np.ascontiguousarray(a[:, : info.d]), np.ascontiguousarray(a[:, info.d]))
        return _CACHE[key]


def available(family: str) -> list:
    """All stored rules of ``family`` (``"gh"`` or ``"le"``).

    Returns a list of :class:`RuleInfo` sorted by dimension and degree. Tensor products, which
    ``pragmatic=True`` builds on demand, are not listed.

    >>> [(r.q, r.p, r.n) for r in available("gh") if r.d == 3][:4]
    [(1, 1, 1), (2, 3, 6), (3, 5, 13), (4, 7, 27)]
    """
    check_family(family)
    return sorted((r for r in INDEX.values() if r.family == family), key=lambda r: (r.d, r.p))
