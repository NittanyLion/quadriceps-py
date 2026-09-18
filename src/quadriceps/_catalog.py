"""The catalog of stored rules (data/index.tsv) and the reader of data/rules.bin.

All rules are in the one binary file rules.bin, format QUADRICEPS1 (see FORMAT.md): an ASCII
header line, an index of little-endian int64 records, then flat little-endian float64 blocks.
"""

from __future__ import annotations

import struct
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
    source_id : int
        The origin as the small integer stored in ``rules.bin`` (0: own; 3: derived from Diallo
        and Worku; 10 and up: a published rule; see FORMAT.md).
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
    source_id: int

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
                            r[7] == "yes", r[8], int(r[9]))
            index[(info.family, info.d, info.p)] = info
    return index


MAGIC = b"QUADRICEPS1"
NIDX = 8


def _read_bin_index() -> dict:
    """The index of rules.bin: (family, d, p) -> (n, offset, nbytes, source_id)."""
    with (_datadir() / "rules.bin").open("rb") as f:
        header = f.readline()
        tok = header.split()
        if not tok or tok[0] != MAGIC:
            raise RuntimeError("rules.bin: not a QUADRICEPS1 file")
        kv = dict(t.split(b"=", 1) for t in tok[1:] if b"=" in t)
        if (kv[b"fmt"], kv[b"endian"], kv[b"float"], kv[b"index_fields"]) != (b"1", b"little", b"binary64", b"8"):
            raise RuntimeError(f"rules.bin: unsupported QUADRICEPS1 variant: {header!r}")
        cells = int(kv[b"cells"])
        raw = f.read(cells * NIDX * 8)
    out = {}
    for i in range(cells):
        fam, d, p, q, n, off, nb, sid = struct.unpack_from("<8q", raw, NIDX * 8 * i)
        if nb != n * (d + 1) * 8:
            raise RuntimeError(f"rules.bin: cell d={d} p={p} has nbytes={nb}")
        out[(FAMILIES[fam], d, p)] = (n, off, nb, sid)
    return out


INDEX = _read_index()
BIN = _read_bin_index()
for _k, _r in INDEX.items():
    if _k not in BIN or BIN[_k][0] != _r.n:
        raise RuntimeError(f"quadriceps: index.tsv and rules.bin disagree at {_k}")
_CACHE: dict = {}
_LOCK = threading.Lock()


def stored(info: RuleInfo):
    """The stored rule behind a catalog entry, in the normalized frame.

    The cached arrays are shared; callers hand out copies.
    """
    key = (info.family, info.d, info.p)
    with _LOCK:
        if key not in _CACHE:
            n, off, nb, _ = BIN[key]
            with (_datadir() / "rules.bin").open("rb") as f:
                f.seek(off)
                a = np.frombuffer(f.read(nb), dtype="<f8").reshape(n, info.d + 1)      # row-major on disk
            _CACHE[key] = (np.array(a[:, : info.d], dtype=float, order="C"), np.array(a[:, info.d], dtype=float))
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
