import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss

from quadriceps import NoRuleError, available, exactness_error, ghpos, lepos, nnodes, ruleinfo

# Every stored rule passed a 1e-11 gate when the data were built; the catalog records the error.
GATE = 1e-11
F = {"gh": ghpos, "le": lepos}
RULES = [r for fam in F for r in available(fam)]


def test_catalog():
    assert available("gh") and available("le")
    assert all(r.p % 2 == 1 and r.d >= 2 and r.n >= 1 and r.q == (r.p + 1) // 2 for r in RULES)
    with pytest.raises(ValueError):
        available("la")


@pytest.mark.parametrize("r", RULES, ids=lambda r: f"{r.family}-d{r.d}-p{r.p}")
def test_stored_rule(r):
    f = F[r.family]
    X, w = f(r.d, p=r.p)
    Xq, wq = f(r.d, r.q)                                        # q form: p = 2q - 1
    assert np.array_equal(X, Xq) and np.array_equal(w, wq)
    assert X.shape == (r.n, r.d) and w.shape == (r.n,)
    assert np.all(w > 0)
    assert np.sum(w) == pytest.approx(1)
    assert exactness_error(X, w, r.p, r.family) < GATE
    if r.family == "le":
        assert np.all((X > 0) & (X < 1))
    assert nnodes(r.family, r.d, p=r.p) <= r.n                  # a higher degree may be cheaper, never dearer
    assert nnodes(r.family, r.d, p=r.p, pragmatic=True) == nnodes(r.family, r.d, p=r.p)   # no product beats a stored rule


def test_one_dimension():
    x, v = hermegauss(4)
    X, w = ghpos(1, 4)
    assert X.shape == (4, 1) and np.allclose(X[:, 0], x) and np.allclose(w, v / np.sqrt(2 * np.pi))
    x, v = hermgauss(4)
    X, w = ghpos(1, p=6, normalize=False)                       # even degree: next odd
    assert np.allclose(X[:, 0], x) and np.allclose(w, v)
    x, v = leggauss(5)
    X, w = lepos(1, 5, normalize=False)
    assert np.allclose(X[:, 0], x) and np.allclose(w, v)
    X, w = lepos(1, 5)
    assert np.sum(w) == pytest.approx(1) and np.all((X > 0) & (X < 1))
    assert exactness_error(X, w, 9, "le") < 1e-14


def test_normalize_false():
    # GH: weight exp(-|x|²); ∫ x₁² x₂⁴ exp(-|x|²) dx = (√π/2)(3√π/4)
    X, w = ghpos(2, 4, normalize=False)
    assert np.sum(w) == pytest.approx(np.pi)
    val = np.sum(w * X[:, 0] ** 2 * X[:, 1] ** 4)
    assert val == pytest.approx((np.sqrt(np.pi) / 2) * (3 * np.sqrt(np.pi) / 4))
    x, v = hermgauss(4)                                         # the same integral from numpy's product grid
    assert val == pytest.approx(np.sum(v * x**2) * np.sum(v * x**4))
    # Le: ∫ over [-1,1]³
    X, w = lepos(3, 3, normalize=False)
    assert np.sum(w) == pytest.approx(8)
    assert np.all((X > -1) & (X < 1))
    assert np.sum(w * X[:, 0] ** 2 * X[:, 2] ** 2) == pytest.approx(8 / 9)
    assert abs(np.sum(w * X[:, 0] * X[:, 1] ** 2)) < 1e-14


def test_q_and_p():
    for a, b in ((ghpos(3, p=6), ghpos(3, p=7)), (ghpos(3, p=7), ghpos(3, 4)), (lepos(2, p=0), lepos(2, 1))):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert nnodes("le", 2, p=10) == nnodes("le", 2, 6)
    with pytest.raises(TypeError):
        ghpos(2)                                                # neither q nor p
    with pytest.raises(TypeError):
        ghpos(2, 3, p=5)                                        # both


@pytest.mark.parametrize("fam", ["gh", "le"])
def test_no_rule(fam):
    f = F[fam]
    q = max(r.q for r in available(fam) if r.d == 3) + 1
    for call in (lambda: f(3, q), lambda: f(3, p=2 * q - 2), lambda: nnodes(fam, 3, q), lambda: f(6, 3)):
        with pytest.raises(NoRuleError, match="pragmatic=True"):
            call()
    assert nnodes(fam, 3, q, pragmatic=True) <= q**3
    for call in (lambda: f(0, 3), lambda: f(2, 0), lambda: f(2, p=-1)):
        with pytest.raises(ValueError):
            call()


def test_pragmatic_fallback():
    # a stored cell is returned unchanged
    assert np.array_equal(ghpos(4, 5, pragmatic=True)[0], ghpos(4, 5)[0])
    assert np.array_equal(lepos(2, 11, pragmatic=True)[1], lepos(2, 11)[1])
    # beyond the stored degrees: a valid rule, cheaper than the product grid
    q = max(r.q for r in available("gh") if r.d == 3) + 1
    X, w = ghpos(3, q, pragmatic=True)
    assert X.shape == (nnodes("gh", 3, q, pragmatic=True), 3)
    assert np.all(w > 0) and np.sum(w) == pytest.approx(1)
    assert exactness_error(X, w, 2 * q - 1, "gh") < GATE
    assert X.shape[0] < q**3
    # beyond the stored dimensions
    for fam, f in F.items():
        for d, q in ((6, 4), (7, 3), (8, 2)):
            X, w = f(d, q, pragmatic=True)
            n = nnodes(fam, d, q, pragmatic=True)
            assert X.shape == (n, d)
            assert np.all(w > 0) and np.sum(w) == pytest.approx(1)
            assert exactness_error(X, w, 2 * q - 1, fam) < GATE
            parts = ruleinfo(fam, d, q, pragmatic=True)
            dims = [r["d"] if isinstance(r, dict) else r.d for r in parts]
            counts = [r["n"] if isinstance(r, dict) else r.n for r in parts]
            assert sum(dims) == d and int(np.prod(counts)) == n
            # cheapest: no two-way split does better
            assert all(n <= nnodes(fam, j, q, pragmatic=True) * nnodes(fam, d - j, q, pragmatic=True)
                       for j in range(1, d // 2 + 1))
    # normalize=False composes with the fallback
    X, w = lepos(6, 2, normalize=False, pragmatic=True)
    assert np.sum(w) == pytest.approx(2**6) and np.sum(w * X[:, 5] ** 2) == pytest.approx(2**6 / 3)
    # absurd requests fail cleanly instead of exhausting memory
    assert nnodes("gh", 60, 16, pragmatic=True) > 2**63
    with pytest.raises(ValueError):
        ghpos(60, 16, pragmatic=True)


def test_results_are_copies():
    X, w = ghpos(2, 3)
    X[:] = 0
    w[:] = 0
    X2, w2 = ghpos(2, 3)
    assert np.sum(w2) == pytest.approx(1) and np.any(X2 != 0)


def test_data_file():
    from importlib import resources

    from quadriceps._catalog import BIN, INDEX

    data = resources.files("quadriceps") / "data"
    with (data / "rules.bin").open("rb") as f:
        header = f.readline()
        f.seek(0, 2)
        size = f.tell()
    assert header.startswith(b"QUADRICEPS1 fmt=1 endian=little cells=%d index_fields=8 float=binary64" % len(INDEX))
    assert BIN.keys() == INDEX.keys()
    assert all(BIN[k][0] == r.n and BIN[k][3] == r.source_id for k, r in INDEX.items())
    assert size == max(off + nb for _, off, nb, _ in BIN.values())             # no slack
    assert sorted(p.name for p in data.iterdir() if not p.name.startswith("__")) == ["index.tsv", "rules.bin"]
