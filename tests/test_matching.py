import numpy as np
import pytest

from facerec.matching import (
    MATCHERS,
    CosineMatcher,
    EuclideanMatcher,
    build_matcher,
    cosine_similarity,
    l2_normalize,
)
from facerec.metrics import UNKNOWN
from tests.conftest import synthetic_identities


def split_gallery(x, y, per_id=6, n_gallery=3):
    idx_g = [i for i in range(len(y)) if i % per_id < n_gallery]
    idx_q = [i for i in range(len(y)) if i % per_id >= n_gallery]
    return x[idx_g], [y[i] for i in idx_g], x[idx_q], [y[i] for i in idx_q]


def test_l2_normalize_unit_norm():
    v = l2_normalize(np.random.default_rng(0).normal(size=(10, 8)))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)


def test_cosine_similarity_self_is_one():
    v = np.random.default_rng(0).normal(size=(4, 8))
    assert np.allclose(np.diag(cosine_similarity(v, v)), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", sorted(MATCHERS))
def test_matchers_identify_known(name, identities):
    x, y, _ = identities
    gx, gy, qx, qy = split_gallery(x, y)
    m = build_matcher(name).fit(gx, gy)
    preds = [m.best(q)[0] for q in qx]
    assert np.mean([p == t for p, t in zip(preds, qy, strict=True)]) >= 0.95


@pytest.mark.parametrize("name", ["cosine", "prototype", "knn"])
def test_unknown_rejected_with_threshold(name, identities):
    x, y, _ = identities
    m = build_matcher(name).fit(x, y)
    stranger = l2_normalize(np.random.default_rng(99).normal(size=x.shape[1]))
    label, score = m.identify(stranger, threshold=0.6)
    assert label == UNKNOWN
    assert score < 0.6


def test_cosine_and_euclidean_give_same_ranking():
    """||a-b||² = 2 - 2cos(a,b) para vetores normalizados -> mesmo vizinho mais próximo."""
    x, y, _ = synthetic_identities(noise=0.6, seed=3)
    gx, gy, qx, _ = split_gallery(x, y)
    c, e = CosineMatcher().fit(gx, gy), EuclideanMatcher().fit(gx, gy)
    for q in qx:
        lc, sc = c.best(q)
        le, se = e.best(q)
        assert lc == le
        assert np.isclose(se**2, 2 - 2 * sc, atol=1e-5)


def test_empty_matcher_returns_unknown():
    m = CosineMatcher()
    assert m.identify(np.ones(4), 0.5)[0] == UNKNOWN


def test_svm_falls_back_with_single_class():
    x, y, _ = synthetic_identities(n_ids=1)
    m = build_matcher("svm").fit(x, y)
    assert m.best(x[0])[0] == "id0"


def test_unknown_matcher_name():
    with pytest.raises(ValueError):
        build_matcher("magic")
