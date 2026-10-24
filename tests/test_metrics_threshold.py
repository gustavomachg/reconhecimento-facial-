import numpy as np
import pytest

from facerec.metrics import binary_metrics, compute_eer, far_frr, open_set_metrics
from facerec.threshold import select_threshold, threshold_sweep


def scores_labels(seed=0, n=500, gap=0.4):
    rng = np.random.default_rng(seed)
    gen = rng.normal(0.7, 0.1, n)
    imp = rng.normal(0.7 - gap, 0.1, n)
    return np.concatenate([gen, imp]), np.array([1] * n + [0] * n)


def test_perfectly_separable_has_zero_eer():
    s = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.15])
    y = np.array([1, 1, 1, 0, 0, 0])
    eer, thr = compute_eer(s, y)
    assert eer == 0.0
    assert binary_metrics(s, y, thr)["accuracy"] == 1.0


def test_far_frr_definitions():
    s = np.array([0.9, 0.4, 0.6, 0.2])
    y = np.array([1, 1, 0, 0])
    far, frr = far_frr(s, y, 0.5)
    assert far == 0.5  # 1 de 2 impostores aceito (0.6)
    assert frr == 0.5  # 1 de 2 genuínos rejeitado (0.4)


def test_eer_threshold_balances_far_and_frr():
    s, y = scores_labels()
    eer, thr = compute_eer(s, y)
    far, frr = far_frr(s, y, thr)
    assert abs(far - frr) < 0.02
    assert 0 < eer < 0.1


def test_confusion_matrix_sums_to_total():
    s, y = scores_labels()
    cm = binary_metrics(s, y, 0.5)["confusion_matrix"]
    assert sum(cm.values()) == len(y)


def test_far_target_strategy_respects_target():
    s, y = scores_labels(gap=0.2)
    thr = select_threshold(s, y, "far_target", far_target=0.01)
    far, _ = far_frr(s, y, thr)
    assert far <= 0.01
    # é o MENOR threshold que cumpre o alvo: baixar um pouco viola o alvo
    lower = s[s < thr].max()
    assert far_frr(s, y, lower)[0] > 0.01


@pytest.mark.parametrize("strategy", ["eer", "max_f1", "max_accuracy"])
def test_strategies_return_threshold_inside_range(strategy):
    s, y = scores_labels(seed=1)
    thr = select_threshold(s, y, strategy)
    assert s.min() <= thr <= s.max()
    assert binary_metrics(s, y, thr)["accuracy"] > 0.9


def test_unknown_strategy_raises():
    s, y = scores_labels()
    with pytest.raises(ValueError):
        select_threshold(s, y, "nope")


def test_sweep_monotonic_far():
    s, y = scores_labels()
    fars = [r["far"] for r in threshold_sweep(s, y, 50)]
    assert all(a >= b for a, b in zip(fars, fars[1:], strict=False))


def test_open_set_metrics():
    true = ["a", "a", "b", "__unknown__", "__unknown__"]
    pred = ["a", "__unknown__", "a", "__unknown__", "b"]
    m = open_set_metrics(true, pred)
    assert m["accuracy"] == pytest.approx(2 / 5)
    assert m["known_accuracy"] == pytest.approx(1 / 3)
    assert m["false_reject_known"] == pytest.approx(1 / 3)
    assert m["misidentification"] == pytest.approx(1 / 3)
    assert m["false_accept_unknown"] == pytest.approx(1 / 2)
