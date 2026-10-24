"""Métricas de verificação facial (numpy/sklearn puro, testável sem GPU).

Convenção: `scores` são similaridades (maior = mais parecido) e `labels` é 1 para
"mesma pessoa" (par genuíno) e 0 para "pessoas diferentes" (par impostor).
Decisão: score >= threshold  =>  "mesma pessoa".

- FAR (False Accept Rate)  = impostores aceitos / total de impostores
- FRR (False Reject Rate)  = genuínos rejeitados / total de genuínos
- EER (Equal Error Rate)   = ponto onde FAR == FRR
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve


def far_frr(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    accept = scores >= threshold
    n_imp = max(int((~labels).sum()), 1)
    n_gen = max(int(labels.sum()), 1)
    far = float((accept & ~labels).sum() / n_imp)
    frr = float((~accept & labels).sum() / n_gen)
    return far, frr


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Retorna (eer, threshold_eer) usando a curva ROC."""
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thr[idx])
    if not np.isfinite(threshold):  # sklearn usa +inf no primeiro ponto
        threshold = float(np.max(scores))
    return eer, threshold


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Accuracy, precision, recall, F1, FAR, FRR e matriz de confusão num threshold."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far, frr = far_frr(scores, labels, threshold)
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "far": far,
        "frr": frr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def verification_report(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Relatório completo num threshold JÁ ESCOLHIDO (nunca escolher o threshold no teste)."""
    eer, eer_thr = compute_eer(scores, labels)
    report = binary_metrics(scores, labels, threshold)
    report.update({"roc_auc": roc_auc(scores, labels), "eer": eer, "eer_threshold": eer_thr})
    return report


UNKNOWN = "__unknown__"


def open_set_metrics(true_labels: list, pred_labels: list, unknown_label: str = UNKNOWN) -> dict:
    """Métricas de identificação em conjunto aberto (conhecidos + desconhecidos).

    - accuracy: acerto geral (identidade correta ou "desconhecido" correto)
    - known_accuracy: entre consultas de pessoas cadastradas, % identificadas corretamente
    - false_accept_unknown: % de desconhecidos aceitos como alguém cadastrado
    - false_reject_known: % de conhecidos rejeitados como desconhecido
    - misidentification: % de conhecidos confundidos com OUTRA pessoa cadastrada
    - macro_f1: F1 macro sobre todas as classes (incluindo "desconhecido")
    """
    t = np.asarray([str(x) for x in true_labels], dtype=object)
    p = np.asarray([str(x) for x in pred_labels], dtype=object)
    known = t != unknown_label
    unknown = ~known

    def _mean(mask: np.ndarray) -> float | None:
        return float(mask.mean()) if mask.size else None

    return {
        "n_queries": int(len(t)),
        "n_known_queries": int(known.sum()),
        "n_unknown_queries": int(unknown.sum()),
        "accuracy": _mean(t == p) or 0.0,
        "known_accuracy": _mean(t[known] == p[known]),
        "false_accept_unknown": _mean(p[unknown] != unknown_label),
        "false_reject_known": _mean(p[known] == unknown_label),
        "misidentification": _mean((p[known] != t[known]) & (p[known] != unknown_label)),
        "macro_f1": float(f1_score(t, p, average="macro")) if len(t) else 0.0,
    }
