"""Escolha do threshold de decisão — SEMPRE no conjunto de VALIDAÇÃO (ADR-003/ADR-004).

Estratégias:
- "eer":        threshold onde FAR == FRR (equilíbrio entre os dois erros)
- "far_target": menor threshold cujo FAR <= alvo (ex.: 1%) — prioriza segurança
- "max_f1" / "max_accuracy": maximiza a métrica na validação
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from facerec.metrics import binary_metrics, compute_eer, far_frr


def threshold_sweep(scores: np.ndarray, labels: np.ndarray, n: int = 201) -> list[dict]:
    """Varre thresholds entre o menor e o maior score e devolve as métricas em cada um."""
    scores = np.asarray(scores, dtype=float)
    grid = np.linspace(scores.min(), scores.max(), n)
    return [binary_metrics(scores, labels, float(t)) for t in grid]


def select_threshold(
    scores: np.ndarray, labels: np.ndarray, strategy: str = "eer", far_target: float = 0.01
) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if strategy == "eer":
        return compute_eer(scores, labels)[1]
    candidates = np.unique(scores)
    if strategy == "far_target":
        # FAR só diminui quando o threshold aumenta: o menor threshold que cumpre o alvo
        # é o que tem o menor FRR possível sob essa restrição.
        for t in candidates:
            far, _ = far_frr(scores, labels, float(t))
            if far <= far_target:
                return float(t)
        return float(candidates[-1] + 1e-6)
    if strategy in ("max_f1", "max_accuracy"):
        key = "f1" if strategy == "max_f1" else "accuracy"
        best = max(candidates, key=lambda t: binary_metrics(scores, labels, float(t))[key])
        return float(best)
    raise ValueError(f"Estratégia de threshold desconhecida: {strategy}")


def main() -> None:
    """E07 — varredura de threshold na validação do dataset privado."""
    from facerec.experiments_private import run_threshold_experiment

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = run_threshold_experiment(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
