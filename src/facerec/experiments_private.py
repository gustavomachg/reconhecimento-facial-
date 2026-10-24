"""Experimentos no dataset privado em conjunto aberto (open-set): E06 e E07.

    python -m facerec.compare_matchers --config configs/experiments/E06_matchers_open_set.yaml
    python -m facerec.threshold        --config configs/experiments/E07_threshold_sweep.yaml

Protocolo (ADR-004):
- galeria  = sessões de cadastro das pessoas "conhecidas";
- validação = outras sessões dos conhecidos + pessoas "desconhecidas" (nunca cadastradas);
- teste    = sessões restantes + OUTRAS pessoas desconhecidas. Usado UMA vez, só no E07.

Para cada consulta, o matcher devolve (melhor identidade, score). A consulta é um
"acerto aceitável" (rótulo 1) se for de uma pessoa conhecida E a melhor identidade estiver
correta; todo o resto (desconhecido, ou conhecido confundido) é rótulo 0. O threshold é
escolhido sobre esses scores na VALIDAÇÃO.

Privacidade: só métricas AGREGADAS são salvas. Nada por participante.
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np

from facerec.config import environment_info, load_yaml, pick_device, resolve_path, set_seed
from facerec.data.datasets import list_private
from facerec.data.splits import Record, split_private_by_session
from facerec.evaluate import embed_paths
from facerec.matching import Matcher, build_matcher
from facerec.metrics import UNKNOWN, binary_metrics, compute_eer, open_set_metrics, roc_auc
from facerec.models import build_embedder
from facerec.threshold import select_threshold


def _prepare(cfg: dict):
    set_seed(cfg.get("seed", 42))
    p = cfg["private"]
    root = resolve_path(p["aligned_root"])
    if not root.exists():
        raise FileNotFoundError(
            f"{root} não existe. Organize as fotos em data/private/raw/<P01>/<sessao>/ e rode "
            "python -m facerec.data.preprocess --dataset private --model <modelo>"
        )
    split = split_private_by_session(
        list_private(root),
        p["gallery_sessions"],
        p["val_sessions"],
        p["test_sessions"],
        p.get("n_unknown_val", 1),
        p.get("n_unknown_test", 1),
        seed=cfg.get("seed", 42),
    )
    model_cfg = dict(cfg["model"])
    if model_cfg.get("checkpoint"):
        model_cfg["checkpoint"] = str(resolve_path(model_cfg["checkpoint"]))
    embedder = build_embedder(model_cfg, pick_device(cfg.get("device", "auto")))
    paths = [r.path for part in ("gallery", "val", "test") for r in split[part]]
    emb = embed_paths(embedder, paths)  # fica só em memória, nunca vai para disco
    out_dir = resolve_path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return split, emb, out_dir


def _queries(records: list[Record], known: set[str], emb: dict):
    x = np.stack([emb[r.path] for r in records])
    y = [r.identity if r.identity in known else UNKNOWN for r in records]
    return x, y


def _top(matcher: Matcher, x: np.ndarray):
    tops = [matcher.best(q) for q in x]
    return [t[0] for t in tops], np.array([t[1] for t in tops])


def _binary_targets(true: list[str], top_labels: list[str]) -> np.ndarray:
    return np.array([int(t != UNKNOWN and t == p) for t, p in zip(true, top_labels, strict=True)])


def _decide(top_labels, scores, thr):
    return [lbl if s >= thr else UNKNOWN for lbl, s in zip(top_labels, scores, strict=True)]


def _fit(cfg_m: dict, split, emb) -> Matcher:
    kwargs = {k: v for k, v in cfg_m.items() if k != "name"}
    g = split["gallery"]
    return build_matcher(cfg_m["name"], **kwargs).fit(
        np.stack([emb[r.path] for r in g]), [r.identity for r in g]
    )


def _dataset_summary(split) -> dict:
    """Só contagens — nenhum identificador de participante."""
    return {
        "n_known_identities": len(split["known_identities"]),
        "n_unknown_val_identities": len(split["unknown_val"]),
        "n_unknown_test_identities": len(split["unknown_test"]),
        "n_gallery_images": len(split["gallery"]),
        "n_val_queries": len(split["val"]),
        "n_test_queries": len(split["test"]),
    }


# --------------------------------------------------------------------------- E06
def run_matcher_comparison(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    split, emb, out_dir = _prepare(cfg)
    shutil.copy(config_path, out_dir / "config_used.yaml")
    known = set(split["known_identities"])
    xv, yv = _queries(split["val"], known, emb)
    strategy = cfg.get("threshold_strategy", "eer")
    rows = {}
    for m_cfg in cfg["matchers"]:
        matcher = _fit(m_cfg, split, emb)
        t0 = time.perf_counter()
        top_lbl, scores = _top(matcher, xv)
        ms = 1000 * (time.perf_counter() - t0) / max(len(xv), 1)
        target = _binary_targets(yv, top_lbl)
        thr = select_threshold(scores, target, strategy) if 0 < target.sum() < len(target) else 0.0
        res = open_set_metrics(yv, _decide(top_lbl, scores, thr))
        res.update(threshold=thr, ms_per_query=ms)
        if 0 < target.sum() < len(target):
            res["roc_auc"] = roc_auc(scores, target)
            res["eer"] = compute_eer(scores, target)[0]
        label = m_cfg["name"] + (f"_k{m_cfg['k']}" if "k" in m_cfg else "")
        rows[label] = res
    best = max(rows, key=lambda k: (rows[k]["macro_f1"], rows[k]["accuracy"]))
    summary = {
        "experiment": cfg["experiment"],
        "model": cfg["model"],
        "dataset": _dataset_summary(split),
        "split_evaluated": "validation",
        "threshold_strategy": strategy,
        "matchers": rows,
        "best_matcher_on_validation": best,
        "environment": environment_info(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    _plot_bars(rows, out_dir)
    return summary


def _plot_bars(rows: dict, out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rows)
    keys = ["accuracy", "macro_f1", "false_accept_unknown", "false_reject_known"]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        ax.bar(np.arange(len(names)) + i * w, [rows[n][k] or 0 for n in names], w, label=k)
    ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2, names)
    ax.set(ylim=(0, 1), title="E06 — comparadores (validação, open-set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "matchers_comparison.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- E07
def run_threshold_experiment(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    split, emb, out_dir = _prepare(cfg)
    shutil.copy(config_path, out_dir / "config_used.yaml")
    known = set(split["known_identities"])
    matcher = _fit(cfg["matcher"], split, emb)
    tcfg = cfg.get("threshold", {"strategy": "eer"})

    # 1) varredura e escolha na VALIDAÇÃO
    xv, yv = _queries(split["val"], known, emb)
    top_v, s_v = _top(matcher, xv)
    target_v = _binary_targets(yv, top_v)
    grid = np.linspace(s_v.min(), s_v.max(), 101)
    sweep = []
    for t in grid:
        m = open_set_metrics(yv, _decide(top_v, s_v, t))
        far, frr = m["false_accept_unknown"], m["false_reject_known"]
        sweep.append(
            {
                "threshold": float(t),
                "accuracy": m["accuracy"],
                "far_unknown": far,
                "frr_known": frr,
                "macro_f1": m["macro_f1"],
            }
        )
    thr = select_threshold(s_v, target_v, tcfg["strategy"], tcfg.get("far_target", 0.01))
    val_metrics = open_set_metrics(yv, _decide(top_v, s_v, thr))

    # 2) teste — UMA única vez, com o threshold congelado
    xt, yt = _queries(split["test"], known, emb)
    top_t, s_t = _top(matcher, xt)
    target_t = _binary_targets(yt, top_t)
    test_metrics = open_set_metrics(yt, _decide(top_t, s_t, thr))
    test_binary = binary_metrics(s_t, target_t, thr)
    if 0 < target_t.sum() < len(target_t):
        test_binary["roc_auc"] = roc_auc(s_t, target_t)
        test_binary["eer"] = compute_eer(s_t, target_t)[0]

    summary = {
        "experiment": cfg["experiment"],
        "model": cfg["model"],
        "matcher": cfg["matcher"],
        "threshold_strategy": tcfg,
        "chosen_threshold": thr,
        "dataset": _dataset_summary(split),
        "validation": val_metrics,
        "test": test_metrics,
        "test_binary_accept_reject": test_binary,
        "sweep_validation": sweep,
        "environment": environment_info(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    _plot_sweep(sweep, thr, out_dir)
    _plot_score_hist(s_t, target_t, thr, out_dir)
    print(
        f"\n>>> Copie para configs/app.yaml:  recognition.threshold: {thr:.4f}  "
        f"(matcher: {cfg['matcher']['name']})"
    )
    return {k: v for k, v in summary.items() if k != "sweep_validation"}


def _plot_sweep(sweep, thr, out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, [r["far_unknown"] for r in sweep], label="FAR (desconhecido aceito)")
    ax.plot(t, [r["frr_known"] for r in sweep], label="FRR (conhecido rejeitado)")
    ax.plot(t, [r["accuracy"] for r in sweep], label="acurácia open-set")
    ax.axvline(thr, color="k", ls="--", lw=0.8, label=f"escolhido = {thr:.3f}")
    ax.set(xlabel="threshold", title="E07 — varredura de threshold (validação)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_sweep.png", dpi=120)
    plt.close(fig)


def _plot_score_hist(scores, target, thr, out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(scores[target == 1], bins=bins, alpha=0.6, density=True, label="conhecido (correto)")
    ax.hist(scores[target == 0], bins=bins, alpha=0.6, density=True, label="desconhecido/errado")
    ax.axvline(thr, color="k", ls="--", lw=0.8)
    ax.set(xlabel="score", title="E07 — scores no teste")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "test_score_hist.png", dpi=120)
    plt.close(fig)
