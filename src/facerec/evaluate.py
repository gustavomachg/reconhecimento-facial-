"""Avaliação de verificação facial (E01–E05).

    python -m facerec.evaluate --config configs/experiments/E01_facenet_baseline.yaml

1) LFW — protocolo oficial de 10 folds (pairs.txt, 6000 pares):
   para cada fold i, o threshold é escolhido nos OUTROS 9 folds e aplicado no fold i.
   Reportamos média ± desvio da acurácia, além de ROC-AUC e EER sobre todos os pares.
2) Dataset privado — VALIDAÇÃO (sessões de validação, pares entre sessões diferentes).
   É a métrica usada para ESCOLHER o modelo (o teste privado só é usado uma vez, no E07).
3) Custo: nº de parâmetros e tempo médio de inferência por face.

Tudo é salvo AGREGADO em experiments/results/<ID>/ (metrics.json, roc.png, hist_*.png).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from facerec.config import environment_info, load_yaml, pick_device, resolve_path, set_seed
from facerec.data.datasets import list_private, make_verification_pairs, parse_lfw_pairs
from facerec.data.splits import split_private_by_session
from facerec.metrics import binary_metrics, compute_eer, roc_auc, verification_report
from facerec.models import Embedder, build_embedder
from facerec.threshold import select_threshold


def load_aligned(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def embed_paths(
    embedder: Embedder, paths: list[str], batch_size: int = 64
) -> dict[str, np.ndarray]:
    uniq = sorted(set(paths))
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(uniq), batch_size):
        chunk = uniq[i : i + batch_size]
        emb = embedder.embed([load_aligned(p, embedder.input_size) for p in chunk])
        out.update(zip(chunk, emb, strict=True))
    return out


def pair_scores(embedder: Embedder, pairs: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    emb = embed_paths(embedder, [p for a, b, _ in pairs for p in (a, b)])
    scores = np.array([float(emb[a] @ emb[b]) for a, b, _ in pairs])
    labels = np.array([int(lbl) for _, _, lbl in pairs])
    return scores, labels


def lfw_10fold(scores_folds, labels_folds, strategy: str = "max_accuracy") -> dict:
    accs, f1s, precs, recs, thrs = [], [], [], [], []
    cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for i in range(len(scores_folds)):
        tr_s = np.concatenate([s for j, s in enumerate(scores_folds) if j != i])
        tr_y = np.concatenate([y for j, y in enumerate(labels_folds) if j != i])
        thr = select_threshold(tr_s, tr_y, strategy)
        m = binary_metrics(scores_folds[i], labels_folds[i], thr)
        accs.append(m["accuracy"])
        f1s.append(m["f1"])
        precs.append(m["precision"])
        recs.append(m["recall"])
        thrs.append(thr)
        for k in cm:
            cm[k] += m["confusion_matrix"][k]
    all_s, all_y = np.concatenate(scores_folds), np.concatenate(labels_folds)
    eer, eer_thr = compute_eer(all_s, all_y)
    thr_mean = float(np.mean(thrs))
    m_mean = binary_metrics(all_s, all_y, thr_mean)
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "threshold_mean": thr_mean,
        "far_at_threshold_mean": m_mean["far"],
        "frr_at_threshold_mean": m_mean["frr"],
        "roc_auc": roc_auc(all_s, all_y),
        "eer": eer,
        "eer_threshold": eer_thr,
        "confusion_matrix_sum": cm,
        "n_pairs": int(len(all_y)),
    }


def plot_hist_and_roc(scores, labels, out_dir: Path, tag: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(scores.min(), scores.max(), 50)
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="mesma pessoa", density=True)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="pessoas diferentes", density=True)
    ax.set(xlabel="similaridade cosseno", ylabel="densidade", title=f"Similaridades — {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"hist_{tag}.png", dpi=120)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc(scores, labels):.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(
        xlabel="FAR (falso positivo)", ylabel="1 - FRR (verdadeiro positivo)", title=f"ROC — {tag}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{tag}.png", dpi=120)
    plt.close(fig)
    # curva subamostrada para o gráfico comparativo entre experimentos (report.py)
    idx = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}


def plot_confusion(cm: dict, out_dir: Path, tag: str) -> None:
    """Matriz de confusão AGREGADA (mesma pessoa x diferentes) no threshold escolhido."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1], ["prev. diferentes", "prev. mesma"])
    ax.set_yticks([0, 1], ["real diferentes", "real mesma"])
    ax.set_title(f"Matriz de confusão — {tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{tag}.png", dpi=120)
    plt.close(fig)


def inference_cost(embedder: Embedder, n: int = 50) -> dict:
    face = np.random.default_rng(0).integers(0, 255, (embedder.input_size,) * 2 + (3,), np.uint8)
    for _ in range(5):  # aquecimento
        embedder.embed(face)
    t0 = time.perf_counter()
    for _ in range(n):
        embedder.embed(face)
    return {
        "inference_ms_per_face": 1000 * (time.perf_counter() - t0) / n,
        "n_parameters": embedder.num_parameters(),
    }


def private_validation_pairs(cfg: dict, seed: int) -> list[tuple] | None:
    pcfg = cfg.get("private")
    if not pcfg or not pcfg.get("enabled", False):
        return None
    root = resolve_path(pcfg["aligned_root"])
    if not root.exists():
        print(f"Dataset privado não encontrado em {root}; pulando.")
        return None
    split = split_private_by_session(
        list_private(root),
        pcfg["gallery_sessions"],
        pcfg["val_sessions"],
        pcfg["test_sessions"],
        pcfg.get("n_unknown_val", 1),
        pcfg.get("n_unknown_test", 1),
        seed=seed,
    )
    # verificação na validação: galeria + consultas de validação (nunca o teste)
    return make_verification_pairs(split["gallery"] + split["val"], pcfg.get("n_pairs", 2000), seed)


def evaluate(config_path: str, which: str = "best") -> dict:
    """which = "best" (checkpoint escolhido na validação) ou "last" (última época treinada)."""
    cfg = load_yaml(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = pick_device(cfg.get("device", "auto"))
    model_cfg = dict(cfg["model"])
    if cfg.get("training", {}).get("strategy", "none") != "none":
        ckpt = resolve_path(cfg["output"]["checkpoint"])
        if which == "last":
            ckpt = ckpt.with_name(ckpt.stem + "_last" + ckpt.suffix)
        model_cfg["checkpoint"] = str(ckpt)
    embedder = build_embedder(model_cfg, device)
    out_dir = resolve_path(cfg["output"]["results_dir"])
    if which == "last":
        out_dir = out_dir / "ultima_epoca"
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = cfg.get("evaluation", {})
    results: dict = {"experiment": cfg["experiment"], "model": model_cfg.get("name")}

    lfw = ev.get("lfw")
    if lfw and resolve_path(lfw["pairs"]).exists():
        folds = parse_lfw_pairs(resolve_path(lfw["pairs"]), resolve_path(lfw["root"]), ext=".png")
        s_folds, y_folds = zip(*(pair_scores(embedder, f) for f in folds), strict=True)
        results["lfw"] = lfw_10fold(
            list(s_folds), list(y_folds), ev.get("lfw_strategy", "max_accuracy")
        )
        s, y = np.concatenate(s_folds), np.concatenate(y_folds)
        results["lfw"]["roc_curve"] = plot_hist_and_roc(s, y, out_dir, "lfw")
        plot_confusion(results["lfw"]["confusion_matrix_sum"], out_dir, "lfw")
    else:
        print("LFW não encontrado; rode scripts/download_lfw.py e o preprocess.")

    pairs = private_validation_pairs(ev, seed)
    if pairs:
        s, y = pair_scores(embedder, pairs)
        thr = select_threshold(s, y, "eer")  # threshold da PRÓPRIA validação, só para referência
        rep = verification_report(s, y, thr)
        rep["roc_curve"] = plot_hist_and_roc(s, y, out_dir, "private_val")
        plot_confusion(rep["confusion_matrix"], out_dir, "private_val")
        results["private_val"] = rep

    results["cost"] = inference_cost(embedder)
    results["device"] = device
    results["environment"] = environment_info()
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação LFW 10-fold + validação privada")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="best = escolhido na validação; last = última época (resultados em ultima_epoca/)",
    )
    args = ap.parse_args()
    r = evaluate(args.config, args.checkpoint)
    short = {k: v for k, v in r.items() if k != "environment"}
    for part in ("lfw", "private_val"):
        if part in short:
            short[part] = {k: v for k, v in short[part].items() if k != "roc_curve"}
    print(json.dumps(short, indent=2, default=str))


if __name__ == "__main__":
    main()
