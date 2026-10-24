"""Avaliação do reconhecimento no dataset do grupo (pasta `pessoas/`).

    python -m facerec.evaluate_folder                         # pessoas/, modelos disponíveis
    python -m facerec.evaluate_folder --pre-aligned           # fotos já recortadas/alinhadas
    python -m facerec.evaluate_folder --models facenet mobilefacenet --matchers cosine prototype

Estrutura esperada (uma subpasta por pessoa; pode haver subpastas de sessão dentro):
    pessoas/<pessoa>/*.jpg

Protocolo (sem vazamento):
1. Verificação: todos os pares de fotos -> "mesma pessoa" × "pessoas diferentes"
   -> ROC, AUC, EER e histograma de similaridade.
2. Identificação em conjunto aberto, para cada comparador:
   - consulta CONHECIDA: cada foto é retirada da galeria e consultada contra o resto
     (a pessoa continua cadastrada com as outras fotos) -> deve ser reconhecida;
   - consulta DESCONHECIDA: cada pessoa é retirada INTEIRA da galeria e suas fotos são
     consultadas -> devem dar "Pessoa desconhecida".
3. Threshold por validação cruzada POR PESSOA: as pessoas são divididas em k grupos; o
   threshold usado nas consultas de um grupo é escolhido (EER) só com as consultas dos
   OUTROS grupos. Assim nenhuma consulta influencia o próprio threshold.

Privacidade: roda só localmente; nomes de pastas/arquivos NUNCA aparecem na saída nem nos
arquivos gerados (só contagens e métricas agregadas em experiments/results/dataset_grupo/).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from facerec.config import environment_info, load_yaml, pick_device, resolve_path, set_seed
from facerec.data.datasets import IMAGE_EXTS
from facerec.matching import build_matcher
from facerec.metrics import UNKNOWN, binary_metrics, compute_eer, open_set_metrics, roc_auc
from facerec.threshold import select_threshold

MODEL_PRESETS = {
    "facenet": {"name": "facenet"},
    "arcface": {"name": "arcface", "path": "checkpoints/public/buffalo_l/w600k_r50.onnx"},
    "mobilefacenet": {"name": "arcface", "path": "checkpoints/public/buffalo_s/w600k_mbf.onnx"},
}
DOWNLOAD_HINT = {
    "arcface": "python scripts/download_arcface.py",
    "mobilefacenet": "python scripts/download_arcface.py --pack buffalo_s",
}


# ------------------------------------------------------------------ leitura da pasta
def list_people(root: Path) -> tuple[list[tuple[Path, int]], int, int]:
    """Retorna [(caminho, índice_da_pessoa)], nº de pessoas e nº de fotos soltas ignoradas.
    A identidade é o índice da subpasta — o nome nunca sai desta função."""
    people = sorted(p for p in root.iterdir() if p.is_dir())
    items = []
    for idx, person in enumerate(people):
        for img in sorted(person.rglob("*")):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                items.append((img, idx))
    loose = sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return items, len(people), loose


def load_pre_aligned(path: Path, size: int) -> np.ndarray:
    """Foto já recortada: corta o quadrado central e redimensiona para a entrada da rede."""
    from PIL import ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def prepare_faces(items, embedder, pre_aligned: bool, device: str, quality: bool, preview_dir=None):
    """Detecta/alinha cada foto. Retorna (faces, rótulos, contagem de falhas por motivo)."""
    from facerec.detection import FaceDetector, laplacian_variance, load_image

    detector = None
    if not pre_aligned:
        detector = FaceDetector(
            device=device,
            align_mode=embedder.align_mode,
            image_size=embedder.input_size,
            prob_threshold=0.9,
        )
    faces, labels, stats = [], [], Counter()
    for path, person in items:
        try:
            if pre_aligned:
                face = load_pre_aligned(path, embedder.input_size)
            else:
                img = load_image(path)
                found = detector.detect(img)
                if not found:
                    stats["sem_face"] += 1
                    continue
                if len(found) > 1:
                    stats["varias_faces_usou_maior"] += 1
                if quality and found[0].size < detector.min_face_px:
                    stats["face_pequena"] += 1
                    continue
                face = detector.align(img, found[0])
                if quality and laplacian_variance(face) < detector.blur_threshold:
                    stats["desfocada"] += 1
                    continue
        except Exception:
            stats["ilegivel"] += 1
            continue
        faces.append(face)
        labels.append(person)
        stats["ok"] += 1
        if preview_dir is not None:  # recorte que entra na rede, para conferência visual
            d = preview_dir / f"pessoa_{person + 1:02d}"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(face).save(d / f"foto_{stats['ok']:03d}.png")
    return faces, labels, stats


# ------------------------------------------------------------------ avaliação (testável)
def verification(emb: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Todos os pares i<j: similaridade cosseno e se é a mesma pessoa."""
    sims = emb @ emb.T
    iu = np.triu_indices(len(labels), k=1)
    scores = sims[iu]
    same = (labels[iu[0]] == labels[iu[1]]).astype(int)
    res = {"n_pairs_same": int(same.sum()), "n_pairs_diff": int((1 - same).sum())}
    if 0 < same.sum() < len(same):
        eer, thr = compute_eer(scores, same)
        res.update(roc_auc=roc_auc(scores, same), eer=eer, eer_threshold=thr)
        res.update({k: v for k, v in binary_metrics(scores, same, thr).items() if k != "threshold"})
    return res, scores, same


def open_set_queries(emb: np.ndarray, labels: np.ndarray, matcher_cfg: dict) -> list[dict]:
    """Consultas conhecidas (deixa uma foto de fora) e desconhecidas (deixa a pessoa de fora)."""
    kwargs = {k: v for k, v in matcher_cfg.items() if k != "name"}
    lab = labels.astype(str)
    counts = Counter(labels.tolist())
    queries = []
    for i in range(len(labels)):
        if counts[labels[i]] < 2:
            continue  # sem outra foto da pessoa, ela não estaria cadastrada
        keep = np.arange(len(labels)) != i
        m = build_matcher(matcher_cfg["name"], **kwargs).fit(emb[keep], list(lab[keep]))
        top, score = m.best(emb[i])
        queries.append({"person": int(labels[i]), "true": lab[i], "top": top, "score": score})
    for person in np.unique(labels):
        keep = labels != person
        if len(np.unique(labels[keep])) < 1:
            continue
        m = build_matcher(matcher_cfg["name"], **kwargs).fit(emb[keep], list(lab[keep]))
        for i in np.where(labels == person)[0]:
            top, score = m.best(emb[i])
            queries.append({"person": int(person), "true": UNKNOWN, "top": top, "score": score})
    return queries


def cross_validated_decisions(queries: list[dict], k: int = 5, seed: int = 42) -> list[float]:
    """Threshold de cada consulta escolhido (EER) só com consultas de OUTRAS pessoas."""
    people = sorted({q["person"] for q in queries})
    random.Random(seed).shuffle(people)
    k = max(2, min(k, len(people)))
    fold_of = {p: i % k for i, p in enumerate(people)}
    scores = np.array([q["score"] for q in queries])
    target = np.array([int(q["true"] != UNKNOWN and q["true"] == q["top"]) for q in queries])
    folds = np.array([fold_of[q["person"]] for q in queries])
    thresholds = np.zeros(len(queries))
    for f in range(k):
        train = folds != f
        if 0 < target[train].sum() < train.sum():
            thr = select_threshold(scores[train], target[train], "eer")
        else:  # sem as duas classes nos outros grupos: usa o EER global (registrado no relatório)
            thr = select_threshold(scores, target, "eer") if 0 < target.sum() < len(target) else 0.0
        thresholds[folds == f] = thr
    return thresholds.tolist()


def evaluate_open_set(emb, labels, matcher_cfg: dict, k: int = 5, seed: int = 42) -> dict:
    queries = open_set_queries(emb, labels, matcher_cfg)
    if not queries:
        return {"erro": "nenhuma consulta possível (cada pessoa precisa de >= 2 fotos)"}
    thr = cross_validated_decisions(queries, k, seed)
    true = [q["true"] for q in queries]
    pred = [q["top"] if q["score"] >= t else UNKNOWN for q, t in zip(queries, thr, strict=True)]
    res = open_set_metrics(true, pred)
    scores = np.array([q["score"] for q in queries])
    target = np.array([int(q["true"] != UNKNOWN and q["true"] == q["top"]) for q in queries])
    accepted = np.array([p != UNKNOWN for p in pred])
    tp = int((accepted & (target == 1)).sum())
    fp = int((accepted & (target == 0)).sum())
    fn = int((~accepted & (target == 1)).sum())
    tn = int((~accepted & (target == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    res.update(
        threshold_mean=float(np.mean(thr)),
        threshold_std=float(np.std(thr)),
        precision=prec,
        recall=rec,
        f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        confusion_accept_reject={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    )
    if 0 < target.sum() < len(target):
        res["roc_auc"] = roc_auc(scores, target)
        res["eer"] = compute_eer(scores, target)[0]
    return res


# ------------------------------------------------------------------ gráficos
def _plots(model: str, scores, same, results: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    if same.sum() and (1 - same).sum():
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(scores.min(), scores.max(), 40)
        ax.hist(scores[same == 1], bins=bins, alpha=0.6, density=True, label="mesma pessoa")
        ax.hist(scores[same == 0], bins=bins, alpha=0.6, density=True, label="pessoas diferentes")
        ax.set(xlabel="similaridade cosseno", ylabel="densidade", title=f"Similaridades — {model}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{model}.png", dpi=120)
        plt.close(fig)
        fpr, tpr, _ = roc_curve(same, scores)
        results.setdefault("_roc", {})[model] = (
            fpr,
            tpr,
            results[model]["verification"]["roc_auc"],
        )

    for mname, r in results[model]["open_set"].items():
        cm = r.get("confusion_accept_reject")
        if not cm:
            continue
        m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.imshow(m, cmap="Blues")
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["rejeitou", "aceitou"])
        ax.set_yticks([0, 1], ["deveria rejeitar", "deveria aceitar"])
        ax.set_title(f"{model} + {mname}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_{model}_{mname}.png", dpi=120)
        plt.close(fig)


def _roc_comparison(results: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = results.pop("_roc", {})
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for model, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{model} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="FAR", ylabel="1 - FRR", title="ROC — dataset do grupo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_comparison.png", dpi=130)
    plt.close(fig)


def _pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def results_table(results: dict) -> str:
    lines = [
        "| Modelo | Comparador | Acurácia (open-set) | F1 | EER | Reconheceu conhecidos | "
        "Rejeitou desconhecidos | Confundiu pessoas | ms/face |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, r in results.items():
        for mname, m in r["open_set"].items():
            if "erro" in m:
                continue
            fau = m["false_accept_unknown"]
            lines.append(
                f"| {model} | {mname} | {_pct(m['accuracy'])} | {_pct(m['f1'])} | "
                f"{_pct(m.get('eer'))} | {_pct(m['known_accuracy'])} | "
                f"{_pct(None if fau is None else 1 - fau)} | {_pct(m['misidentification'])} | "
                f"{r['cost']['inference_ms_per_face']:.1f} |"
            )
    return "\n".join(lines)


# ------------------------------------------------------------------ principal
def run(
    root: Path,
    models: list[str],
    matchers: list[dict],
    pre_aligned: bool,
    quality: bool,
    out_dir: Path,
    device: str = "auto",
    k: int = 5,
    seed: int = 42,
    embedder_factory=None,
    preview_dir: Path | None = None,
) -> dict:
    from facerec.evaluate import inference_cost
    from facerec.models import build_embedder

    set_seed(seed)
    device = pick_device(device)
    items, n_people, loose = list_people(root)
    if loose:
        print(
            f"Aviso: {loose} foto(s) soltas em {root.name}/ foram ignoradas — "
            "coloque cada pessoa numa subpasta própria."
        )
    if n_people < 2:
        raise SystemExit(
            f"São necessárias pelo menos 2 pessoas (subpastas) em {root}; encontradas: {n_people}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for model in models:
        if embedder_factory is not None:
            embedder = embedder_factory(model)
        else:
            cfg = dict(MODEL_PRESETS[model])
            if "path" in cfg and not resolve_path(cfg["path"]).exists():
                print(f"[{model}] pesos não encontrados — pulei. Baixe com: {DOWNLOAD_HINT[model]}")
                continue
            embedder = build_embedder(cfg, device)
        print(f"\n[{model}] preparando faces...")
        pdir = preview_dir / model if preview_dir is not None else None
        faces, labels, stats = prepare_faces(items, embedder, pre_aligned, device, quality, pdir)
        if pdir is not None:
            print(f"[{model}] recortes salvos em {pdir} (local, ignorado pelo git)")
        labels = np.array(labels)
        print(f"[{model}] fotos: {dict(stats)}")
        if len(np.unique(labels)) < 2:
            print(f"[{model}] menos de 2 pessoas com face válida — pulei.")
            continue
        emb = embedder.embed(faces)
        ver, scores, same = verification(emb, labels)
        res = {
            "dataset": {
                "n_pessoas": int(len(np.unique(labels))),
                "n_fotos_validas": int(len(labels)),
                "fotos_por_pessoa_min": int(min(Counter(labels.tolist()).values())),
                "fotos_por_pessoa_max": int(max(Counter(labels.tolist()).values())),
                "preprocessamento": dict(stats),
                "pre_alinhadas": pre_aligned,
            },
            "verification": ver,
            "open_set": {},
            "cost": inference_cost(embedder),
        }
        for mc in matchers:
            label = mc["name"] + (f"_k{mc['k']}" if "k" in mc else "")
            res["open_set"][label] = evaluate_open_set(emb, labels, mc, k, seed)
        results[model] = res
        _plots(model, scores, same, results, out_dir)
    _roc_comparison(results, out_dir)
    summary = {
        "protocolo": "leave-one-out (conhecidos) + leave-one-person-out (desconhecidos); "
        f"threshold por validação cruzada em {k} grupos de pessoas (EER)",
        "results": results,
        "environment": environment_info(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    table = results_table(results)
    (out_dir / "results_table.md").write_text(table + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", default="pessoas")
    ap.add_argument("--models", nargs="+", default=list(MODEL_PRESETS), choices=list(MODEL_PRESETS))
    ap.add_argument(
        "--matchers",
        nargs="+",
        default=["cosine", "prototype", "knn"],
        choices=["cosine", "euclidean", "prototype", "knn", "svm"],
    )
    ap.add_argument(
        "--pre-aligned",
        action="store_true",
        help="as fotos já são recortes do rosto: pula a detecção",
    )
    ap.add_argument(
        "--quality",
        action="store_true",
        help="descarta faces pequenas/desfocadas (mesmos limites da aplicação)",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="experiments/results/dataset_grupo")
    ap.add_argument(
        "--save-aligned",
        action="store_true",
        help="salva os recortes que entram na rede em data/private/aligned_preview/ (conferência)",
    )
    args = ap.parse_args()

    app_cfg = load_yaml(resolve_path("configs/app.yaml"))
    matchers = [{"name": m, **({"k": 3} if m == "knn" else {})} for m in args.matchers]
    summary = run(
        resolve_path(args.root),
        args.models,
        matchers,
        args.pre_aligned,
        args.quality,
        resolve_path(args.out),
        device=app_cfg.get("device", "auto"),
        k=args.folds,
        preview_dir=resolve_path("data/private/aligned_preview") if args.save_aligned else None,
    )
    print("\n" + results_table(summary["results"]))
    for model, r in summary["results"].items():
        v = r["verification"]
        if "eer" in v:
            print(
                f"[{model}] verificação de pares: AUC {v['roc_auc']:.3f} | "
                f"EER {100 * v['eer']:.1f}% | {v['n_pairs_same']} pares mesma pessoa, "
                f"{v['n_pairs_diff']} diferentes"
            )
    print(f"\nResultados agregados em {resolve_path(args.out)}")


if __name__ == "__main__":
    main()
