"""Verifica sobreposição de identidades entre CelebA (treino do fine-tuning) e LFW (benchmark).

    python scripts/check_identity_overlap.py --model facenet --threshold 0.80

Por que: se uma pessoa do LFW também está no CelebA usado no fine-tuning, a avaliação no LFW
fica otimista (o modelo "já viu" aquela pessoa) — vazamento de identidade.

Problema: o CelebA NÃO publica nomes, só IDs numéricos. Então a comparação por nome é
impossível e fazemos uma checagem EMPÍRICA por embedding:
  1. centroide (média normalizada) de até N faces por identidade, nos dois datasets;
  2. similaridade cosseno entre todos os centroides CelebA × LFW;
  3. pares acima de um threshold alto são "suspeitos de mesma pessoa".
Os IDs CelebA suspeitos vão para data/public/celeba_exclude.txt, e o treino os remove
(`data.exclude_identities_file` na config). Só contagens agregadas vão para experiments/results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facerec.config import pick_device  # noqa: E402
from facerec.data.datasets import list_identity_folder  # noqa: E402
from facerec.evaluate import embed_paths  # noqa: E402
from facerec.matching import l2_normalize  # noqa: E402
from facerec.models import build_embedder  # noqa: E402


def centroids(records, embedder, per_identity: int) -> tuple[list[str], np.ndarray]:
    by_id = defaultdict(list)
    for r in records:
        if len(by_id[r.identity]) < per_identity:
            by_id[r.identity].append(r.path)
    emb = embed_paths(embedder, [p for ps in by_id.values() for p in ps])
    ids = sorted(by_id)
    return ids, l2_normalize(np.stack([np.mean([emb[p] for p in by_id[i]], axis=0) for i in ids]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="facenet")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--per-identity", type=int, default=5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    base = ROOT / "data" / "public" / f"aligned_{args.model}"
    embedder = build_embedder({"name": args.model}, pick_device(args.device))
    c_ids, c_cent = centroids(list_identity_folder(base / "celeba"), embedder, args.per_identity)
    l_ids, l_cent = centroids(list_identity_folder(base / "lfw"), embedder, args.per_identity)
    sims = c_cent @ l_cent.T
    ci, li = np.where(sims >= args.threshold)
    suspects = sorted({c_ids[i] for i in ci})
    (ROOT / "data" / "public" / "celeba_exclude.txt").write_text(
        "\n".join(suspects), encoding="utf-8"
    )
    best = sims.max(axis=1)
    summary = {
        "model": args.model,
        "threshold": args.threshold,
        "n_celeba_identities": len(c_ids),
        "n_lfw_identities": len(l_ids),
        "n_suspect_pairs": int(len(ci)),
        "n_suspect_celeba_identities_excluded": len(suspects),
        "max_similarity_percentiles": {
            str(q): float(np.percentile(best, q)) for q in (50, 90, 99, 100)
        },
    }
    out = ROOT / "experiments" / "results" / "identity_overlap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
