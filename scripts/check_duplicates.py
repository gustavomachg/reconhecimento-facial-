"""Checa quase-duplicatas (hash perceptual) ENTRE os conjuntos de cada split (ADR-004).

    python scripts/check_duplicates.py --dataset celeba --config configs/experiments/E03_...yaml
    python scripts/check_duplicates.py --dataset private --config configs/experiments/E06_...yaml

Reproduz exatamente o mesmo split usado no experimento (mesma seed) e salva só as CONTAGENS
em experiments/results/duplicates_<dataset>.json. Com --exclude-out, grava a lista de
arquivos duplicados (caminhos locais) em data/<public|private>/duplicates_exclude.txt —
arquivo ignorado pelo git — para serem removidos antes do treino.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facerec.config import load_yaml, resolve_path  # noqa: E402
from facerec.data.datasets import list_identity_folder, list_private  # noqa: E402
from facerec.data.dedup import find_cross_duplicates  # noqa: E402
from facerec.data.splits import (  # noqa: E402
    filter_records,
    split_by_identity,
    split_private_by_session,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=["celeba", "private"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-distance", type=int, default=4)
    ap.add_argument("--exclude-out", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = cfg.get("seed", 42)

    if args.dataset == "celeba":
        d = cfg["data"]
        recs = list_identity_folder(resolve_path(d["aligned_root"]))
        split = split_by_identity(
            [r.identity for r in recs],
            d.get("val_fraction", 0.15),
            d.get("test_fraction", 0.15),
            seed,
        )
        sets = {
            k: [r.path for r in filter_records(recs, split[k])] for k in ("train", "val", "test")
        }
        exclude_file = resolve_path("data/public/duplicates_exclude.txt")
    else:
        p = cfg["private"]
        split = split_private_by_session(
            list_private(resolve_path(p["aligned_root"])),
            p["gallery_sessions"],
            p["val_sessions"],
            p["test_sessions"],
            p.get("n_unknown_val", 1),
            p.get("n_unknown_test", 1),
            seed,
        )
        sets = {k: [r.path for r in split[k]] for k in ("gallery", "val", "test")}
        exclude_file = resolve_path("data/private/duplicates_exclude.txt")

    dups = find_cross_duplicates(sets, args.max_distance)
    counts: dict[str, int] = {}
    for a, _, b, _, _ in dups:
        counts[f"{a}x{b}"] = counts.get(f"{a}x{b}", 0) + 1
    summary = {
        "dataset": args.dataset,
        "max_hamming_distance": args.max_distance,
        "set_sizes": {k: len(v) for k, v in sets.items()},
        "n_cross_duplicates": len(dups),
        "by_pair": counts,
    }
    out = resolve_path(f"experiments/results/duplicates_{args.dataset}.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.exclude_out and dups:
        # remove sempre a cópia do conjunto de avaliação (b), preservando o treino
        exclude_file.write_text(
            "\n".join(sorted({pb for _, _, _, pb, _ in dups})), encoding="utf-8"
        )
        print(f"Lista de exclusão (local, ignorada pelo git): {exclude_file}")


if __name__ == "__main__":
    main()
