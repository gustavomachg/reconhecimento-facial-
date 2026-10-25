"""Pré-processamento offline: detecta e alinha as faces de um dataset e salva os recortes.

Uso:
    python -m facerec.data.preprocess --dataset lfw     --model facenet
    python -m facerec.data.preprocess --dataset celeba  --model facenet
    python -m facerec.data.preprocess --dataset private --model arcface

Em datasets (diferente da aplicação) não levantamos erro para múltiplas faces: escolhemos
uma. Em LFW e CelebA, a face MAIS PRÓXIMA DO CENTRO (as imagens são centralizadas na pessoa
rotulada; a maior face pode ser de outra pessoa); no privado, a MAIOR (`--face-selection`).
Se nenhuma face for encontrada:
- LFW: recorte central (prática comum no protocolo LFW, para não descartar pares);
- CelebA / privado: a imagem é descartada.
Só estatísticas AGREGADAS (contagens) são salvas em experiments/results/preprocess_*.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from facerec.config import pick_device, resolve_path
from facerec.data.datasets import list_identity_folder, list_private, load_celeba_records
from facerec.detection import FaceDetector, load_image, pick_face

MODEL_ALIGN = {"facenet": ("crop", 160), "arcface": ("similarity", 112)}


def center_crop(img: Image.Image, size: int) -> np.ndarray:
    w, h = img.size
    s = int(min(w, h) * 0.6)
    box = ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
    return np.asarray(img.crop(box).resize((size, size)), dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detecta + alinha faces de um dataset")
    ap.add_argument("--dataset", choices=["lfw", "celeba", "private"], required=True)
    ap.add_argument("--model", choices=list(MODEL_ALIGN), default="facenet")
    ap.add_argument("--celeba-min-images", type=int, default=15)
    ap.add_argument("--celeba-max-identities", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--face-selection",
        choices=["auto", "center", "largest"],
        default="auto",
        help="auto = center para lfw/celeba, largest para private",
    )
    args = ap.parse_args()

    align_mode, size = MODEL_ALIGN[args.model]
    detector = FaceDetector(
        device=pick_device(args.device), align_mode=align_mode, image_size=size, prob_threshold=0.9
    )

    if args.dataset == "lfw":
        src_root = resolve_path("data/public/lfw")
        records = list_identity_folder(src_root)
        out_root = resolve_path(f"data/public/aligned_{args.model}/lfw")
    elif args.dataset == "celeba":
        src_root = resolve_path("data/public/celeba/img_align_celeba")
        records = load_celeba_records(
            resolve_path("data/public/celeba/identity_CelebA.txt"),
            src_root,
            min_images=args.celeba_min_images,
            max_identities=args.celeba_max_identities,
        )
        out_root = resolve_path(f"data/public/aligned_{args.model}/celeba")
    else:
        src_root = resolve_path("data/private/raw")
        records = list_private(src_root)
        out_root = resolve_path(f"data/private/aligned_{args.model}")

    stats = {"dataset": args.dataset, "model": args.model, "total": len(records)}
    selection = args.face_selection
    if selection == "auto":
        selection = "largest" if args.dataset == "private" else "center"
    stats["face_selection"] = selection
    stats.update(ok=0, no_face=0, multiple_faces=0, fallback_center=0, unreadable=0)
    for rec in tqdm(records, desc=f"alinhando {args.dataset}"):
        try:
            img = load_image(rec.path)
        except Exception:
            stats["unreadable"] += 1
            continue
        faces = detector.detect(img)
        if len(faces) > 1:
            stats["multiple_faces"] += 1
        if faces:
            crop = detector.align(img, pick_face(faces, img.size, selection))
            stats["ok"] += 1
        elif args.dataset == "lfw":
            crop = center_crop(img, size)
            stats["fallback_center"] += 1
        else:
            stats["no_face"] += 1
            continue
        sub = Path(rec.identity) if args.dataset != "private" else Path(rec.identity, rec.session)
        dst = out_root / sub / (Path(rec.path).stem + ".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(dst)

    if args.dataset == "lfw":  # o protocolo precisa do pairs.txt junto das faces alinhadas
        pairs = src_root / "pairs.txt"
        if pairs.exists():
            (out_root / "pairs.txt").write_text(pairs.read_text(encoding="utf-8"))

    out = resolve_path(f"experiments/results/preprocess_{args.dataset}_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
