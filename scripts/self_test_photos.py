"""Teste local com fotos de UMA pessoa (o aluno): o sistema reconhece com fotos pré-processadas?

    python scripts/self_test_photos.py pessoas
    python scripts/self_test_photos.py pessoas --save-aligned     # salva os recortes alinhados

Para cada foto: detecção (MTCNN) -> alinhamento -> embedding, com o mesmo modelo, detector,
matcher e threshold de configs/app.yaml. Depois faz *leave-one-out*: cadastra todas as fotos
menos uma e tenta reconhecer a que ficou de fora, repetindo para cada foto.

Privacidade (docs/privacy.md):
- roda só localmente; nada é enviado para fora e nada vai para o banco da aplicação;
- a saída usa índices anônimos (Foto 1, 2, ...), nunca nomes de arquivo;
- com --save-aligned, os recortes vão para data/private/aligned_preview/ (ignorado pelo git),
  para VOCÊ conferir o pré-processamento. Apague depois (ou use purge_private_data.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facerec.config import load_yaml, pick_device, resolve_path  # noqa: E402
from facerec.data.datasets import IMAGE_EXTS  # noqa: E402
from facerec.detection import FaceDetector  # noqa: E402
from facerec.exceptions import FaceRecError  # noqa: E402
from facerec.matching import build_matcher  # noqa: E402
from facerec.metrics import UNKNOWN  # noqa: E402
from facerec.models import build_embedder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("folder", help="pasta com fotos de UMA mesma pessoa")
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument("--save-aligned", action="store_true")
    ap.add_argument("--no-quality", action="store_true", help="não rejeitar por tamanho/nitidez")
    args = ap.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    device = pick_device(cfg.get("device", "auto"))
    embedder = build_embedder(cfg["model"], device)
    det_cfg = cfg.get("detection", {})
    detector = FaceDetector(
        device=device,
        align_mode=embedder.align_mode,
        image_size=embedder.input_size,
        prob_threshold=det_cfg.get("prob_threshold", 0.95),
        min_face_px=det_cfg.get("min_face_px", 60),
        blur_threshold=det_cfg.get("blur_threshold", 15.0),
    )
    rec = cfg.get("recognition", {})
    threshold = float(rec["threshold"])
    matcher_name = rec.get("matcher", "cosine")

    folder = resolve_path(args.folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit("Nenhuma imagem encontrada na pasta.")
    out_dir = resolve_path("data/private/aligned_preview")

    print(f"Modelo: {embedder.name} | matcher: {matcher_name} | threshold: {threshold}")
    print(f"{len(files)} foto(s) encontradas\n\n1) Detecção e pré-processamento")
    faces, idx_ok = [], []
    for i, path in enumerate(files, start=1):
        try:
            face = detector.detect_single(path, check_quality=not args.no_quality)
        except FaceRecError as exc:
            print(f"  Foto {i}: REJEITADA — {exc}")
            continue
        faces.append(face.aligned)
        idx_ok.append(i)
        print(
            f"  Foto {i}: ok (confiança {face.prob:.3f}, face {face.size:.0f}px, "
            f"nitidez {face.blur_score:.0f})"
        )
        if args.save_aligned:
            from PIL import Image

            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(face.aligned).save(out_dir / f"foto_{i:02d}.png")
    if args.save_aligned and faces:
        print(f"  Recortes alinhados salvos em {out_dir} (local, ignorado pelo git)")

    if len(faces) < 2:
        sys.exit("\nSão necessárias pelo menos 2 fotos aceitas para o teste.")

    emb = embedder.embed(faces)
    sims = emb @ emb.T
    print("\n2) Similaridade cosseno entre as fotos (1 = idênticas)")
    print("        " + "".join(f"F{i:<6}" for i in idx_ok))
    for a, i in enumerate(idx_ok):
        print(f"  F{i:<4} " + "".join(f"{sims[a, b]:<7.3f}" for b in range(len(idx_ok))))
    off = sims[~np.eye(len(idx_ok), dtype=bool)]
    print(f"  média {off.mean():.3f} | mínimo {off.min():.3f} | máximo {off.max():.3f}")

    print("\n3) Leave-one-out: cadastra as outras fotos e reconhece a que ficou de fora")
    hits = 0
    for a, i in enumerate(idx_ok):
        gallery = np.delete(emb, a, axis=0)
        matcher = build_matcher(matcher_name, **(rec.get("matcher_kwargs") or {}))
        matcher.fit(gallery, ["voce"] * len(gallery))
        label, score = matcher.identify(emb[a], threshold)
        ok = label != UNKNOWN
        hits += ok
        status = "RECONHECIDA" if ok else "desconhecida"
        print(f"  Foto {i}: {status} (score {score:.3f} vs threshold {threshold})")
    print(f"\nResultado: {hits}/{len(idx_ok)} reconhecidas ({100 * hits / len(idx_ok):.0f}%)")
    print("Obs.: este teste mede só se VOCÊ é reconhecido (FRR). Ele não mede se outras pessoas")
    print("seriam aceitas por engano (FAR) — isso exige fotos de outras pessoas (E06/E07).")


if __name__ == "__main__":
    main()
