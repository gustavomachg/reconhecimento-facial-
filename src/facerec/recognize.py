"""Reconhece fotos usando a pasta `pessoas/` como cadastro (sem webcam e sem banco).

    python -m facerec.recognize minha_foto.jpg
    python -m facerec.recognize pasta_de_teste/
    python -m facerec.recognize pessoas/                 # cada foto contra as OUTRAS fotos
    python -m facerec.recognize foto.jpg --pre-aligned   # fotos já recortadas no rosto

Cadastro = cada subpasta de `pessoas/` é uma pessoa (nome = nome da subpasta). Modelo,
comparador e threshold vêm de configs/app.yaml, como na aplicação. Os embeddings ficam só
em memória. Se a foto consultada estiver dentro de `pessoas/`, ela é retirada do cadastro
antes da consulta (senão a resposta seria trivial: a foto se compararia consigo mesma).

Privacidade: roda só localmente. A saída mostra o nome da pessoa reconhecida (é a função do
sistema); não compartilhe essa saída fora do grupo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from facerec.config import load_yaml, pick_device, resolve_path
from facerec.data.datasets import IMAGE_EXTS
from facerec.evaluate_folder import list_people, load_pre_aligned
from facerec.matching import build_matcher
from facerec.metrics import UNKNOWN
from facerec.models import build_embedder


class FaceLoader:
    """Transforma um arquivo em recorte alinhado, do mesmo jeito para cadastro e consulta."""

    def __init__(self, embedder, pre_aligned: bool, device: str):
        self.size = embedder.input_size
        self.pre_aligned = pre_aligned
        self.detector = None
        if not pre_aligned:
            from facerec.detection import FaceDetector

            self.detector = FaceDetector(
                device=device,
                align_mode=embedder.align_mode,
                image_size=self.size,
                prob_threshold=0.9,
            )

    def __call__(self, path: Path) -> tuple[np.ndarray | None, str]:
        """(recorte, aviso). Recorte None = foto sem face utilizável."""
        if self.pre_aligned:
            return load_pre_aligned(path, self.size), ""
        from facerec.detection import load_image

        try:
            img = load_image(path)
        except Exception:
            return None, "arquivo ilegível"
        faces = self.detector.detect(img)
        if not faces:
            return None, "nenhuma face detectada"
        warn = "várias faces — usei a maior" if len(faces) > 1 else ""
        return self.detector.align(img, faces[0]), warn


def build_gallery(root: Path, loader: FaceLoader, embedder):
    items, n_people, loose = list_people(root)
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    faces, labels, paths, skipped = [], [], [], 0
    for path, idx in items:
        face, _ = loader(path)
        if face is None:
            skipped += 1
            continue
        faces.append(face)
        labels.append(names[idx])
        paths.append(path.resolve())
    if not faces:
        raise SystemExit(f"Nenhuma foto utilizável no cadastro ({root}).")
    return embedder.embed(faces), labels, paths, n_people, skipped, loose


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("query", help="foto ou pasta de fotos a reconhecer")
    ap.add_argument(
        "--gallery", default="pessoas", help="pasta do cadastro (uma subpasta por pessoa)"
    )
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument("--pre-aligned", action="store_true")
    ap.add_argument("--threshold", type=float, help="sobrescreve o threshold do app.yaml")
    args = ap.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    device = pick_device(cfg.get("device", "auto"))
    rec = cfg.get("recognition", {})
    threshold = args.threshold if args.threshold is not None else float(rec["threshold"])
    matcher_name = rec.get("matcher", "cosine")
    embedder = build_embedder(cfg["model"], device)
    loader = FaceLoader(embedder, args.pre_aligned, device)

    # caminhos relativos partem do diretório atual (como a foto consultada); se não existir
    # ali, tenta a raiz do projeto
    gallery_root = Path(args.gallery)
    if not gallery_root.is_absolute():
        cwd_path = Path.cwd() / gallery_root
        gallery_root = cwd_path if cwd_path.is_dir() else resolve_path(args.gallery)
    if not gallery_root.is_dir():
        sys.exit(f"Pasta do cadastro não encontrada: {args.gallery}")
    emb, labels, paths, n_people, skipped, loose = build_gallery(gallery_root, loader, embedder)
    print(f"Modelo: {embedder.name} | comparador: {matcher_name} | threshold: {threshold}")
    print(
        f"Cadastro: {n_people} pessoa(s), {len(labels)} foto(s)"
        + (f", {skipped} sem face" if skipped else "")
        + (f" ({loose} foto(s) soltas ignoradas)" if loose else "")
    )

    q = Path(args.query)
    q = q if q.is_absolute() else Path.cwd() / q
    if q.is_dir():
        queries = sorted(p for p in q.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    elif q.is_file():
        queries = [q]
    else:
        sys.exit(f"Não encontrado: {args.query}")

    kwargs = rec.get("matcher_kwargs") or {}
    full = build_matcher(matcher_name, **kwargs).fit(emb, labels)
    hits = total = 0
    print()
    for path in queries:
        face, warn = loader(path)
        shown = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path.name
        if face is None:
            print(f"{shown}: {warn}")
            continue
        qemb = embedder.embed(face)[0]
        matcher, truth = full, None
        rp = path.resolve()
        if rp in paths:  # a foto está no cadastro: tira ela antes de consultar
            keep = [i for i, p in enumerate(paths) if p != rp]
            truth = labels[paths.index(rp)]
            if not keep:
                print(f"{shown}: única foto do cadastro — nada para comparar")
                continue
            matcher = build_matcher(matcher_name, **kwargs).fit(
                emb[keep], [labels[i] for i in keep]
            )
        label, score = matcher.identify(qemb, threshold)
        result = f"Pessoa reconhecida: {label}" if label != UNKNOWN else "Pessoa desconhecida"
        extra = f" [{warn}]" if warn else ""
        check = ""
        if truth is not None and truth in set(matcher.labels):
            total += 1
            hits += label == truth
            check = "  -> CERTO" if label == truth else "  -> ERRADO"
        print(f"{shown}: {result} (score {score:.3f}){extra}{check}")
    if total:
        print(f"\nAcertos nas fotos do cadastro: {hits}/{total} ({100 * hits / total:.0f}%)")


if __name__ == "__main__":
    main()
