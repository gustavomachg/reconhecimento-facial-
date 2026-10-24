"""Cadastra na APLICAÇÃO todas as pessoas da pasta `pessoas/` (uma subpasta por pessoa).

    python scripts/import_people.py
    python scripts/import_people.py --pre-aligned    # fotos já recortadas no rosto

A aplicação tem o próprio cadastro (SQLite em embeddings/facerec.db, configurado em
configs/app.yaml) e NÃO lê a pasta pessoas/ sozinha. Este script faz a ponte: para cada
subpasta, usa o mesmo detector/modelo da aplicação e cadastra a pessoa com o nome da pasta
(o nome fica só no banco local; não é impresso nem vai para logs).

- Fotos sem face, com várias faces ou de baixa qualidade são puladas (as outras são usadas).
- Pessoas já cadastradas com o mesmo nome são puladas (pode rodar de novo sem duplicar).
- Saída anônima: "Pessoa 1, Pessoa 2, ..." — nada de nomes.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facerec.config import load_yaml, resolve_path  # noqa: E402
from facerec.data.datasets import IMAGE_EXTS  # noqa: E402
from facerec.detection import DetectedFace  # noqa: E402
from facerec.exceptions import DuplicatePerson, FaceRecError, InconsistentEnrollment  # noqa: E402
from facerec.service import build_service  # noqa: E402


class PreAlignedDetector:
    """Para fotos que já são recortes do rosto: só redimensiona (sem MTCNN)."""

    def __init__(self, size: int):
        self.size = size

    def detect_single(self, data: bytes, check_quality: bool = True) -> DetectedFace:
        import io

        import numpy as np
        from PIL import Image, ImageOps

        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        aligned = np.asarray(img.resize((self.size, self.size)), dtype=np.uint8)
        return DetectedFace(np.array([0, 0, s, s], dtype=float), 1.0, None, aligned)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", default="pessoas")
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument("--pre-aligned", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    svc = build_service(cfg)
    if args.pre_aligned:
        svc.detector = PreAlignedDetector(svc.embedder.input_size)
    already = {p.display_name for p in svc.list_persons()}
    people = sorted(p for p in resolve_path(args.root).iterdir() if p.is_dir())
    if not people:
        sys.exit("Nenhuma subpasta encontrada: crie uma subpasta por pessoa dentro de pessoas/.")

    print(f"Modelo: {svc.embedder.name} | matcher: {svc.matcher_name} | threshold: {svc.threshold}")
    totals = Counter()
    for i, folder in enumerate(people, start=1):
        tag = f"Pessoa {i}"
        if folder.name in already:
            print(f"  {tag}: já cadastrada — pulei")
            totals["ja_cadastrada"] += 1
            continue
        photos = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
        ok, rejected = [], Counter()
        for p in photos:
            data = p.read_bytes()
            try:
                svc.detector.detect_single(data)
                ok.append(data)
            except FaceRecError as exc:
                rejected[exc.code] += 1
        ok = ok[: svc.max_enroll_images]
        rej = f" ({sum(rejected.values())} rejeitada(s): {dict(rejected)})" if rejected else ""
        if not ok:
            print(f"  {tag}: nenhuma foto válida{rej}")
            totals["sem_foto_valida"] += 1
            continue
        try:
            svc.enroll(folder.name, ok)
            print(f"  {tag}: cadastrada com {len(ok)} foto(s){rej}")
            totals["cadastradas"] += 1
        except DuplicatePerson:
            print(f"  {tag}: NÃO cadastrada — muito parecida com alguém já cadastrado")
            totals["duplicada"] += 1
        except InconsistentEnrollment:
            print(
                f"  {tag}: NÃO cadastrada — as fotos parecem ser de pessoas diferentes "
                "(confira a pasta ou ajuste recognition.consistency_threshold)"
            )
            totals["inconsistente"] += 1
    print(f"\nResumo: {dict(totals)} | total no cadastro: {len(svc.list_persons())}")
    print("Agora rode a aplicação: python -m uvicorn facerec.app.main:app")


if __name__ == "__main__":
    main()
