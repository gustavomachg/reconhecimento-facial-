"""Baixa pacotes oficiais de modelos do insightface (ArcFace).

    python scripts/download_arcface.py                    # w600k_r50.onnx (iResNet-50)
    python scripts/download_arcface.py --pack buffalo_s   # w600k_mbf.onnx (MobileFaceNet)

Fonte oficial: https://github.com/deepinsight/insightface (releases, v0.7).
LICENÇA: modelos pré-treinados do insightface são liberados somente para pesquisa
acadêmica / uso NÃO comercial. Os pesos ficam em checkpoints/public/ (ignorado pelo git).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/{pack}.zip"
PACKS = {"buffalo_l": "w600k_r50.onnx", "buffalo_s": "w600k_mbf.onnx"}
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", choices=sorted(PACKS), default="buffalo_l")
    args = ap.parse_args()
    dest = ROOT / "checkpoints" / "public" / args.pack
    target = dest / PACKS[args.pack]
    if target.exists():
        print(f"Já existe: {target}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / f"{args.pack}.zip"
    with requests.get(BASE_URL.format(pack=args.pack), stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(archive, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    with zipfile.ZipFile(archive) as z:
        # extrai só o modelo de reconhecimento; o pacote também traz detector, landmarks e
        # um estimador de gênero/idade, que não são usados (minimização)
        for member in z.namelist():
            if Path(member).name == target.name:
                target.write_bytes(z.read(member))
    archive.unlink()
    print(f"Modelos extraídos em {dest}: {sorted(p.name for p in dest.glob('*.onnx'))}")
    if not target.exists():
        print(f"ATENÇÃO: {target.name} não veio no pacote; ajuste `model.path` na config.")


if __name__ == "__main__":
    main()
