"""Baixa o LFW (Labeled Faces in the Wild) e o pairs.txt para data/public/lfw/ (ignorado pelo git).

    python scripts/download_lfw.py

Fonte oficial: http://vis-www.cs.umass.edu/lfw/ (University of Massachusetts, Amherst).
Como o servidor oficial fica fora do ar com frequência, há como alternativa o espelho
mantido pelo scikit-learn (figshare), que contém os mesmos arquivos (`lfw.tgz`, `pairs.txt`).
O dataset NÃO é redistribuído por este repositório.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "public"

SOURCES = {
    "lfw.tgz": [
        "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
        "https://ndownloader.figshare.com/files/5976018",  # espelho usado pelo scikit-learn
    ],
    "pairs.txt": [
        "http://vis-www.cs.umass.edu/lfw/pairs.txt",
        "https://ndownloader.figshare.com/files/5976006",
    ],
}


def download(urls: list[str], dst: Path) -> None:
    for url in urls:
        try:
            print(f"Baixando {url}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            return
        except requests.RequestException as exc:
            print(f"  falhou ({exc}); tentando próxima fonte...")
    sys.exit(f"Não foi possível baixar {dst.name}. Baixe manualmente da fonte oficial.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "lfw.tgz"
    if not (DEST / "lfw").exists():
        if not archive.exists():
            download(SOURCES["lfw.tgz"], archive)
        print(f"sha256(lfw.tgz) = {sha256(archive)}  (registre no README para reprodutibilidade)")
        with tarfile.open(archive) as tar:
            try:  # filtro de segurança (Python >= 3.11.4); versões anteriores não o aceitam
                tar.extractall(DEST, filter="data")
            except TypeError:
                tar.extractall(DEST)  # noqa: S202 - arquivo da fonte oficial do LFW
        if not args.keep_archive:
            archive.unlink()
    pairs = DEST / "lfw" / "pairs.txt"
    if not pairs.exists():
        tmp = DEST / "pairs.txt"
        download(SOURCES["pairs.txt"], tmp)
        shutil.move(tmp, pairs)
    n_people = sum(1 for p in (DEST / "lfw").iterdir() if p.is_dir())
    print(f"LFW pronto em {DEST / 'lfw'} — {n_people} identidades.")
    print("Próximo passo: python -m facerec.data.preprocess --dataset lfw --model facenet")


if __name__ == "__main__":
    main()
