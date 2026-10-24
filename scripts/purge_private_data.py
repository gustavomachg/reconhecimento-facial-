"""Eliminação dos dados privados ao fim da disciplina (LGPD — docs/privacy.md).

    python scripts/purge_private_data.py            # simulação: só lista o que seria apagado
    python scripts/purge_private_data.py --yes      # apaga de fato
    python scripts/purge_private_data.py --yes --include-consent

Apaga: data/private/, pessoas/, faces/, embeddings/ (banco SQLite da aplicação),
checkpoints/private/ (modelos ajustados com dados privados) e bancos *.db/*.sqlite soltos.
Os termos de consentimento (consent/) só são apagados com --include-consent, pois podem
precisar ser guardados como comprovação pelo prazo definido em docs/privacy.md.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = ["data/private", "pessoas", "faces", "embeddings", "checkpoints/private"]
DB_PATTERNS = ["*.db", "*.sqlite", "*.sqlite3"]
SKIP = {".git", ".venv", "venv", "node_modules"}


def targets(include_consent: bool) -> list[Path]:
    dirs = TARGET_DIRS + (["consent"] if include_consent else [])
    found = [ROOT / d for d in dirs if (ROOT / d).exists()]
    for pattern in DB_PATTERNS:
        for p in ROOT.rglob(pattern):
            if not SKIP.intersection(p.relative_to(ROOT).parts) and not any(
                p.is_relative_to(d) for d in found
            ):
                found.append(p)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="apaga de fato (sem isso é simulação)")
    ap.add_argument("--include-consent", action="store_true")
    args = ap.parse_args()
    items = targets(args.include_consent)
    if not items:
        print("Nada a apagar.")
        return
    for p in items:
        n = sum(1 for _ in p.rglob("*")) if p.is_dir() else 1
        print(f"{'APAGANDO' if args.yes else '[simulação]'} {p.relative_to(ROOT)} ({n} item(ns))")
        if args.yes:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    if not args.yes:
        print("\nNada foi apagado. Rode com --yes para confirmar.")
    else:
        print("\nConcluído. Lembre-se de esvaziar a lixeira/backup (ex.: OneDrive) também.")


if __name__ == "__main__":
    main()
