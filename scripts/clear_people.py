"""Esvazia os cadastros da APLICAÇÃO (contas, e-mails, senhas e embeddings), mantendo o banco.

    python scripts/clear_people.py          # simulação: só conta o que seria apagado
    python scripts/clear_people.py --yes    # apaga de fato

Usa o banco configurado em configs/app.yaml (storage.db_path). Cada pessoa é removida pela
mesma rotina da exclusão de conta (secure_delete + VACUUM, ver docs/specs/006), então nada
fica recuperável no arquivo. Depois disso o início mostra "Nenhuma pessoa cadastrada.".
Saída anônima: só números, nunca nomes ou e-mails.

Não mexe em data/private/, pessoas/ nem nos modelos: para apagar TUDO ao fim da disciplina,
use scripts/purge_private_data.py. Com o servidor ligado, quem estava logado é desconectado
na próxima ação (a conta deixa de existir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facerec.config import load_yaml, resolve_path  # noqa: E402
from facerec.storage import FaceRepository  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="apaga de fato (sem isso é simulação)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "app.yaml"))
    args = ap.parse_args()

    st = load_yaml(args.config).get("storage", {})
    db = resolve_path(st.get("db_path", "embeddings/facerec.db"))
    if not Path(db).exists():
        print("Nenhum banco encontrado: não há cadastros para apagar.")
        return
    photos = resolve_path(st["photos_dir"]) if st.get("store_photos") else None
    repo = FaceRepository(db, photos)
    try:
        people = repo.list_persons()
        n_emb = repo.count_embeddings()
        if not people:
            print("Nenhuma pessoa cadastrada.")
            return
        print(f"{len(people)} pessoa(s) e {n_emb} embedding(s) no banco.")
        if not args.yes:
            print("Simulação: nada foi apagado. Rode de novo com --yes para apagar.")
            return
        for p in people:
            repo.delete_person(p.id)
        print(f"Apagado: {len(people)} pessoa(s). Restam {repo.count_persons()}.")
    finally:
        repo.close()


if __name__ == "__main__":
    main()
