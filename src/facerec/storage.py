"""Repositório local de pessoas e embeddings (SQLite) — ADR-002.

Esquema:
    persons(id TEXT PK, display_name TEXT, created_at TEXT)
    embeddings(id INTEGER PK, person_id TEXT FK -> persons ON DELETE CASCADE,
               vector BLOB float32, dim INTEGER, photo_path TEXT NULL)

Decisões de privacidade:
- `id` é um UUID opaco: é o único identificador que aparece em logs.
- `PRAGMA secure_delete=ON` sobrescreve com zeros o conteúdo apagado, e `VACUUM` após a
  exclusão reconstrói o arquivo, para que embeddings removidos não fiquem recuperáveis.
- Fotos só são guardadas se `photos_dir` for configurado (por padrão NÃO são — minimização).
"""

from __future__ import annotations

import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from facerec.exceptions import PersonNotFound

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL,
    photo_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Person:
    id: str
    display_name: str
    created_at: str
    n_embeddings: int = 0


class FaceRepository:
    def __init__(self, db_path: str | Path, photos_dir: str | Path | None = None):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.photos_dir = Path(photos_dir) if photos_dir else None
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA secure_delete = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------- escrita ----------
    def add_person(
        self,
        display_name: str,
        embeddings: np.ndarray,
        photos: list[bytes] | None = None,
    ) -> Person:
        embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        person_id = uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        photo_paths: list[str | None] = [None] * len(embeddings)
        if photos and self.photos_dir is not None:
            person_dir = self.photos_dir / person_id
            person_dir.mkdir(parents=True, exist_ok=True)
            for i, data in enumerate(photos[: len(embeddings)]):
                path = person_dir / f"{i:03d}.jpg"
                path.write_bytes(data)
                photo_paths[i] = str(path)
        with self._conn:
            self._conn.execute(
                "INSERT INTO persons (id, display_name, created_at) VALUES (?, ?, ?)",
                (person_id, display_name, created_at),
            )
            self._conn.executemany(
                "INSERT INTO embeddings (person_id, vector, dim, photo_path) VALUES (?, ?, ?, ?)",
                [
                    (person_id, vec.tobytes(), int(vec.shape[0]), p)
                    for vec, p in zip(embeddings, photo_paths, strict=True)
                ],
            )
        return Person(person_id, display_name, created_at, len(embeddings))

    def delete_person(self, person_id: str) -> None:
        """Remove a pessoa, TODOS os embeddings (cascade) e as fotos armazenadas."""
        if self.get_person(person_id) is None:
            raise PersonNotFound()
        photo_paths = [
            row[0]
            for row in self._conn.execute(
                "SELECT photo_path FROM embeddings WHERE person_id = ? AND photo_path IS NOT NULL",
                (person_id,),
            )
        ]
        with self._conn:
            self._conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        for p in photo_paths:
            Path(p).unlink(missing_ok=True)
        if self.photos_dir is not None:
            shutil.rmtree(self.photos_dir / person_id, ignore_errors=True)
        if self.db_path != ":memory:":
            self._conn.execute("VACUUM")

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ---------- leitura ----------
    def get_person(self, person_id: str) -> Person | None:
        row = self._conn.execute(
            "SELECT p.id, p.display_name, p.created_at, COUNT(e.id) FROM persons p "
            "LEFT JOIN embeddings e ON e.person_id = p.id WHERE p.id = ? GROUP BY p.id",
            (person_id,),
        ).fetchone()
        return Person(*row) if row else None

    def list_persons(self) -> list[Person]:
        rows = self._conn.execute(
            "SELECT p.id, p.display_name, p.created_at, COUNT(e.id) FROM persons p "
            "LEFT JOIN embeddings e ON e.person_id = p.id GROUP BY p.id ORDER BY p.created_at"
        ).fetchall()
        return [Person(*r) for r in rows]

    def count_persons(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0])

    def count_embeddings(self, person_id: str | None = None) -> int:
        if person_id is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (person_id,)
            ).fetchone()[0]
        )

    def all_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """Todos os embeddings da galeria e o person_id de cada um."""
        rows = self._conn.execute("SELECT person_id, vector FROM embeddings").fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []
        vecs = np.stack([np.frombuffer(v, dtype=np.float32) for _, v in rows])
        return vecs, [pid for pid, _ in rows]
