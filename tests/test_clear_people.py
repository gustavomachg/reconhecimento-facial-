"""scripts/clear_people.py: simula sem --yes, apaga com --yes e não imprime dados pessoais."""

import runpy
import sys
from pathlib import Path

import numpy as np

from facerec.storage import FaceRepository

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clear_people.py"


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["clear_people.py", *args])
    runpy.run_path(str(SCRIPT), run_name="__main__")


def make_db(tmp_path):
    db = tmp_path / "app.db"
    repo = FaceRepository(db)
    rng = np.random.default_rng(0)
    for i in range(2):
        repo.add_person(f"Pessoa Sintética {i}", rng.normal(size=(3, 8)), email=f"p{i}@exemplo.com")
    repo.close()
    cfg = tmp_path / "app.yaml"
    cfg.write_text(f"storage:\n  db_path: {db.as_posix()}\n", encoding="utf-8")
    return db, cfg


def test_without_yes_only_counts(tmp_path, monkeypatch, capsys):
    db, cfg = make_db(tmp_path)
    run(monkeypatch, "--config", str(cfg))
    out = capsys.readouterr().out
    assert "2 pessoa(s) e 6 embedding(s)" in out and "Simulação" in out
    assert FaceRepository(db).count_persons() == 2


def test_with_yes_empties_the_database(tmp_path, monkeypatch, capsys):
    db, cfg = make_db(tmp_path)
    run(monkeypatch, "--config", str(cfg), "--yes")
    out = capsys.readouterr().out
    repo = FaceRepository(db)
    assert repo.count_persons() == 0 and repo.count_embeddings() == 0
    assert "Sintética" not in out and "@" not in out  # saída anônima
    run(monkeypatch, "--config", str(cfg))
    assert "Nenhuma pessoa cadastrada." in capsys.readouterr().out
