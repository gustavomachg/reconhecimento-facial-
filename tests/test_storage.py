import sqlite3

import numpy as np
import pytest

from facerec.exceptions import EmailAlreadyRegistered, PersonNotFound
from facerec.storage import FaceRepository


@pytest.fixture
def repo(tmp_path):
    r = FaceRepository(tmp_path / "test.db", photos_dir=tmp_path / "faces")
    yield r
    r.close()


def rand(n=3, d=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def test_initial_state_is_empty(repo):
    assert repo.count_persons() == 0
    assert repo.list_persons() == []
    vecs, ids = repo.all_embeddings()
    assert ids == [] and vecs.size == 0


def test_add_and_read_back(repo):
    emb = rand()
    p = repo.add_person("Pessoa Sintética", emb)
    assert len(p.id) == 32  # uuid4 hex opaco
    assert repo.get_person(p.id).n_embeddings == 3
    vecs, ids = repo.all_embeddings()
    assert ids == [p.id] * 3
    assert np.allclose(vecs, emb)


def test_delete_removes_person_embeddings_and_photos(repo, tmp_path):
    keep = repo.add_person("Fica", rand(seed=1))
    gone = repo.add_person("Sai", rand(seed=2), photos=[b"x", b"y", b"z"])
    photo_dir = tmp_path / "faces" / gone.id
    assert len(list(photo_dir.iterdir())) == 3

    repo.delete_person(gone.id)

    assert repo.get_person(gone.id) is None
    assert repo.count_embeddings(gone.id) == 0
    assert not photo_dir.exists()
    assert repo.get_person(keep.id) is not None
    assert repo.count_embeddings(keep.id) == 3
    # conferência direta no arquivo SQLite, por outra conexão
    con = sqlite3.connect(tmp_path / "test.db")
    assert (
        con.execute("SELECT COUNT(*) FROM embeddings WHERE person_id=?", (gone.id,)).fetchone()[0]
        == 0
    )
    assert con.execute("SELECT COUNT(*) FROM persons WHERE id=?", (gone.id,)).fetchone()[0] == 0
    con.close()


def test_deleted_bytes_not_left_in_db_file(repo, tmp_path):
    """secure_delete + VACUUM: os bytes do embedding apagado não ficam no arquivo."""
    emb = np.full((1, 64), 1234.5678, dtype=np.float32)
    p = repo.add_person("Temp", emb)
    repo.delete_person(p.id)
    raw = (tmp_path / "test.db").read_bytes()
    assert emb.tobytes()[:32] not in raw
    assert b"Temp" not in raw


def test_delete_unknown_raises(repo):
    with pytest.raises(PersonNotFound):
        repo.delete_person("nao-existe")


def test_photos_not_stored_by_default(tmp_path):
    r = FaceRepository(tmp_path / "x.db")  # sem photos_dir
    r.add_person("A", rand(), photos=[b"1", b"2", b"3"])
    assert not (tmp_path / "faces").exists()
    r.close()


def test_model_mismatch_is_detected(tmp_path):
    from facerec.service import ModelMismatchError, check_model_compatibility

    repo = FaceRepository(tmp_path / "m.db")
    check_model_compatibility(repo, {"name": "facenet"})  # banco vazio: grava o modelo
    repo.add_person("A", rand())
    check_model_compatibility(repo, {"name": "facenet"})  # mesmo modelo: ok
    with pytest.raises(ModelMismatchError):
        check_model_compatibility(repo, {"name": "arcface", "path": "x.onnx"})
    repo.delete_person(repo.list_persons()[0].id)
    check_model_compatibility(repo, {"name": "arcface", "path": "x.onnx"})  # vazio: pode trocar
    assert repo.get_meta("model") == "name=arcface|path=x.onnx"
    repo.close()


def test_unit_roundtrip_and_old_database_is_migrated(tmp_path):
    """Bancos criados antes do campo `unit` continuam abrindo, sem perder cadastros."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE persons (id TEXT PRIMARY KEY, display_name TEXT NOT NULL,"
        " created_at TEXT NOT NULL);"
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, person_id TEXT NOT NULL"
        " REFERENCES persons(id) ON DELETE CASCADE, vector BLOB NOT NULL, dim INTEGER NOT NULL,"
        " photo_path TEXT);"
        "INSERT INTO persons VALUES ('abc', 'Antigo', '2026-01-01T00:00:00');"
    )
    conn.commit()
    conn.close()
    r = FaceRepository(db)
    old = r.get_person("abc")
    assert old.display_name == "Antigo" and old.unit == "" and old.kind == "morador"
    new = r.add_person("Novo", rand(), unit="Apto 12", kind="visitante")
    assert r.get_person(new.id).unit == "Apto 12"
    assert r.get_person(new.id).kind == "visitante"
    assert [p.unit for p in r.list_persons()] == ["", "Apto 12"]
    r.close()


# ------------------------------------------------------------------ contas (spec 009)
def test_account_fields_roundtrip(repo):
    p = repo.add_person("A", rand(), email="a@exemplo.com", password_hash="hash-a")
    assert p.email == "a@exemplo.com"
    assert repo.get_person(p.id).email == "a@exemplo.com"
    assert repo.get_credentials("a@exemplo.com") == (p.id, "hash-a")
    assert repo.get_password_hash(p.id) == "hash-a"
    assert repo.get_credentials("outro@exemplo.com") is None
    assert repo.email_exists("a@exemplo.com") and not repo.email_exists("outro@exemplo.com")


def test_email_is_unique_but_many_people_can_have_none(repo):
    repo.add_person("A", rand(seed=1), email="a@exemplo.com", password_hash="h")
    with pytest.raises(EmailAlreadyRegistered):
        repo.add_person("B", rand(seed=2), email="a@exemplo.com", password_hash="h")
    assert repo.count_persons() == 1  # o cadastro recusado não deixou nada para trás
    assert repo.count_embeddings() == 3
    repo.add_person("C", rand(seed=3))
    repo.add_person("D", rand(seed=4))  # sem e-mail: vários são permitidos
    assert repo.count_persons() == 3
    assert repo.get_credentials("") is None  # sem e-mail => sem login


def test_refused_signup_leaves_no_orphan_photos(repo, tmp_path):
    repo.add_person("A", rand(seed=1), email="a@exemplo.com", password_hash="h")
    with pytest.raises(EmailAlreadyRegistered):
        repo.add_person(
            "B", rand(seed=2), photos=[b"1", b"2", b"3"], email="a@exemplo.com", password_hash="h"
        )
    assert list((tmp_path / "faces").glob("*")) == []


def test_update_person_and_password(repo):
    a = repo.add_person("A", rand(seed=1), email="a@exemplo.com", password_hash="h1")
    b = repo.add_person("B", rand(seed=2), email="b@exemplo.com", password_hash="h2")
    repo.update_person(a.id, "A Novo", "Apto 9", "visitante", "novo@exemplo.com")
    got = repo.get_person(a.id)
    assert (got.display_name, got.unit, got.kind, got.email) == (
        "A Novo", "Apto 9", "visitante", "novo@exemplo.com",
    )  # fmt: skip
    assert got.n_embeddings == 3  # o rosto não muda
    with pytest.raises(EmailAlreadyRegistered):
        repo.update_person(b.id, "B", "", "morador", "novo@exemplo.com")
    with pytest.raises(PersonNotFound):
        repo.update_person("nao-existe", "X", "", "morador", "x@exemplo.com")
    repo.set_password_hash(a.id, "h3")
    assert repo.get_password_hash(a.id) == "h3"
    with pytest.raises(PersonNotFound):
        repo.set_password_hash("nao-existe", "h")


def test_delete_person_removes_credentials(repo):
    p = repo.add_person("A", rand(), email="a@exemplo.com", password_hash="h")
    repo.delete_person(p.id)
    assert repo.get_credentials("a@exemplo.com") is None
    assert not repo.email_exists("a@exemplo.com")


def test_database_from_before_accounts_is_migrated(tmp_path):
    db = tmp_path / "v1.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE persons (id TEXT PRIMARY KEY, display_name TEXT NOT NULL,"
        " created_at TEXT NOT NULL, unit TEXT NOT NULL DEFAULT '',"
        " kind TEXT NOT NULL DEFAULT 'morador');"
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, person_id TEXT NOT NULL"
        " REFERENCES persons(id) ON DELETE CASCADE, vector BLOB NOT NULL, dim INTEGER NOT NULL,"
        " photo_path TEXT);"
        "INSERT INTO persons (id, display_name, created_at) VALUES ('abc', 'Antigo', '2026-01-01');"
    )
    conn.commit()
    conn.close()
    r = FaceRepository(db)
    old = r.get_person("abc")
    assert old.display_name == "Antigo" and old.email == ""
    assert r.get_credentials("antigo@exemplo.com") is None  # sem conta: não entra
    new = r.add_person("Novo", rand(), email="novo@exemplo.com", password_hash="h")
    assert r.get_credentials("novo@exemplo.com") == (new.id, "h")
    r.close()
