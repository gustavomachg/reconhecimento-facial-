import sqlite3

import numpy as np
import pytest

from facerec.exceptions import PersonNotFound
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
