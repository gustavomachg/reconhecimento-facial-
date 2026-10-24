"""API com serviço falso (detector/embedder sintéticos)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from facerec.app.main import create_app  # noqa: E402
from facerec.service import FaceRecognitionService  # noqa: E402
from facerec.storage import FaceRepository  # noqa: E402
from tests.conftest import FakeDetector, FakeEmbedder, make_image  # noqa: E402


@pytest.fixture
def client(tmp_path):
    repo = FaceRepository(tmp_path / "app.db")
    svc = FaceRecognitionService(FakeDetector(), FakeEmbedder(), repo, threshold=0.7)
    yield TestClient(create_app(svc))
    repo.close()


def files(identity, n=2):
    return [
        ("files", (f"{i}.png", make_image(identity, f"face{i}"), "image/png")) for i in range(n)
    ]


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Reconhecedor Facial" in r.text


def test_initial_state(client):
    assert client.get("/api/persons").json() == {
        "persons": [],
        "message": "Nenhuma pessoa cadastrada.",
    }


def test_full_flow(client):
    r = client.post("/api/persons", data={"name": "Pessoa A"}, files=files(1))
    assert r.status_code == 201
    pid = r.json()["id"]

    rec = client.post(
        "/api/recognize", files={"file": ("q.png", make_image(1, "face9"), "image/png")}
    )
    assert rec.json()["recognized"] and rec.json()["name"] == "Pessoa A"

    unk = client.post(
        "/api/recognize", files={"file": ("q.png", make_image(7, "face1"), "image/png")}
    )
    assert unk.json()["message"] == "Pessoa desconhecida"

    assert client.delete(f"/api/persons/{pid}").status_code == 204
    assert client.get("/api/persons").json()["message"] == "Nenhuma pessoa cadastrada."
    assert client.delete(f"/api/persons/{pid}").status_code == 404


def test_domain_errors_are_422(client):
    r = client.post(
        "/api/persons",
        data={"name": "X"},
        files=[("files", ("a.png", make_image(1, "noface"), "image/png"))],
    )
    assert r.status_code == 422 and r.json()["code"] == "no_face"
    client.post("/api/persons", data={"name": "A"}, files=files(1))
    r = client.post(
        "/api/recognize", files={"file": ("q.png", make_image(1, "multi"), "image/png")}
    )
    assert r.status_code == 422 and r.json()["code"] == "multiple_faces"


def test_duplicate_is_422(client):
    client.post("/api/persons", data={"name": "A"}, files=files(1))
    r = client.post("/api/persons", data={"name": "A2"}, files=files(1))
    assert r.status_code == 422 and r.json()["code"] == "duplicate_person"
