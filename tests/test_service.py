import json
import logging

import pytest

from facerec.exceptions import (
    DuplicatePerson,
    InconsistentEnrollment,
    LowQualityImage,
    MultipleFacesDetected,
    NoFaceDetected,
    PersonNotFound,
)
from facerec.logging_utils import log_event
from facerec.service import NO_PERSONS_MESSAGE, FaceRecognitionService
from facerec.storage import FaceRepository
from tests.conftest import FakeDetector, FakeEmbedder, make_image


@pytest.fixture
def service(tmp_path):
    repo = FaceRepository(tmp_path / "app.db", photos_dir=tmp_path / "faces")
    svc = FaceRecognitionService(
        FakeDetector(),
        FakeEmbedder(),
        repo,
        threshold=0.7,
        matcher_name="prototype",
        store_photos=True,
    )
    yield svc
    repo.close()


def imgs(identity, n=3):
    return [make_image(identity, f"face{v}") for v in range(n)]


def test_initial_state_message(service):
    assert service.list_persons() == []
    r = service.recognize(make_image(1, "face9"))
    assert not r.recognized and r.message == NO_PERSONS_MESSAGE


def test_enroll_and_recognize_known(service):
    p = service.enroll("Pessoa A", imgs(1))
    r = service.recognize(make_image(1, "face7"))  # foto nova da mesma identidade
    assert r.recognized and r.person_id == p.id
    assert r.message.startswith("Pessoa reconhecida: Pessoa A")
    assert r.score >= 0.7


def test_unknown_person(service):
    service.enroll("Pessoa A", imgs(1))
    service.enroll("Pessoa B", imgs(2))
    r = service.recognize(make_image(5, "face1"))
    assert not r.recognized and r.message == "Pessoa desconhecida"


@pytest.mark.parametrize(
    "kind,exc",
    [("noface", NoFaceDetected), ("multi", MultipleFacesDetected), ("blurry", LowQualityImage)],
)
def test_enroll_errors(service, kind, exc):
    with pytest.raises(exc):
        service.enroll("X", [make_image(1, kind)])
    assert service.list_persons() == []  # nada é salvo em caso de erro


@pytest.mark.parametrize("kind,exc", [("noface", NoFaceDetected), ("multi", MultipleFacesDetected)])
def test_recognize_errors(service, kind, exc):
    service.enroll("A", imgs(1))
    with pytest.raises(exc):
        service.recognize(make_image(1, kind))


def test_duplicate_enrollment_rejected(service):
    service.enroll("Pessoa A", imgs(1))
    with pytest.raises(DuplicatePerson):
        service.enroll("Pessoa A de novo", [make_image(1, "face5")])
    assert len(service.list_persons()) == 1


def test_inconsistent_enrollment_rejected(service):
    with pytest.raises(InconsistentEnrollment):
        service.enroll("Mistura", [make_image(1, "face0"), make_image(2, "face0")])


def test_empty_name_or_images(service):
    with pytest.raises(ValueError):
        service.enroll("  ", imgs(1))
    with pytest.raises(ValueError):
        service.enroll("A", [])


def test_delete_removes_everything_and_person_becomes_unknown(service, tmp_path):
    a = service.enroll("Pessoa A", imgs(1))
    service.enroll("Pessoa B", imgs(2))
    assert service.recognize(make_image(1, "face8")).recognized
    assert (tmp_path / "faces" / a.id).exists()

    service.delete(a.id)

    assert service.repo.get_person(a.id) is None
    assert service.repo.count_embeddings(a.id) == 0
    assert not (tmp_path / "faces" / a.id).exists()
    r = service.recognize(make_image(1, "face8"))
    assert not r.recognized  # o cache do matcher foi invalidado
    assert [p.display_name for p in service.list_persons()] == ["Pessoa B"]


def test_delete_unknown_id(service):
    with pytest.raises(PersonNotFound):
        service.delete("inexistente")


def test_logs_have_no_names(service, caplog):
    caplog.set_level(logging.INFO, logger="facerec")
    p = service.enroll("Nome Secreto", imgs(3))
    service.recognize(make_image(3, "face4"))
    service.delete(p.id)
    text = caplog.text
    assert "Nome Secreto" not in text
    assert p.id in text
    events = [json.loads(r.message)["event"] for r in caplog.records]
    assert {"enroll_ok", "recognize_ok", "delete_ok"} <= set(events)


def test_log_event_refuses_sensitive_keys():
    with pytest.raises(ValueError):
        log_event("x", name="Fulano")
