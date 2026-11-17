import json
import logging

import pytest

from facerec.exceptions import (
    DuplicatePerson,
    EmailAlreadyRegistered,
    InconsistentEnrollment,
    InvalidCredentials,
    LowQualityImage,
    MultipleFacesDetected,
    NoFaceDetected,
    PersonNotFound,
    TooManyAttempts,
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
        password_iterations=1000,  # rápido nos testes
        max_failed_logins=3,
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


# ------------------------------------------------------------------ spec 007 (vídeo)
def test_recognition_returns_face_box(service):
    """A caixa da face vem no resultado para a interface desenhar sobre o vídeo."""
    assert service.recognize(make_image(1, "face9")).box is None  # galeria vazia
    service.enroll("Pessoa A", imgs(1))
    known = service.recognize(make_image(1, "face7"))
    unknown = service.recognize(make_image(5, "face1"))
    for r in (known, unknown):
        assert r.box == [0.0, 0.0, 100.0, 100.0]  # caixa do FakeDetector


def test_unit_in_recognition_result_and_not_in_logs(service, caplog):
    caplog.set_level(logging.INFO, logger="facerec")
    service.enroll("Morador A", imgs(1), unit="Apto 101")
    known = service.recognize(make_image(1, "face7"))
    unknown = service.recognize(make_image(5, "face1"))
    assert known.unit == "Apto 101" and unknown.unit is None
    assert "Apto 101" not in caplog.text and "Morador A" not in caplog.text


def test_name_and_unit_length_limits(service):
    with pytest.raises(ValueError):
        service.enroll("n" * 101, imgs(1))
    with pytest.raises(ValueError):
        service.enroll("A", imgs(1), unit="u" * 31)
    assert service.list_persons() == []


# ------------------------------------------------------------------ tipo e porta (spec 008)
def door_events(caplog):
    return [json.loads(r.message) for r in caplog.records if '"door_open"' in r.message]


def test_face_recognition_opens_door_and_logs_without_name(service, caplog):
    caplog.set_level(logging.INFO, logger="facerec")
    p = service.enroll("Visitante B", imgs(2), kind="visitante")
    r = service.recognize(make_image(2, "face7"))
    assert r.recognized and r.kind == "visitante"
    events = door_events(caplog)
    assert len(events) == 1
    assert events[0]["method"] == "face" and events[0]["person_id"] == p.id
    assert "Visitante B" not in caplog.text


def test_unknown_face_does_not_open_door(service, caplog):
    caplog.set_level(logging.INFO, logger="facerec")
    service.enroll("Morador A", imgs(1))
    r = service.recognize(make_image(5, "face1"))
    assert not r.recognized and r.kind is None
    assert door_events(caplog) == []


def test_manual_open_is_logged(service, caplog):
    caplog.set_level(logging.INFO, logger="facerec")
    assert service.open_door("manual") == service.door_open_seconds
    events = door_events(caplog)
    assert len(events) == 1 and events[0]["method"] == "manual" and "person_id" not in events[0]
    with pytest.raises(ValueError):
        service.open_door("controle-remoto")


def test_invalid_kind_rejected(service):
    with pytest.raises(ValueError):
        service.enroll("A", imgs(1), kind="porteiro")
    assert service.list_persons() == []


# ------------------------------------------------------------------ contas e login (spec 009)
PW = "senha-forte-1"


def account(service, identity=1, email="a@exemplo.com", name="Morador A", **kw):
    return service.enroll(name, imgs(identity), email=email, password=PW, **kw)


def test_enroll_with_account_stores_hash_not_password(service):
    p = account(service, email="  Maria@Exemplo.com ")
    assert p.email == "maria@exemplo.com"
    stored = service.repo.get_password_hash(p.id)
    assert stored.startswith("pbkdf2_sha256$") and PW not in stored


def test_enroll_account_validation(service):
    with pytest.raises(ValueError):
        service.enroll("A", imgs(1), email="sem-arroba", password=PW)
    with pytest.raises(ValueError):
        service.enroll("A", imgs(1), email="a@exemplo.com", password="curta")
    with pytest.raises(ValueError):
        service.enroll("A", imgs(1), email="a@exemplo.com")  # e-mail sem senha
    assert service.list_persons() == []


def test_duplicate_email_rejected_before_any_face_work(service):
    account(service)
    with pytest.raises(EmailAlreadyRegistered):
        account(service, identity=2, email="A@exemplo.com", name="Outro")
    assert len(service.list_persons()) == 1


def test_duplicate_face_message_does_not_name_the_other_person(service):
    account(service, name="Fulano Reservado")
    with pytest.raises(DuplicatePerson) as exc:
        account(service, identity=1, email="b@exemplo.com", name="Impostor")
    assert "Fulano" not in str(exc.value)


def test_authenticate_ok_and_failures(service):
    p = account(service)
    assert service.authenticate(" A@Exemplo.com ", PW).id == p.id
    for email, pw in [
        ("a@exemplo.com", "errada-123"),
        ("x@exemplo.com", PW),
        ("", ""),
        (None, None),
    ]:
        with pytest.raises(InvalidCredentials):
            service.authenticate(email, pw)


def test_person_without_account_cannot_log_in(service):
    service.enroll("Legado", imgs(3))  # cadastro antigo/importado, sem e-mail e senha
    with pytest.raises(InvalidCredentials):
        service.authenticate("", "")


def test_authenticate_throttles_after_max_failures(service):
    account(service)
    for _ in range(3):
        with pytest.raises(InvalidCredentials):
            service.authenticate("a@exemplo.com", "errada-123")
    with pytest.raises(TooManyAttempts):
        service.authenticate("a@exemplo.com", PW)


def test_logs_never_contain_email_password_or_name(service, caplog):
    caplog.set_level(logging.INFO, logger="facerec")
    p = account(service, email="segredo@exemplo.com", name="Nome Secreto", unit="Apto 7")
    with pytest.raises(InvalidCredentials):
        service.authenticate("segredo@exemplo.com", "errada-123")
    service.authenticate("segredo@exemplo.com", PW)
    service.verify_face(p.id, make_image(1, "face9"))
    service.change_password(p.id, PW, "outra-senha-9")
    for secret in ("segredo@exemplo.com", "Nome Secreto", "Apto 7", PW, "outra-senha-9", "errada"):
        assert secret not in caplog.text


def test_log_event_rejects_email_and_password_fields():
    for field in ("email", "password", "password_hash", "senha"):
        with pytest.raises(ValueError):
            log_event("x", **{field: "valor"})


def test_verify_face_match_other_and_unknown(service, caplog):
    caplog.set_level(logging.INFO, logger="facerec")
    a = account(service, identity=1)
    b = account(service, identity=2, email="b@exemplo.com", name="Morador B")
    ok = service.verify_face(a.id, make_image(1, "face9"))
    assert ok.matched and ok.reason == "match" and ok.door_open and ok.door_seconds > 0
    other = service.verify_face(a.id, make_image(2, "face9"))  # rosto de B na conta de A
    assert not other.matched and other.reason == "other_person" and not other.door_open
    assert "Morador B" not in other.message and b.id not in other.message
    unknown = service.verify_face(a.id, make_image(9, "face1"))
    assert not unknown.matched and unknown.reason == "unknown" and not unknown.door_open
    opens = [r for r in caplog.records if '"door_open"' in r.message]
    assert len(opens) == 1  # só o caso "match" abriu a porta


def test_verify_face_errors_propagate(service):
    a = account(service)
    for kind, exc in (("noface", NoFaceDetected), ("multi", MultipleFacesDetected)):
        with pytest.raises(exc):
            service.verify_face(a.id, make_image(1, kind))


def test_update_profile_and_email_change_rules(service):
    a = account(service)
    account(service, identity=2, email="b@exemplo.com", name="B")
    u = service.update_profile(a.id, "Novo", "Apto 3", "visitante", "a@exemplo.com")
    assert (u.display_name, u.unit, u.kind, u.email) == ("Novo", "Apto 3", "visitante", a.email)
    with pytest.raises(ValueError, match="Senha atual"):
        service.update_profile(a.id, "Novo", "", "morador", "novo@exemplo.com")
    with pytest.raises(EmailAlreadyRegistered):
        service.update_profile(a.id, "Novo", "", "morador", "b@exemplo.com", PW)
    u = service.update_profile(a.id, "Novo", "", "morador", "novo@exemplo.com", PW)
    assert u.email == "novo@exemplo.com"
    assert service.authenticate("novo@exemplo.com", PW).id == a.id
    with pytest.raises(PersonNotFound):
        service.update_profile("nao-existe", "X", "", "morador", "x@exemplo.com", PW)


def test_update_profile_keeps_the_face(service):
    a = account(service, identity=1)
    service.update_profile(a.id, "Outro Nome", "", "morador", "a@exemplo.com")
    assert service.verify_face(a.id, make_image(1, "face9")).matched


def test_change_password(service):
    a = account(service)
    with pytest.raises(ValueError, match="Senha atual"):
        service.change_password(a.id, "errada-123", "outra-senha-9")
    with pytest.raises(ValueError):
        service.change_password(a.id, PW, "123")
    service.change_password(a.id, PW, "outra-senha-9")
    with pytest.raises(InvalidCredentials):
        service.authenticate("a@exemplo.com", PW)
    assert service.authenticate("a@exemplo.com", "outra-senha-9").id == a.id


def test_deleting_account_frees_email_and_face(service):
    a = account(service)
    service.delete(a.id)
    with pytest.raises(InvalidCredentials):
        service.authenticate("a@exemplo.com", PW)
    account(service)  # mesmo e-mail e mesmo rosto podem voltar
