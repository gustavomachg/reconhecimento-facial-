"""Autenticação (ADR-005): hash de senha, sessões e limite de tentativas."""

import pytest

from facerec.auth import (
    LoginThrottle,
    SessionStore,
    hash_password,
    normalize_email,
    validate_email,
    validate_password,
    verify_password,
)
from facerec.exceptions import TooManyAttempts

FAST = 1000  # iterações baixas só nos testes


def test_hash_roundtrip_and_format():
    h = hash_password("senha-forte-1", FAST)
    assert h.startswith("pbkdf2_sha256$1000$")
    assert "senha-forte-1" not in h
    assert verify_password("senha-forte-1", h)
    assert not verify_password("senha-forte-2", h)


def test_same_password_gets_different_salt():
    assert hash_password("mesma-senha", FAST) != hash_password("mesma-senha", FAST)


@pytest.mark.parametrize("stored", ["", "lixo", "a$b$c$d", "pbkdf2_sha256$x$zz$zz", "md5$1$00$00"])
def test_verify_never_raises_on_malformed_hash(stored):
    assert verify_password("qualquer", stored) is False


def test_verify_rejects_huge_password_without_hashing():
    assert verify_password("x" * 10_000, hash_password("abc12345", FAST)) is False


def test_email_normalization_and_validation():
    assert normalize_email("  Maria@Exemplo.COM ") == "maria@exemplo.com"
    assert validate_email(" Maria@Exemplo.com ") == "maria@exemplo.com"
    for bad in ["", None, "sem-arroba", "a@b", "a b@c.com", "@x.com", "a@" + "x" * 260 + ".com"]:
        with pytest.raises(ValueError):
            validate_email(bad)


def test_password_rules():
    validate_password("12345678")
    for bad in [None, "", "1234567", "x" * 129]:
        with pytest.raises(ValueError):
            validate_password(bad)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_session_create_get_revoke():
    s = SessionStore()
    token = s.create("p1", 60)
    assert len(token) >= 32 and s.get(token) == "p1"
    assert s.get("outro") is None and s.get(None) is None and s.get("") is None
    s.revoke(token)
    assert s.get(token) is None


def test_session_expires():
    clock = Clock()
    s = SessionStore(clock)
    token = s.create("p1", 60)
    clock.t += 59
    assert s.get(token) == "p1"
    clock.t += 2
    assert s.get(token) is None


def test_revoke_person_ends_all_their_sessions_only():
    s = SessionStore()
    a1, a2, b = s.create("a", 60), s.create("a", 60), s.create("b", 60)
    s.revoke_person("a")
    assert s.get(a1) is None and s.get(a2) is None and s.get(b) == "b"


def test_throttle_locks_after_max_failures_then_unlocks():
    clock = Clock()
    t = LoginThrottle(max_failures=3, lockout_seconds=60, clock=clock)
    for _ in range(2):
        t.failure("a@x.com")
        t.check("a@x.com")  # ainda não bloqueou
    t.failure("a@x.com")
    with pytest.raises(TooManyAttempts):
        t.check("a@x.com")
    t.check("b@x.com")  # outro e-mail não é afetado
    clock.t += 61
    t.check("a@x.com")  # bloqueio acabou


def test_throttle_success_resets_counter():
    t = LoginThrottle(max_failures=3, lockout_seconds=60)
    t.failure("a@x.com")
    t.failure("a@x.com")
    t.success("a@x.com")
    t.failure("a@x.com")
    t.failure("a@x.com")
    t.check("a@x.com")  # só 2 falhas desde o último sucesso
