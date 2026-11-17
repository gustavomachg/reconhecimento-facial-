"""Autenticação local da aplicação (ADR-005): senha, sessão e limite de tentativas.

Só biblioteca padrão — nada de serviço externo:
- senhas: PBKDF2-HMAC-SHA256 com sal aleatório; só o hash é guardado (nunca a senha);
- sessões: token aleatório guardado NA MEMÓRIA do servidor (some ao reiniciar) e entregue em
  cookie HttpOnly + SameSite=Strict;
- tentativas de login: bloqueio temporário após falhas seguidas no mesmo e-mail.

E-mail e senha são dados pessoais: nunca vão para logs (`logging_utils.FORBIDDEN_KEYS`).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time

from facerec.exceptions import TooManyAttempts

PBKDF2_ITERATIONS = 600_000  # recomendação OWASP (2023) para PBKDF2-HMAC-SHA256
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 128  # limite superior: evita gastar CPU com senhas gigantes
MAX_EMAIL_LEN = 254
_ALGORITHM = "pbkdf2_sha256"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ------------------------------------------------------------------ e-mail e senha
def normalize_email(email: str | None) -> str:
    return (email or "").strip().lower()


def validate_email(email: str | None) -> str:
    """Devolve o e-mail normalizado ou levanta ValueError com mensagem para o usuário."""
    value = normalize_email(email)
    if not value or len(value) > MAX_EMAIL_LEN or not _EMAIL_RE.match(value):
        raise ValueError("Informe um e-mail válido.")
    return value


def validate_password(password: str | None) -> None:
    if not password or len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"A senha precisa ter pelo menos {MIN_PASSWORD_LEN} caracteres.")
    if len(password) > MAX_PASSWORD_LEN:
        raise ValueError(f"A senha pode ter no máximo {MAX_PASSWORD_LEN} caracteres.")


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    """`pbkdf2_sha256$<iterações>$<sal hex>$<hash hex>` — as iterações ficam no próprio hash,
    então dá para aumentá-las no futuro sem invalidar senhas antigas."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Compara em tempo constante. Nunca levanta exceção: hash malformado => False."""
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != _ALGORITHM or len(password) > MAX_PASSWORD_LEN:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


# ------------------------------------------------------------------ sessões
class SessionStore:
    """Sessões em memória: token -> (person_id, expiração). Seguro para várias threads."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._items: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def create(self, person_id: str, ttl_seconds: float) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._items[token] = (person_id, self._clock() + ttl_seconds)
        return token

    def get(self, token: str | None) -> str | None:
        if not token:
            return None
        with self._lock:
            item = self._items.get(token)
            if item is None:
                return None
            person_id, expires = item
            if expires <= self._clock():
                del self._items[token]
                return None
            return person_id

    def revoke(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._items.pop(token, None)

    def revoke_person(self, person_id: str) -> None:
        """Encerra todas as sessões de uma pessoa (ex.: ao excluir a conta)."""
        with self._lock:
            for token in [t for t, (pid, _) in self._items.items() if pid == person_id]:
                del self._items[token]


# ------------------------------------------------------------------ tentativas de login
class LoginThrottle:
    """Depois de `max_failures` erros seguidos num e-mail, bloqueia novas tentativas por
    `lockout_seconds`. Dificulta adivinhar senhas por força bruta."""

    def __init__(self, max_failures: int = 5, lockout_seconds: float = 60, clock=time.monotonic):
        self.max_failures = max_failures
        self.lockout_seconds = lockout_seconds
        self._clock = clock
        self._state: dict[str, tuple[int, float]] = {}  # chave -> (falhas, bloqueado até)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        with self._lock:
            _, locked_until = self._state.get(key, (0, 0.0))
            remaining = locked_until - self._clock()
        if remaining > 0:
            raise TooManyAttempts(
                f"Muitas tentativas de login. Tente de novo em {int(remaining) + 1} s."
            )

    def failure(self, key: str) -> None:
        with self._lock:
            count, locked_until = self._state.get(key, (0, 0.0))
            count += 1
            if count >= self.max_failures:
                count, locked_until = 0, self._clock() + self.lockout_seconds
            self._state[key] = (count, locked_until)

    def success(self, key: str) -> None:
        with self._lock:
            self._state.pop(key, None)
