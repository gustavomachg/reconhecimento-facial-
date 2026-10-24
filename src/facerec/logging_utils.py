"""Logs estruturados (JSON lines) SEM dados pessoais.

Regra (docs/privacy.md): registrar apenas IDs internos opacos, scores e tempos.
Nunca nomes, imagens ou embeddings. `log_event` recusa chaves sensíveis.
"""

from __future__ import annotations

import json
import logging
import time

FORBIDDEN_KEYS = {"name", "display_name", "nome", "image", "photo", "embedding", "vector"}

_logger = logging.getLogger("facerec")


def configure_logging(level: str = "INFO") -> None:
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)
    _logger.setLevel(level)


def log_event(event: str, **fields) -> dict:
    bad = FORBIDDEN_KEYS.intersection(fields)
    if bad:
        raise ValueError(f"Campos sensíveis não podem ir para o log: {sorted(bad)}")
    record = {"ts": round(time.time(), 3), "event": event, **fields}
    _logger.info(json.dumps(record, ensure_ascii=False, default=str))
    return record
