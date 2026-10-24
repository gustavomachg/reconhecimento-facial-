"""Splits que evitam data leakage (ADR-004).

- Públicos (CelebA/LFW): identidades DISJUNTAS entre treino / validação / teste.
  Se a mesma pessoa aparecesse no treino e no teste, o modelo seria avaliado em rostos
  "já vistos" e a métrica ficaria otimista.
- Privado: split por SESSÃO de captura. Fotos da mesma sessão são quase idênticas
  (mesma luz, roupa, fundo); separá-las aleatoriamente entre treino e teste inflaria o resultado.
- Open-set: algumas identidades privadas ficam FORA da galeria e só aparecem como
  "desconhecidas" na validação/teste.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    path: str
    identity: str
    session: str = "0"


def split_by_identity(
    identities: Iterable[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    ids = sorted(set(identities))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    split = {
        "test": ids[:n_test],
        "val": ids[n_test : n_test + n_val],
        "train": ids[n_test + n_val :],
    }
    assert_disjoint(split)
    return split


def assert_disjoint(groups: dict[str, Sequence[str]]) -> None:
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = set(groups[a]) & set(groups[b])
            if inter:
                raise ValueError(f"Vazamento: {len(inter)} itens em comum entre {a} e {b}")


def filter_records(records: Iterable[Record], identities: Iterable[str]) -> list[Record]:
    keep = set(identities)
    return [r for r in records if r.identity in keep]


def split_private_by_session(
    records: Sequence[Record],
    gallery_sessions: Sequence[str],
    val_sessions: Sequence[str],
    test_sessions: Sequence[str],
    n_unknown_val: int = 1,
    n_unknown_test: int = 1,
    seed: int = 42,
) -> dict[str, list[Record]]:
    """Divide o dataset privado por sessão e reserva identidades "desconhecidas".

    Retorna:
        gallery: fotos de cadastro (sessões de galeria) das identidades conhecidas
        val:     consultas de validação = conhecidos (sessões de val) + desconhecidos-val
        test:    consultas de teste     = conhecidos (sessões de teste) + desconhecidos-test
    Os desconhecidos de val e de teste são pessoas DIFERENTES entre si, para que o threshold
    calibrado na validação seja testado em desconhecidos realmente novos.
    """
    assert_disjoint({"gallery": gallery_sessions, "val": val_sessions, "test": test_sessions})
    ids = sorted({r.identity for r in records})
    if len(ids) < n_unknown_val + n_unknown_test + 2:
        raise ValueError("Poucas identidades para reservar desconhecidos e ainda ter conhecidos")
    rng = random.Random(seed)
    rng.shuffle(ids)
    unk_val = set(ids[:n_unknown_val])
    unk_test = set(ids[n_unknown_val : n_unknown_val + n_unknown_test])
    known = set(ids) - unk_val - unk_test

    def pick(sessions, identities):
        s = set(sessions)
        return [r for r in records if r.session in s and r.identity in identities]

    return {
        "gallery": pick(gallery_sessions, known),
        "val": pick(val_sessions, known) + [r for r in records if r.identity in unk_val],
        "test": pick(test_sessions, known) + [r for r in records if r.identity in unk_test],
        "known_identities": sorted(known),
        "unknown_val": sorted(unk_val),
        "unknown_test": sorted(unk_test),
    }
