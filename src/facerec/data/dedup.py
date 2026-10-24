"""Detecção de duplicatas / quase-duplicatas entre conjuntos (hash perceptual).

dHash (difference hash): reduz a imagem para 9x8 em tons de cinza e compara cada pixel
com o vizinho da direita -> 64 bits. Imagens quase iguais (mesma foto recomprimida,
redimensionada, levemente editada) têm hashes com distância de Hamming pequena.
Usado para garantir que a mesma foto não esteja em treino E validação/teste.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np
from PIL import Image


def dhash(image: Image.Image | str, hash_size: int = 8) -> int:
    img = image if isinstance(image, Image.Image) else Image.open(image)
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distâncias de Hamming entre dois vetores de hashes uint64 -> (len(a), len(b))."""
    x = np.bitwise_xor.outer(a, b)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def find_cross_duplicates(
    sets: dict[str, Iterable[str]], max_distance: int = 4, chunk: int = 2048
) -> list[tuple[str, str, str, str, int]]:
    """Retorna (conjuntoA, arquivoA, conjuntoB, arquivoB, distância) para quase-duplicatas
    entre conjuntos DIFERENTES (vetorizado em blocos para escalar a dezenas de milhares)."""
    paths = {name: list(ps) for name, ps in sets.items()}
    hashes = {n: np.array([dhash(p) for p in ps], dtype=np.uint64) for n, ps in paths.items()}
    found = []
    for a, b in combinations(paths, 2):
        for i0 in range(0, len(hashes[a]), chunk):
            d = _hamming_matrix(hashes[a][i0 : i0 + chunk], hashes[b])
            for i, j in zip(*np.nonzero(d <= max_distance), strict=True):
                found.append((a, paths[a][i0 + i], b, paths[b][j], int(d[i, j])))
    return found
