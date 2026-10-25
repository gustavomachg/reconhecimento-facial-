"""Detecção de duplicatas / quase-duplicatas entre conjuntos (hash perceptual).

dHash (difference hash): reduz a imagem para (n+1)×n em tons de cinza e compara cada pixel
com o vizinho da direita -> n² bits. Imagens quase iguais (mesma foto recomprimida,
redimensionada, levemente editada) têm hashes com distância de Hamming pequena.
Usado para garantir que a mesma foto não esteja em treino E validação/teste.

Calibração (faces alinhadas do CelebA, 1500 imagens): com 64 bits (n=8), pessoas DIFERENTES
chegavam a distância 2 — faces alinhadas são parecidas demais nessa resolução e o teste dava
milhares de falsos positivos. Com 256 bits (n=16): pessoas diferentes ≥ 30, cópia real
(redimensionada + JPEG q70) ≤ 9. Por isso o padrão é n=16 e limite 16.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np
from PIL import Image

HASH_SIZE = 16
MAX_DISTANCE = 16


def dhash_bits(image: Image.Image | str, hash_size: int = HASH_SIZE) -> np.ndarray:
    """Hash como vetor de bits empacotado (uint8, hash_size²/8 bytes)."""
    img = image if isinstance(image, Image.Image) else Image.open(image)
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    return np.packbits((px[:, 1:] > px[:, :-1]).flatten())


def dhash(image: Image.Image | str, hash_size: int = HASH_SIZE) -> int:
    """Hash como inteiro (conveniente para comparar poucas imagens)."""
    return int.from_bytes(dhash_bits(image, hash_size).tobytes(), "big")


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distâncias de Hamming entre hashes empacotados (N,B) e (M,B) -> (N, M)."""
    x = np.bitwise_xor(a[:, None, :], b[None, :, :])
    return np.unpackbits(x, axis=-1).sum(-1)


def find_cross_duplicates(
    sets: dict[str, Iterable[str]],
    max_distance: int = MAX_DISTANCE,
    hash_size: int = HASH_SIZE,
    chunk: int = 256,
) -> list[tuple[str, str, str, str, int]]:
    """Retorna (conjuntoA, arquivoA, conjuntoB, arquivoB, distância) para quase-duplicatas
    entre conjuntos DIFERENTES (vetorizado em blocos para escalar a dezenas de milhares)."""
    paths = {name: list(ps) for name, ps in sets.items()}
    hashes = {
        n: np.stack([dhash_bits(p, hash_size) for p in ps])
        if ps
        else np.zeros((0, hash_size**2 // 8), np.uint8)
        for n, ps in paths.items()
    }
    found = []
    for a, b in combinations(paths, 2):
        if not len(hashes[a]) or not len(hashes[b]):
            continue
        for i0 in range(0, len(hashes[a]), chunk):
            d = _hamming_matrix(hashes[a][i0 : i0 + chunk], hashes[b])
            for i, j in zip(*np.nonzero(d <= max_distance), strict=True):
                found.append((a, paths[a][i0 + i], b, paths[b][j], int(d[i, j])))
    return found
