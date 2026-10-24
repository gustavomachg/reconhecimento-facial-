"""Fixtures SINTÉTICAS: embeddings aleatórios, imagens geradas e detector/embedder falsos.
Nenhum teste usa rostos reais (docs/privacy.md)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from facerec.detection import DetectedFace
from facerec.exceptions import LowQualityImage, MultipleFacesDetected, NoFaceDetected
from facerec.matching import l2_normalize

DIM = 32


def synthetic_identities(n_ids=5, per_id=6, dim=DIM, noise=0.15, seed=0):
    """Clusters gaussianos na esfera: cada identidade = um centro aleatório + ruído."""
    rng = np.random.default_rng(seed)
    centers = l2_normalize(rng.normal(size=(n_ids, dim)))
    x, y = [], []
    for i, c in enumerate(centers):
        x.append(l2_normalize(c + noise * rng.normal(size=(per_id, dim))))
        y += [f"id{i}"] * per_id
    return np.concatenate(x), y, centers


def make_image(seed: int, kind: str = "face") -> bytes:
    """Imagem sintética. O `kind` é lido pelo detector falso para simular cada caso."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.info["kind"] = kind
    buf = io.BytesIO()
    meta = {"comment": f"{kind}:{seed}".encode()}
    img.save(buf, format="PNG", pnginfo=_pnginfo(meta))
    return buf.getvalue()


def _pnginfo(meta: dict):
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    for k, v in meta.items():
        info.add_text(k, v.decode())
    return info


def parse_image(data: bytes) -> tuple[str, int]:
    kind, seed = Image.open(io.BytesIO(data)).text["comment"].split(":")
    return kind, int(seed)


class FakeDetector:
    """Simula o MTCNN: o tipo da imagem decide o resultado. O recorte "alinhado"
    carrega a identidade sintética no primeiro pixel, lido pelo FakeEmbedder."""

    def detect_single(self, data: bytes) -> DetectedFace:
        kind, seed = parse_image(data)
        if kind == "noface":
            raise NoFaceDetected()
        if kind == "multi":
            raise MultipleFacesDetected()
        if kind == "blurry":
            raise LowQualityImage()
        aligned = np.zeros((8, 8, 3), dtype=np.uint8)
        aligned[0, 0, 0] = seed  # identidade
        aligned[0, 0, 1] = int(kind[4:] or 0) if kind.startswith("face") else 0  # variação
        return DetectedFace(np.array([0, 0, 100, 100.0]), 0.99, None, aligned)


class FakeEmbedder:
    """Identidade k -> centro k + pequena variação determinística."""

    input_size = 8
    align_mode = "crop"

    def __init__(self, n_ids=20, dim=DIM, noise=0.05):
        rng = np.random.default_rng(123)
        self.centers = l2_normalize(rng.normal(size=(n_ids, dim)))
        self.noise = noise
        self.dim = dim

    def embed(self, faces):
        out = []
        for f in faces:
            ident, var = int(f[0, 0, 0]), int(f[0, 0, 1])
            rng = np.random.default_rng(1000 * ident + var)
            out.append(self.centers[ident] + self.noise * rng.normal(size=self.dim))
        return l2_normalize(np.stack(out))

    def num_parameters(self):
        return 0


@pytest.fixture
def identities():
    return synthetic_identities()
