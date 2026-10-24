"""Interface comum dos extratores de embedding."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from facerec.matching import l2_normalize


class Embedder(ABC):
    """Recebe faces já alinhadas (uint8, HxWx3, RGB) e devolve embeddings L2-normalizados."""

    name: str = "base"
    input_size: int = 160
    align_mode: str = "crop"
    embedding_dim: int = 512

    @abstractmethod
    def _forward(self, batch: np.ndarray) -> np.ndarray:
        """batch: (N, H, W, 3) uint8 -> (N, D) float32 (sem normalizar)."""

    @abstractmethod
    def num_parameters(self) -> int: ...

    def embed(self, faces: np.ndarray | list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        faces = np.asarray(faces, dtype=np.uint8)
        if faces.ndim == 3:
            faces = faces[None]
        out = [self._forward(faces[i : i + batch_size]) for i in range(0, len(faces), batch_size)]
        return l2_normalize(np.concatenate(out, axis=0))
