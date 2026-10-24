"""Amostrador P×K: cada batch tem P identidades com K imagens cada.

A Triplet Loss com batch-hard mining precisa que cada âncora tenha positivos (mesma
pessoa) e negativos (outras pessoas) DENTRO do batch; amostragem aleatória pura quase
nunca garante isso quando há milhares de identidades.
"""

from __future__ import annotations

import random
from collections import defaultdict


class PKBatchSampler:
    def __init__(self, labels: list[int], p: int = 16, k: int = 4, seed: int = 42):
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.by_label[lbl].append(idx)
        self.valid = [lbl for lbl, idxs in self.by_label.items() if len(idxs) >= 2]
        if len(self.valid) < 2:
            raise ValueError("PKBatchSampler precisa de >=2 identidades com >=2 imagens")
        self.p, self.k = min(p, len(self.valid)), k
        self.rng = random.Random(seed)
        self.n_batches = max(1, len(labels) // (self.p * self.k))

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for lbl in self.rng.sample(self.valid, self.p):
                idxs = self.by_label[lbl]
                if len(idxs) >= self.k:
                    batch += self.rng.sample(idxs, self.k)
                else:
                    batch += self.rng.choices(idxs, k=self.k)
            yield batch
