"""Comparadores (matchers) de embeddings para identificação em conjunto aberto.

Todos seguem a mesma interface:
    matcher.fit(gallery_embeddings, gallery_labels)
    label, score = matcher.identify(query_embedding, threshold)

`score` é sempre "maior = mais parecido", então a regra de decisão é a mesma para todos:
    score >= threshold -> pessoa reconhecida ; caso contrário -> desconhecida.
A escala do score muda de um matcher para outro, por isso cada um tem o SEU threshold,
calibrado na validação (E06/E07).

Observação importante para a análise: com embeddings L2-normalizados,
    ||a - b||^2 = 2 - 2 cos(a, b)
ou seja, distância euclidiana e cosseno produzem o MESMO ranking; só muda a escala do score.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from facerec.metrics import UNKNOWN


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similaridade cosseno entre linhas de `a` (N,D) e de `b` (M,D) -> (N,M)."""
    return l2_normalize(np.atleast_2d(a)) @ l2_normalize(np.atleast_2d(b)).T


class Matcher:
    name = "base"

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> Matcher:
        self.embeddings = l2_normalize(np.atleast_2d(embeddings))
        self.labels = np.asarray([str(lbl) for lbl in labels], dtype=object)
        if len(self.embeddings) != len(self.labels):
            raise ValueError("embeddings e labels com tamanhos diferentes")
        self._fit()
        return self

    def _fit(self) -> None:  # pragma: no cover - sobrescrito quando necessário
        pass

    @property
    def is_empty(self) -> bool:
        return not hasattr(self, "labels") or len(self.labels) == 0

    def scores(self, query: np.ndarray) -> dict[str, float]:
        """Score por identidade da galeria."""
        raise NotImplementedError

    def best(self, query: np.ndarray) -> tuple[str | None, float]:
        if self.is_empty:
            return None, float("-inf")
        s = self.scores(query)
        label = max(s, key=s.get)
        return label, float(s[label])

    def identify(self, query: np.ndarray, threshold: float) -> tuple[str, float]:
        label, score = self.best(query)
        if label is None or score < threshold:
            return UNKNOWN, score
        return label, score


class CosineMatcher(Matcher):
    """Vizinho mais próximo por cosseno; score da pessoa = maior similaridade entre suas fotos."""

    name = "cosine"

    def scores(self, query):
        sims = cosine_similarity(query, self.embeddings)[0]
        out: dict[str, float] = defaultdict(lambda: -1.0)
        for lbl, s in zip(self.labels, sims, strict=True):
            out[lbl] = max(out[lbl], float(s))
        return dict(out)


class EuclideanMatcher(Matcher):
    """Vizinho mais próximo por distância euclidiana; score = -distância (maior = melhor)."""

    name = "euclidean"

    def scores(self, query):
        q = l2_normalize(np.atleast_2d(query))
        dists = np.linalg.norm(self.embeddings - q, axis=1)
        out: dict[str, float] = defaultdict(lambda: float("-inf"))
        for lbl, d in zip(self.labels, dists, strict=True):
            out[lbl] = max(out[lbl], -float(d))
        return dict(out)


class PrototypeMatcher(Matcher):
    """Cada pessoa vira um protótipo (média normalizada dos seus embeddings); score = cosseno."""

    name = "prototype"

    def _fit(self):
        self.classes = sorted(set(self.labels))
        self.prototypes = l2_normalize(
            np.stack([self.embeddings[self.labels == c].mean(axis=0) for c in self.classes])
        )

    def scores(self, query):
        sims = cosine_similarity(query, self.prototypes)[0]
        return {c: float(s) for c, s in zip(self.classes, sims, strict=True)}


class KNNMatcher(Matcher):
    """k vizinhos mais próximos (cosseno). Vence a pessoa com mais votos; o score é a
    similaridade média dos vizinhos dessa pessoa (para permitir rejeitar desconhecidos)."""

    name = "knn"

    def __init__(self, k: int = 3):
        self.k = k

    def scores(self, query):
        sims = cosine_similarity(query, self.embeddings)[0]
        k = min(self.k, len(sims))
        idx = np.argsort(-sims)[:k]
        votes: dict[str, list[float]] = defaultdict(list)
        for i in idx:
            votes[self.labels[i]].append(float(sims[i]))
        # desempate por similaridade média; o score final penaliza poucos votos
        return {lbl: float(np.sum(v) / k) for lbl, v in votes.items()}


class SVMMatcher(Matcher):
    """SVM linear one-vs-rest sobre os embeddings; score = probabilidade calibrada da classe.

    Com uma única pessoa cadastrada não há como treinar um classificador: cai para cosseno.
    """

    name = "svm"

    def __init__(self, C: float = 1.0):
        self.C = C

    def _fit(self):
        from sklearn.svm import SVC

        self.classes = sorted(set(self.labels))
        self._fallback = None
        counts = [int((self.labels == c).sum()) for c in self.classes]
        if len(self.classes) < 2 or min(counts) < 2:
            self._fallback = CosineMatcher().fit(self.embeddings, list(self.labels))
            return
        self.clf = SVC(kernel="linear", C=self.C, probability=True, random_state=0)
        self.clf.fit(self.embeddings, self.labels)

    def scores(self, query):
        if self._fallback is not None:
            return self._fallback.scores(query)
        proba = self.clf.predict_proba(l2_normalize(np.atleast_2d(query)))[0]
        return {str(c): float(p) for c, p in zip(self.clf.classes_, proba, strict=True)}


MATCHERS = {
    "cosine": CosineMatcher,
    "euclidean": EuclideanMatcher,
    "prototype": PrototypeMatcher,
    "knn": KNNMatcher,
    "svm": SVMMatcher,
}


def build_matcher(name: str, **kwargs) -> Matcher:
    if name not in MATCHERS:
        raise ValueError(f"Matcher desconhecido: {name}. Opções: {sorted(MATCHERS)}")
    return MATCHERS[name](**kwargs)
