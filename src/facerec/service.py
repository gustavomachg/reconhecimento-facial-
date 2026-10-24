"""Casos de uso da aplicação: cadastrar, reconhecer, listar e excluir pessoas.

O serviço só conhece INTERFACES (detector, embedder, repositório, matcher): por isso os
testes usam versões falsas/sintéticas de cada peça, sem rostos reais e sem GPU.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np

from facerec.exceptions import (
    DuplicatePerson,
    FaceRecError,
    InconsistentEnrollment,
    PersonNotFound,
)
from facerec.logging_utils import log_event
from facerec.matching import Matcher, build_matcher, cosine_similarity
from facerec.metrics import UNKNOWN
from facerec.storage import FaceRepository, Person

NO_PERSONS_MESSAGE = "Nenhuma pessoa cadastrada."


@dataclass
class RecognitionResult:
    recognized: bool
    person_id: str | None
    display_name: str | None
    score: float | None
    threshold: float
    message: str


class FaceRecognitionService:
    def __init__(
        self,
        detector,
        embedder,
        repository: FaceRepository,
        threshold: float,
        matcher_name: str = "cosine",
        matcher_kwargs: dict | None = None,
        duplicate_threshold: float = 0.75,
        consistency_threshold: float = 0.40,
        store_photos: bool = False,
        max_enroll_images: int = 10,
    ):
        self.detector = detector
        self.embedder = embedder
        self.repo = repository
        self.threshold = float(threshold)
        self.matcher_name = matcher_name
        self.matcher_kwargs = matcher_kwargs or {}
        self.duplicate_threshold = duplicate_threshold
        self.consistency_threshold = consistency_threshold
        self.store_photos = store_photos
        self.max_enroll_images = max_enroll_images
        self._matcher: Matcher | None = None

    # ------------------------------------------------------------------ internos
    def _face_embeddings(self, images: list[bytes]) -> tuple[np.ndarray, list[np.ndarray]]:
        aligned = [self.detector.detect_single(img).aligned for img in images]
        return self.embedder.embed(aligned), aligned

    def _get_matcher(self) -> Matcher | None:
        if self._matcher is None:
            vecs, ids = self.repo.all_embeddings()
            if not ids:
                return None
            self._matcher = build_matcher(self.matcher_name, **self.matcher_kwargs).fit(vecs, ids)
        return self._matcher

    def _invalidate(self) -> None:
        self._matcher = None

    @staticmethod
    def _to_jpeg(face: np.ndarray) -> bytes:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.asarray(face, dtype=np.uint8)).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    # ------------------------------------------------------------------ casos de uso
    def enroll(self, display_name: str, images: list[bytes]) -> Person:
        t0 = time.perf_counter()
        display_name = (display_name or "").strip()
        if not display_name:
            raise ValueError("Informe o nome da pessoa.")
        if not images:
            raise ValueError("Envie ao menos uma imagem.")
        if len(images) > self.max_enroll_images:
            raise ValueError(f"Envie no máximo {self.max_enroll_images} imagens.")
        try:
            emb, aligned = self._face_embeddings(images)
            # 1) todas as imagens do cadastro precisam ser da mesma pessoa:
            #    a MENOR similaridade entre dois pares de fotos do cadastro deve ser alta
            if len(emb) > 1 and float((emb @ emb.T).min()) < self.consistency_threshold:
                raise InconsistentEnrollment()
            # 2) cadastro duplicado: alguma imagem muito parecida com alguém já cadastrado
            vecs, ids = self.repo.all_embeddings()
            if ids:
                sims = cosine_similarity(emb, vecs)
                i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
                if sims[i, j] >= self.duplicate_threshold:
                    existing = self.repo.get_person(ids[j])
                    log_event(
                        "enroll_rejected",
                        reason="duplicate",
                        existing_person_id=ids[j],
                        score=round(float(sims[i, j]), 4),
                    )
                    raise DuplicatePerson(
                        f"Face muito parecida com '{existing.display_name}' já cadastrada "
                        f"(similaridade {sims[i, j]:.2f})."
                    )
        except FaceRecError as exc:
            if not isinstance(exc, DuplicatePerson):
                log_event("enroll_rejected", reason=exc.code)
            raise
        photos = [self._to_jpeg(a) for a in aligned] if self.store_photos else None
        person = self.repo.add_person(display_name, emb, photos)
        self._invalidate()
        log_event(
            "enroll_ok",
            person_id=person.id,
            n_images=len(images),
            ms=round(1000 * (time.perf_counter() - t0), 1),
        )
        return person

    def recognize(self, image: bytes) -> RecognitionResult:
        t0 = time.perf_counter()
        matcher = self._get_matcher()
        if matcher is None:
            return RecognitionResult(False, None, None, None, self.threshold, NO_PERSONS_MESSAGE)
        try:
            emb, _ = self._face_embeddings([image])
        except FaceRecError as exc:
            log_event("recognize_error", reason=exc.code)
            raise
        label, score = matcher.identify(emb[0], self.threshold)
        ms = round(1000 * (time.perf_counter() - t0), 1)
        if label == UNKNOWN:
            log_event("recognize_unknown", score=round(score, 4), ms=ms)
            return RecognitionResult(
                False, None, None, score, self.threshold, "Pessoa desconhecida"
            )
        person = self.repo.get_person(label)
        log_event("recognize_ok", person_id=label, score=round(score, 4), ms=ms)
        return RecognitionResult(
            True,
            label,
            person.display_name,
            score,
            self.threshold,
            f"Pessoa reconhecida: {person.display_name} ({score:.2f})",
        )

    def delete(self, person_id: str) -> None:
        try:
            self.repo.delete_person(person_id)
        except PersonNotFound:
            log_event("delete_not_found", person_id=person_id)
            raise
        self._invalidate()
        log_event("delete_ok", person_id=person_id)

    def list_persons(self) -> list[Person]:
        return self.repo.list_persons()


class ModelMismatchError(RuntimeError):
    """O cadastro foi feito com outro modelo: embeddings de modelos diferentes não são
    comparáveis (cada rede tem o seu próprio espaço de embeddings)."""


def model_fingerprint(model_cfg: dict) -> str:
    keys = ("name", "path", "checkpoint")
    return "|".join(f"{k}={model_cfg[k]}" for k in keys if model_cfg.get(k))


def check_model_compatibility(repo: FaceRepository, model_cfg: dict) -> None:
    """Grava o modelo no primeiro uso; se o cadastro existente veio de outro modelo, para
    com uma mensagem clara em vez de simplesmente nunca reconhecer ninguém."""
    current = model_fingerprint(model_cfg)
    stored = repo.get_meta("model")
    if stored is None or repo.count_persons() == 0:
        repo.set_meta("model", current)
        return
    if stored != current:
        raise ModelMismatchError(
            f"O cadastro em {repo.db_path} foi feito com o modelo [{stored}], mas a config usa "
            f"[{current}]. Embeddings de modelos diferentes não são comparáveis: volte o modelo "
            "em configs/app.yaml ou apague o banco e cadastre de novo "
            "(ex.: scripts/import_people.py)."
        )


def build_service(cfg: dict) -> FaceRecognitionService:
    """Monta o serviço real (MTCNN + modelo + SQLite) a partir de configs/app.yaml."""
    from facerec.config import pick_device, resolve_path
    from facerec.detection import FaceDetector
    from facerec.models import build_embedder

    device = pick_device(cfg.get("device", "auto"))
    embedder = build_embedder(cfg["model"], device)
    det = cfg.get("detection", {})
    detector = FaceDetector(
        device=device,
        align_mode=embedder.align_mode,
        image_size=embedder.input_size,
        prob_threshold=det.get("prob_threshold", 0.95),
        min_face_px=det.get("min_face_px", 60),
        blur_threshold=det.get("blur_threshold", 15.0),
    )
    st = cfg.get("storage", {})
    photos_dir = resolve_path(st["photos_dir"]) if st.get("store_photos") else None
    repo = FaceRepository(resolve_path(st.get("db_path", "embeddings/facerec.db")), photos_dir)
    check_model_compatibility(repo, cfg["model"])
    rec = cfg.get("recognition", {})
    return FaceRecognitionService(
        detector,
        embedder,
        repo,
        threshold=rec["threshold"],
        matcher_name=rec.get("matcher", "cosine"),
        matcher_kwargs=rec.get("matcher_kwargs") or {},
        duplicate_threshold=rec.get("duplicate_threshold", 0.75),
        consistency_threshold=rec.get("consistency_threshold", 0.40),
        store_photos=bool(st.get("store_photos", False)),
        max_enroll_images=rec.get("max_enroll_images", 10),
    )
