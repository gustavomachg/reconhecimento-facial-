"""Casos de uso da aplicação: cadastrar, entrar, reconhecer, editar e excluir pessoas.

O serviço só conhece INTERFACES (detector, embedder, repositório, matcher): por isso os
testes usam versões falsas/sintéticas de cada peça, sem rostos reais e sem GPU.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np

from facerec.auth import (
    PBKDF2_ITERATIONS,
    LoginThrottle,
    hash_password,
    normalize_email,
    validate_email,
    validate_password,
    verify_password,
)
from facerec.exceptions import (
    DuplicatePerson,
    EmailAlreadyRegistered,
    FaceRecError,
    InconsistentEnrollment,
    InvalidCredentials,
    PersonNotFound,
)
from facerec.logging_utils import log_event
from facerec.matching import Matcher, build_matcher, cosine_similarity
from facerec.metrics import UNKNOWN
from facerec.storage import FaceRepository, Person

NO_PERSONS_MESSAGE = "Nenhuma pessoa cadastrada."
MAX_NAME_LEN = 100
MAX_UNIT_LEN = 30
PERSON_KINDS = ("morador", "visitante")


@dataclass
class RecognitionResult:
    recognized: bool
    person_id: str | None
    display_name: str | None
    score: float | None
    threshold: float
    message: str
    # caixa da face na imagem enviada (x1, y1, x2, y2, em pixels) — a interface desenha
    # o retângulo sobre o vídeo. None quando não houve detecção (ex.: ninguém cadastrado).
    box: list[float] | None = None
    unit: str | None = None  # apartamento/bloco da pessoa reconhecida
    kind: str | None = None  # "morador" | "visitante" (None se não reconhecida)


@dataclass
class FaceVerification:
    """Resultado de conferir um rosto com a conta que está logada (porta aberta pelo rosto).

    Nunca carrega nome ou dados de OUTRA pessoa: se o rosto for de alguém cadastrado que não é o
    dono da conta, a resposta diz apenas que não corresponde.
    """

    matched: bool
    reason: str  # "match" | "other_person" | "unknown"
    score: float | None
    threshold: float
    box: list[float] | None
    message: str
    door_open: bool
    door_seconds: float


@dataclass
class _Identification:
    label: str  # person_id, ou UNKNOWN
    score: float
    box: list[float]


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
        model_label: str = "",
        door_open_seconds: float = 5.0,
        password_iterations: int = PBKDF2_ITERATIONS,
        session_seconds: float = 8 * 3600,
        max_failed_logins: int = 5,
        lockout_seconds: float = 60,
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
        self.door_open_seconds = float(door_open_seconds)  # tempo de porta aberta
        self.model_label = model_label  # ex.: "name=arcface|path=...w600k_mbf.onnx"
        self.password_iterations = password_iterations
        self.session_seconds = float(session_seconds)
        self._throttle = LoginThrottle(max_failed_logins, lockout_seconds)
        self._dummy_hash: str | None = None
        self._matcher: Matcher | None = None

    # ------------------------------------------------------------------ internos
    def _detect(self, images: list[bytes]) -> list:
        return [self.detector.detect_single(img) for img in images]

    def _face_embeddings(self, images: list[bytes]) -> tuple[np.ndarray, list[np.ndarray]]:
        aligned = [face.aligned for face in self._detect(images)]
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

    def _identify(self, image: bytes) -> _Identification | None:
        """Detecta a face, calcula o embedding e procura na galeria. None = galeria vazia."""
        matcher = self._get_matcher()
        if matcher is None:
            return None
        try:
            (face,) = self._detect([image])
        except FaceRecError as exc:
            log_event("recognize_error", reason=exc.code)
            raise
        emb = self.embedder.embed([face.aligned])
        box = [round(float(v), 1) for v in face.box]
        label, score = matcher.identify(emb[0], self.threshold)
        return _Identification(label, float(score), box)

    @staticmethod
    def _check_profile(display_name: str, unit: str, kind: str) -> tuple[str, str, str]:
        display_name = (display_name or "").strip()
        unit = (unit or "").strip()
        kind = (kind or "").strip().lower()
        if not display_name:
            raise ValueError("Informe o nome da pessoa.")
        if kind not in PERSON_KINDS:
            raise ValueError("Informe se a pessoa é morador ou visitante.")
        if len(display_name) > MAX_NAME_LEN:
            raise ValueError(f"Nome muito longo (máximo {MAX_NAME_LEN} caracteres).")
        if len(unit) > MAX_UNIT_LEN:
            raise ValueError(f"Apartamento/bloco muito longo (máximo {MAX_UNIT_LEN} caracteres).")
        return display_name, unit, kind

    # ------------------------------------------------------------------ cadastro
    def enroll(
        self,
        display_name: str,
        images: list[bytes],
        unit: str = "",
        kind: str = "morador",
        email: str | None = None,
        password: str | None = None,
    ) -> Person:
        """Cadastra uma pessoa. Com `email` e `password`, cria também a conta de login."""
        t0 = time.perf_counter()
        display_name, unit, kind = self._check_profile(display_name, unit, kind)
        email_norm = None
        if email is not None or password is not None:
            email_norm = validate_email(email)
            validate_password(password)
            if self.repo.email_exists(email_norm):
                raise EmailAlreadyRegistered()
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
                    log_event(
                        "enroll_rejected",
                        reason="duplicate",
                        existing_person_id=ids[j],
                        score=round(float(sims[i, j]), 4),
                    )
                    # não diz QUEM é: o cadastro é aberto e o nome seria dado pessoal de terceiro
                    raise DuplicatePerson(
                        "Este rosto já está cadastrado. Se a conta é sua, entre com o seu "
                        "e-mail e senha."
                    )
        except FaceRecError as exc:
            if not isinstance(exc, DuplicatePerson):
                log_event("enroll_rejected", reason=exc.code)
            raise
        password_hash = (
            hash_password(password, self.password_iterations) if email_norm is not None else None
        )
        photos = [self._to_jpeg(a) for a in aligned] if self.store_photos else None
        person = self.repo.add_person(
            display_name, emb, photos, unit, kind, email=email_norm, password_hash=password_hash
        )
        self._invalidate()
        log_event(
            "enroll_ok",
            person_id=person.id,
            kind=kind,
            has_account=email_norm is not None,
            n_images=len(images),
            ms=round(1000 * (time.perf_counter() - t0), 1),
        )
        return person

    # ------------------------------------------------------------------ conta
    def authenticate(self, email: str | None, password: str | None) -> Person:
        """Confere e-mail e senha. A mesma mensagem vale para e-mail inexistente e senha errada,
        e o tempo gasto é parecido nos dois casos (hash de mentira quando o e-mail não existe)."""
        key = normalize_email(email)
        self._throttle.check(key)
        creds = self.repo.get_credentials(key) if key else None
        if creds is None and self._dummy_hash is None:
            self._dummy_hash = hash_password("senha-inexistente", self.password_iterations)
        stored = creds[1] if creds else self._dummy_hash
        ok = verify_password(password or "", stored)
        if creds is None or not ok:
            self._throttle.failure(key)
            log_event("login_failed")
            raise InvalidCredentials()
        self._throttle.success(key)
        person = self.repo.get_person(creds[0])
        log_event("login_ok", person_id=person.id)
        return person

    def get_person(self, person_id: str) -> Person | None:
        return self.repo.get_person(person_id)

    def update_profile(
        self,
        person_id: str,
        display_name: str,
        unit: str,
        kind: str,
        email: str,
        current_password: str | None = None,
    ) -> Person:
        """Altera os dados pessoais. Trocar o e-mail exige a senha atual."""
        person = self.repo.get_person(person_id)
        if person is None:
            raise PersonNotFound()
        display_name, unit, kind = self._check_profile(display_name, unit, kind)
        new_email = validate_email(email)
        if new_email != person.email:
            self._require_current_password(person_id, current_password)
            if self.repo.email_exists(new_email):
                raise EmailAlreadyRegistered()
        self.repo.update_person(person_id, display_name, unit, kind, new_email)
        log_event("profile_updated", person_id=person_id, email_changed=new_email != person.email)
        return self.repo.get_person(person_id)

    def change_password(self, person_id: str, current_password: str | None, new_password: str):
        self._require_current_password(person_id, current_password)
        validate_password(new_password)
        self.repo.set_password_hash(
            person_id, hash_password(new_password, self.password_iterations)
        )
        log_event("password_changed", person_id=person_id)

    def _require_current_password(self, person_id: str, current_password: str | None) -> None:
        stored = self.repo.get_password_hash(person_id)
        if not stored or not verify_password(current_password or "", stored):
            raise ValueError("Senha atual incorreta.")

    # ------------------------------------------------------------------ reconhecimento e porta
    def recognize(self, image: bytes) -> RecognitionResult:
        """Identifica QUEM é (1:N) e, se for alguém cadastrado, abre a porta. Devolve os dados da
        pessoa: use só em contexto de operador; a interface web usa `verify_face`."""
        t0 = time.perf_counter()
        ident = self._identify(image)
        if ident is None:
            return RecognitionResult(False, None, None, None, self.threshold, NO_PERSONS_MESSAGE)
        ms = round(1000 * (time.perf_counter() - t0), 1)
        if ident.label == UNKNOWN:
            log_event("recognize_unknown", score=round(ident.score, 4), ms=ms)
            return RecognitionResult(
                False, None, None, ident.score, self.threshold, "Pessoa desconhecida", ident.box
            )
        person = self.repo.get_person(ident.label)
        log_event(
            "recognize_ok",
            person_id=ident.label,
            kind=person.kind,
            score=round(ident.score, 4),
            ms=ms,
        )
        # reconhecer um cadastrado (morador ou visitante) destranca a porta
        self.open_door("face", person_id=ident.label)
        return RecognitionResult(
            True,
            ident.label,
            person.display_name,
            ident.score,
            self.threshold,
            f"Pessoa reconhecida: {person.display_name} ({ident.score:.2f})",
            ident.box,
            person.unit or None,
            person.kind,
        )

    def verify_face(self, person_id: str, image: bytes) -> FaceVerification:
        """Porta pelo rosto: procura o rosto entre TODOS os cadastrados (1:N) e só abre se o
        mais parecido for o dono da conta logada. Não revela quem é uma pessoa diferente."""
        t0 = time.perf_counter()
        ident = self._identify(image)
        ms = round(1000 * (time.perf_counter() - t0), 1)
        score = None if ident is None else round(ident.score, 4)
        box = None if ident is None else ident.box
        if ident is not None and ident.label == person_id:
            log_event("face_verify_ok", person_id=person_id, score=score, ms=ms)
            seconds = self.open_door("face", person_id=person_id)
            return FaceVerification(
                True, "match", score, self.threshold, box,
                "Rosto reconhecido. Porta liberada.", True, seconds,
            )  # fmt: skip
        if ident is None or ident.label == UNKNOWN:
            log_event("face_verify_unknown", person_id=person_id, score=score, ms=ms)
            return FaceVerification(
                False, "unknown", score, self.threshold, box,
                "Pessoa não cadastrada. A porta continua trancada.", False, 0.0,
            )  # fmt: skip
        log_event("face_verify_mismatch", person_id=person_id, score=score, ms=ms)
        return FaceVerification(
            False, "other_person", score, self.threshold, box,
            "Este rosto não corresponde à sua conta. A porta continua trancada.", False, 0.0,
        )  # fmt: skip

    def open_door(self, method: str, person_id: str | None = None) -> float:
        """Destranca a porta (simulada) e devolve por quantos segundos ela fica aberta.

        `method` = "face" (reconhecimento) ou "manual" (deslizar/botão). Não há fechadura real: o
        estado da porta aparece na interface; aqui fica o registro de auditoria (sem nome).
        """
        if method not in ("face", "manual"):
            raise ValueError("Método de abertura inválido.")
        if person_id is None:
            log_event("door_open", method=method)
        else:
            log_event("door_open", method=method, person_id=person_id)
        return self.door_open_seconds

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
    auth = cfg.get("auth", {})
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
        model_label=model_fingerprint(cfg["model"]),
        door_open_seconds=cfg.get("door", {}).get("open_seconds", 5),
        session_seconds=3600 * auth.get("session_hours", 8),
        max_failed_logins=auth.get("max_failed_logins", 5),
        lockout_seconds=auth.get("lockout_seconds", 60),
    )
