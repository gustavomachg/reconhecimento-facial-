"""Detecção (MTCNN) + checagem de qualidade + alinhamento da face — spec 001.

Fluxo de `FaceDetector.detect_single`:
    imagem -> MTCNN (caixas, probabilidades, 5 landmarks)
           -> 0 faces confiáveis  -> NoFaceDetected
           -> >1 face confiável   -> MultipleFacesDetected
           -> face pequena/borrada -> LowQualityImage
           -> alinhamento -> recorte RGB (uint8) pronto para a rede

Dois modos de alinhamento:
- "crop":       recorte da caixa com margem e redimensionamento (como o facenet-pytorch faz;
                é o pré-processamento com que a InceptionResnetV1/VGGFace2 foi treinada).
- "similarity": transformação de similaridade (rotação+escala+translação) que leva os
                5 landmarks (olhos, nariz, cantos da boca) a um template fixo. É o padrão
                do ArcFace/insightface (112x112).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from facerec.exceptions import InvalidImage, LowQualityImage, MultipleFacesDetected, NoFaceDetected

# Template de landmarks do ArcFace para 112x112 (olho esq., olho dir., nariz, boca esq., boca dir.)
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


@dataclass
class DetectedFace:
    box: np.ndarray  # (4,) x1, y1, x2, y2
    prob: float
    landmarks: np.ndarray | None  # (5, 2)
    aligned: np.ndarray | None = field(default=None, repr=False)  # (H, W, 3) uint8
    blur_score: float | None = None

    @property
    def size(self) -> float:
        """Menor lado da caixa em pixels."""
        x1, y1, x2, y2 = self.box
        return float(min(x2 - x1, y2 - y1))


# ---------------------------------------------------------------- utilidades de imagem
def load_image(source: bytes | str | Path | Image.Image) -> Image.Image:
    """Abre a imagem em RGB respeitando a orientação EXIF. Erro de domínio se inválida."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    try:
        img = Image.open(io.BytesIO(source)) if isinstance(source, bytes) else Image.open(source)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImage() from exc


def laplacian_variance(rgb: np.ndarray) -> float:
    """Medida de nitidez: variância do Laplaciano (baixa = imagem borrada)."""
    gray = np.asarray(rgb, dtype=np.float64)
    if gray.ndim == 3:
        gray = gray @ np.array([0.299, 0.587, 0.114])
    lap = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(lap.var())


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Matriz 2x3 da transformação de similaridade que leva `src` a `dst` (Umeyama, 1991)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    rot = u @ np.diag(d) @ vt
    scale = (sig * d).sum() / s_c.var(0).sum()
    t = mu_d - scale * rot @ mu_s
    return np.hstack([scale * rot, t[:, None]])


def warp_affine(img: Image.Image, matrix: np.ndarray, size: int) -> np.ndarray:
    """Aplica a matriz 2x3 (entrada -> saída) e devolve recorte size x size em uint8."""
    full = np.vstack([matrix, [0, 0, 1]])
    inv = np.linalg.inv(full)  # PIL pede o mapeamento saída -> entrada
    coeffs = tuple(inv[:2].ravel())
    out = img.transform((size, size), Image.Transform.AFFINE, coeffs, Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.uint8)


def align_similarity(img: Image.Image, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    template = ARCFACE_TEMPLATE_112 * (size / 112.0)
    return warp_affine(img, estimate_similarity_transform(landmarks, template), size)


def crop_with_margin(img: Image.Image, box: np.ndarray, size: int = 160, margin: int = 0):
    """Mesmo critério do facenet-pytorch: `margin` é medida em pixels da imagem FINAL."""
    x1, y1, x2, y2 = (float(v) for v in box)
    mx = margin * (x2 - x1) / max(size - margin, 1)
    my = margin * (y2 - y1) / max(size - margin, 1)
    w, h = img.size
    crop_box = (
        int(max(x1 - mx / 2, 0)),
        int(max(y1 - my / 2, 0)),
        int(min(x2 + mx / 2, w)),
        int(min(y2 + my / 2, h)),
    )
    crop = img.crop(crop_box).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.uint8)


def pick_face(faces: list[DetectedFace], image_size: tuple[int, int], strategy: str = "largest"):
    """Escolhe UMA face quando a imagem tem várias.

    - "largest": a maior face (pessoa mais próxima da câmera) — usada na pasta do grupo;
    - "center":  a face cujo centro está mais perto do centro da imagem. É o critério certo
                 para LFW e CelebA, cujas imagens foram montadas CENTRALIZADAS na pessoa
                 rotulada; a maior face pode ser de outra pessoa ao lado.
    """
    if not faces:
        return None
    if strategy == "largest":
        return max(faces, key=lambda f: f.size)
    if strategy == "center":
        cx, cy = image_size[0] / 2, image_size[1] / 2

        def dist(f: DetectedFace) -> float:
            x1, y1, x2, y2 = f.box
            return float(((x1 + x2) / 2 - cx) ** 2 + ((y1 + y2) / 2 - cy) ** 2)

        return min(faces, key=dist)
    raise ValueError(f"Estratégia de escolha de face desconhecida: {strategy}")


# ---------------------------------------------------------------- detector
class FaceDetector:
    def __init__(
        self,
        device: str = "cpu",
        align_mode: str = "crop",
        image_size: int = 160,
        margin: int = 0,
        prob_threshold: float = 0.95,
        min_face_px: int = 60,
        blur_threshold: float = 15.0,
        mtcnn_min_face_size: int = 20,
        backend=None,
    ):
        if align_mode not in ("crop", "similarity"):
            raise ValueError("align_mode deve ser 'crop' ou 'similarity'")
        self.align_mode = align_mode
        self.image_size = image_size
        self.margin = margin
        self.prob_threshold = prob_threshold
        self.min_face_px = min_face_px
        self.blur_threshold = blur_threshold
        if backend is None:
            from facenet_pytorch import MTCNN

            backend = MTCNN(keep_all=True, device=device, min_face_size=mtcnn_min_face_size)
        self.backend = backend

    def detect(self, img: Image.Image) -> list[DetectedFace]:
        """Todas as faces com probabilidade >= prob_threshold, da maior para a menor."""
        boxes, probs, points = self.backend.detect(img, landmarks=True)
        if boxes is None:
            return []
        faces = [
            DetectedFace(
                np.asarray(b, dtype=np.float64),
                float(p),
                None if points is None else np.asarray(points[i]),
            )
            for i, (b, p) in enumerate(zip(boxes, probs, strict=True))
            if p is not None and p >= self.prob_threshold
        ]
        return sorted(faces, key=lambda f: f.size, reverse=True)

    def align(self, img: Image.Image, face: DetectedFace) -> np.ndarray:
        if self.align_mode == "similarity" and face.landmarks is not None:
            return align_similarity(img, face.landmarks, self.image_size)
        return crop_with_margin(img, face.box, self.image_size, self.margin)

    def detect_single(self, source, check_quality: bool = True) -> DetectedFace:
        img = load_image(source)
        faces = self.detect(img)
        if not faces:
            raise NoFaceDetected()
        if len(faces) > 1:
            raise MultipleFacesDetected()
        face = faces[0]
        face.aligned = self.align(img, face)
        face.blur_score = laplacian_variance(face.aligned)
        if check_quality:
            if face.size < self.min_face_px:
                raise LowQualityImage(
                    f"Face muito pequena ({face.size:.0f}px; mínimo {self.min_face_px}px). "
                    "Aproxime-se da câmera."
                )
            if face.blur_score < self.blur_threshold:
                raise LowQualityImage("Imagem desfocada. Tente novamente com a câmera parada.")
        return face
