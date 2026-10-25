"""Detecção com backend MTCNN FALSO (não baixa pesos, não usa rostos)."""

import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

from facerec.detection import (
    ARCFACE_TEMPLATE_112,
    FaceDetector,
    align_similarity,
    estimate_similarity_transform,
    laplacian_variance,
    load_image,
)
from facerec.exceptions import InvalidImage, LowQualityImage, MultipleFacesDetected, NoFaceDetected


class FakeMTCNN:
    def __init__(self, boxes, probs):
        self.boxes, self.probs = boxes, probs

    def detect(self, img, landmarks=True):
        if not self.boxes:
            return None, [None], None
        pts = np.array([ARCFACE_TEMPLATE_112 + b[:2] for b in np.array(self.boxes)])
        return np.array(self.boxes, float), np.array(self.probs), pts


def noisy_image(size=200, blur=0):
    arr = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    return img.filter(ImageFilter.GaussianBlur(blur)) if blur else img


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def det(boxes, probs, **kw):
    return FaceDetector(backend=FakeMTCNN(boxes, probs), **kw)


def test_no_face():
    with pytest.raises(NoFaceDetected):
        det([], []).detect_single(noisy_image())


def test_low_confidence_counts_as_no_face():
    with pytest.raises(NoFaceDetected):
        det([[10, 10, 150, 150]], [0.5]).detect_single(noisy_image())


def test_multiple_faces():
    with pytest.raises(MultipleFacesDetected):
        det([[0, 0, 90, 90], [100, 100, 190, 190]], [0.99, 0.98]).detect_single(noisy_image())


def test_small_face_is_low_quality():
    with pytest.raises(LowQualityImage):
        det([[10, 10, 40, 40]], [0.99], min_face_px=60).detect_single(noisy_image())


def test_blurry_face_is_low_quality():
    img = noisy_image(blur=6)
    with pytest.raises(LowQualityImage):
        det([[10, 10, 180, 180]], [0.99], blur_threshold=50).detect_single(img)


def test_good_face_returns_aligned_crop():
    face = det([[10, 10, 180, 180]], [0.99]).detect_single(png_bytes(noisy_image()))
    assert face.aligned.shape == (160, 160, 3) and face.aligned.dtype == np.uint8


def test_similarity_mode_size():
    d = det([[10, 10, 180, 180]], [0.99], align_mode="similarity", image_size=112)
    assert d.detect_single(noisy_image()).aligned.shape == (112, 112, 3)


def test_invalid_image_bytes():
    with pytest.raises(InvalidImage):
        load_image(b"isto nao e uma imagem")


def test_similarity_transform_recovers_known_transform():
    theta, scale, t = np.deg2rad(20), 1.7, np.array([5.0, -3.0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = ARCFACE_TEMPLATE_112
    dst = (scale * rot @ src.T).T + t
    m = estimate_similarity_transform(src, dst)
    assert np.allclose(m[:, :2], scale * rot, atol=1e-6)
    assert np.allclose(m[:, 2], t, atol=1e-6)


def test_align_similarity_maps_landmarks_to_template():
    """Desenha pontos brancos nos landmarks; após alinhar, devem cair perto do template."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    lm = ARCFACE_TEMPLATE_112 * 2 + 40
    for x, y in lm.astype(int):
        img[y - 2 : y + 3, x - 2 : x + 3] = 255
    out = align_similarity(Image.fromarray(img), lm, 112)
    for x, y in ARCFACE_TEMPLATE_112.astype(int):
        assert out[y - 1 : y + 2, x - 1 : x + 2].max() > 100


def test_laplacian_variance_lower_when_blurred():
    sharp = np.asarray(noisy_image())
    blurred = np.asarray(noisy_image(blur=4))
    assert laplacian_variance(blurred) < laplacian_variance(sharp)


def test_pick_face_center_vs_largest():
    from facerec.detection import DetectedFace, pick_face

    central = DetectedFace(np.array([100, 100, 150, 150.0]), 0.99, None)  # pequena, no centro
    big_side = DetectedFace(np.array([0, 0, 90, 90.0]), 0.99, None)  # maior, no canto
    faces = [big_side, central]
    assert pick_face(faces, (250, 250), "center") is central
    assert pick_face(faces, (250, 250), "largest") is big_side
    assert pick_face([], (250, 250), "center") is None
