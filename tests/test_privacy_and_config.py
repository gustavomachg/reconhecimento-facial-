import sys
from pathlib import Path

import pytest

from facerec.config import PROJECT_ROOT, load_yaml

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from check_private_files import violation  # noqa: E402


@pytest.mark.parametrize(
    "path",
    [
        "data/private/raw/P01/s1/a.jpg",
        "data/public/lfw/x.txt",
        "embeddings/facerec.db",
        "checkpoints/finetuned/E03.pt",
        "consent/termo.pdf",
        "pessoas/foto.JPG",
        "foto.jpeg",
        "src/model.onnx",
        "x/emb.npy",
        "app.sqlite",
        ".env",
        "experiments/notebooks/face.png",
        "data/familia/notas.txt",
        "data/amigos/lista.csv",
        "data/teste_local/x.json",
        "database/pessoas.json",
    ],
)
def test_forbidden_files_blocked(path):
    assert violation(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "src/facerec/service.py",
        "docs/specs/001-face-detection.md",
        "docs/img/arquitetura.png",
        "experiments/results/E01/roc_lfw.png",
        "experiments/results/E01/metrics.json",
        "configs/app.yaml",
        "data/README.md",
    ],
)
def test_allowed_files_pass(path):
    assert violation(path) is None


def test_gitignore_has_mandatory_entries():
    content = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    mandatory = [
        "data/public/",
        "data/private/",
        "faces/",
        "embeddings/",
        "checkpoints/",
        "consent/",
        "*.db",
        "*.sqlite",
        "*.sqlite3",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.onnx",
        "*.safetensors",
        "*.npy",
        "*.npz",
        "*.pkl",
        "*.jpg",
        "*.jpeg",
        "*.png",
        "!docs/**/*.png",
        "!experiments/results/**/*.png",
        ".env",
        "__pycache__/",
        ".venv/",
    ]
    assert not [m for m in mandatory if m not in content]


REQUIRED_EXPERIMENT_KEYS = {"experiment", "seed", "model", "output"}


@pytest.mark.parametrize(
    "cfg_path",
    sorted((PROJECT_ROOT / "configs" / "experiments").glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_experiment_configs_are_complete(cfg_path: Path):
    cfg = load_yaml(cfg_path)
    assert set(cfg) >= REQUIRED_EXPERIMENT_KEYS
    assert cfg["experiment"]["hypothesis"]
    assert cfg["experiment"]["id"] == cfg_path.stem.split("_")[0]
    if cfg.get("training", {}).get("strategy") in ("partial", "full"):
        for k in ("epochs", "optimizer", "lr"):
            assert k in cfg["training"]
        assert "augment" in cfg["data"] and "name" in cfg["loss"]


def test_app_config():
    cfg = load_yaml(PROJECT_ROOT / "configs" / "app.yaml")
    assert cfg["storage"]["store_photos"] is False  # minimização por padrão
    assert 0 < cfg["recognition"]["threshold"] < 1
    assert cfg["storage"]["db_path"].startswith("embeddings/")
