"""Leitura de configurações YAML (experimentos e aplicação) e utilidades de ambiente."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    """Raiz do projeto (onde fica configs/). Ordem: variável FACEREC_ROOT, pasta do código
    (instalação com `pip install -e .` ou execução direta), diretório atual."""
    env = os.environ.get("FACEREC_ROOT")
    if env:
        return Path(env).resolve()
    from_code = Path(__file__).resolve().parents[2]
    if (from_code / "configs").is_dir():
        return from_code
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config inválida (esperado mapeamento): {path}")
    return data


def resolve_path(p: str | os.PathLike) -> Path:
    """Caminhos relativos nas configs são relativos à raiz do projeto."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    """Fixa seeds de python/numpy/torch para reprodutibilidade."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def environment_info() -> dict[str, Any]:
    """Versões e hardware, registrados junto com cada experimento."""
    import platform

    import numpy as np

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    from importlib.metadata import PackageNotFoundError, version

    for pkg in ("torchvision", "facenet-pytorch", "onnxruntime", "scikit-learn", "Pillow"):
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = None
    try:
        import os

        info["cpu_count"] = os.cpu_count()
        if info.get("cuda_available"):
            import torch

            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
    except Exception:
        pass
    return info


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
