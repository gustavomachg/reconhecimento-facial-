"""Fábrica de extratores de embedding a partir da config."""

from __future__ import annotations

from facerec.models.base import Embedder


def build_embedder(cfg: dict, device: str = "cpu") -> Embedder:
    """cfg = {"name": "facenet"|"arcface", "checkpoint": opcional, "path": opcional}."""
    name = cfg.get("name", "facenet")
    if name == "facenet":
        from facerec.models.facenet import FaceNetEmbedder

        return FaceNetEmbedder(device=device, checkpoint=cfg.get("checkpoint"))
    if name == "arcface":
        from facerec.config import resolve_path
        from facerec.models.arcface import DEFAULT_ARCFACE_PATH, ArcFaceEmbedder

        return ArcFaceEmbedder(resolve_path(cfg.get("path", DEFAULT_ARCFACE_PATH)), device=device)
    raise ValueError(f"Modelo desconhecido: {name}")


__all__ = ["Embedder", "build_embedder"]
