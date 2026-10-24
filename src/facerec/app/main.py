"""API e front-end local — spec 005.

    uvicorn facerec.app.main:app --reload          (abre em http://127.0.0.1:8000)

A config vem de configs/app.yaml (ou da variável de ambiente FACEREC_CONFIG).
A aplicação escuta apenas em localhost por padrão: as imagens nunca saem da máquina.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from facerec.config import load_yaml, resolve_path
from facerec.exceptions import FaceRecError, PersonNotFound
from facerec.logging_utils import configure_logging
from facerec.service import NO_PERSONS_MESSAGE, FaceRecognitionService, build_service

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Imagem maior que 10 MB.")
    if not data:
        raise HTTPException(400, "Arquivo vazio.")
    return data


def create_app(service: FaceRecognitionService | None = None) -> FastAPI:
    app = FastAPI(title="Reconhecedor Facial — UFMS")
    state: dict = {"service": service}

    def svc() -> FaceRecognitionService:
        if state["service"] is None:
            cfg_path = os.environ.get("FACEREC_CONFIG", str(resolve_path("configs/app.yaml")))
            cfg = load_yaml(cfg_path)
            configure_logging(cfg.get("logging", {}).get("level", "INFO"))
            state["service"] = build_service(cfg)
        return state["service"]

    @app.exception_handler(PersonNotFound)
    async def _not_found(_, exc: PersonNotFound):
        return JSONResponse({"code": exc.code, "message": str(exc)}, status_code=404)

    @app.exception_handler(FaceRecError)
    async def _domain_error(_, exc: FaceRecError):
        return JSONResponse({"code": exc.code, "message": str(exc)}, status_code=422)

    @app.exception_handler(ValueError)
    async def _bad_request(_, exc: ValueError):
        return JSONResponse({"code": "bad_request", "message": str(exc)}, status_code=400)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/persons")
    def list_persons():
        persons = svc().list_persons()
        return {
            "persons": [
                {
                    "id": p.id,
                    "name": p.display_name,
                    "created_at": p.created_at,
                    "n_embeddings": p.n_embeddings,
                }
                for p in persons
            ],
            "message": NO_PERSONS_MESSAGE
            if not persons
            else f"{len(persons)} pessoa(s) cadastrada(s).",
        }

    @app.post("/api/persons", status_code=201)
    async def enroll(name: str = Form(...), files: list[UploadFile] = File(...)):
        images = [await _read(f) for f in files]
        p = svc().enroll(name, images)
        return {
            "id": p.id,
            "name": p.display_name,
            "n_embeddings": p.n_embeddings,
            "message": f"Pessoa cadastrada: {p.display_name}",
        }

    @app.post("/api/recognize")
    async def recognize(file: UploadFile = File(...)):
        r = svc().recognize(await _read(file))
        return {
            "recognized": r.recognized,
            "person_id": r.person_id,
            "name": r.display_name,
            "score": r.score,
            "threshold": r.threshold,
            "message": r.message,
        }

    @app.delete("/api/persons/{person_id}", status_code=204)
    def delete(person_id: str):
        svc().delete(person_id)
        return Response(status_code=204)

    return app


app = create_app()
