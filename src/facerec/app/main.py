"""API e front-end local — specs 005, 007, 008 e 009.

    uvicorn facerec.app.main:app --reload          (abre em http://127.0.0.1:8000)

A config vem de configs/app.yaml (ou da variável de ambiente FACEREC_CONFIG).
A aplicação escuta apenas em localhost por padrão: as imagens nunca saem da máquina.

Cada pessoa tem a sua conta (e-mail + senha) e só enxerga/altera os PRÓPRIOS dados: não existe
rota que liste ou devolva dados de outras pessoas.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from facerec.auth import SessionStore
from facerec.config import load_yaml, resolve_path
from facerec.exceptions import FaceRecError, NotAuthenticated
from facerec.logging_utils import configure_logging
from facerec.service import NO_PERSONS_MESSAGE, FaceRecognitionService, build_service
from facerec.storage import Person

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SESSION_COOKIE = "portaria_session"
# nomes legíveis dos modelos da Parte 1, procurados no "fingerprint" da config
MODEL_NAMES = {
    "w600k_mbf": "MobileFaceNet (E08)",
    "w600k_r50": "ArcFace iResNet-50 (E02)",
    "checkpoint=": "FaceNet fine-tunada",
    "name=facenet": "FaceNet (E01)",
}


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Imagem maior que 10 MB.")
    if not data:
        raise HTTPException(400, "Arquivo vazio.")
    return data


def _profile(p: Person) -> dict:
    return {
        "id": p.id,
        "name": p.display_name,
        "unit": p.unit,
        "kind": p.kind,
        "email": p.email,
        "n_embeddings": p.n_embeddings,
        "created_at": p.created_at,
    }


def create_app(service: FaceRecognitionService | None = None) -> FastAPI:
    app = FastAPI(title="Portaria Facial — UFMS")
    state: dict = {"service": service}
    sessions = SessionStore()

    def svc() -> FaceRecognitionService:
        if state["service"] is None:
            cfg_path = os.environ.get("FACEREC_CONFIG", str(resolve_path("configs/app.yaml")))
            cfg = load_yaml(cfg_path)
            configure_logging(cfg.get("logging", {}).get("level", "INFO"))
            state["service"] = build_service(cfg)
        return state["service"]

    # ------------------------------------------------------------ sessão
    def start_session(request: Request, response: Response, person_id: str) -> None:
        ttl = svc().session_seconds
        token = sessions.create(person_id, ttl)
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=int(ttl),
            httponly=True,  # JavaScript da página não consegue ler o cookie
            samesite="strict",  # não é enviado em requisições vindas de outros sites
            secure=request.url.scheme == "https",
            path="/",
        )

    def current_person(request: Request) -> Person:
        person_id = sessions.get(request.cookies.get(SESSION_COOKIE))
        person = svc().get_person(person_id) if person_id else None
        if person is None:
            raise NotAuthenticated()
        return person

    # ------------------------------------------------------------ erros
    @app.exception_handler(FaceRecError)
    async def _domain_error(_, exc: FaceRecError):
        return JSONResponse({"code": exc.code, "message": str(exc)}, status_code=exc.http_status)

    @app.exception_handler(ValueError)
    async def _bad_request(_, exc: ValueError):
        return JSONResponse({"code": "bad_request", "message": str(exc)}, status_code=400)

    @app.exception_handler(Exception)
    async def _internal_error(_, exc: Exception):
        # ex.: modelo ONNX não baixado, banco feito com outro modelo (ModelMismatchError).
        # A mensagem é da própria aplicação (sem dados pessoais) e aparece na interface,
        # em vez de a página quebrar ao tentar ler uma resposta que não é JSON.
        logging.getLogger("facerec").exception("erro interno")
        return JSONResponse(
            {"code": "internal_error", "message": f"Erro interno: {exc}"}, status_code=500
        )

    @app.middleware("http")
    async def _no_store(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"  # dados pessoais: sem cache
        elif path == "/" or path.startswith("/static/"):
            # a interface pode ficar em cache, mas o navegador confere (ETag) se há versão nova
            response.headers["Cache-Control"] = "no-cache"
        return response

    # ------------------------------------------------------------ página e status
    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    # scripts da interface (cena 3D e three.js), servidos pelo próprio app: nada vem de CDN
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/api/status")
    def status():
        """Configuração em uso e quantas pessoas existem (sem dados pessoais)."""
        s = svc()
        label = s.model_label
        friendly = next((v for k, v in MODEL_NAMES.items() if k in label), label or "desconhecido")
        n = s.repo.count_persons()
        return {
            "model": friendly,
            "matcher": s.matcher_name,
            "threshold": s.threshold,
            "door_seconds": s.door_open_seconds,
            "n_persons": n,
            "message": NO_PERSONS_MESSAGE if n == 0 else f"{n} pessoa(s) cadastrada(s).",
        }

    # ------------------------------------------------------------ cadastro e login
    @app.post("/api/auth/register", status_code=201)
    async def register(
        request: Request,
        response: Response,
        name: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
        files: list[UploadFile] = File(...),
        unit: str = Form(""),
        kind: str = Form("morador"),
        consent: bool = Form(False),
    ):
        # LGPD: dado biométrico só com consentimento de quem está sendo cadastrado. Aqui é a
        # própria pessoa que se cadastra; o termo assinado fica com ela, fora do sistema.
        if not consent:
            raise ValueError("Confirme o consentimento para o uso da sua imagem.")
        images = [await _read(f) for f in files]
        person = svc().enroll(name, images, unit, kind, email=email, password=password)
        start_session(request, response, person.id)
        return {**_profile(person), "message": f"Bem-vindo(a), {person.display_name}!"}

    @app.post("/api/auth/login")
    def login(
        request: Request,
        response: Response,
        email: str = Form(""),
        password: str = Form(""),
    ):
        person = svc().authenticate(email, password)
        start_session(request, response, person.id)
        return _profile(person)

    @app.post("/api/auth/logout", status_code=204)
    def logout(request: Request):
        sessions.revoke(request.cookies.get(SESSION_COOKIE))
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    # ------------------------------------------------------------ minha conta
    @app.get("/api/me")
    def me(person: Person = Depends(current_person)):
        return _profile(person)

    @app.put("/api/me")
    def update_me(
        name: str = Form(...),
        email: str = Form(...),
        unit: str = Form(""),
        kind: str = Form("morador"),
        current_password: str = Form(""),
        person: Person = Depends(current_person),
    ):
        updated = svc().update_profile(person.id, name, unit, kind, email, current_password)
        return {**_profile(updated), "message": "Dados atualizados."}

    @app.post("/api/me/password")
    def change_password(
        request: Request,
        response: Response,
        current_password: str = Form(""),
        new_password: str = Form(""),
        person: Person = Depends(current_person),
    ):
        svc().change_password(person.id, current_password, new_password)
        # senha nova => encerra todas as sessões antigas e abre uma nova só para este navegador
        sessions.revoke_person(person.id)
        start_session(request, response, person.id)
        return {"message": "Senha alterada."}

    @app.delete("/api/me", status_code=204)
    def delete_me(person: Person = Depends(current_person)):
        """Exclui a própria conta: cadastro, embeddings e fotos (LGPD art. 18)."""
        svc().delete(person.id)
        sessions.revoke_person(person.id)
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    # ------------------------------------------------------------ porta
    @app.post("/api/me/door/open")
    def open_door(person: Person = Depends(current_person)):
        """Abertura manual (deslizar na página): vale porque quem pede já fez login."""
        seconds = svc().open_door("manual", person_id=person.id)
        return {"door_open": True, "door_seconds": seconds, "method": "manual"}

    @app.post("/api/me/door/face")
    async def open_door_by_face(
        file: UploadFile = File(...), person: Person = Depends(current_person)
    ):
        """Abertura pelo rosto: só abre se o rosto for o do dono da conta (spec 003/007/009)."""
        v = svc().verify_face(person.id, await _read(file))
        return {
            "matched": v.matched,
            "reason": v.reason,
            "score": v.score,
            "threshold": v.threshold,
            "message": v.message,
            "box": v.box,
            "door_open": v.door_open,
            "door_seconds": v.door_seconds,
        }

    return app


app = create_app()
