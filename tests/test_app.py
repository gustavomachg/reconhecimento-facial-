"""API com serviço falso (detector/embedder sintéticos): contas, login, porta e privacidade."""

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from facerec.app.main import create_app  # noqa: E402
from facerec.service import FaceRecognitionService  # noqa: E402
from facerec.storage import FaceRepository  # noqa: E402
from tests.conftest import FakeDetector, FakeEmbedder, make_image  # noqa: E402

PASSWORD = "senha-forte-1"


@pytest.fixture
def client(tmp_path):
    repo = FaceRepository(tmp_path / "app.db")
    svc = FaceRecognitionService(
        FakeDetector(),
        FakeEmbedder(),
        repo,
        threshold=0.7,
        password_iterations=1000,  # rápido nos testes
        max_failed_logins=3,
        lockout_seconds=60,
    )
    yield TestClient(create_app(svc))
    repo.close()


def other_client(client):
    """Outro navegador (cookies próprios) falando com o MESMO servidor."""
    return TestClient(client.app)


def face_files(identity, n=2):
    return [
        ("files", (f"{i}.png", make_image(identity, f"face{i}"), "image/png")) for i in range(n)
    ]


def probe(identity, kind="face9"):
    return {"file": ("q.png", make_image(identity, kind), "image/png")}


def register(client, identity=1, name="Morador A", email="a@exemplo.com", **extra):
    data = {"name": name, "email": email, "password": PASSWORD, "consent": "true", **extra}
    return client.post("/api/auth/register", data=data, files=face_files(identity))


def login(client, email="a@exemplo.com", password=PASSWORD):
    return client.post("/api/auth/login", data={"email": email, "password": password})


# ------------------------------------------------------------------ página e estado inicial
def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Portaria Facial" in r.text


def test_initial_state(client):
    s = client.get("/api/status").json()
    assert s["n_persons"] == 0 and s["message"] == "Nenhuma pessoa cadastrada."


def test_status_shows_config_without_personal_data(tmp_path):
    repo = FaceRepository(tmp_path / "s.db")
    svc = FaceRecognitionService(
        FakeDetector(),
        FakeEmbedder(),
        repo,
        threshold=0.25,
        matcher_name="prototype",
        model_label="name=arcface|path=checkpoints/public/buffalo_s/w600k_mbf.onnx",
        password_iterations=1000,
    )
    c = TestClient(create_app(svc))
    register(c, name="Pessoa Secreta", email="segredo@exemplo.com")
    r = c.get("/api/status")
    body = r.json()
    assert body["model"] == "MobileFaceNet (E08)" and body["matcher"] == "prototype"
    assert body["threshold"] == 0.25 and body["door_seconds"] == 5.0 and body["n_persons"] == 1
    assert "Pessoa Secreta" not in r.text and "segredo@exemplo.com" not in r.text
    repo.close()


def test_unexpected_error_is_json_500():
    class BrokenService:
        model_label = ""
        matcher_name = ""
        threshold = 0
        door_open_seconds = 0

        @property
        def repo(self):
            raise RuntimeError("modelo não encontrado")

    c = TestClient(create_app(BrokenService()), raise_server_exceptions=False)
    r = c.get("/api/status")
    assert r.status_code == 500
    assert r.json()["code"] == "internal_error" and "modelo não encontrado" in r.json()["message"]


# ------------------------------------------------------------------ cadastro
def test_register_creates_account_and_logs_in(client):
    r = register(client, email="  Maria@Exemplo.COM ", unit="Apto 101", kind="visitante")
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "Morador A" and body["email"] == "maria@exemplo.com"
    assert body["unit"] == "Apto 101" and body["kind"] == "visitante"
    assert PASSWORD not in r.text and "password" not in body
    me = client.get("/api/me")
    assert me.status_code == 200 and me.json()["id"] == body["id"]


def test_register_requires_consent(client):
    data = {"name": "A", "email": "a@exemplo.com", "password": PASSWORD}
    r = client.post("/api/auth/register", data=data, files=face_files(1))
    assert r.status_code == 400 and "consentimento" in r.json()["message"]
    r = client.post("/api/auth/register", data={**data, "consent": "false"}, files=face_files(1))
    assert r.status_code == 400
    assert client.get("/api/status").json()["n_persons"] == 0
    assert client.get("/api/me").status_code == 401


@pytest.mark.parametrize(
    "extra,fragment",
    [
        ({"email": "sem-arroba"}, "e-mail"),
        ({"password": "curta"}, "senha"),
        ({"kind": "porteiro"}, "morador ou visitante"),
        ({"unit": "x" * 31}, "muito longo"),
        ({"name": "   "}, "nome"),
    ],
)
def test_register_validation_errors_are_400(client, extra, fragment):
    data = {"name": "A", "email": "a@exemplo.com", "password": PASSWORD, "consent": "true"}
    r = client.post("/api/auth/register", data={**data, **extra}, files=face_files(1))
    assert r.status_code == 400 and fragment in r.json()["message"].lower()
    assert client.get("/api/status").json()["n_persons"] == 0


def test_register_face_errors_are_422_and_save_nothing(client):
    for kind, code in (("noface", "no_face"), ("multi", "multiple_faces")):
        r = client.post(
            "/api/auth/register",
            data={"name": "A", "email": "a@exemplo.com", "password": PASSWORD, "consent": "true"},
            files=[("files", ("a.png", make_image(1, kind), "image/png"))],
        )
        assert r.status_code == 422 and r.json()["code"] == code
    assert client.get("/api/status").json()["n_persons"] == 0


def test_duplicate_email_is_409(client):
    register(client)
    r = register(other_client(client), identity=2, name="Outro")
    assert r.status_code == 409 and r.json()["code"] == "email_in_use"


def test_duplicate_face_is_422_and_does_not_reveal_who(client):
    register(client, name="Fulano Reservado")
    r = register(other_client(client), identity=1, name="Impostor", email="b@exemplo.com")
    assert r.status_code == 422 and r.json()["code"] == "duplicate_person"
    assert "Fulano" not in r.text and "a@exemplo.com" not in r.text


# ------------------------------------------------------------------ login e logout
def test_login_logout_flow(client):
    register(client)
    client.cookies.clear()
    assert client.get("/api/me").status_code == 401
    assert login(client).status_code == 200
    assert client.get("/api/me").json()["email"] == "a@exemplo.com"
    assert login(client, email="A@EXEMPLO.com").status_code == 200  # e-mail não diferencia caixa
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/me").status_code == 401


def test_wrong_email_and_wrong_password_give_same_answer(client):
    register(client)
    client.cookies.clear()
    bad_pw = login(client, password="senha-errada-9")
    bad_mail = login(client, email="naoexiste@exemplo.com")
    assert bad_pw.status_code == bad_mail.status_code == 401
    assert bad_pw.json() == bad_mail.json()
    assert bad_pw.json()["code"] == "invalid_credentials"


def test_login_locks_after_repeated_failures(client):
    register(client)
    client.cookies.clear()
    for _ in range(3):
        assert login(client, password="errada-123").status_code == 401
    r = login(client)  # até a senha certa é recusada durante o bloqueio
    assert r.status_code == 429 and r.json()["code"] == "too_many_attempts"


def test_session_cookie_flags_and_no_store(client):
    r = register(client)
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.get("/api/me").headers["cache-control"] == "no-store"


# ------------------------------------------------------------------ rotas protegidas
PROTECTED = [
    ("GET", "/api/me", {}),
    ("PUT", "/api/me", {"data": {"name": "X", "email": "x@exemplo.com"}}),
    ("POST", "/api/me/password", {"data": {"current_password": "a", "new_password": "b"}}),
    ("DELETE", "/api/me", {}),
    ("POST", "/api/me/door/open", {}),
    ("POST", "/api/me/door/face", {"files": probe(1)}),
]


@pytest.mark.parametrize("method,url,kwargs", PROTECTED)
def test_protected_routes_require_login(client, method, url, kwargs):
    register(client)
    client.cookies.clear()
    r = client.request(method, url, **kwargs)
    assert r.status_code == 401 and r.json()["code"] == "not_authenticated"


def test_there_is_no_route_that_lists_other_people(client):
    register(client)
    for url in ("/api/persons", "/api/recognize"):
        assert client.get(url).status_code in (404, 405)
        assert client.post(url).status_code in (404, 405)


# ------------------------------------------------------------------ porta
def test_manual_door_open_needs_login(client):
    assert client.post("/api/me/door/open").status_code == 401
    register(client)
    r = client.post("/api/me/door/open")
    assert r.status_code == 200
    assert r.json() == {"door_open": True, "door_seconds": 5.0, "method": "manual"}


def test_door_opens_by_face_of_the_account_owner(client):
    register(client, identity=1)
    r = client.post("/api/me/door/face", files=probe(1, "face9"))
    body = r.json()
    assert r.status_code == 200 and body["matched"] and body["reason"] == "match"
    assert body["door_open"] is True and body["door_seconds"] == 5.0
    assert len(body["box"]) == 4 and body["score"] >= body["threshold"]


def test_other_registered_face_does_not_open_door_and_is_not_revealed(client):
    register(client, identity=1, name="Dona da Conta")
    other = other_client(client)
    register(other, identity=2, name="Vizinha Reservada", email="v@exemplo.com")
    r = client.post("/api/me/door/face", files=probe(2, "face9"))  # rosto da vizinha
    body = r.json()
    assert r.status_code == 200 and not body["matched"] and body["reason"] == "other_person"
    assert body["door_open"] is False and body["door_seconds"] == 0
    assert "Vizinha" not in r.text and "v@exemplo.com" not in r.text
    assert set(body) == {
        "matched",
        "reason",
        "score",
        "threshold",
        "message",
        "box",
        "door_open",
        "door_seconds",
    }


def test_unknown_face_does_not_open_door(client):
    register(client, identity=1)
    r = client.post("/api/me/door/face", files=probe(7, "face1"))
    body = r.json()
    assert not body["matched"] and body["reason"] == "unknown" and body["door_open"] is False
    assert "não cadastrada" in body["message"].lower()


def test_door_by_face_domain_errors(client):
    register(client, identity=1)
    for kind, code in (("noface", "no_face"), ("multi", "multiple_faces")):
        r = client.post("/api/me/door/face", files=probe(1, kind))
        assert r.status_code == 422 and r.json()["code"] == code


# ------------------------------------------------------------------ editar dados pessoais
def test_update_profile(client):
    register(client)
    r = client.put(
        "/api/me",
        data={
            "name": "Novo Nome",
            "email": "a@exemplo.com",
            "unit": "Apto 12",
            "kind": "visitante",
        },
    )
    body = r.json()
    assert r.status_code == 200 and body["name"] == "Novo Nome"
    assert body["unit"] == "Apto 12" and body["kind"] == "visitante"
    assert client.get("/api/me").json()["name"] == "Novo Nome"


def test_update_profile_validation(client):
    register(client)
    base = {"name": "A", "email": "a@exemplo.com"}
    assert client.put("/api/me", data={**base, "kind": "porteiro"}).status_code == 400
    assert client.put("/api/me", data={**base, "unit": "x" * 31}).status_code == 400
    assert client.put("/api/me", data={**base, "name": " "}).status_code == 400


def test_changing_email_needs_current_password_and_checks_uniqueness(client):
    register(client)
    register(other_client(client), identity=2, name="B", email="b@exemplo.com")
    data = {"name": "A", "email": "novo@exemplo.com"}
    r = client.put("/api/me", data=data)
    assert r.status_code == 400 and "senha atual" in r.json()["message"].lower()
    r = client.put("/api/me", data={**data, "current_password": "errada-123"})
    assert r.status_code == 400
    r = client.put("/api/me", data={**data, "email": "b@exemplo.com", "current_password": PASSWORD})
    assert r.status_code == 409
    r = client.put("/api/me", data={**data, "current_password": PASSWORD})
    assert r.status_code == 200 and r.json()["email"] == "novo@exemplo.com"
    client.cookies.clear()
    assert login(client, email="a@exemplo.com").status_code == 401
    assert login(client, email="novo@exemplo.com").status_code == 200


def test_change_password(client):
    register(client)
    other_browser = other_client(client)
    login(other_browser)
    new = "outra-senha-9"
    wrong = {"current_password": "errada-123", "new_password": new}
    assert client.post("/api/me/password", data=wrong).status_code == 400
    weak = {"current_password": PASSWORD, "new_password": "123"}
    assert client.post("/api/me/password", data=weak).status_code == 400
    good = {"current_password": PASSWORD, "new_password": new}
    assert client.post("/api/me/password", data=good).status_code == 200
    assert client.get("/api/me").status_code == 200  # este navegador segue logado
    assert other_browser.get("/api/me").status_code == 401  # as outras sessões caíram
    client.cookies.clear()
    assert login(client, password=PASSWORD).status_code == 401
    assert login(client, password=new).status_code == 200


# ------------------------------------------------------------------ excluir a própria conta
def test_delete_account_removes_everything(client):
    register(client, identity=1)
    assert client.delete("/api/me").status_code == 204
    assert client.get("/api/me").status_code == 401
    assert login(client).status_code == 401
    s = client.get("/api/status").json()
    assert s["n_persons"] == 0 and s["message"] == "Nenhuma pessoa cadastrada."
    # o mesmo rosto e o mesmo e-mail podem ser cadastrados de novo
    assert register(client, identity=1).status_code == 201


def test_delete_account_ends_all_sessions(client):
    register(client)
    other_browser = other_client(client)
    login(other_browser)
    client.delete("/api/me")
    assert other_browser.get("/api/me").status_code == 401


# ------------------------------------------------------------------ interface 3D (spec 010)
STATIC = Path(__file__).resolve().parents[1] / "src" / "facerec" / "app" / "static"


def test_scene_and_three_are_served_locally(client):
    scene = client.get("/static/scene.js")
    assert scene.status_code == 200 and "javascript" in scene.headers["content-type"]
    assert "./vendor/three.module.min.js" in scene.text  # importa a cópia local
    three = client.get("/static/vendor/three.module.min.js")
    assert three.status_code == 200 and "SPDX-License-Identifier: MIT" in three.text[:300]
    assert client.get("/static/vendor/THREE-LICENSE.txt").status_code == 200


def test_page_loads_the_scene_as_optional_module(client):
    html = client.get("/").text
    assert '<script type="module" src="/static/scene.js"></script>' in html
    assert '<canvas id="bg3d" aria-hidden="true">' in html
    assert "prefers-reduced-motion" in html  # respeita quem pede menos movimento


def test_interface_requests_nothing_from_other_hosts():
    """Promessa de privacidade: a página não busca fonte, script, estilo ou imagem na internet."""
    own = ["index.html", "scene.js"]
    for name in own:
        text = (STATIC / name).read_text(encoding="utf-8")
        urls = re.findall(r"""(?:src|href)\s*=\s*["'](https?:)?//""", text)
        urls += re.findall(r"""(?:import|from|fetch\(|url\()\s*["'](?:https?:)?//""", text)
        assert urls == [], f"{name} referencia outro host: {urls}"
    three = (STATIC / "vendor" / "three.module.min.js").read_text(encoding="utf-8")
    assert not re.search(r"""\bimport\s*["']https?://|\bfrom\s*["']https?://""", three)


def test_home_door_has_the_buttons_at_its_foot(client):
    """A porta 3D se ancora no bloco #doorAnchor; os botões são adesivos espelhados no pé dela."""
    html = client.get("/").text
    assert '<div class="doorway" id="doorAnchor"></div>' in html
    m = re.search(r'<div class="stickers">(.*?)</div>', html, re.S)
    assert m and 'class="sticker" id="goLogin"' in m.group(1)
    # espelhado: o ponto fica do lado de fora de cada adesivo
    assert 'class="sticker mirror" id="goRegister"' in m.group(1)
    assert "aspect-ratio:2.2 / 2.6" in html  # mesma proporção da porta em scene.js
    scene = (STATIC / "scene.js").read_text(encoding="utf-8")
    assert re.search(r"const W = 2\.2, H = 2\.6;", scene)
    # topo no estilo da inspiração: marca + "Entrar" + pílula "Cadastrar-se" (some quando há sessão)
    assert 'id="navLogin"' in html and 'class="pill" id="navRegister"' in html
    assert '$("topnav").hidden = !!me;' in html


def test_intro_screen_is_optional_and_shown_once(client):
    """A tela de carregamento aparece uma vez por aba, respeita "reduzir movimento", nunca trava."""
    html = client.get("/").text
    assert '<div id="intro" aria-hidden="true" hidden>' in html  # sem JS, não aparece
    assert 'sessionStorage.getItem("portaria:intro")' in html
    assert 'prefers-reduced-motion: reduce)").matches) return;' in html
    assert "if (el > 4000) target = 100;" in html  # termina mesmo se algo não carregar


def test_static_files_are_revalidated(client):
    for url in ("/", "/static/scene.js"):
        assert client.get(url).headers["cache-control"] == "no-cache"
    assert client.get("/api/status").headers["cache-control"] == "no-store"


def test_font_is_served_locally(client):
    font = client.get("/static/vendor/PlayfairDisplay-latin-400.woff2")
    assert font.status_code == 200 and font.content[:4] == b"wOF2"
    assert "SIL OPEN FONT LICENSE" in client.get("/static/vendor/PlayfairDisplay-OFL.txt").text
    html = client.get("/").text
    assert "/static/vendor/PlayfairDisplay-latin-400.woff2" in html
    word = re.search(r'<div class="bigword" aria-hidden="true">(.*?)</div>', html).group(1)
    assert re.sub(r"<[^>]+>", "", word) == "PORTARIA"  # palavra gigante decorativa, letra a letra
    assert '<h1 class="sr-only">' in html  # o título continua acessível


def test_home_layout_is_driven_by_the_word_size(client):
    """A palavra gigante (--fs) é limitada pela largura e pela altura da tela; a porta e os espaços
    derivam dela, então tudo mantém a proporção. A palavra cabe INTEIRA na largura da tela."""
    html = client.get("/").text
    assert "--fs:min(19.5vw, 40vh);" in html  # "PORTARIA" tem ~4,7x --fs de largura: ~91% da tela
    assert "--door-w:clamp(220px, min(calc(var(--fs) * 1.9), 42vh), 520px);" in html
    assert "--door-w:min(62vw, 260px);" in html  # celular
    assert "font-size:var(--fs);" in html and "width:var(--door-w)" in html
    assert "calc(var(--door-w) * .22 + 18px)" in html  # texto abaixo do anel de luz do chão
    assert "width:100vw; overflow-x:clip;" in html  # a névoa larga não cria rolagem lateral
    assert 'section[data-view="home"]:not([hidden]) { flex:1;' in html  # centralizada na vertical
    # o <body> não pode ter data-view: o roteador esconde todo elemento com esse atributo
    assert "document.body.dataset.screen = name;" in html and "body.dataset.view" not in html
