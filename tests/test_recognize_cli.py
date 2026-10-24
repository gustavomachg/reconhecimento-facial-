"""`python -m facerec.recognize` com cadastro e consultas SINTÉTICOS (caminhos absolutos)."""

import sys

import facerec.recognize as rec
from tests.test_pipeline_smoke import PixelEmbedder, make_identities


def run_cli(monkeypatch, capsys, *args):
    monkeypatch.setattr(rec, "build_embedder", lambda *a, **k: PixelEmbedder())
    monkeypatch.setattr(sys, "argv", ["recognize", *args])
    rec.main()
    return capsys.readouterr().out


def test_recognizes_gallery_photos_leaving_each_out(tmp_path, monkeypatch, capsys):
    gallery = tmp_path / "galeria"
    make_identities(gallery, n_ids=3, per=3, seed=5)
    out = run_cli(
        monkeypatch,
        capsys,
        str(gallery),
        "--gallery",
        str(gallery),
        "--pre-aligned",
        "--threshold",
        "0.5",
    )
    assert "Cadastro: 3 pessoa(s), 9 foto(s)" in out
    assert "Acertos nas fotos do cadastro: 9/9" in out


def test_unknown_person_is_rejected(tmp_path, monkeypatch, capsys):
    gallery, stranger = tmp_path / "galeria", tmp_path / "estranho"
    make_identities(gallery, n_ids=3, per=3, seed=5)
    make_identities(stranger, n_ids=1, per=2, seed=99)
    out = run_cli(
        monkeypatch,
        capsys,
        str(stranger),
        "--gallery",
        str(gallery),
        "--pre-aligned",
        "--threshold",
        "0.9",
    )
    assert out.count("Pessoa desconhecida") == 2
    assert "Pessoa reconhecida" not in out
