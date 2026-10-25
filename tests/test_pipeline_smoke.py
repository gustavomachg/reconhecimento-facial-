"""Teste de ponta a ponta da Parte 1 com dados SINTÉTICOS (sem internet, sem datasets).

Usa os YAMLs REAIS de configs/experiments/, trocando apenas caminhos e tamanhos, e roda:
treino (E03/E04), avaliação LFW 10-fold + validação privada (E01), comparadores (E06) e
threshold (E07). Garante que o pipeline roda em qualquer máquina com o requirements.txt.

As "faces" são padrões de ruído: cada identidade tem um padrão base, e cada foto é esse
padrão com ruído — imitam a estrutura "mesma pessoa = parecido".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from facerec.config import PROJECT_ROOT
from facerec.matching import l2_normalize

EXP = PROJECT_ROOT / "configs" / "experiments"


def _face(rng, base, size=160):
    noisy = np.clip(base.astype(int) + rng.integers(-25, 25, base.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy).resize((size, size), Image.Resampling.NEAREST)


def make_identities(root: Path, n_ids: int, per: int, sessions=None, lfw_names=False, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n_ids):
        base = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        name = f"Pessoa{i}" if lfw_names else f"P{i:02d}"
        for s in sessions or [None]:
            folder = root / name / s if s else root / name
            folder.mkdir(parents=True, exist_ok=True)
            for k in range(per):
                fname = f"{name}_{k + 1:04d}.png" if lfw_names else f"{k:03d}.png"
                _face(rng, base).save(folder / fname)


class PixelEmbedder:
    """Embedder determinístico e barato (pixels reduzidos) — substitui as redes nos testes
    de avaliação, para que os scores façam sentido sem baixar pesos."""

    name = "pixel"
    input_size = 160
    align_mode = "crop"

    def embed(self, faces, batch_size=64):
        faces = np.asarray(faces, dtype=np.float32)
        if faces.ndim == 3:
            faces = faces[None]
        small = faces.reshape(len(faces), 8, 20, 8, 20, 3).mean(axis=(2, 4))
        return l2_normalize(small.reshape(len(faces), -1) - 127.5)

    def num_parameters(self):
        return 0


def load_cfg(name: str) -> dict:
    with open(EXP / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump(cfg: dict, path: Path) -> str:
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


def private_block(cfg: dict, root: Path) -> None:
    cfg.update(gallery_sessions=["s1"], val_sessions=["s2"], test_sessions=["s3"])
    cfg.update(aligned_root=str(root), n_unknown_val=1, n_unknown_test=1)


@pytest.fixture(scope="module")
def private_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("private")
    make_identities(root, n_ids=6, per=3, sessions=["s1", "s2", "s3"], seed=1)
    return root


# ------------------------------------------------------------------ E01: avaliação
def test_evaluate_lfw_protocol_and_private_val(tmp_path, private_root, monkeypatch):
    import facerec.evaluate as ev

    lfw = tmp_path / "lfw"
    make_identities(lfw, n_ids=8, per=3, lfw_names=True, seed=2)
    rng = np.random.default_rng(0)
    lines, n_folds, n_per = [], 3, 4
    for _ in range(n_folds):
        for _ in range(n_per):
            i = rng.integers(8)
            lines.append(f"Pessoa{i}\t1\t{rng.integers(2, 4)}")
        for _ in range(n_per):
            i, j = rng.choice(8, 2, replace=False)
            lines.append(f"Pessoa{i}\t1\tPessoa{j}\t{rng.integers(1, 4)}")
    (lfw / "pairs.txt").write_text(f"{n_folds}\t{n_per}\n" + "\n".join(lines), encoding="utf-8")

    cfg = load_cfg("E01_facenet_baseline.yaml")
    cfg["evaluation"]["lfw"] = {"root": str(lfw), "pairs": str(lfw / "pairs.txt")}
    private_block(cfg["evaluation"]["private"], private_root)
    cfg["evaluation"]["private"]["n_pairs"] = 60
    cfg["output"]["results_dir"] = str(tmp_path / "E01")
    cfg["device"] = "cpu"
    monkeypatch.setattr(ev, "build_embedder", lambda *a, **k: PixelEmbedder())

    r = ev.evaluate(dump(cfg, tmp_path / "e01.yaml"))

    assert r["lfw"]["n_pairs"] == n_folds * 2 * n_per
    assert 0.0 <= r["lfw"]["eer"] <= 1.0 and r["lfw"]["accuracy_mean"] > 0.5
    assert {"accuracy", "f1", "eer", "roc_auc", "far", "frr"} <= set(r["private_val"])
    out = tmp_path / "E01"
    for f in (
        "metrics.json",
        "roc_lfw.png",
        "hist_lfw.png",
        "confusion_lfw.png",
        "roc_private_val.png",
        "confusion_private_val.png",
    ):
        assert (out / f).exists(), f
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "environment" in saved and "cost" in saved


# ------------------------------------------------------------------ E06 / E07
def test_private_open_set_experiments(tmp_path, private_root, monkeypatch):
    import facerec.experiments_private as ep

    monkeypatch.setattr(ep, "build_embedder", lambda *a, **k: PixelEmbedder())

    e6 = load_cfg("E06_matchers_open_set.yaml")
    private_block(e6["private"], private_root)
    e6["output"]["results_dir"] = str(tmp_path / "E06")
    r6 = ep.run_matcher_comparison(dump(e6, tmp_path / "e06.yaml"))
    assert set(r6["matchers"]) == {"cosine", "euclidean", "prototype", "knn_k3", "svm"}
    assert r6["split_evaluated"] == "validation"
    assert (tmp_path / "E06" / "matchers_comparison.png").exists()

    e7 = load_cfg("E07_threshold_sweep.yaml")
    private_block(e7["private"], private_root)
    e7["output"]["results_dir"] = str(tmp_path / "E07")
    r7 = ep.run_threshold_experiment(dump(e7, tmp_path / "e07.yaml"))
    assert r7["dataset"]["n_unknown_test_identities"] == 1
    assert r7["test"]["n_unknown_queries"] > 0
    for f in ("metrics.json", "threshold_sweep.png", "test_score_hist.png"):
        assert (tmp_path / "E07" / f).exists(), f
    # privacidade: o metrics.json não contém IDs dos participantes
    text = (tmp_path / "E07" / "metrics.json").read_text(encoding="utf-8")
    assert "P00" not in text and "P05" not in text


# ------------------------------------------------------------------ E03 / E04: treino
@pytest.mark.parametrize(
    "cfg_name", ["E03_facenet_partial_triplet.yaml", "E04_facenet_partial_arcface.yaml"]
)
def test_training_runs_end_to_end(tmp_path, cfg_name):
    pytest.importorskip("torch")
    pytest.importorskip("facenet_pytorch")
    from facerec.train import train

    data = tmp_path / "celeba_sintetico"
    make_identities(data, n_ids=9, per=4, seed=3)
    cfg = load_cfg(cfg_name)
    cfg["model"]["pretrained"] = None  # sem download de pesos no teste
    cfg["device"] = "cpu"
    cfg["data"].update(
        aligned_root=str(data), val_fraction=0.34, test_fraction=0.0, val_pairs=30, num_workers=0
    )
    cfg["training"].update(epochs=1, batch_size=8, batch={"p": 4, "k": 2})
    cfg["output"] = {"checkpoint": str(tmp_path / "ck.pt"), "results_dir": str(tmp_path / "out")}

    s = train(dump(cfg, tmp_path / "cfg.yaml"))

    assert s["best_epoch"] in (0, 1)
    assert 0 < s["n_trainable_parameters"] < s["n_parameters"]  # parcial: parte congelada
    assert (tmp_path / "ck.pt").exists()
    assert (tmp_path / "ck_last.pt").exists()  # última época, para análise de overfitting
    for f in ("train_summary.json", "loss_curves.png", "split.json", "config_used.yaml"):
        assert (tmp_path / "out" / f).exists(), f
    split = json.loads((tmp_path / "out" / "split.json").read_text(encoding="utf-8"))
    assert not set(split["train"]) & set(split["val"])  # identidades disjuntas
