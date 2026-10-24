"""Avaliação da pasta `pessoas/` — somente dados SINTÉTICOS."""

import json

import numpy as np
import pytest

from facerec.evaluate_folder import (
    cross_validated_decisions,
    evaluate_open_set,
    list_people,
    open_set_queries,
    run,
    verification,
)
from tests.conftest import synthetic_identities
from tests.test_pipeline_smoke import PixelEmbedder, make_identities


def clustered(n_ids=6, per=5, noise=0.15, seed=0):
    x, y, _ = synthetic_identities(n_ids=n_ids, per_id=per, noise=noise, seed=seed)
    return x, np.array([int(lbl[2:]) for lbl in y])


def test_verification_separates_same_and_different():
    emb, labels = clustered()
    res, scores, same = verification(emb, labels)
    assert res["n_pairs_same"] == 6 * (5 * 4 // 2)
    assert res["n_pairs_same"] + res["n_pairs_diff"] == len(labels) * (len(labels) - 1) // 2
    assert res["roc_auc"] > 0.99 and res["eer"] < 0.05


def test_open_set_queries_known_and_unknown():
    emb, labels = clustered(n_ids=4, per=3)
    q = open_set_queries(emb, labels, {"name": "cosine"})
    assert sum(x["true"] != "__unknown__" for x in q) == 12  # uma consulta por foto
    assert sum(x["true"] == "__unknown__" for x in q) == 12  # cada pessoa tirada inteira
    # numa consulta desconhecida, a melhor identidade nunca é a própria pessoa (foi removida)
    assert all(x["top"] != str(x["person"]) for x in q if x["true"] == "__unknown__")


@pytest.mark.parametrize("matcher", ["cosine", "prototype", "knn"])
def test_open_set_recognizes_known_and_rejects_unknown(matcher):
    emb, labels = clustered()
    cfg = {"name": matcher, **({"k": 3} if matcher == "knn" else {})}
    r = evaluate_open_set(emb, labels, cfg)
    assert r["known_accuracy"] >= 0.9
    assert r["false_accept_unknown"] <= 0.1
    assert sum(r["confusion_accept_reject"].values()) == r["n_queries"]


def test_threshold_of_a_person_ignores_its_own_queries():
    """Sem vazamento: alterar os scores das consultas de um grupo não muda o threshold dele."""
    rng = np.random.default_rng(0)
    queries = [
        {
            "person": p,
            "true": str(p) if k % 2 else "__unknown__",
            "top": str(p),
            "score": float(rng.uniform(0.6, 0.9) if k % 2 else rng.uniform(0.1, 0.5)),
        }
        for p in range(10)
        for k in range(4)
    ]
    thr_a = cross_validated_decisions(queries, k=5, seed=1)
    target_person = queries[0]["person"]
    altered = [
        dict(q, score=q["score"] * 0.1) if q["person"] == target_person else q for q in queries
    ]
    thr_b = cross_validated_decisions(altered, k=5, seed=1)
    idx = [i for i, q in enumerate(queries) if q["person"] == target_person]
    assert all(thr_a[i] == thr_b[i] for i in idx)


def test_list_people_ignores_loose_files(tmp_path):
    make_identities(tmp_path, n_ids=3, per=2)
    (tmp_path / "solta.jpg").write_bytes(b"x")
    items, n, loose = list_people(tmp_path)
    assert n == 3 and len(items) == 6 and loose == 1
    assert {lbl for _, lbl in items} == {0, 1, 2}  # identidades = índices, não nomes


def test_run_end_to_end_without_leaking_names(tmp_path):
    root = tmp_path / "pessoas"
    make_identities(root, n_ids=5, per=4, seed=7)
    for i, d in enumerate(sorted(root.iterdir())):  # nomes "reais" que não podem vazar
        d.rename(root / f"NomeSecreto{i}")
    out = tmp_path / "out"
    summary = run(
        root,
        ["pixel"],
        [{"name": "cosine"}, {"name": "prototype"}],
        pre_aligned=True,
        quality=False,
        out_dir=out,
        device="cpu",
        embedder_factory=lambda m: PixelEmbedder(),
    )
    r = summary["results"]["pixel"]
    assert r["dataset"]["n_pessoas"] == 5 and r["dataset"]["n_fotos_validas"] == 20
    assert r["open_set"]["cosine"]["known_accuracy"] > 0.8
    for f in (
        "metrics.json",
        "results_table.md",
        "hist_pixel.png",
        "roc_comparison.png",
        "confusion_pixel_cosine.png",
    ):
        assert (out / f).exists(), f
    for f in ("metrics.json", "results_table.md"):
        text = (out / f).read_text(encoding="utf-8")
        assert "NomeSecreto" not in text
    json.loads((out / "metrics.json").read_text(encoding="utf-8"))


def test_run_requires_two_people(tmp_path):
    make_identities(tmp_path, n_ids=1, per=3)
    with pytest.raises(SystemExit):
        run(
            tmp_path,
            ["pixel"],
            [{"name": "cosine"}],
            True,
            False,
            tmp_path / "o",
            embedder_factory=lambda m: PixelEmbedder(),
        )


def test_save_aligned_previews_use_anonymous_names(tmp_path):
    root = tmp_path / "pessoas"
    make_identities(root, n_ids=2, per=2)
    for i, d in enumerate(sorted(root.iterdir())):
        d.rename(root / f"NomeSecreto{i}")
    prev = tmp_path / "preview"
    run(
        root,
        ["pixel"],
        [{"name": "cosine"}],
        True,
        False,
        tmp_path / "o",
        embedder_factory=lambda m: PixelEmbedder(),
        preview_dir=prev,
    )
    files = sorted(p.relative_to(prev).as_posix() for p in prev.rglob("*.png"))
    assert len(files) == 4 and all(f.startswith("pixel/pessoa_0") for f in files)
    assert not any("NomeSecreto" in f for f in files)
