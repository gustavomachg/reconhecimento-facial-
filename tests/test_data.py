import numpy as np
import pytest
from PIL import Image, ImageFilter

from facerec.data.datasets import make_verification_pairs, parse_lfw_pairs
from facerec.data.dedup import dhash, find_cross_duplicates, hamming
from facerec.data.samplers import PKBatchSampler
from facerec.data.splits import (
    Record,
    assert_disjoint,
    split_by_identity,
    split_private_by_session,
)


def test_identity_split_is_disjoint_and_complete():
    ids = [f"id{i}" for i in range(100)]
    s = split_by_identity(ids, 0.15, 0.15, seed=1)
    assert set(s["train"]) | set(s["val"]) | set(s["test"]) == set(ids)
    assert_disjoint(s)
    assert len(s["test"]) == 15 and len(s["val"]) == 15


def test_identity_split_is_reproducible():
    ids = [f"id{i}" for i in range(50)]
    assert split_by_identity(ids, seed=7) == split_by_identity(ids, seed=7)


def test_assert_disjoint_detects_leak():
    with pytest.raises(ValueError):
        assert_disjoint({"train": ["a", "b"], "test": ["b"]})


def private_records(n_ids=6, sessions=("s1", "s2", "s3"), per=4):
    return [
        Record(f"P{i}/{s}/{k}.png", f"P{i}", s)
        for i in range(n_ids)
        for s in sessions
        for k in range(per)
    ]


def test_private_split_by_session_no_leakage():
    recs = private_records()
    sp = split_private_by_session(recs, ["s1"], ["s2"], ["s3"], 1, 1, seed=0)
    known = set(sp["known_identities"])
    assert {r.session for r in sp["gallery"]} == {"s1"}
    # nenhuma FOTO aparece em dois conjuntos
    assert_disjoint({k: [r.path for r in sp[k]] for k in ("gallery", "val", "test")})
    # conhecidos em val/test vêm só das sessões certas
    assert {r.session for r in sp["val"] if r.identity in known} == {"s2"}
    assert {r.session for r in sp["test"] if r.identity in known} == {"s3"}
    # desconhecidos nunca estão na galeria, e os de val != os de teste
    gallery_ids = {r.identity for r in sp["gallery"]}
    assert not gallery_ids & set(sp["unknown_val"])
    assert not gallery_ids & set(sp["unknown_test"])
    assert not set(sp["unknown_val"]) & set(sp["unknown_test"])


def test_private_split_rejects_overlapping_sessions():
    with pytest.raises(ValueError):
        split_private_by_session(private_records(), ["s1"], ["s1"], ["s3"])


def test_private_split_needs_enough_identities():
    with pytest.raises(ValueError):
        split_private_by_session(private_records(n_ids=3), ["s1"], ["s2"], ["s3"], 1, 1)


def test_verification_pairs_balanced_and_cross_session():
    recs = private_records()
    pairs = make_verification_pairs(recs, n_pairs=200, seed=0)
    labels = [lbl for *_, lbl in pairs]
    assert abs(sum(labels) - len(labels) / 2) <= 2
    sess = {r.path: r.session for r in recs}
    ident = {r.path: r.identity for r in recs}
    for a, b, lbl in pairs:
        assert (ident[a] == ident[b]) == bool(lbl)
        if lbl:
            assert sess[a] != sess[b]


def test_parse_lfw_pairs(tmp_path):
    f = tmp_path / "pairs.txt"
    f.write_text("2\t1\nAna\t1\t2\nAna\t1\tBob\t3\nCid\t2\t4\nCid\t1\tDan\t1\n", encoding="utf-8")
    folds = parse_lfw_pairs(f, tmp_path / "lfw", ext=".png")
    assert len(folds) == 2 and len(folds[0]) == 2
    a, b, lbl = folds[0][0]
    assert lbl == 1 and a.endswith("Ana_0001.png") and b.endswith("Ana_0002.png")
    assert folds[0][1][2] == 0 and folds[0][1][1].endswith("Bob_0003.png")


def test_pk_sampler_batches_have_positives():
    labels = [i // 5 for i in range(100)]  # 20 identidades x 5
    sampler = PKBatchSampler(labels, p=4, k=3, seed=0)
    for batch in sampler:
        assert len(batch) == 12
        ids = [labels[i] for i in batch]
        assert all(ids.count(x) == 3 for x in set(ids))


def synthetic_image(seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((128, 128), Image.Resampling.BICUBIC)


def test_dhash_near_duplicates(tmp_path):
    a = synthetic_image(0)
    a_resized = a.resize((100, 100)).filter(ImageFilter.GaussianBlur(0.5))
    b = synthetic_image(1)
    assert hamming(dhash(a), dhash(a_resized)) <= 16  # limite padrão (256 bits)
    assert hamming(dhash(a), dhash(b)) > 30

    paths = {}
    for name, img in {"a": a, "a2": a_resized, "b": b}.items():
        paths[name] = str(tmp_path / f"{name}.png")
        img.save(paths[name])
    dups = find_cross_duplicates({"train": [paths["a"]], "test": [paths["a2"], paths["b"]]})
    assert len(dups) == 1 and dups[0][3] == paths["a2"]
