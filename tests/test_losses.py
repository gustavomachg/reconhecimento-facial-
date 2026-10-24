"""Losses e treino com tensores sintéticos (pulados se o PyTorch não estiver instalado)."""

import pytest

torch = pytest.importorskip("torch")

from facerec.losses import ArcFaceLoss, TripletLoss, build_loss, pairwise_distances  # noqa: E402


def clustered(n_ids=4, per=4, dim=16, sep=5.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_ids, dim, generator=g) * sep
    x = centers.repeat_interleave(per, 0) + 0.1 * torch.randn(n_ids * per, dim, generator=g)
    y = torch.arange(n_ids).repeat_interleave(per)
    return x, y


def test_pairwise_distances_match_cdist():
    x = torch.randn(10, 8)
    assert torch.allclose(pairwise_distances(x), torch.cdist(x, x), atol=1e-4)


def test_triplet_zero_when_well_separated():
    x, y = clustered()
    assert TripletLoss(margin=0.2)(x, y).item() == pytest.approx(0.0, abs=1e-6)


def test_triplet_positive_when_mixed():
    x = torch.randn(16, 16)
    y = torch.arange(4).repeat(4)
    assert TripletLoss(margin=0.2)(x, y).item() > 0


def test_triplet_has_gradients():
    x = torch.randn(16, 8, requires_grad=True)
    y = torch.arange(4).repeat(4)
    TripletLoss()(x, y).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_arcface_margin_increases_loss():
    x, y = clustered()
    torch.manual_seed(0)
    no_margin = ArcFaceLoss(16, 4, s=30, m=0.0)
    with_margin = ArcFaceLoss(16, 4, s=30, m=0.5)
    with_margin.load_state_dict(no_margin.state_dict())
    assert with_margin(x, y).item() > no_margin(x, y).item()


def test_arcface_learns_on_synthetic_data():
    torch.manual_seed(0)
    x, y = clustered(sep=1.0)
    emb = torch.nn.Linear(16, 8)
    loss_fn = build_loss({"name": "arcface", "s": 16, "m": 0.3}, n_classes=4, embedding_dim=8)
    opt = torch.optim.Adam(list(emb.parameters()) + list(loss_fn.parameters()), lr=0.05)
    first = loss_fn(emb(x), y).item()
    for _ in range(100):
        opt.zero_grad()
        loss = loss_fn(emb(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_configure_trainable_partial_freezes_rest():
    pytest.importorskip("facenet_pytorch")
    from facenet_pytorch import InceptionResnetV1

    from facerec.train import configure_trainable, count_trainable

    # classify=True cria a camada `logits`, como acontece ao carregar os pesos do VGGFace2
    model = InceptionResnetV1(pretrained=None, classify=True, num_classes=10)
    configure_trainable(model, "partial", ["block8", "last_linear", "last_bn"])
    n_train, n_total = count_trainable(model)
    assert 0 < n_train < n_total
    assert not any(p.requires_grad for p in model.conv2d_1a.parameters())
    configure_trainable(model, "full", [])
    assert count_trainable(model)[0] == n_total
    assert not any(p.requires_grad for p in model.logits.parameters())
