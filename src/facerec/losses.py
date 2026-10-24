"""Funções de perda para aprendizado de embeddings faciais.

TripletLoss (FaceNet, Schroff et al. 2015) com *batch-hard mining* (Hermans et al. 2017):
    para cada âncora a do batch:
        d(a,p)  = distância ao POSITIVO mais distante (mesma pessoa)
        d(a,n)  = distância ao NEGATIVO mais próximo (outra pessoa)
        L = max(0, d(a,p) - d(a,n) + margem)
    Os pares "difíceis" são os que mais ensinam; triplets fáceis dão perda zero.

ArcFace / Additive Angular Margin (Deng et al. 2019):
    logits = s * cos(theta_y + m) para a classe correta e s * cos(theta_j) para as demais,
    onde theta é o ângulo entre o embedding normalizado e o peso normalizado de cada classe.
    A margem m em ÂNGULO força cada identidade a ocupar uma região angular compacta,
    separada das outras -> embeddings mais discriminativos. Depois aplica-se cross-entropy.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def pairwise_distances(emb: torch.Tensor) -> torch.Tensor:
    """Distâncias euclidianas entre todas as linhas (N,D) -> (N,N)."""
    dot = emb @ emb.T
    sq = dot.diag()
    d2 = (sq[:, None] - 2 * dot + sq[None, :]).clamp(min=0)
    # sqrt(0) tem gradiente infinito: soma um epsilon só onde é zero
    mask = (d2 == 0).float()
    return torch.sqrt(d2 + mask * 1e-16) * (1 - mask)


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(embeddings, dim=1)
        dist = pairwise_distances(emb)
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        pos_mask = same & ~eye
        neg_mask = ~same
        # positivo mais distante (hard positive)
        hardest_pos = (dist * pos_mask.float()).max(dim=1).values
        # negativo mais próximo (hard negative): soma um valor grande onde não é negativo
        big = dist.max().detach() + 1.0
        hardest_neg = (dist + big * (~neg_mask).float()).min(dim=1).values
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss[valid].mean()


class ArcMarginProduct(nn.Module):
    """Cabeça de classificação do ArcFace. Só é usada no TREINO; descartada na inferência."""

    def __init__(self, in_features: int, n_classes: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        # para theta + m > pi a função deixa de ser monotônica; usa a correção do paper
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(
            -1 + 1e-7, 1 - 1e-7
        )
        sine = torch.sqrt(1.0 - cosine**2)
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, cosine.size(1)).float()
        return self.s * (one_hot * phi + (1 - one_hot) * cosine)


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features: int, n_classes: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.head = ArcMarginProduct(in_features, n_classes, s, m)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.head(embeddings, labels), labels)


def build_loss(cfg: dict, n_classes: int, embedding_dim: int = 512) -> nn.Module:
    name = cfg["name"]
    if name == "triplet":
        return TripletLoss(margin=cfg.get("margin", 0.2))
    if name == "arcface":
        return ArcFaceLoss(embedding_dim, n_classes, s=cfg.get("s", 30.0), m=cfg.get("m", 0.5))
    raise ValueError(f"Loss desconhecida: {name}")
