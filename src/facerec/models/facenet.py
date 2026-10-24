"""FaceNet: InceptionResnetV1 pré-treinada no VGGFace2 (facenet-pytorch).

Os pesos pré-treinados são baixados automaticamente pelo facenet-pytorch da release
oficial do projeto (timesler/facenet-pytorch) para o cache do torch (fora do repositório).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from facerec.models.base import Embedder

# Nomes dos blocos da InceptionResnetV1, do início para o fim da rede.
FACENET_BLOCKS = [
    "conv2d_1a",
    "conv2d_2a",
    "conv2d_2b",
    "maxpool_3a",
    "conv2d_3b",
    "conv2d_4a",
    "conv2d_4b",
    "repeat_1",
    "mixed_6a",
    "repeat_2",
    "mixed_7a",
    "repeat_3",
    "block8",
    "avgpool_1a",
    "dropout",
    "last_linear",
    "last_bn",
]


def standardize(batch_uint8: np.ndarray):
    """(N,H,W,3) uint8 -> tensor (N,3,H,W) com a padronização do facenet-pytorch."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(batch_uint8)).permute(0, 3, 1, 2).float()
    return (x - 127.5) / 128.0


def build_facenet(pretrained: str | None = "vggface2", checkpoint: str | Path | None = None):
    import torch
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(pretrained=pretrained, classify=False)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state))
    return model


class FaceNetEmbedder(Embedder):
    name = "facenet"
    input_size = 160
    align_mode = "crop"
    embedding_dim = 512

    def __init__(self, device: str = "cpu", checkpoint: str | Path | None = None, model=None):
        import torch

        self.torch = torch
        self.device = device
        self.model = (model or build_facenet(checkpoint=checkpoint)).to(device).eval()

    def _forward(self, batch):
        with self.torch.no_grad():
            x = standardize(batch).to(self.device)
            return self.model(x).cpu().numpy().astype(np.float32)

    def num_parameters(self) -> int:
        # ignora `logits` (classificador das 8631 identidades do VGGFace2): não é usado no embedding
        return int(
            sum(p.numel() for n, p in self.model.named_parameters() if not n.startswith("logits"))
        )
