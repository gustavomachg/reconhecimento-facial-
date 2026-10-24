"""ArcFace (insightface): iResNet-50 treinada com Additive Angular Margin Loss (WebFace600K).

Usamos o arquivo ONNX `w600k_r50.onnx` do pacote oficial `buffalo_l` do insightface,
executado diretamente com onnxruntime (evita compilar o pacote `insightface` no Windows;
ver ADR-001). Baixe com `python scripts/download_arcface.py`.

LICENÇA: os pesos do insightface são liberados apenas para uso NÃO comercial / acadêmico.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from facerec.models.base import Embedder

DEFAULT_ARCFACE_PATH = "checkpoints/public/buffalo_l/w600k_r50.onnx"


class ArcFaceEmbedder(Embedder):
    name = "arcface"
    input_size = 112
    align_mode = "similarity"
    embedding_dim = 512

    def __init__(self, model_path: str | Path = DEFAULT_ARCFACE_PATH, device: str = "cpu"):
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo ArcFace não encontrado em {path}. Rode: python scripts/download_arcface.py"
            )
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.model_path = path
        opts = ort.SessionOptions()
        # o ONNX declara batch fixo = 1 nos metadados, mas aceita batch maior; esconde o aviso
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _forward(self, batch):
        # insightface: entrada RGB, (x - 127.5) / 127.5, layout NCHW
        x = (batch.astype(np.float32) - 127.5) / 127.5
        x = np.transpose(x, (0, 3, 1, 2))
        return self.session.run(None, {self.input_name: x})[0].astype(np.float32)

    def num_parameters(self) -> int:
        import onnx
        from onnx import numpy_helper

        graph = onnx.load(str(self.model_path)).graph
        return int(sum(numpy_helper.to_array(t).size for t in graph.initializer))
