"""Transforms. Data augmentation SÓ no treino e SÓ depois do split (ADR-004).

Augmentation razoável para faces alinhadas: flip horizontal, pequenas variações de
brilho/contraste/saturação, rotação e recorte leves. Nada que destrua a identidade
(ex.: flip vertical ou recortes grandes).
"""

from __future__ import annotations


class Standardize:
    """Tensor [0,1] -> (x*255 - 127.5) / 128, a padronização do facenet-pytorch.
    Classe (e não lambda) para funcionar com DataLoader multiprocesso no Windows."""

    def __call__(self, x):
        return (x * 255.0 - 127.5) / 128.0


def build_transforms(train: bool, image_size: int = 160, augment: dict | None = None):
    from torchvision import transforms as T

    augment = augment or {}
    if not train or not augment.get("enabled", True):
        return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), Standardize()])
    return T.Compose(
        [
            T.RandomResizedCrop(
                image_size, scale=(augment.get("min_scale", 0.85), 1.0), ratio=(0.95, 1.05)
            ),
            T.RandomHorizontalFlip(),
            T.RandomApply(
                [T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)],
                p=augment.get("color_p", 0.8),
            ),
            T.RandomRotation(augment.get("rotation", 10)),
            T.RandomGrayscale(p=augment.get("gray_p", 0.1)),
            T.ToTensor(),
            Standardize(),
        ]
    )
