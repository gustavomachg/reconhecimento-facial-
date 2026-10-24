"""Leitura dos datasets (LFW, CelebA, privado) e Dataset PyTorch de faces alinhadas.

Estruturas esperadas (todas IGNORADAS pelo git):
    data/public/lfw/<Nome>/<Nome>_0001.jpg  + data/public/lfw/pairs.txt
    data/public/celeba/img_align_celeba/*.jpg + data/public/celeba/identity_CelebA.txt
    data/private/raw/<pessoa_id>/<sessao>/*.jpg     (pessoa_id é um código opaco, ex. P01)
Após `python -m facerec.data.preprocess`, as faces alinhadas ficam em
    data/<public|private>/aligned_<modelo>/<identidade>/[<sessao>/]arquivo.png
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from facerec.data.splits import Record

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def list_identity_folder(root: str | Path) -> list[Record]:
    """root/<identidade>/*.jpg -> registros."""
    root = Path(root)
    return [
        Record(str(img), d.name)
        for d in sorted(p for p in root.iterdir() if p.is_dir())
        for img in _images(d)
    ]


def list_private(root: str | Path) -> list[Record]:
    """root/<pessoa>/<sessao>/*.jpg -> registros com sessão."""
    root = Path(root)
    records = []
    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        for session in sorted(s for s in person.iterdir() if s.is_dir()):
            records += [Record(str(img), person.name, session.name) for img in _images(session)]
    return records


def load_celeba_records(
    identity_file: str | Path,
    image_dir: str | Path,
    min_images: int = 15,
    max_identities: int | None = 1000,
    seed: int = 42,
) -> list[Record]:
    """Subconjunto do CelebA: identidades com pelo menos `min_images` fotos."""
    by_id: dict[str, list[str]] = defaultdict(list)
    with open(identity_file, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                by_id[parts[1]].append(parts[0])
    eligible = sorted(i for i, imgs in by_id.items() if len(imgs) >= min_images)
    random.Random(seed).shuffle(eligible)
    if max_identities:
        eligible = eligible[:max_identities]
    image_dir = Path(image_dir)
    return [Record(str(image_dir / img), f"celeba_{i}") for i in eligible for img in by_id[i]]


def parse_lfw_pairs(
    pairs_file: str | Path, lfw_root: str | Path, ext: str = ".jpg"
) -> list[list[tuple]]:
    """Lê o pairs.txt oficial do LFW -> 10 folds de (caminho1, caminho2, mesma_pessoa).

    `lfw_root` pode ser a pasta original (.jpg) ou a pasta de faces alinhadas (.png)."""
    lfw_root = Path(lfw_root)

    def path(name: str, n: str) -> str:
        return str(lfw_root / name / f"{name}_{int(n):04d}{ext}")

    with open(pairs_file, encoding="utf-8") as f:
        header = f.readline().split()
        n_folds, n_per = int(header[0]), int(header[1])
        lines = [ln.split() for ln in f if ln.strip()]
    folds, i = [], 0
    for _ in range(n_folds):
        fold = []
        for parts in lines[i : i + n_per]:  # pares positivos: nome n1 n2
            fold.append((path(parts[0], parts[1]), path(parts[0], parts[2]), 1))
        for parts in lines[i + n_per : i + 2 * n_per]:  # negativos: nome1 n1 nome2 n2
            fold.append((path(parts[0], parts[1]), path(parts[2], parts[3]), 0))
        folds.append(fold)
        i += 2 * n_per
    return folds


def make_verification_pairs(
    records: list[Record], n_pairs: int = 3000, seed: int = 42, cross_session: bool = True
) -> list[tuple[str, str, int]]:
    """Gera pares balanceados (metade genuínos, metade impostores) para validação.

    Com `cross_session=True`, pares genuínos sempre combinam sessões diferentes — é o caso
    realista (cadastro num dia, reconhecimento em outro).
    """
    rng = random.Random(seed)
    by_id: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_id[r.identity].append(r)
    ids = [i for i, rs in by_id.items() if len(rs) >= 2]
    if len(by_id) < 2 or not ids:
        raise ValueError("São necessárias >=2 identidades e alguma com >=2 imagens")
    pairs: set[tuple[str, str, int]] = set()
    attempts = 0
    while len(pairs) < n_pairs // 2 and attempts < n_pairs * 50:
        attempts += 1
        a, b = rng.sample(by_id[rng.choice(ids)], 2)
        multi_session = len({r.session for r in by_id[a.identity]}) > 1
        if cross_session and multi_session and a.session == b.session:
            continue
        pairs.add((a.path, b.path, 1))
    all_ids = list(by_id)
    while len(pairs) < n_pairs and attempts < n_pairs * 100:
        attempts += 1
        i, j = rng.sample(all_ids, 2)
        pairs.add((rng.choice(by_id[i]).path, rng.choice(by_id[j]).path, 0))
    return sorted(pairs)


class FaceImageDataset:
    """Dataset PyTorch de faces JÁ alinhadas: devolve (tensor, índice_da_identidade)."""

    def __init__(self, records: list[Record], transform, label_map: dict[str, int] | None = None):
        self.records = records
        self.transform = transform
        ids = sorted({r.identity for r in records})
        self.label_map = label_map or {k: i for i, k in enumerate(ids)}
        self.labels = [self.label_map[r.identity] for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        from PIL import Image

        img = Image.open(self.records[idx].path).convert("RGB")
        return self.transform(img), self.labels[idx]
