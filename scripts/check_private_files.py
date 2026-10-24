"""Hook de pre-commit: bloqueia o commit de dados/modelos privados (LGPD — ver docs/privacy.md).

Uso pelo pre-commit:   python scripts/check_private_files.py <arquivos staged...>
Varredura manual:      python scripts/check_private_files.py --all       (arquivos rastreados)
Histórico do git:      python scripts/check_private_files.py --history   (todos os commits)

Sai com código 1 (bloqueando o commit) se algum arquivo:
- estiver dentro de uma pasta proibida (qualquer coisa em data/ exceto data/README.md,
  database, faces, embeddings, checkpoints, consent, pessoas); ou
- tiver extensão proibida (imagens, vídeos, bancos, checkpoints, arrays) FORA das pastas
  permitidas (docs/**.png e experiments/results/**.png); ou
- for um .env.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath

FORBIDDEN_DIRS = (
    "data/public/",
    "data/private/",
    "database/",
    "faces/",
    "embeddings/",
    "checkpoints/",
    "consent/",
    "pessoas/",
)
FORBIDDEN_EXTS = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".bin",
    ".safetensors",
    ".npy",
    ".npz",
    ".pkl",
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".webp",
    ".bmp",
    ".gif",
    ".mp4",
    ".avi",
    ".mov",
    ".zip",
    ".tgz",
    ".tar",
    ".gz",
}
# Só gráficos agregados (PNG) podem ser versionados, e só nestas pastas.
ALLOWED_PREFIXES = {".png": ("docs/", "experiments/results/")}


def violation(path: str) -> str | None:
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    low = p.lower()
    if low.startswith("data/") and low != "data/readme.md":
        return "pasta de dados (data/ só pode conter README.md)"
    for d in FORBIDDEN_DIRS:
        if low.startswith(d) or f"/{d}" in f"/{low}":
            return f"pasta proibida ({d})"
    name = PurePosixPath(low).name
    if name == ".env" or name.startswith(".env."):
        return "arquivo de ambiente (.env)"
    ext = PurePosixPath(low).suffix
    if ext in FORBIDDEN_EXTS:
        allowed = ALLOWED_PREFIXES.get(ext, ())
        if not any(low.startswith(a) for a in allowed):
            return f"extensão proibida ({ext})"
    return None


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--all":
        files = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    elif argv and argv[0] == "--history":
        # adicionar ao .gitignore depois NÃO remove o que já foi commitado: varre todo o histórico
        out = subprocess.run(
            ["git", "log", "--all", "--name-only", "--pretty=format:"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        files = sorted({f for f in out.splitlines() if f.strip()})
        print(f"Arquivos que já passaram pelo histórico: {len(files)}")
    else:
        files = argv
    bad = [(f, v) for f in files if (v := violation(f))]
    if bad:
        print("COMMIT BLOQUEADO — possíveis dados/modelos privados (LGPD):")
        for f, v in bad:
            print(f"  - {f}: {v}")
        print("Remova do stage (git restore --staged <arquivo>). Se já estiver no histórico,")
        print("é preciso reescrevê-lo (ex.: git filter-repo) — o .gitignore não basta.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
