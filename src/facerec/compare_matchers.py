"""E06 — comparação de comparadores no dataset privado (open-set, validação)."""

from __future__ import annotations

import argparse
import json

from facerec.experiments_private import run_matcher_comparison


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    s = run_matcher_comparison(args.config)
    print(json.dumps({k: v for k, v in s.items() if k != "environment"}, indent=2, default=str))


if __name__ == "__main__":
    main()
