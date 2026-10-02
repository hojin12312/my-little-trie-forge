#!/usr/bin/env python3
"""Refresh the official model IDs bundled with shell completion.

Uses `install/catalog.py` to update the versioned catalog, independently of
packaging. The model-catalog workflow runs this script.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "install/completions/official-models.txt"


def main(argv=None):
    sys.path.insert(0, str(ROOT))
    from install import catalog

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CATALOG)
    args = parser.parse_args(argv)

    if not catalog.refresh(destination=args.output):
        raise SystemExit(f"could not refresh the catalog from {catalog.COLLECTION}")
    model_ids = args.output.read_text(encoding="utf-8").splitlines()
    print(f"Wrote {len(model_ids)} official model IDs to {args.output}")


if __name__ == "__main__":
    main()
