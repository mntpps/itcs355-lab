"""Write the drift reference profile the serving image ships with.

    python scripts/build_reference_profile.py      # -> service/reference_profile.json

The reference is the TRAINING partition — the distribution the model actually learned
from — not the whole file. tests/test_unit.py regenerates it and fails if the committed
copy is stale, because a dashboard comparing live traffic against the wrong reference
shows drift that is not there, or hides drift that is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, seeds
from src.drift_stats import reference_profile

OUT = config.REPO_ROOT / "service" / "reference_profile.json"


def build() -> dict:
    cfg = config.load(strict=False)
    train_df, _, _ = data.split(data.load_raw(cfg.raw_path), seed=seeds.DEFAULT_SEED)
    return {
        "source": "training split of data/raw/sensors.csv",
        "seed": seeds.DEFAULT_SEED,
        "fingerprint": data.data_fingerprint(cfg.raw_path),
        "features": {f: reference_profile(train_df[f].to_numpy()) for f in data.FEATURES},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
