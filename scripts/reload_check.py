"""Lab 2 — prove the registered model can be reloaded by version, from the registry.

    python scripts/reload_check.py --name itcs355-<studentid> --version 3

This is the lab's quiet test. Models that cannot be reloaded six months later are the
commonest form of dead work in industry, and the cause is nearly always a serialization
assumption: a custom class that no longer exists, a library version that moved, a
preprocessing step that only ever lived in a notebook.

Loading from a local file instead of the registry defeats the purpose and is checked.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow

from src import config, data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="registered model name")
    ap.add_argument("--version", required=True)
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()

    cfg = config.load(strict=False)
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    uri = f"models:/{args.name}/{args.version}"
    print(f"loading {uri}")
    try:
        model = mlflow.sklearn.load_model(uri)
    except TypeError as exc:
        if "azureml_artifacts_builder() got an unexpected keyword argument" not in str(exc):
            raise

        az = shutil.which("az")
        if az is None:
            raise RuntimeError("Azure CLI (az) is required for the registry fallback") from exc

        match = re.search(r"/workspaces/([^/]+)", cfg.mlflow_tracking_uri)
        if match is None:
            raise RuntimeError("Could not determine Azure ML workspace from MLFLOW_TRACKING_URI") from exc
        workspace = match.group(1)

        with tempfile.TemporaryDirectory(prefix="lab2-reload-") as tmp:
            subprocess.run([
                az, "ml", "model", "download",
                "--name", args.name,
                "--version", str(args.version),
                "--download-path", tmp,
                "--resource-group", cfg.project_id,
                "--workspace-name", workspace,
                "--only-show-errors",
            ], check=True)

            model_dirs = list(Path(tmp).rglob("MLmodel"))
            if not model_dirs:
                raise RuntimeError("Azure ML registry download did not contain an MLmodel file")
            model = mlflow.sklearn.load_model(str(model_dirs[0].parent))

    df = data.load_raw(cfg.raw_path)
    _, _, test_df = data.split(df, seed=20260101)
    sample = test_df.head(args.rows)
    preds = model.predict_proba(sample[data.FEATURES])[:, 1]

    for rid, p in zip(sample[data.ID], preds):
        print(f"  reading {rid}: p(failure)={p:.4f}")
    print("\nPASS  model reloaded from the registry and scored rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
