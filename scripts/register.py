"""Lab 5 — register the candidate, only if the gate passed, with its lineage.

    python scripts/register.py --metrics m/metrics.json --gate gate/ --registered-out out/
    python scripts/register.py --adopt 2      # one-time: mark an existing version production

The pipeline only runs this step when the gate's decision is "pass" (the `condition` in
pipeline/pipeline.yaml). This script checks the decision again anyway: a register step
that trusts its caller is one refactor away from registering unconditionally.

Every registered version carries tags tracing it back to what made it:
  run_id            the training run (its params, metrics, commit, data fingerprint)
  git_commit        the image the pipeline ran, which is tagged by commit
  data_fingerprint  the exact batch it was trained on
  pipeline_run      the pipeline execution
  gate_*            what the gate measured: paired gain, the version it beat, the floors
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path)
    ap.add_argument("--gate", type=Path, help="the evaluate step's decision folder")
    ap.add_argument("--registered-out", type=Path, help="writes `version` here")
    ap.add_argument("--adopt", metavar="VERSION",
                    help="mark an already-registered version as production (bootstrap)")
    args = ap.parse_args()

    cfg = config.load(strict=False)
    name = cfg.model_registry_name

    if args.adopt:
        c = registry.client(cfg.mlflow_tracking_uri)
        replaced = registry.promote(c, name, args.adopt, {"adopted": "bootstrap"})
        print(f"{name} v{args.adopt} is now production"
              + (f" (v{replaced} retired)" if replaced else ""))
        return 0

    if not (args.metrics and args.gate and args.registered_out):
        ap.error("--metrics, --gate and --registered-out are required (or use --adopt)")
    args.registered_out.mkdir(parents=True, exist_ok=True)
    decision = (args.gate / "decision").read_text().strip()
    if decision != "pass":
        print(f"gate decision is {decision!r}: NOT registering")
        (args.registered_out / "SKIPPED").write_text(f"gate decision {decision}\n")
        return 0

    import mlflow

    c = registry.client(cfg.mlflow_tracking_uri)
    metrics = json.loads(args.metrics.read_text())
    verdict = json.loads((args.gate / "gate.json").read_text())
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mv = mlflow.register_model(f"runs:/{metrics['run_id']}/model", name)

    tags = {
        "run_id": metrics["run_id"],
        "git_commit": metrics.get("git_commit", "unknown"),
        "data_fingerprint": metrics["data_fingerprint"],
        "model_kind": metrics.get("model", "rf"),
        "test_roc_auc": round(metrics["test_roc_auc"], 4),
        "gate_paired_gain": verdict["paired_gain"],
        "gate_beat_version": verdict["incumbent_version"],
        "gate_min_improvement": verdict["min_improvement"],
        "pipeline_run": os.environ.get("PIPELINE_RUN_ID", "manual"),
    }
    replaced = registry.promote(c, name, mv.version, {k: v for k, v in tags.items() if v is not None})
    (args.registered_out / "version").write_text(f"{mv.version}\n")
    print(f"registered {name} v{mv.version} from run {metrics['run_id']} and made it production"
          + (f"; v{replaced} retired" if replaced else ""))
    for k, v in tags.items():
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
