"""Lab 5 — deploy a newly registered model version to staging, smoke-test it, roll back
if it misbehaves.

    python scripts/deploy.py --target staging --registered out/

The serving image does not change: it loads the model version named in MODEL_VERSION at
startup (service/app.py). So deploying a model is a configuration rollout. The adapter's
`rollout_model` creates a new revision of the staging app with the new MODEL_VERSION,
then the same three-payload smoke test CD uses (scripts/smoke_test.py) must pass with
/ready reporting the new version. If it does not, the previous version is rolled back
and the step fails, so the pipeline run shows red.

The registry still says the new version is production when a rollback happens, so this
step also re-promotes the previous version: the registry and staging never disagree.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_test import run as smoke  # noqa: E402

from cloudlayer.factory import get_adapter  # noqa: E402
from src import config, registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("staging",), default="staging")
    ap.add_argument("--registered", type=Path, required=True,
                    help="the register step's output folder (holds `version`)")
    ap.add_argument("--app", default=os.environ.get("CONTAINER_APP", "itcs355-predict"))
    ap.add_argument("--ready-timeout", type=float, default=420)
    args = ap.parse_args()

    marker = args.registered / "version"
    if not marker.exists():
        print("nothing was registered: no deploy")
        return 0
    version = marker.read_text().strip()

    cfg = config.load(strict=False)
    adapter = get_adapter(cfg)
    url, previous = adapter.rollout_model(args.app, version)
    print(f"{args.target}: {url} now loading {cfg.model_registry_name} v{version} "
          f"(was v{previous})")

    if smoke(url, expect_version=version, ready_timeout=args.ready_timeout) == 0:
        print(f"deployed v{version} to {args.target}")
        return 0

    print(f"smoke test failed: rolling {args.target} back to v{previous}")
    adapter.rollout_model(args.app, previous)
    c = registry.client(cfg.mlflow_tracking_uri)
    registry.promote(c, cfg.model_registry_name, previous, {"restored_after_failed": version})
    c.set_model_version_tag(cfg.model_registry_name, version, "deploy", "failed-smoke-test")
    smoke(url, expect_version=previous, ready_timeout=args.ready_timeout)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
