"""Lab 4 — entry point for the scheduled drift job (Azure Container Apps job, cron).

    python monitoring/scheduled.py

Everything comes from the environment, so the job definition is one command with no
arguments to quote:

    STAGING_URL          base URL of the service whose rolling window is scored
    MONITORING_TOKEN     shared token guarding /monitoring/window (job secret)
    ALERT_WEBHOOK_URL    Slack or Discord webhook (job secret)
    CLOUD_PROVIDER, PROJECT_ID, REGION, MODEL_REGISTRY_NAME, AZURE_SUBSCRIPTION_ID,
    CONTAINER_APP, AZURE_CLIENT_ID   for the Azure Monitor metric write

It rebuilds the training reference (deterministic, byte-identical to the DVC-tracked
data), then runs exactly the same detector as `python -m monitoring.drift`, with the
same thresholds. Exit codes are drift.py's, so an alerting run shows as a failed job
execution in the portal: 0 ok, 2 drift, 3 broken inputs, 1 the check itself failed.

Why a Container Apps job rather than GitHub Actions' schedule: on 2026-10-24/25 the
GitHub schedule (every 15 minutes) fired 4 times in 9 hours and then not at all for
5 hours, including the hour after the injected drift. A scheduler that does not run
cannot detect anything.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REFERENCE = Path(os.environ.get("REFERENCE_CSV", "/tmp/itcs355/sensors.csv"))
REPORT = Path(os.environ.get("DRIFT_REPORT", "/tmp/itcs355/drift.json"))


def build_reference(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("make_dataset", ROOT / "scripts" / "make_dataset.py")
    make_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_dataset)
    from src import seeds

    path.parent.mkdir(parents=True, exist_ok=True)
    make_dataset.build(seeds.DEFAULT_SEED).to_csv(path, index=False, lineterminator="\n")


def main() -> int:
    url = os.environ.get("STAGING_URL")
    if not url:
        print("STAGING_URL is not set", file=sys.stderr)
        return 1
    build_reference(REFERENCE)
    from monitoring import drift

    sys.argv = ["drift", "--reference", str(REFERENCE), "--current-url", url,
                "--out", str(REPORT), "--emit", "--notify"]
    return drift.main()


if __name__ == "__main__":
    raise SystemExit(main())
