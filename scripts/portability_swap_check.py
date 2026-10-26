"""Lab 5 Task 4 — prove the portability seam is real.

    python scripts/portability_swap_check.py --second-provider aws --env cloud.aws.env

Exercises upload, download, and invoke against a SECOND provider's adapter. You are not
migrating the whole system; you are proving the seam exists.

It runs twice, on purpose:

  1. naive swap: the primary Config with only CLOUD_PROVIDER changed, which is what the
     scaffold did. It fails, because BLOB_URI and the endpoint are Azure values. That
     failure is the first finding: the seam is the adapter AND a per-provider config.
  2. real swap: the second provider's own env file (--env). This one has to pass.

invoke sends one real reading (the smoke test's healthy machine) to the serving
endpoint and checks a probability comes back. An empty payload proves nothing about
whether the service works.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_test import HEALTHY  # noqa: E402

from cloudlayer.factory import get_adapter  # noqa: E402
from src import config  # noqa: E402


def exercise(cfg, endpoint: str) -> dict[str, str]:
    adapter = get_adapter(cfg)
    results: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.txt"
        probe.write_text("itcs355 portability probe\n")
        back = Path(tmp) / "back.txt"

        def download():
            adapter.download(f"{cfg.blob_uri.rstrip('/')}/portability/probe.txt", str(back))
            if back.read_text() != probe.read_text():
                raise AssertionError("downloaded bytes differ from uploaded")
            return "round trip identical"

        def invoke():
            out = adapter.invoke(endpoint, HEALTHY)
            if not 0.0 <= float(out["probability"]) <= 1.0:
                raise AssertionError(f"no probability in {out}")
            return f"probability {out['probability']:.4f}, model_version {out.get('model_version')}"

        for name, call in (
            ("upload", lambda: adapter.upload(str(probe), "portability/probe.txt")),
            ("download", download),
            ("invoke", invoke),
        ):
            try:
                results[name] = f"PASS — {call()}"
            except NotImplementedError:
                results[name] = "NOT IMPLEMENTED"
            except Exception as exc:  # noqa: BLE001 - every failure is a finding
                results[name] = f"FAIL — {type(exc).__name__}: {str(exc)[:160]}"
    return results


def show(title: str, results: dict[str, str]) -> None:
    print(title)
    for name, outcome in results.items():
        verdict, _, detail = outcome.partition(" — ")
        print(f"  [{verdict:<15}] {name:<8} {detail}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--second-provider", required=True, choices=["aws", "azure", "gcp"])
    ap.add_argument("--env", type=Path, help="the second provider's env file, e.g. cloud.aws.env")
    ap.add_argument("--skip-naive", action="store_true")
    args = ap.parse_args()

    primary = config.load(strict=False)
    if primary.provider == args.second_provider:
        print("The second provider must differ from CLOUD_PROVIDER.")
        return 1
    print(f"primary   {primary.provider}\nsecondary {args.second_provider}\n")

    if not args.skip_naive:
        naive = config.Config(**{**primary.__dict__, "provider": args.second_provider})
        show("1. naive swap: same cloud.env, only the provider changed",
             exercise(naive, primary.project_id))

    if not args.env:
        print("Pass --env <file> with the second provider's capability slots for the real swap.")
        return 1
    second = config.load_file(args.env)
    if second.provider != args.second_provider:
        print(f"{args.env} says CLOUD_PROVIDER={second.provider}, not {args.second_provider}")
        return 1
    endpoint = config.read_env_file(args.env).get("SERVING_ENDPOINT", second.project_id)
    results = exercise(second, endpoint)
    show(f"2. real swap: {args.env} (endpoint {endpoint})", results)

    print("Write-up (Lab 5 Task 4): README, 'Portability test'.")
    return 0 if all(v.startswith("PASS") for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
