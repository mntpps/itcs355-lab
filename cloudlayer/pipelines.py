"""Lab 5 — compile pipeline/pipeline.yaml into your provider's pipeline service.

This module is Layer 3, so provider specifics are allowed here and nowhere else. The DAG
stays in YAML; only the translation lives here. That separation is what you argue for or
against in Lab 5 Task 4.

    python -m cloudlayer.pipelines compile                     # writes pipeline/build/, no cloud calls
    python -m cloudlayer.pipelines submit [--set model_kind=logreg] [--wait]
    python -m cloudlayer.pipelines schedule                    # create/update the weekly schedule
    python -m cloudlayer.pipelines unschedule

Azure: compile_azure emits an Azure ML pipeline-job YAML and submission goes through the
Azure ML CLI (`az ml job create`). No azure-ai-ml SDK is needed locally; the CLI carries
its own, and the SDK does not install cleanly on every Python this course meets.

Two translation choices worth defending:

  * Conditions are evaluated INSIDE the step, by a shell guard the compiler wraps around
    its command, not with Azure ML's `dsl.condition`. That node needs a "control output"
    written through Azure's mldesigner library, which would put a provider import in the
    gate script. The guard reads the gate's decision file, which works identically on any
    provider. The cost: a skipped step still starts a container (about a minute) and shows
    as Completed rather than Skipped, so its log says why it did nothing.
  * Every run is forced to re-execute (`force_rerun`). Azure ML otherwise reuses a step
    whose inputs and command are unchanged, which for `deploy_staging` means a pipeline
    that reports a deploy it never did.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "pipeline" / "pipeline.yaml"
BUILD = ROOT / "pipeline" / "build"
AZ = shutil.which("az") or "az"

_VAR = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_REF = re.compile(r"\{(inputs|outputs|params)\.([a-z_][a-z0-9_]*)\}")

# Azure ML schedules take Windows time-zone names.
_WINDOWS_TZ = {"Asia/Bangkok": "SE Asia Standard Time", "UTC": "UTC"}


def _defaults() -> None:
    """cloud.env first (src.config loads it), then values derivable from the checkout."""
    sys.path.insert(0, str(ROOT))
    from src import config  # noqa: F401  (side effect: cloud.env -> os.environ)

    if not os.environ.get("IMAGE_TAG"):
        try:
            os.environ["IMAGE_TAG"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass


def load_spec(path: Path = SPEC) -> dict[str, Any]:
    """Read the neutral DAG and expand ${VAR} references from the environment."""
    _defaults()
    raw = path.read_text()
    missing = sorted({m for m in _VAR.findall(raw) if not os.environ.get(m)})
    if missing:
        raise RuntimeError(f"{path.name} needs {', '.join(missing)}: add them to cloud.env "
                           "(scripts/azure_lab5_setup.sh prints the values)")
    return yaml.safe_load(_VAR.sub(lambda m: os.environ[m.group(1)], raw))


def validate_spec(spec: dict[str, Any]) -> None:
    """Catch wiring mistakes before any provider sees them."""
    names = [s["name"] for s in spec["steps"]]
    produced = {s["name"]: set(s.get("outputs", [])) for s in spec["steps"]}
    for i, step in enumerate(spec["steps"]):
        for inp, src in (step.get("inputs") or {}).items():
            if src not in names[:i]:
                raise ValueError(f"{step['name']}: input {inp} comes from {src}, which is not "
                                 "an earlier step")
            if inp not in produced[src]:
                raise ValueError(f"{step['name']}: {src} has no output named {inp}")
        for kind, ref in _REF.findall(step["command"]):
            pool = {"inputs": step.get("inputs") or {}, "outputs": step.get("outputs", []),
                    "params": spec.get("params", {})}[kind]
            if ref not in pool:
                raise ValueError(f"{step['name']}: command uses {{{kind}.{ref}}}, not declared")
        cond = step.get("condition")
        if cond and cond["input"] not in (step.get("inputs") or {}):
            raise ValueError(f"{step['name']}: condition reads {cond['input']}, not an input")


def compile_aws(spec: dict[str, Any]):
    """SageMaker Pipelines.

    Each step becomes a ProcessingStep or TrainingStep; `condition` becomes a
    ConditionStep. Note that SageMaker's condition steps branch rather than abort, so
    `on_failure: abort` needs a FailStep on the else branch — an asymmetry worth
    mentioning in your Lab 5 write-up.
    """
    raise NotImplementedError("Not implemented: this project's pipeline runs on Azure ML")


# --- Azure ML ---------------------------------------------------------------------------
def _azure_command(step: dict[str, Any], workdir: str) -> str:
    def bind(m: re.Match) -> str:
        kind, ref = m.groups()
        return f"${{{{{'inputs' if kind == 'params' else kind}.{ref}}}}}"

    body = " ".join(_REF.sub(bind, step["command"]).split())
    # Azure ML runs commands in its own working directory; the image's code is at workdir.
    # The pipeline run's id is Azure's AZUREML_ROOT_RUN_ID; steps only ever see the
    # neutral PIPELINE_RUN_ID.
    prefix = f'export PIPELINE_RUN_ID="${{AZUREML_ROOT_RUN_ID:-unknown}}"; cd {workdir} && '
    cond = step.get("condition")
    if not cond:
        return prefix + body
    flag = f"${{{{inputs.{cond['input']}}}}}/{cond['file']}"
    skipped = " ".join(f"&& echo skipped > ${{{{outputs.{o}}}}}/SKIPPED" for o in step.get("outputs", []))
    return (f'{prefix}if [ "$(cat {flag} 2>/dev/null | tr -d "[:space:]")" = "{cond["equals"]}" ]; '
            f"then {body}; else echo \"condition {cond['input']} == {cond['equals']} is false: "
            f'{step["name"]} skipped" {skipped}; fi')


def compile_azure(spec: dict[str, Any]) -> dict[str, Any]:
    """Azure ML pipeline job (CLI v2 YAML schema).

    Each step becomes an inline command job running the training image. Inputs are bound
    to the producing step's output (`${{parent.jobs.<step>.outputs.<name>}}`), params to
    pipeline inputs, and the step's identity to a user-assigned managed identity attached
    to the compute cluster. AZURE_CLIENT_ID tells DefaultAzureCredential inside the step
    which of the cluster's identities to use.
    """
    validate_spec(spec)
    identities = spec.get("identities", {})
    env_common = {
        "CLOUD_PROVIDER": "azure",
        "PROJECT_ID": os.environ["PROJECT_ID"],
        "REGION": os.environ.get("REGION", ""),
        "MODEL_REGISTRY_NAME": os.environ["MODEL_REGISTRY_NAME"],
        "AZURE_SUBSCRIPTION_ID": os.environ["AZURE_SUBSCRIPTION_ID"],
        "CONTAINER_APP": os.environ.get("CONTAINER_APP", "itcs355-predict"),
        "GIT_COMMIT": os.environ["IMAGE_TAG"],
        # MLFLOW_TRACKING_URI is deliberately absent: Azure ML injects the workspace's
        # own, authenticated for the job.
    }
    jobs: dict[str, Any] = {}
    for step in spec["steps"]:
        if step.get("spot"):
            print(f"warning: {step['name']} asks for spot; the cluster's tier decides on Azure "
                  "ML, and this subscription's low-priority quota is 0, so it runs dedicated",
                  file=sys.stderr)
        inputs = {name: f"${{{{parent.jobs.{src}.outputs.{name}}}}}"
                  for name, src in (step.get("inputs") or {}).items()}
        for _, ref in (r for r in _REF.findall(step["command"]) if r[0] == "params"):
            inputs[ref] = f"${{{{parent.inputs.{ref}}}}}"
        job: dict[str, Any] = {
            "type": "command",
            "display_name": step["name"],
            "command": _azure_command(step, spec.get("workdir", "/app")),
            "environment": {"image": spec["image"]},
            "inputs": inputs,
            "outputs": {o: {"type": "uri_folder", "mode": "rw_mount"} for o in step.get("outputs", [])},
            "environment_variables": dict(env_common),
        }
        client_id = identities.get(step.get("identity", "training"))
        if client_id:
            job["identity"] = {"type": "managed", "client_id": client_id}
            job["environment_variables"]["AZURE_CLIENT_ID"] = client_id
        jobs[step["name"]] = job

    return {
        "$schema": "https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json",
        "type": "pipeline",
        "display_name": spec["name"],
        "experiment_name": spec.get("experiment", spec["name"]),
        "description": "ingest -> validate -> train -> evaluate -> register (gate) -> deploy_staging",
        "tags": {k: str(v) for k, v in spec.get("tags", {}).items()},
        "settings": {
            "default_compute": f"azureml:{spec['compute']}",
            "continue_on_step_failure": False,
            "force_rerun": True,
        },
        "inputs": {k: str(v) for k, v in spec.get("params", {}).items()},
        "jobs": jobs,
    }


def compile_gcp(spec: dict[str, Any]):
    """Vertex AI Pipelines.

    Kubeflow Pipelines underneath: steps become container components, conditions become
    dsl.If blocks. The compiled artifact is a JSON file you submit, so your build step
    produces a file rather than an object — different from the other two.
    """
    raise NotImplementedError("Not implemented: this project's pipeline runs on Azure ML")


COMPILERS = {"aws": compile_aws, "azure": compile_azure, "gcp": compile_gcp}


def compile_for(provider: str, spec: dict[str, Any] | None = None):
    spec = spec or load_spec()
    try:
        compiler = COMPILERS[provider.lower()]
    except KeyError:
        raise ValueError(f"No pipeline compiler for {provider!r}") from None
    return compiler(spec)


# --- Azure ML CLI: submit and schedule ---------------------------------------------------
def write_azure(spec: dict[str, Any]) -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / "azure-pipeline.yaml"
    out.write_text(yaml.safe_dump(compile_azure(spec), sort_keys=False, width=1000))
    return out


def _az(*args: str) -> str:
    ws = os.environ.get("AZURE_ML_WORKSPACE")
    if not ws:
        raise RuntimeError("AZURE_ML_WORKSPACE is not set in cloud.env")
    cmd = [AZ, "ml", *args, "-g", os.environ["PROJECT_ID"], "-w", ws]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[1:4])} failed:\n{result.stderr.strip()}")
    return result.stdout.replace("\r", "")


def submit_azure(spec: dict[str, Any], params: dict[str, str], wait: bool = False) -> str:
    path = write_azure(spec)
    sets: list[str] = []
    for k, v in params.items():
        if k not in spec.get("params", {}):
            raise ValueError(f"unknown param {k!r}; pipeline params: {sorted(spec['params'])}")
        sets += ["--set", f"inputs.{k}={v}"]
    job = json.loads(_az("job", "create", "--file", str(path), *sets, "-o", "json"))
    print(f"submitted {job['name']}  params {dict(spec['params'], **params)}")
    print(f"studio: {job.get('services', {}).get('Studio', {}).get('endpoint', '?')}")
    if wait:
        subprocess.run([AZ, "ml", "job", "stream", "-n", job["name"], "-g", os.environ["PROJECT_ID"],
                        "-w", os.environ["AZURE_ML_WORKSPACE"]], check=False)
        status = json.loads(_az("job", "show", "-n", job["name"], "-o", "json"))["status"]
        print(f"{job['name']}: {status}")
    return job["name"]


def schedule_azure(spec: dict[str, Any], name: str = "itcs355-retrain-weekly") -> str:
    path = write_azure(spec)
    sched = spec["schedule"]
    doc = {
        "$schema": "https://azuremlschemas.azureedge.net/latest/schedule.schema.json",
        "name": name,
        "display_name": f"{spec['name']} (weekly)",
        "description": "Scheduled retraining. The gate decides whether anything ships.",
        "tags": {k: str(v) for k, v in spec.get("tags", {}).items()},
        "trigger": {"type": "cron", "expression": sched["cron"],
                    "time_zone": _WINDOWS_TZ.get(sched.get("timezone", "UTC"), "UTC")},
        "create_job": f"./{path.name}",
    }
    sched_path = BUILD / "azure-schedule.yaml"
    sched_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    out = json.loads(_az("schedule", "create", "--file", str(sched_path), "-o", "json"))
    print(f"schedule {out['name']}: {sched['cron']} ({doc['trigger']['time_zone']}), "
          f"enabled={out.get('is_enabled')}")
    return out["name"]


def unschedule_azure(name: str = "itcs355-retrain-weekly") -> None:
    # A schedule must be disabled before it can be deleted.
    try:
        _az("schedule", "disable", "-n", name, "--no-wait")
    except RuntimeError as exc:
        if "not found" in str(exc).lower() or "ResourceNotFound" in str(exc):
            print(f"no schedule {name}")
            return
        raise
    _az("schedule", "delete", "-n", name, "--yes")
    print(f"deleted schedule {name}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m cloudlayer.pipelines")
    ap.add_argument("action", choices=("compile", "submit", "schedule", "unschedule"))
    ap.add_argument("--set", action="append", default=[], metavar="PARAM=VALUE")
    ap.add_argument("--wait", action="store_true", help="stream the run until it finishes")
    args = ap.parse_args(argv)

    spec = load_spec()
    provider = os.environ.get("CLOUD_PROVIDER", "local").lower()
    if provider != "azure":
        compile_for(provider, spec)
        return 0
    if args.action == "compile":
        print(f"wrote {write_azure(spec).relative_to(ROOT)}  (image {spec['image']})")
    elif args.action == "submit":
        submit_azure(spec, dict(kv.split("=", 1) for kv in args.set), wait=args.wait)
    elif args.action == "schedule":
        schedule_azure(spec)
    else:
        unschedule_azure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
