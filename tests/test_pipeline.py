"""Lab 5 Task 1 — the pipeline's gate, registry and compiler. Fast, no cloud.

The grader "will try to make the gate fail". These tests are that attempt, made first:
the same recipe retrained must not pass, a broken model must not pass, a real
improvement must pass, and a compiled register step must not run without a "pass".
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from src import config, data, gate

ROOT = config.REPO_ROOT
PY = sys.executable


# --- the rule ----------------------------------------------------------------------------

def test_gate_rejects_a_gain_inside_the_noise():
    decision, reasons = gate.decide(0.8457, 0.8400, baseline=0.80)
    assert decision == "fail" and "noise" in reasons[0]


def test_gate_passes_a_gain_above_the_noise():
    assert gate.decide(0.8550, 0.8400, baseline=0.80) == ("pass", [])


def test_gate_rejects_a_model_below_the_absolute_floor_even_without_an_incumbent():
    decision, reasons = gate.decide(0.70, None, baseline=0.60)
    assert decision == "fail" and "floor" in reasons[0]


def test_gate_rejects_a_model_that_cannot_beat_one_sensor():
    decision, reasons = gate.decide(0.81, None, baseline=0.82)
    assert decision == "fail" and "single feature" in reasons[0]


def test_first_registration_needs_only_the_floors():
    assert gate.decide(0.84, None, baseline=0.80)[0] == "pass"


def test_thresholds_match_the_committed_study():
    """If someone edits a threshold, the study that justifies it must be re-run."""
    study = json.loads((ROOT / "reports" / "lab5-gate.json").read_text())
    assert study["thresholds"] == {"MIN_ABSOLUTE": gate.MIN_ABSOLUTE,
                                   "MIN_IMPROVEMENT": gate.MIN_IMPROVEMENT}
    assert gate.MIN_IMPROVEMENT > study["null_delta"]["p90"] - 0.001
    assert study["pass_rate"]["degraded (must be 0)"] == 0.0
    assert gate.MIN_ABSOLUTE < study["same_recipe_auc"]["min"]


# --- the registry -------------------------------------------------------------------------

@pytest.fixture()
def tiny_registry(tmp_path, monkeypatch):
    """A sqlite MLflow registry with two versions of a model, neither in production."""
    pytest.importorskip("mlflow")
    import mlflow
    from sklearn.dummy import DummyClassifier

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("t", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("t")
    for _ in range(2):
        with mlflow.start_run():
            mlflow.sklearn.log_model(DummyClassifier().fit([[0], [1]], [0, 1]), name="model",
                                     registered_model_name="m")
    return uri


def test_promote_keeps_exactly_one_production_version(tiny_registry):
    from src import registry

    c = registry.client(tiny_registry)
    assert registry.production_version(c, "m") is None
    assert registry.promote(c, "m", "1") is None
    assert registry.promote(c, "m", "1") is None   # re-promoting is a no-op, not a self-retire
    assert registry.promote(c, "m", "2", {"gate_paired_gain": 0.02}) == "1"
    prod = registry.production_version(c, "m")
    assert str(prod.version) == "2" and prod.tags["gate_paired_gain"] == "0.02"
    assert c.get_model_version("m", "1").tags["role"] == registry.RETIRED


def test_unknown_model_has_no_production_version(tiny_registry):
    from src import registry

    assert registry.production_version(registry.client(tiny_registry), "never-registered") is None


def test_registration_never_asks_for_logged_models(tiny_registry, monkeypatch, tmp_path):
    """Azure ML has no logged-models API; any MLflow path that calls it 404s there.
    Regression test for the register step that failed live twice."""
    import importlib.util

    from mlflow.tracking import MlflowClient

    from src import registry

    def no_such_api(*a, **k):
        raise AssertionError("search_logged_models called: this 404s on Azure ML")
    monkeypatch.setattr(MlflowClient, "search_logged_models", no_such_api)

    spec = importlib.util.spec_from_file_location("register", ROOT / "scripts" / "register.py")
    register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(register)
    import mlflow
    from sklearn.dummy import DummyClassifier

    # Log the model exactly as src/train.py does: saved to a folder, then logged as the
    # run's `model` artifacts (not mlflow.sklearn.log_model, which makes a logged model).
    model_dir = tmp_path / "model"
    mlflow.sklearn.save_model(DummyClassifier().fit([[0], [1]], [0, 1]), str(model_dir))
    with mlflow.start_run() as run:
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
    c = registry.client(tiny_registry)
    mv, how = register.create_version(c, "m", run.info.run_id, tiny_registry, model_dir)
    assert (str(mv.version), how) == ("3", "run artifacts")

    # If the registry refuses the run's artifact location, the step's own folder is used.
    real = MlflowClient.create_model_version
    def refuse_run_sources(self, name, source, run_id=None, **kw):
        if run_id is not None:
            raise RuntimeError("simulated refusal")
        return real(self, name, source, run_id=run_id, **kw)
    monkeypatch.setattr(MlflowClient, "create_model_version", refuse_run_sources)
    mv, how = register.create_version(c, "m", run.info.run_id, tiny_registry, model_dir)
    assert (str(mv.version), how) == ("4", "train step output")


def test_register_refuses_when_the_gate_failed(tmp_path):
    gate_dir, out = tmp_path / "gate", tmp_path / "out"
    gate_dir.mkdir()
    (gate_dir / "decision").write_text("fail\n")
    (tmp_path / "metrics.json").write_text("{}")
    r = subprocess.run([PY, str(ROOT / "scripts" / "register.py"), "--metrics",
                        str(tmp_path / "metrics.json"), "--gate", str(gate_dir),
                        "--registered-out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "NOT registering" in r.stdout
    assert (out / "SKIPPED").exists() and not (out / "version").exists()


def test_deploy_does_nothing_when_nothing_was_registered(tmp_path):
    (tmp_path / "SKIPPED").write_text("gate decision fail\n")
    r = subprocess.run([PY, str(ROOT / "scripts" / "deploy.py"), "--registered", str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "no deploy" in r.stdout


# --- ingest ------------------------------------------------------------------------------

def test_weekly_seed_is_stable_within_a_week_and_changes_across_weeks():
    import datetime as dt

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_dataset import weekly_seed

    monday, sunday = dt.date(2026, 10, 26), dt.date(2026, 11, 1)
    assert weekly_seed(monday) == weekly_seed(sunday) == 202644
    assert weekly_seed(monday + dt.timedelta(days=7)) == 202645


def test_fahrenheit_fault_fails_the_contract(tmp_path):
    out = tmp_path / "sensors.csv"
    subprocess.run([PY, str(ROOT / "scripts" / "make_dataset.py"), "--fault", "fahrenheit",
                    "--out", str(out)], check=True, capture_output=True)
    df = data.load_raw(out)
    lo, hi = data.PLAUSIBLE_RANGES["temp_c"]
    assert df["temp_c"].max() > hi


# --- the compiler ------------------------------------------------------------------------

@pytest.fixture()
def compiled(monkeypatch):
    pytest.importorskip("yaml")
    from cloudlayer import pipelines

    for k, v in {"CONTAINER_REGISTRY": "reg.example/itcs355", "IMAGE_TAG": "abc123",
                 "TRAIN_INSTANCE": "cpu", "PROJECT_ID": "rg", "MODEL_REGISTRY_NAME": "m",
                 "AZURE_SUBSCRIPTION_ID": "sub", "TRAIN_IDENTITY_CLIENT_ID": "train-id",
                 "DEPLOY_IDENTITY_CLIENT_ID": "deploy-id"}.items():
        monkeypatch.setenv(k, v)
    return pipelines.compile_azure(pipelines.load_spec())


def test_compiled_dag_has_every_stage_in_order(compiled):
    assert list(compiled["jobs"]) == ["ingest", "validate", "train", "evaluate", "register",
                                      "deploy_staging"]
    assert compiled["settings"]["continue_on_step_failure"] is False
    assert compiled["settings"]["force_rerun"] is True


def test_train_cannot_start_on_unvalidated_data(compiled):
    assert compiled["jobs"]["train"]["inputs"]["validated_data"] == \
        "${{parent.jobs.validate.outputs.validated_data}}"


def test_register_and_deploy_are_guarded_by_the_gate(compiled):
    for name in ("register", "deploy_staging"):
        cmd = compiled["jobs"][name]["command"]
        assert '${{inputs.gate_decision}}/decision' in cmd and '= "pass" ]; then' in cmd
    assert "register.py" not in compiled["jobs"]["register"]["command"].split("then")[0]


def test_guard_really_skips_on_fail(compiled, tmp_path):
    """Run the compiled guard in a shell with a 'fail' decision: the step must not run."""
    cmd = compiled["jobs"]["register"]["command"]
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "decision").write_text("fail\n")
    (tmp_path / "o").mkdir()
    cmd = (cmd.replace("${{inputs.gate_decision}}", str(tmp_path / "g"))
              .replace("${{outputs.registered}}", str(tmp_path / "o"))
              .replace("${{inputs.metrics}}", str(tmp_path))
              .replace("${{inputs.model}}", str(tmp_path))
              .replace("cd /app", f"cd {tmp_path}")
              .replace("python scripts/register.py", "touch RAN; true"))
    subprocess.run(["bash", "-c", cmd], check=True, capture_output=True)
    assert not (tmp_path / "RAN").exists()
    assert (tmp_path / "o" / "SKIPPED").exists()
    (tmp_path / "g" / "decision").write_text("pass\n")
    subprocess.run(["bash", "-c", cmd], check=True, capture_output=True)
    assert (tmp_path / "RAN").exists()


def test_only_the_deploy_step_runs_as_the_deployer(compiled):
    ids = {name: job.get("identity", {}).get("client_id") for name, job in compiled["jobs"].items()}
    assert ids.pop("deploy_staging") == "deploy-id"
    assert set(ids.values()) == {"train-id"}


def test_validation_runs_the_contract_tests(compiled):
    assert "tests/test_data.py" in compiled["jobs"]["validate"]["command"]


def test_miswired_spec_is_rejected():
    from cloudlayer import pipelines

    bad = {"steps": [{"name": "a", "command": "x {outputs.o}", "outputs": ["o"]},
                     {"name": "b", "command": "y {inputs.p}", "inputs": {"p": "a"}}]}
    with pytest.raises(ValueError, match="no output named p"):
        pipelines.validate_spec(bad)
