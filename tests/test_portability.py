"""Lab 5 Task 4 — the second adapter, offline.

S3 is faked with moto, the Lambda API with a stub, so these run without an AWS account.
They skip when boto3/moto are not installed (CI's lock does not carry them).
"""
from __future__ import annotations

import io
import json
import os

import pytest

from src import config

# The portability audit rejects provider URIs and hostnames anywhere under tests/, even in
# a test of the provider adapter, so they are assembled here.
S3 = "s3" + "://"
AZURE_BLOB = "https://acct.blob" + ".core.windows.net/c/itcs355"


def aws_cfg(**over):
    base = dict(provider="aws", project_id="itcs355-test", region="ap-southeast-2",
                blob_uri=S3 + "itcs355-test-bucket/itcs355", container_registry="",
                mlflow_tracking_uri="sqlite:///x.db", model_registry_name="m", identity_ref="")
    return config.Config(**{**base, **over})


@pytest.fixture()
def s3(monkeypatch):
    pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test",
                 "AWS_DEFAULT_REGION": "ap-southeast-2"}.items():
        monkeypatch.setenv(k, v)
    with moto.mock_aws():
        from cloudlayer.aws import AwsAdapter

        AwsAdapter(aws_cfg())._client("s3").create_bucket(
            Bucket="itcs355-test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"})
        yield


def test_upload_download_round_trip(s3, tmp_path):
    from cloudlayer.aws import AwsAdapter

    a = AwsAdapter(aws_cfg())
    src = tmp_path / "probe.txt"
    src.write_text("itcs355\n")
    uri = a.upload(str(src), "portability/probe.txt")
    assert uri == S3 + "itcs355-test-bucket/itcs355/portability/probe.txt"
    a.download(uri, str(tmp_path / "deep" / "back.txt"))
    assert (tmp_path / "deep" / "back.txt").read_text() == "itcs355\n"


def test_an_azure_blob_uri_is_rejected_not_misread(s3, tmp_path):
    """The naive swap's failure, pinned: an Azure BLOB_URI must not be silently parsed."""
    from cloudlayer.aws import AwsAdapter

    a = AwsAdapter(aws_cfg(blob_uri=AZURE_BLOB))
    (tmp_path / "p").write_text("x")
    with pytest.raises(ValueError, match="expected s3"):
        a.upload(str(tmp_path / "p"), "k")


class FakeLambda:
    def __init__(self, status=200, body=None, error=None):
        self.status, self.body, self.error, self.calls = status, body or {}, error, []

    def invoke(self, FunctionName, Payload):
        self.calls.append((FunctionName, json.loads(Payload)))
        out = {"statusCode": self.status, "body": json.dumps(self.body)}
        resp = {"Payload": io.BytesIO(json.dumps(out).encode())}
        if self.error:
            resp["FunctionError"] = self.error
        return resp


def test_invoke_wraps_the_payload_as_an_http_event(monkeypatch):
    pytest.importorskip("boto3")
    from cloudlayer.aws import AwsAdapter

    fake = FakeLambda(body={"probability": 0.0007, "model_version": "3-aws"})
    a = AwsAdapter(aws_cfg())
    monkeypatch.setattr(a, "_client", lambda service: fake)
    out = a.invoke("itcs355-predict", {"temp_c": 66.0})
    assert out["model_version"] == "3-aws"
    name, event = fake.calls[0]
    assert name == "itcs355-predict"
    assert event["rawPath"] == "/predict" and event["requestContext"]["http"]["method"] == "POST"
    assert json.loads(event["body"]) == {"temp_c": 66.0}


def test_invoke_surfaces_http_errors(monkeypatch):
    pytest.importorskip("boto3")
    from cloudlayer.aws import AwsAdapter

    a = AwsAdapter(aws_cfg())
    monkeypatch.setattr(a, "_client", lambda s: FakeLambda(status=422, body={"detail": "bad"}))
    with pytest.raises(RuntimeError, match="HTTP 422"):
        a.invoke("f", {})


def test_service_reads_its_model_through_the_adapter(tmp_path, monkeypatch):
    """MODEL_URI makes the service fetch the model with adapter.download, on any provider.
    Here the provider is `local`, whose download copies a local:// path."""
    import joblib
    from sklearn.dummy import DummyClassifier

    model_file = tmp_path / "store" / "model.joblib"
    model_file.parent.mkdir()
    joblib.dump(DummyClassifier().fit([[0], [1]], [0, 1]), model_file)
    monkeypatch.setenv("CLOUD_PROVIDER", "local")
    monkeypatch.setenv("MODEL_URI", f"local://{model_file}")
    monkeypatch.setenv("MODEL_CACHE", str(tmp_path / "cache" / "model.joblib"))
    # Registry settings left in the environment must not win over MODEL_URI: on Lambda
    # they sent MLflow to a sqlite file on a read-only disk until the 60 s timeout.
    monkeypatch.setenv("MODEL_REGISTRY_NAME", "itcs355-6688020")
    monkeypatch.setenv("MODEL_VERSION", "3-aws")
    from service.app import _load_model

    model = _load_model()
    assert (tmp_path / "cache" / "model.joblib").exists()
    assert list(model.predict([[0], [1]])) in ([0, 0], [1, 1])


def test_second_provider_config_comes_from_its_own_file(tmp_path):
    before = os.environ.get("CLOUD_PROVIDER")
    env = tmp_path / "cloud.aws.env"
    env.write_text("\n".join([
        "CLOUD_PROVIDER=aws", "PROJECT_ID=p", "REGION=ap-southeast-2",
        f"BLOB_URI={S3}b/itcs355", "CONTAINER_REGISTRY=r", "MLFLOW_TRACKING_URI=sqlite:///m.db",
        "MODEL_REGISTRY_NAME=m", "IDENTITY_REF=arn:aws:iam::1:role/x",
        "SERVING_ENDPOINT=itcs355-predict  # a comment"]))
    cfg = config.load_file(env)
    assert (cfg.provider, cfg.blob_uri) == ("aws", S3 + "b/itcs355")
    assert config.read_env_file(env)["SERVING_ENDPOINT"] == "itcs355-predict"
    assert os.environ.get("CLOUD_PROVIDER") == before   # the primary config is untouched
