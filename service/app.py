"""Lab 3 — inference service.

Provider-neutral by construction: the model arrives through the adapter, and the same
container image deploys to SageMaker, Azure ML, or Vertex AI. Route paths differ per
platform; that difference belongs in cloudlayer/, never here.

Run locally:  uvicorn service.app:app --port 8080
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from service import monitoring
from service.schemas import BatchRequest, BatchResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}',
)
log = logging.getLogger("service")

STATE: dict[str, Any] = {
    "model": None,
    "version": os.environ.get("MODEL_VERSION", "unknown"),
    # The image's commit SHA, set by CD. MODEL_VERSION says which weights; this says
    # which code. During an incident you need both.
    "build": os.environ.get("BUILD_SHA", "local"),
}


def _load_model():
    """Load once, at startup. Never per request.

    Loading per request is the commonest cause of a p99 that looks nothing like p50, and
    it is the first thing to check when your latency distribution has a long tail.
    """
    from pathlib import Path

    import joblib

    # Lab 5 Task 4: on a provider with no MLflow registry wired up (AWS here), the model
    # file is read from object storage THROUGH THE ADAPTER: the same download() the swap
    # check exercises, so the service never names a provider. Checked first: an explicit
    # file wins over registry settings that may be left over in the environment (on
    # Lambda they sent MLflow to a sqlite file on a read-only disk, retrying until the
    # 60 s timeout).
    uri = os.environ.get("MODEL_URI")
    if uri:
        from cloudlayer.factory import get_adapter
        from src import config

        local = Path(os.environ.get("MODEL_CACHE", "/tmp/model.joblib"))
        get_adapter(config.load(strict=False)).download(uri, str(local))
        return joblib.load(local)

    name = os.environ.get("MODEL_REGISTRY_NAME")
    version = os.environ.get("MODEL_VERSION")
    if name and version:
        import mlflow.sklearn  # imported lazily so tests can run without a registry

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        return mlflow.sklearn.load_model(f"models:/{name}/{version}")

    # Fallback for local development and tests only. Submitting this is not acceptable:
    # your deployed service must load a registered version.
    path = Path(os.environ.get("MODEL_PATH", "reports/model.joblib"))
    if not path.exists():
        raise RuntimeError(
            "No model available. Set MODEL_REGISTRY_NAME and MODEL_VERSION, or MODEL_PATH."
        )
    return joblib.load(path)


def prepare_for_serving(model):
    """Serve single-threaded, whatever the model was trained with.

    Training uses n_jobs=-1, and scikit-learn keeps that setting for prediction. For one
    row that means spinning up a thread pool per request: ~35 ms instead of ~8 ms on a
    2-core machine, and worse on a fractional-vCPU container that reports the host's
    core count. tests/test_model_behaviour.py times the model AS SERVED, so deleting
    this fails CI rather than quietly costing the latency budget.
    """
    if hasattr(model, "n_jobs"):
        model.n_jobs = int(os.environ.get("SERVE_N_JOBS", "1"))
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Read at startup, not import: the version belongs to this process's lifetime, and an
    # import-time read let whichever test imported the module first fix it for all.
    STATE["version"] = os.environ.get("MODEL_VERSION", "unknown")
    STATE["build"] = os.environ.get("BUILD_SHA", "local")
    try:
        STATE["model"] = prepare_for_serving(_load_model())
        log.info('"model loaded, version=%s build=%s"', STATE["version"], STATE["build"])
        monitoring.set_model_info(str(STATE["version"]), str(STATE["build"]))
    except Exception as exc:  # readiness stays false; liveness still passes
        STATE["model"] = None
        log.error('"model load failed: %s"', exc)
    yield
    STATE["model"] = None


app = FastAPI(title="ITCS355 inference", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # An unhandled error must still count as a 5xx, or the error-rate panel lies.
        route = request.scope.get("route")
        monitoring.record_request(getattr(route, "path", "unmatched"), request.method, 500,
                                  (time.perf_counter() - started) * 1000)
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    # The route template, not the raw path: a scanner probing random URLs must not mint
    # a new time series per URL.
    route = request.scope.get("route")
    monitoring.record_request(getattr(route, "path", "unmatched"), request.method,
                              response.status_code, latency_ms)
    response.headers["x-request-id"] = request_id
    response.headers["x-model-version"] = str(STATE["version"])
    response.headers["x-build-sha"] = str(STATE["build"])
    log.info(
        '{"request_id":"%s","path":"%s","status":%d,"latency_ms":%.2f,"model_version":"%s"}',
        request_id, request.url.path, response.status_code, latency_ms, STATE["version"],
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness. The process is up. Says nothing about whether it can serve."""
    return {"status": "alive"}


@app.get("/ready")
def ready():
    """Readiness. The model is loaded and can score.

    These two are genuinely different, and confusing them causes a specific production
    failure: traffic routed to a container whose model has not finished loading. All three
    providers distinguish them, and Quiz 3 asks about it.
    """
    if STATE["model"] is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": "model not loaded"})
    return {"status": "ready", "model_version": STATE["version"], "build_sha": STATE["build"]}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus exposition. See service/monitoring.py for what each series means."""
    return Response(generate_latest(monitoring.REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring/window", include_in_schema=False)
def monitoring_window(x_monitoring_token: str | None = Header(default=None)):
    """The rolling window of accepted inputs, for the scheduled drift detector."""
    if not monitoring.window_access_allowed(x_monitoring_token):
        raise HTTPException(status_code=403, detail="monitoring token required")
    return {**monitoring.WINDOW.snapshot(), "model_version": STATE["version"],
            "build_sha": STATE["build"]}


def _score(rows: list[dict]) -> list[float]:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    import pandas as pd

    from src.data import FEATURES

    frame = pd.DataFrame(rows)[FEATURES]
    return [float(p) for p in STATE["model"].predict_proba(frame)[:, 1]]


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    row = payload.model_dump()
    score = _score([row])[0]
    monitoring.WINDOW.add([row])
    return PredictResponse(probability=score, model_version=str(STATE["version"]))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest) -> BatchResponse:
    rows = [row.model_dump() for row in payload.rows]
    scores = _score(rows)
    monitoring.WINDOW.add(rows)
    return BatchResponse(probabilities=scores, model_version=str(STATE["version"]))
