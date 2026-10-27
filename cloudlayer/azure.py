"""Azure adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install azure-storage-blob azure-identity azure-containerregistry
Docs: BlobServiceClient for storage; ACR push goes through `docker push` after
      `az acr login --name <registry>`.

Hints for Lab 1:
  * BLOB_URI is either abfss://container@account.dfs.core.windows.net/prefix or
    https://account.blob.core.windows.net/container/prefix. Pick one form and parse
    it here, never in src/.
  * Use DefaultAzureCredential rather than a connection string. It picks up your CLI
    login locally and your managed identity in CI, which is what Lab 4 needs.
  * push_image must return the digest reference: registry.azurecr.io/repo@sha256:...
  * Azure tags live on the resource, not the blob. Tag the storage account, the
    registry, and later the workspace with cfg.tags(1).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from cloudlayer.base import CloudAdapter

# On Windows the Azure CLI is `az.cmd`, which subprocess will not find as plain "az";
# shutil.which applies PATHEXT and returns the real path. Elsewhere this is just "az".
AZ = shutil.which("az") or "az"


# Provider SDKs are imported inside the methods that use them, not here. CI pushes
# images and deploys through this adapter, and those paths only shell out to `az` and
# `docker`; a module-level `import azure.ai.ml` would drag the whole Azure ML stack
# (and its ruamel.yaml pin, which cannot build on Python 3.11) into every CI job.


class AzureAdapter(CloudAdapter):
    def _blob_parts(self) -> tuple[str, str, str]:
        """Parse BLOB_URI into account URL, container, and prefix."""
        parsed = urlparse(self.cfg.blob_uri)

        if parsed.scheme != "https" or not parsed.netloc.endswith(
            ".blob.core.windows.net"
        ):
            raise ValueError(
                "BLOB_URI must use https://<account>.blob.core.windows.net/"
                "<container>/<prefix>"
            )

        account_url = f"{parsed.scheme}://{parsed.netloc}"
        parts = parsed.path.strip("/").split("/")

        if not parts or not parts[0]:
            raise ValueError("BLOB_URI must include a container")

        container = parts[0]
        prefix = "/".join(parts[1:]).strip("/")

        return account_url, container, prefix

    def upload(self, local_path: str, key: str) -> str:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        account_url, container, prefix = self._blob_parts()

        blob_name = "/".join(part for part in (prefix, key) if part)

        credential = DefaultAzureCredential()
        service = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )

        blob = service.get_blob_client(
            container=container,
            blob=blob_name,
        )

        with open(local_path, "rb") as file:
            blob.upload_blob(file, overwrite=True)

        return f"{account_url}/{container}/{blob_name}"

    def download(self, uri: str, local_path: str) -> None:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        parsed = urlparse(uri)

        if parsed.scheme != "https" or not parsed.netloc.endswith(
            ".blob.core.windows.net"
        ):
            raise ValueError(
                "Azure blob URI must use https://<account>.blob.core.windows.net/"
            )

        account_url = f"{parsed.scheme}://{parsed.netloc}"
        parts = parsed.path.strip("/").split("/")

        if len(parts) < 2:
            raise ValueError("Azure blob URI must include container and blob name")

        container = parts[0]
        blob_name = "/".join(parts[1:])

        credential = DefaultAzureCredential()
        service = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )

        blob = service.get_blob_client(
            container=container,
            blob=blob_name,
        )

        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "wb") as file:
            blob.download_blob().readinto(file)

    def push_image(self, local_tag: str) -> str:
        registry_repo = self.cfg.container_registry.rstrip("/")

        registry_host, repo = registry_repo.split("/", 1)
        registry_name = registry_host.split(".", 1)[0]

        tag = local_tag.rsplit(":", 1)[1]
        remote_tag = f"{registry_repo}:{tag}"

        subprocess.run(
            [AZ, "acr", "login", "--name", registry_name],
            check=True,
        )

        subprocess.run(
            ["docker", "tag", local_tag, remote_tag],
            check=True,
        )

        result = subprocess.run(
            ["docker", "push", remote_tag],
            check=True,
            text=True,
            capture_output=True,
        )

        output = result.stdout + result.stderr

        digest = None
        for line in output.splitlines():
            if "digest:" in line.lower():
                digest = line.split(":", 1)[1].strip()
                break

        if not digest or not digest.startswith("sha256:"):
            inspect = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    remote_tag,
                    "--format",
                    "{{index .RepoDigests 0}}",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            repo_digest = inspect.stdout.strip()

            if "@sha256:" not in repo_digest:
                raise RuntimeError(
                    f"Could not determine pushed image digest. Docker output:\n{output}"
                )

            digest = repo_digest.split("@", 1)[1]

        return f"{registry_repo}@{digest}"

    def _ml_client(self):
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        workspace = __import__("os").environ.get("AZURE_ML_WORKSPACE")
        if not workspace:
            raise RuntimeError("AZURE_ML_WORKSPACE is not set")

        credential = DefaultAzureCredential()
        return MLClient(
            credential=credential,
            subscription_id=__import__("os").environ["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=self.cfg.project_id,
            workspace_name=workspace,
        )

    def submit_training(self, image_uri: str, args: dict[str, object]) -> str:
        """Submit an Azure ML command job for Lab 2."""
        from azure.ai.ml import Input, Output, command
        from azure.ai.ml.entities import Environment

        ml_client = self._ml_client()

        input_uri = str(args.get("input_uri", self.cfg.blob_uri))
        output_uri = str(args["output_uri"])
        compute = str(args.get("compute", "itcs3556688020-cpu"))
        experiment_name = str(args.get("experiment", "itcs355-lab2"))

        job = command(
            display_name="itcs355-lab2-training",
            experiment_name=experiment_name,
            code=".",
            command=str(args["command"]),
            environment=Environment(
                image=image_uri,
                name="itcs355-lab2-env",
            ),
            compute=compute,
            inputs={
                "data": Input(
                    type="uri_file",
                    path=input_uri,
                    mode="download",
                )
            },
            outputs={
                "artifacts": Output(
                    type="uri_folder",
                    path=output_uri,
                    mode="rw_mount",
                )
            },
        )

        created = ml_client.jobs.create_or_update(job)
        return created.name

    def wait_training(self, job_id: str) -> dict[str, object]:
        import time

        ml_client = self._ml_client()

        while True:
            job = ml_client.jobs.get(job_id)
            status = str(job.status)

            print(f"Azure ML job {job_id}: {status}")

            if status.upper().endswith(("COMPLETED", "FAILED", "CANCELED", "NOT_RESPONDING")):
                break

            time.sleep(10)

        result = {
            "job_id": job.name,
            "status": status,
            "studio_url": getattr(job, "studio_url", None),
        }

        if not status.upper().endswith("COMPLETED"):
            raise RuntimeError(f"Azure ML training job failed: {result}")

        return result

    def register_model(self, model_uri: str, name: str) -> str:
        '''Register an MLflow model in Azure ML and return its version.'''
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Model

        ml_client = self._ml_client()

        model = Model(
            path=model_uri,
            name=name,
            type=AssetTypes.MLFLOW_MODEL,
        )

        registered = ml_client.models.create_or_update(model)
        return str(registered.version)

    # submit_training / register_model  -> Lab 2 (Azure ML command job + model registry)

    def deploy(self, model_ref: str, endpoint: str, instance: str) -> str:
        """Deploy the serving image to Azure Container Apps. Returns the app's URL.

        `model_ref` is the image (tagged by commit SHA in CD); the model weights are the
        registry version in MODEL_VERSION, loaded by the service at startup. BUILD_SHA is
        passed through so /ready and every response header say which code is serving.
        """
        import os

        parts = instance.split(":")
        if len(parts) != 2:
            raise ValueError(
                "instance must be '<cpu>:<memory>', e.g. '0.25:0.5Gi'"
            )

        cpu, memory = parts

        command = [
            AZ,
            "containerapp",
            "update",
            "--name",
            endpoint,
            "--resource-group",
            self.cfg.project_id,
            "--image",
            model_ref,
            "--cpu",
            cpu,
            "--memory",
            memory,
            "--set-env-vars",
            f"MODEL_REGISTRY_NAME={self.cfg.model_registry_name}",
            f"MLFLOW_TRACKING_URI={os.environ.get('MLFLOW_TRACKING_URI', self.cfg.mlflow_tracking_uri)}",
            f"BUILD_SHA={os.environ.get('BUILD_SHA') or model_ref.rsplit(':', 1)[-1]}",
        ]
        # Since Lab 5 the training pipeline owns MODEL_VERSION (rollout_model, after the
        # gate). A code deploy keeps whatever version is serving unless told otherwise;
        # otherwise every push would silently roll staging back to an old model.
        if os.environ.get("MODEL_VERSION"):
            command.append(f"MODEL_VERSION={os.environ['MODEL_VERSION']}")

        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            # Surface az's own message: "does not have authorization" here is the classic
            # CI-can-build-but-not-deploy failure, and it should be readable in the log.
            raise RuntimeError(f"az containerapp update failed:\n{result.stderr.strip()}")

        fqdn = subprocess.run(
            [AZ, "containerapp", "show", "--name", endpoint, "--resource-group",
             self.cfg.project_id, "--query", "properties.configuration.ingress.fqdn", "-o", "tsv"],
            check=True, text=True, capture_output=True,
        ).stdout.strip()
        return f"https://{fqdn}"

    def invoke(self, endpoint: str, payload: dict[str, object], path: str = "/predict") -> dict[str, object]:
        """Invoke the deployed Azure Container App (`path`: /predict or /predict/batch)."""
        import json
        import urllib.request

        url = endpoint.rstrip("/") + path

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def teardown(self, tags: dict[str, str]) -> list[str]:
        """Delete Azure resources matching all supplied tags."""
        import json

        result = subprocess.run(
            [
                AZ,
                "resource",
                "list",
                "--resource-group",
                self.cfg.project_id,
                "-o",
                "json",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        resources = json.loads(result.stdout)

        # Some resources carry a lab's tags without belonging to it: the capstone's
        # compute cluster was tagged lab=5 when it was created. TEARDOWN_KEEP in cloud.env
        # lists names teardown must never delete, whatever their tags say.
        import os

        keep = {n.strip() for n in os.environ.get("TEARDOWN_KEEP", "").split(",") if n.strip()}
        matches = [
            resource
            for resource in resources
            if all((resource.get("tags") or {}).get(key) == value for key, value in tags.items())
            and resource["name"].split("/")[-1] not in keep
        ]

        priority = {
            "Microsoft.App/jobs": 0,              # the drift schedule goes first
            "Microsoft.App/containerApps": 0,
            "Microsoft.App/managedEnvironments": 1,
        }
        matches.sort(key=lambda resource: priority.get(resource.get("type"), 2))

        deleted = []
        for resource in matches:
            subprocess.run(
                [
                    AZ,
                    "resource",
                    "delete",
                    "--ids",
                    resource["id"],
                    "--only-show-errors",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            deleted.append(resource["name"])

        return deleted

    # --- Lab 5 ---------------------------------------------------------------
    ARM = "https://management.azure.com"
    APPS_API = "2024-03-01"

    def _arm(self, method: str, path: str, body: dict | None = None) -> dict:
        """One Azure Resource Manager call as whoever DefaultAzureCredential resolves to.
        In a pipeline step that is the managed identity named by AZURE_CLIENT_ID, so this
        needs no `az` in the image and no stored secret."""
        import json
        import urllib.error
        import urllib.request

        from azure.identity import DefaultAzureCredential

        token = DefaultAzureCredential().get_token(f"{self.ARM}/.default").token
        sep = "&" if "?" in path else "?"
        req = urllib.request.Request(
            f"{self.ARM}{path}{sep}api-version={self.APPS_API}",
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
        except urllib.error.HTTPError as exc:
            # 403 AuthorizationFailed names the exact action that was missing, which is
            # what Lab 5 Task 2's remove-one-permission test needs to see.
            raise RuntimeError(f"ARM {method} {path}: HTTP {exc.code} "
                               f"{exc.read().decode(errors='replace')[:600]}") from None
        return json.loads(raw) if raw else {}

    def rollout_model(self, endpoint: str, version: str) -> tuple[str, str]:
        """New revision of the Container App with MODEL_VERSION=version; same image."""
        import os
        import time

        path = (f"/subscriptions/{os.environ['AZURE_SUBSCRIPTION_ID']}/resourceGroups/"
                f"{self.cfg.project_id}/providers/Microsoft.App/containerApps/{endpoint}")
        app = self._arm("GET", path)
        template = app["properties"]["template"]
        template.pop("revisionSuffix", None)  # a reused suffix would reject the new revision
        env = template["containers"][0].setdefault("env", [])
        previous = next((e.get("value") for e in env if e["name"] == "MODEL_VERSION"), None)
        env[:] = [e for e in env if e["name"] != "MODEL_VERSION"]
        env.append({"name": "MODEL_VERSION", "value": str(version)})

        self._arm("PATCH", path, {"properties": {"template": template}})
        deadline = time.time() + 300
        while True:
            state = self._arm("GET", path)["properties"].get("provisioningState")
            if state == "Succeeded":
                break
            if state in ("Failed", "Canceled") or time.time() > deadline:
                raise RuntimeError(f"Container App {endpoint} rollout ended {state}")
            time.sleep(5)
        fqdn = app["properties"]["configuration"]["ingress"]["fqdn"]
        return f"https://{fqdn}", str(previous)

    # --- Lab 4 ---------------------------------------------------------------
    def emit_metric(self, name: str, value: float, unit: str = "None") -> None:
        """Write one data point as an Azure Monitor custom metric.

        Custom metrics must be attached to a real resource (not a resource group) and
        posted to the regional endpoint where that resource lives. The staging
        Container App is the natural owner: the drift score describes its inputs, so
        it sits next to its request and replica metrics in the portal and an alert
        rule can be scoped to it.

        The caller needs `Monitoring Metrics Publisher` on that resource. In CI that is
        the OIDC service principal; scripts/azure_lab4_setup.sh grants it.

        Azure Monitor has no units for custom metrics, so `unit` is accepted for the
        interface and ignored. Dots are not allowed in names; they become underscores.
        """
        import datetime as dt
        import json
        import os
        import urllib.request

        from azure.identity import DefaultAzureCredential

        resource_id = os.environ.get("AZURE_METRICS_RESOURCE_ID") or (
            f"/subscriptions/{os.environ['AZURE_SUBSCRIPTION_ID']}"
            f"/resourceGroups/{self.cfg.project_id}"
            f"/providers/Microsoft.App/containerApps/"
            f"{os.environ.get('CONTAINER_APP', 'itcs355-predict')}"
        )
        token = DefaultAzureCredential().get_token("https://monitoring.azure.com/.default").token
        body = {
            "time": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data": {
                "baseData": {
                    "metric": name.replace(".", "_"),
                    "namespace": "itcs355",
                    "dimNames": ["model"],
                    "series": [{
                        "dimValues": [self.cfg.model_registry_name],
                        "min": value, "max": value, "sum": value, "count": 1,
                    }],
                }
            },
        }
        request = urllib.request.Request(
            f"https://{self.cfg.region}.monitoring.azure.com{resource_id}/metrics",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 300:
                raise RuntimeError(f"emit_metric {name}: HTTP {response.status}")

    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
