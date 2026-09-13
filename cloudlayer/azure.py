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

import subprocess
from pathlib import Path
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from cloudlayer.base import CloudAdapter


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
            ["az", "acr", "login", "--name", registry_name],
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

    # submit_training / register_model  -> Lab 2 (Azure ML command job + model registry)
    # deploy / invoke                   -> Lab 3 (managed online endpoint + deployment)
    # emit_metric                       -> Lab 4 (Azure Monitor custom metric)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resource graph query by tag)
