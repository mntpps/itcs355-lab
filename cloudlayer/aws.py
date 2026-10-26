"""AWS adapter: the second provider for Lab 5 Task 4.

Implemented: upload, download, invoke (the three the task asks for), plus push_image and
teardown, because the setup has to put an image somewhere and Task 6 has to remove it.
Everything else stays NotImplementedError: this proves the seam, it is not a migration.

    S3      upload / download          BLOB_URI = s3://<bucket>/<prefix>
    ECR     push_image                 CONTAINER_REGISTRY = <acct>.dkr.ecr.<region>.amazonaws.com/<repo>
    Lambda  invoke                     the same serving image as Azure, plus the AWS Lambda
                                       Web Adapter (service/Dockerfile.lambda)

SDK: boto3 (pip install boto3). Credentials come from the standard chain: a named
profile on a laptop (`aws configure`), never keys in cloud.env or the repo.

Where AWS and Azure differ in a way the seam cannot hide is written up in the README's
Lab 5 Task 4 section. Two show up right here:
  * `endpoint` means a URL on Azure (Container Apps ingress) and a function name on AWS.
    The service is the same, but how you address it is not.
  * `invoke` on Azure is a plain HTTPS POST. A Lambda is invoked through the Lambda API
    with a request *event*, which the Web Adapter turns back into an HTTP request for the
    unchanged FastAPI app. The payload has to be wrapped as an event to get there.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cloudlayer.base import CloudAdapter

AWS = shutil.which("aws") or "aws"


class AwsAdapter(CloudAdapter):
    def _client(self, service: str):
        import boto3

        return boto3.client(service, region_name=self.cfg.region)

    @staticmethod
    def _split(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"expected s3://bucket/key, got {uri!r}")
        return parsed.netloc, parsed.path.lstrip("/")

    # --- the seam: upload / download -------------------------------------------------
    def upload(self, local_path: str, key: str) -> str:
        bucket, prefix = self._split(self.cfg.blob_uri)
        object_key = "/".join(p for p in (prefix.rstrip("/"), key) if p)
        self._client("s3").upload_file(str(local_path), bucket, object_key)
        return f"s3://{bucket}/{object_key}"

    def download(self, uri: str, local_path: str) -> None:
        bucket, key = self._split(uri)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client("s3").download_file(bucket, key, str(local_path))

    # --- the seam: invoke -------------------------------------------------------------
    def invoke(self, endpoint: str, payload: dict[str, Any], path: str = "/predict") -> dict[str, Any]:
        """POST `payload` to the service's `path` on the Lambda function `endpoint`.

        The event is the Lambda function-URL format, which the Web Adapter translates into
        an ordinary HTTP request to the FastAPI app inside the container, so neither the
        app nor its routes change. Invoking through the Lambda API (rather than a public
        function URL) keeps the function private: only IAM principals with
        lambda:InvokeFunction can call it.
        """
        body = json.dumps(payload)
        event = {
            "version": "2.0", "routeKey": "$default", "rawPath": path, "rawQueryString": "",
            "headers": {"content-type": "application/json", "host": "lambda"},
            "requestContext": {"http": {"method": "POST", "path": path, "protocol": "HTTP/1.1",
                                        "sourceIp": "127.0.0.1", "userAgent": "itcs355"},
                               "domainName": "lambda", "stage": "$default"},
            "body": body, "isBase64Encoded": False,
        }
        resp = self._client("lambda").invoke(FunctionName=endpoint, Payload=json.dumps(event).encode())
        raw = resp["Payload"].read()
        if resp.get("FunctionError"):
            raise RuntimeError(f"Lambda {endpoint} failed: {raw[:500]!r}")
        out = json.loads(raw)
        status = int(out.get("statusCode", 200))
        result = json.loads(out.get("body") or "{}")
        if status >= 400:
            raise RuntimeError(f"{endpoint}{path} returned HTTP {status}: {result}")
        return result

    # --- supporting: push_image, teardown ---------------------------------------------
    def push_image(self, local_tag: str) -> str:
        registry_repo = self.cfg.container_registry.rstrip("/")
        host = registry_repo.split("/", 1)[0]
        remote = f"{registry_repo}:{local_tag.rsplit(':', 1)[1]}"
        password = subprocess.run([AWS, "ecr", "get-login-password", "--region", self.cfg.region],
                                  check=True, text=True, capture_output=True).stdout
        subprocess.run(["docker", "login", "--username", "AWS", "--password-stdin", host],
                       input=password, check=True, text=True, capture_output=True)
        subprocess.run(["docker", "tag", local_tag, remote], check=True)
        out = subprocess.run(["docker", "push", remote], check=True, text=True, capture_output=True)
        digest = next((w for line in out.stdout.splitlines() for w in line.split()
                       if w.startswith("sha256:")), None)
        if not digest:
            raise RuntimeError(f"no digest in docker push output:\n{out.stdout[-500:]}")
        return f"{registry_repo}@{digest}"

    def teardown(self, tags: dict[str, str]) -> list[str]:
        """Delete the Lambda functions, ECR repositories, S3 buckets and IAM roles carrying
        all of `tags`. The Resource Groups Tagging API finds the first three; IAM roles
        are global and are not listed there, so they are checked one by one."""
        filters = [{"Key": k, "Values": [v]} for k, v in tags.items()]
        found = []
        pages = self._client("resourcegroupstaggingapi").get_paginator("get_resources")
        for page in pages.paginate(TagFilters=filters):
            found += [r["ResourceARN"] for r in page["ResourceTagMappingList"]]

        deleted = []
        for arn in sorted(found, key=lambda a: (":lambda:" not in a, ":ecr:" not in a)):
            name = arn.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
            if ":lambda:" in arn and ":function:" in arn:
                self._client("lambda").delete_function(FunctionName=name)
            elif ":ecr:" in arn:
                self._client("ecr").delete_repository(repositoryName=arn.split("repository/", 1)[1],
                                                      force=True)
            elif arn.startswith("arn:aws:s3:::"):
                import boto3

                bucket = boto3.resource("s3", region_name=self.cfg.region).Bucket(name)
                bucket.object_versions.delete()
                bucket.objects.delete()
                bucket.delete()
            else:
                continue
            deleted.append(arn)

        iam = self._client("iam")
        for role in iam.list_roles()["Roles"]:
            if not role["RoleName"].startswith("itcs355"):
                continue
            role_tags = {t["Key"]: t["Value"] for t in iam.list_role_tags(RoleName=role["RoleName"])["Tags"]}
            if all(role_tags.get(k) == v for k, v in tags.items()):
                for p in iam.list_attached_role_policies(RoleName=role["RoleName"])["AttachedPolicies"]:
                    iam.detach_role_policy(RoleName=role["RoleName"], PolicyArn=p["PolicyArn"])
                for p in iam.list_role_policies(RoleName=role["RoleName"])["PolicyNames"]:
                    iam.delete_role_policy(RoleName=role["RoleName"], PolicyName=p)
                iam.delete_role(RoleName=role["RoleName"])
                deleted.append(role["Arn"])
        return deleted

    # submit_training / register_model / deploy / emit_metric: not ported (Task 4 is three
    # methods). The README's Task 4 write-up estimates what porting them would take.
