"""Which registered version is production, and how a version becomes it.

Plain MLflow client calls, so this works against any MLflow registry: Azure ML's, a
self-hosted server, or the local sqlite file the tests use. The registry name and the
tracking URI arrive through src.config like everything else.

Production is a version TAG (role=production), not an MLflow stage. Stages are deprecated
in MLflow 3, Azure ML does not show them in the studio, and a tag is visible everywhere.
Exactly one version carries role=production. Promoting a new one retires the old one, so
a rollback is one tag change back.
"""
from __future__ import annotations

ROLE = "role"
PRODUCTION = "production"
RETIRED = "retired"


def client(tracking_uri: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def versions(c, name: str) -> list:
    """All versions of `name`, newest first. An unregistered name has none."""
    try:
        found = c.search_model_versions(f"name='{name}'")
    except Exception as exc:  # noqa: BLE001 - a missing model raises, by provider
        if "RESOURCE_DOES_NOT_EXIST" in str(exc) or "not found" in str(exc).lower():
            return []
        raise
    return sorted(found, key=lambda v: int(v.version), reverse=True)


def production_version(c, name: str):
    """The version tagged role=production, or None when nothing is in production yet."""
    tagged = [v for v in versions(c, name) if (v.tags or {}).get(ROLE) == PRODUCTION]
    if len(tagged) > 1:
        # Should not happen (promote retires the old one), but if two versions claim it,
        # the newest wins and the gate compares against that.
        print(f"warning: {len(tagged)} versions tagged {ROLE}={PRODUCTION}; using the newest")
    return tagged[0] if tagged else None


def promote(c, name: str, version: str, tags: dict[str, str] | None = None) -> str | None:
    """Make `version` production. Returns the version it replaced, if any."""
    previous = production_version(c, name)
    for key, value in (tags or {}).items():
        c.set_model_version_tag(name, str(version), key, str(value))
    c.set_model_version_tag(name, str(version), ROLE, PRODUCTION)
    # Versions come back as int from some registries and str from others.
    if previous is not None and str(previous.version) != str(version):
        c.set_model_version_tag(name, str(previous.version), ROLE, RETIRED)
        c.set_model_version_tag(name, str(previous.version), "replaced_by", str(version))
        return str(previous.version)
    return None
