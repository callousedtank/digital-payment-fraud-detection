import json
import re
from pathlib import Path


REGISTRY_VERSION = 1
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_version(version):
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            "Model versions may contain only letters, numbers, dots, underscores, "
            "and hyphens."
        )
    return version


def load_registry(path):
    registry_path = Path(path)
    if not registry_path.exists():
        return {
            "registry_version": REGISTRY_VERSION,
            "active_version": None,
            "models": {},
        }

    with registry_path.open(encoding="utf-8") as file:
        registry = json.load(file)

    if registry.get("registry_version") != REGISTRY_VERSION:
        raise ValueError(f"Unsupported model registry: {registry_path}")
    if not isinstance(registry.get("models"), dict):
        raise ValueError(f"Invalid model registry: {registry_path}")
    return registry


def write_registry(registry, path):
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = registry_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary_path.replace(registry_path)


def register_model(version, artifact_path, metadata, registry_path):
    version = validate_version(version)
    registry = load_registry(registry_path)
    registry["models"][version] = {
        "artifact_path": str(artifact_path),
        "metadata": metadata,
        "validation_status": "validated",
    }
    registry["active_version"] = version
    write_registry(registry, registry_path)


def activate_model(version, registry_path):
    version = validate_version(version)
    registry = load_registry(registry_path)
    entry = registry["models"].get(version)
    if entry is None:
        raise ValueError(f"Model version not found: {version}")
    if entry.get("validation_status") != "validated":
        raise ValueError(f"Model version is not validated: {version}")
    if not Path(entry["artifact_path"]).exists():
        raise FileNotFoundError(
            f"Model artifact for version {version} was not found: "
            f"{entry['artifact_path']}"
        )
    registry["active_version"] = version
    write_registry(registry, registry_path)


def resolve_model_path(version, registry_path, legacy_model_path):
    registry = load_registry(registry_path)
    selected_version = version or registry["active_version"]

    if selected_version:
        entry = registry["models"].get(selected_version)
        if entry is None:
            raise ValueError(f"Model version not found: {selected_version}")
        if entry.get("validation_status") != "validated":
            raise ValueError(f"Model version is not validated: {selected_version}")
        artifact_path = Path(entry["artifact_path"])
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact for version {selected_version} was not found: "
                f"{artifact_path}"
            )
        return artifact_path, selected_version

    return Path(legacy_model_path), "legacy"
