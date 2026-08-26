import pytest

from src.model_registry import activate_model, register_model, resolve_model_path


def test_resolve_and_activate_registered_model(tmp_path):
    registry_path = tmp_path / "model_registry.json"
    first_model = tmp_path / "model-1.0.0.joblib"
    second_model = tmp_path / "model-1.1.0.joblib"
    first_model.touch()
    second_model.touch()

    register_model("1.0.0", first_model, {"model_version": "1.0.0"}, registry_path)
    register_model("1.1.0", second_model, {"model_version": "1.1.0"}, registry_path)

    active_path, active_version = resolve_model_path(
        None, registry_path, tmp_path / "legacy.joblib"
    )
    assert (active_path, active_version) == (second_model, "1.1.0")

    activate_model("1.0.0", registry_path)
    rollback_path, rollback_version = resolve_model_path(
        None, registry_path, tmp_path / "legacy.joblib"
    )
    assert (rollback_path, rollback_version) == (first_model, "1.0.0")


def test_unknown_model_version_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Model version not found"):
        resolve_model_path("missing", tmp_path / "registry.json", tmp_path / "legacy.joblib")


def test_duplicate_model_version_is_rejected(tmp_path):
    registry_path = tmp_path / "model_registry.json"
    artifact_path = tmp_path / "model-1.0.0.joblib"
    artifact_path.touch()

    register_model("1.0.0", artifact_path, {"model_version": "1.0.0"}, registry_path)

    with pytest.raises(ValueError, match="already exists"):
        register_model("1.0.0", artifact_path, {"model_version": "1.0.0"}, registry_path)
