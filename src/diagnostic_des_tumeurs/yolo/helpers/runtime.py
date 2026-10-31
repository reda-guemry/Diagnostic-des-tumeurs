"""Runtime tweaks for Ultralytics."""

from __future__ import annotations


def disable_native_mlflow() -> None:
    """Use the project's explicit MLflow logging instead of Ultralytics' one."""
    from ultralytics import settings

    settings.update({"mlflow": False})
