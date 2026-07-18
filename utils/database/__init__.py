"""Lazy access to shared database helpers."""

try:
    from retrieval_augmented_service.modules.database import load
except ModuleNotFoundError as exc:
    if exc.name != "retrieval_augmented_service":
        raise
    from modules.database import load

__all__ = ["load"]
