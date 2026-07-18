"""Lazy access to experiment result and logging helpers."""

try:
    from retrieval_augmented_service.modules._registry import load as _load
except ModuleNotFoundError as exc:
    if exc.name != "retrieval_augmented_service":
        raise
    from modules._registry import load as _load

SOURCE_MODULES = {"results": "scripts.retrieval.results"}


def load():
    return _load(SOURCE_MODULES, "results")


__all__ = ["load"]
