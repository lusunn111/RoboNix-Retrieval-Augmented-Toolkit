"""Lazy access to the embedding HTTP client."""

try:
    from retrieval_augmented_service.modules._registry import load as _load
except ModuleNotFoundError as exc:
    if exc.name != "retrieval_augmented_service":
        raise
    from modules._registry import load as _load

SOURCE_MODULES = {"client": "scripts.common.embedding_client"}


def load():
    return _load(SOURCE_MODULES, "client")


__all__ = ["load"]
