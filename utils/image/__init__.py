"""Lazy access to RT-Cache image utilities."""

try:
    from retrieval_augmented_service.modules._registry import load as _load
except ModuleNotFoundError as exc:
    if exc.name != "retrieval_augmented_service":
        raise
    from modules._registry import load as _load

SOURCE_MODULES = {"default": "scripts.common.image_utils"}


def load():
    return _load(SOURCE_MODULES, "default")


__all__ = ["load"]
