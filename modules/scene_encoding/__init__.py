"""OpenVLA, CLIP, and multi-view scene encoders."""

from .._registry import load as _load

SOURCE_MODULES = {
    "openvla": "scripts.embedding.embedding_server",
    "multi_view": "scripts.embedding.embedding_server_mix",
    "offline": "scripts.embedding.custom_embedding_generator",
}


def load(name: str = "openvla"):
    return _load(SOURCE_MODULES, name)


__all__ = ["SOURCE_MODULES", "load"]
