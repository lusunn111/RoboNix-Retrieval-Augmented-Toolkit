"""Exact, vector, incremental, and hierarchical index builders."""

from .._registry import load as _load

SOURCE_MODULES = {
    "bridge": "scripts.retrieval.build_bridge_increments",
    "bridge_multi_view": "scripts.retrieval.build_bridge_increments_mix",
}


def load(name: str = "bridge"):
    return _load(SOURCE_MODULES, name)


__all__ = ["SOURCE_MODULES", "load"]
