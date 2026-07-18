"""Experience persistence and robot data acquisition."""

from .._registry import load as _load

SOURCE_MODULES = {
    "connections": "scripts.common.database",
    "collection_server": "scripts.data_acquisition.data_collection_server",
}


def load(name: str = "connections"):
    return _load(SOURCE_MODULES, name)


__all__ = ["SOURCE_MODULES", "load"]
