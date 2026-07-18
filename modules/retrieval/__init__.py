"""Online experience and trajectory retrieval engines."""

from .._registry import load as _load

SOURCE_MODULES = {
    "realtime": "scripts.retrieval.retrieval_server",
    "libero_goal": "scripts.retrieval.retrieval_libero_goal",
    "libero_goal_multi_view": "scripts.retrieval.retrieval_libero_goal_mix",
    "baselines": "scripts.retrieval.models",
}


def load(name: str = "realtime"):
    return _load(SOURCE_MODULES, name)


__all__ = ["SOURCE_MODULES", "load"]
