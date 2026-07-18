"""Incremental updates, backup, restore, and memory cleanup."""

from .._registry import load as _load

SOURCE_MODULES = {
    "backup": "scripts.retrieval.backup_qdrant",
    "restore": "scripts.retrieval.restore_qdrant",
    "sizes": "scripts.retrieval.report_backup_sizes",
    "clear": "scripts.data_processing.clear_databases",
}


def load(name: str):
    return _load(SOURCE_MODULES, name)


__all__ = ["SOURCE_MODULES", "load"]
