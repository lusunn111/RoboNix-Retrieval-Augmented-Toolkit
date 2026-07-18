"""Lazy module registry shared by the organized retrieval modules."""

from __future__ import annotations

import importlib
from typing import Mapping

try:
    from retrieval_augmented_service.service_bootstrap import activate_vendor
except ModuleNotFoundError as exc:
    if exc.name != "retrieval_augmented_service":
        raise
    from service_bootstrap import activate_vendor


def load(source_modules: Mapping[str, str], name: str):
    try:
        module_name = source_modules[name]
    except KeyError as exc:
        choices = ", ".join(sorted(source_modules))
        raise KeyError(f"Unknown component {name!r}; choose one of: {choices}") from exc
    activate_vendor()
    return importlib.import_module(module_name)
