"""Standalone organization of the HeiSD retrieval-augmented execution code."""

try:
    from .service_bootstrap import activate_vendor
except ImportError:  # Support direct checkout/root imports during validation.
    from service_bootstrap import activate_vendor

__all__ = ["activate_vendor"]
