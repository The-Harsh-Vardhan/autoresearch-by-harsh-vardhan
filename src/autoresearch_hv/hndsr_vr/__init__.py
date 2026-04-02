"""Backwards-compatible re-exports. Prefer autoresearch_hv.domains.hndsr_vr."""

# Explicit re-exports — avoids wildcard import pulling in torch via LifecycleHooks
from autoresearch_hv.domains.hndsr_vr.lifecycle import LifecycleHooks  # noqa: F401

__all__ = ["LifecycleHooks"]
