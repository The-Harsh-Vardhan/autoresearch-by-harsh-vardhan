"""Backwards-compatible re-exports. Prefer chakra.domains.hndsr_vr."""

# Explicit re-exports — avoids wildcard import pulling in torch via LifecycleHooks
from chakra.domains.hndsr_vr.lifecycle import LifecycleHooks  # noqa: F401

__all__ = ["LifecycleHooks"]
