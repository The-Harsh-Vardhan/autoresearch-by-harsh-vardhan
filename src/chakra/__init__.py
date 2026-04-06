"""Chakra — Autonomous Research System."""

__all__ = ["__version__"]

try:
    from importlib.metadata import version as _meta_version

    __version__ = _meta_version("chakra_auto_research")
except Exception:
    __version__ = "0.4.0"
