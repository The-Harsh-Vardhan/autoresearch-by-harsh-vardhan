"""Shared exception hierarchy for Chakra core runtime and orchestration."""

from __future__ import annotations


class ChakraError(Exception):
    """Base exception for user-visible Chakra failures."""


class ManifestError(ChakraError):
    """Raised when a domain manifest is invalid or inconsistent."""


class ConfigValidationError(ChakraError):
    """Raised when runtime configuration values are missing or malformed."""


class ExecutionError(ChakraError):
    """Raised when execution strategy selection or runner invocation fails."""


class StrategyError(ExecutionError):
    """Raised when execution strategy selection cannot resolve safely."""
