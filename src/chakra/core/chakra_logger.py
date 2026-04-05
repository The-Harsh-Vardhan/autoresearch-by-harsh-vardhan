"""Chakra — structured stage-aware logging for the autonomous research cycle.

Each stage of the Chakra cycle emits formatted output:
    Sutra    (Plan)    — Scaffold and freeze experiment configs
    Yantra   (Execute) — Train and evaluate models
    Rakshak  (Guard)   — Validate contracts and catch errors
    Vimarsh  (Review)  — Sync results and generate reviews
    Manthan  (Improve) — Propose bounded ablation suggestions
    Aavart   (Cycle)   — Orchestrate a complete end-to-end loop
"""

from __future__ import annotations

import sys
from typing import TextIO


# Stage definitions: (chakra_name, english_name, symbol)
_STAGES = {
    "sutra":   ("Sutra",   "Plan",    "📜"),
    "yantra":  ("Yantra",  "Execute", "⚙️"),
    "rakshak": ("Rakshak", "Guard",   "🛡️"),
    "vimarsh": ("Vimarsh", "Review",  "🔍"),
    "manthan": ("Manthan", "Improve", "🔄"),
}


class ChakraLogger:
    """Structured logger for the Chakra research cycle."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr

    def _emit(self, message: str) -> None:
        self._stream.write(message + "\n")
        self._stream.flush()

    def _stage(self, key: str, message: str) -> None:
        chakra, english, symbol = _STAGES[key]
        self._emit(f"  {symbol} {chakra} ({english}): {message}")

    # -- Stage methods -------------------------------------------------------

    def sutra(self, message: str) -> None:
        """Log a Sutra (Plan) stage message."""
        self._stage("sutra", message)

    def yantra(self, message: str) -> None:
        """Log a Yantra (Execute) stage message."""
        self._stage("yantra", message)

    def rakshak(self, message: str) -> None:
        """Log a Rakshak (Guard) stage message."""
        self._stage("rakshak", message)

    def vimarsh(self, message: str) -> None:
        """Log a Vimarsh (Review) stage message."""
        self._stage("vimarsh", message)

    def manthan(self, message: str) -> None:
        """Log a Manthan (Improve) stage message."""
        self._stage("manthan", message)

    # -- Cycle boundary methods ----------------------------------------------

    def aavart_start(self, domain: str, version: str) -> None:
        """Log the start of an Aavart (Full Cycle)."""
        self._emit("")
        self._emit(f"🔁 [Chakra] Starting Aavart (Full Cycle) — {domain} {version}")
        self._emit(f"   Plan → Execute → Guard → Review → Improve")
        self._emit("")

    def aavart_end(self, domain: str, version: str, decision: str = "") -> None:
        """Log the completion of an Aavart (Full Cycle)."""
        self._emit("")
        suffix = f" Decision: {decision}" if decision else ""
        self._emit(f"✅ [Chakra] Aavart complete — {domain} {version}.{suffix}")
        self._emit("")

    def aavart_fail(self, domain: str, version: str, stage: str, error: str) -> None:
        """Log a failed Aavart cycle."""
        self._emit("")
        self._emit(f"❌ [Chakra] Aavart failed at {stage} — {domain} {version}")
        self._emit(f"   Error: {error}")
        self._emit("")
