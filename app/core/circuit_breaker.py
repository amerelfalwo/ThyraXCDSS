"""
Circuit Breaker for LLM API calls.

Tracks consecutive failures per service. When a threshold is crossed,
the circuit "opens" and subsequent calls immediately fall back to
local-only logic (XGBoost/ONNX predictions) without burning API quota.

The circuit auto-resets after a cooldown period.

States:
  CLOSED  → normal operation, calls go through
  OPEN    → API is down, skip LLM calls, use fallback
  HALF    → cooldown expired, allow ONE probe call to test recovery
"""

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────
FAILURE_THRESHOLD = 5       # consecutive failures to trip
COOLDOWN_SECONDS = 120      # seconds before retrying after trip


@dataclass
class _CircuitState:
    """Internal mutable state for a single circuit."""
    failures: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False


# One circuit per service name
_circuits: dict[str, _CircuitState] = {}


def _get_circuit(service: str) -> _CircuitState:
    if service not in _circuits:
        _circuits[service] = _CircuitState()
    return _circuits[service]


def is_circuit_open(service: str) -> bool:
    """Check if the circuit is open (API considered down).

    If the cooldown has elapsed, transitions to HALF-OPEN state
    by returning False to allow a single probe call.
    """
    circuit = _get_circuit(service)
    if not circuit.is_open:
        return False

    elapsed = time.monotonic() - circuit.last_failure_time
    if elapsed >= COOLDOWN_SECONDS:
        logger.info(
            f"Circuit breaker [{service}]: cooldown expired "
            f"({elapsed:.0f}s), allowing probe call (HALF-OPEN)"
        )
        return False  # allow one probe

    return True


def record_success(service: str) -> None:
    """Record a successful API call — resets the circuit to CLOSED."""
    circuit = _get_circuit(service)
    if circuit.failures > 0 or circuit.is_open:
        logger.info(f"Circuit breaker [{service}]: recovered → CLOSED")
    circuit.failures = 0
    circuit.is_open = False


def record_failure(service: str) -> None:
    """Record a failed API call. Opens circuit if threshold exceeded."""
    circuit = _get_circuit(service)
    circuit.failures += 1
    circuit.last_failure_time = time.monotonic()

    if circuit.failures >= FAILURE_THRESHOLD and not circuit.is_open:
        circuit.is_open = True
        logger.warning(
            f"Circuit breaker [{service}]: OPEN after "
            f"{circuit.failures} consecutive failures. "
            f"Falling back to local-only for {COOLDOWN_SECONDS}s."
        )


def get_circuit_status() -> dict:
    """Return status of all circuits (for /health endpoint)."""
    return {
        name: {
            "is_open": state.is_open,
            "failures": state.failures,
            "last_failure_ago": (
                f"{time.monotonic() - state.last_failure_time:.0f}s"
                if state.last_failure_time > 0
                else "never"
            ),
        }
        for name, state in _circuits.items()
    }
