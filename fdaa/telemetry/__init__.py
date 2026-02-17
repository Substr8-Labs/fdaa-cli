"""
FDAA Telemetry - OpenTelemetry instrumentation with dual export.

Exports to:
1. Jaeger (standard tracing, Grafana integration)
2. FDAA Console (agent-specific views with reasoning visibility)
"""

from .tracer import (
    init_telemetry,
    get_tracer,
    get_current_span,
    record_llm_call,
    record_verification_result,
)
from .exporter import FDAAExporter, FDAATrace, FDAASpan

__all__ = [
    "init_telemetry",
    "get_tracer",
    "get_current_span",
    "record_llm_call",
    "record_verification_result",
    "FDAAExporter",
    "FDAATrace",
    "FDAASpan",
]
