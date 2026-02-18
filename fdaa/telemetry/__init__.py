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
    get_fdaa_exporter,
    record_llm_call,
    record_verification_result,
    record_sandbox_execution,
    record_signing,
    trace_tier,
)
from .exporter import FDAAExporter, FDAATrace, FDAASpan
from .instrumented_pipeline import (
    traced_verify_skill,
    get_recent_traces,
    get_trace,
)

__all__ = [
    # Tracer
    "init_telemetry",
    "get_tracer",
    "get_current_span",
    "get_fdaa_exporter",
    "trace_tier",
    # Recording helpers
    "record_llm_call",
    "record_verification_result",
    "record_sandbox_execution",
    "record_signing",
    # Exporter
    "FDAAExporter",
    "FDAATrace",
    "FDAASpan",
    # Instrumented pipeline
    "traced_verify_skill",
    "get_recent_traces",
    "get_trace",
]
