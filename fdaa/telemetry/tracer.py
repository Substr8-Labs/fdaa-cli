"""
FDAA Tracer - OpenTelemetry setup with dual export.
"""

import os
from typing import Optional, Any
from contextlib import contextmanager

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, Span
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    trace = None
    TracerProvider = None

# Jaeger exporter
try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    HAS_JAEGER = True
except ImportError:
    HAS_JAEGER = False
    JaegerExporter = None

from .exporter import FDAAExporter

# Global tracer instance
_tracer: Optional[Any] = None
_fdaa_exporter: Optional[FDAAExporter] = None


def init_telemetry(
    service_name: str = "fdaa-cli",
    jaeger_host: str = None,
    jaeger_port: int = 6831,
    fdaa_backend_url: str = None,
    enable_jaeger: bool = True,
    enable_fdaa: bool = True,
) -> None:
    """Initialize OpenTelemetry with dual export.
    
    Args:
        service_name: Service name for traces
        jaeger_host: Jaeger agent host (default: localhost or JAEGER_AGENT_HOST)
        jaeger_port: Jaeger agent port (default: 6831)
        fdaa_backend_url: FDAA Console backend URL (default: local file storage)
        enable_jaeger: Whether to export to Jaeger
        enable_fdaa: Whether to export to FDAA Console
    """
    global _tracer, _fdaa_exporter
    
    if not HAS_OTEL:
        print("Warning: OpenTelemetry not installed. Telemetry disabled.")
        return
    
    # Create resource with service info
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_VERSION: "0.4.0",
        "fdaa.component": "cli",
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add Jaeger exporter
    if enable_jaeger and HAS_JAEGER:
        jaeger_host = jaeger_host or os.environ.get("JAEGER_AGENT_HOST", "localhost")
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_host,
                agent_port=jaeger_port,
            )
            provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            print(f"Jaeger exporter enabled: {jaeger_host}:{jaeger_port}")
        except Exception as e:
            print(f"Warning: Could not initialize Jaeger exporter: {e}")
    
    # Add FDAA exporter (always enabled for local storage)
    if enable_fdaa:
        _fdaa_exporter = FDAAExporter(backend_url=fdaa_backend_url)
        # Use SimpleSpanProcessor for immediate export (better for CLI)
        provider.add_span_processor(SimpleSpanProcessor(_fdaa_exporter))
        print(f"FDAA exporter enabled: {_fdaa_exporter.storage_path}")
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    # Get tracer
    _tracer = trace.get_tracer("fdaa.pipeline", "0.3.0")


def get_tracer():
    """Get the FDAA tracer instance."""
    global _tracer
    
    if _tracer is None:
        if not HAS_OTEL:
            return NoOpTracer()
        # Auto-initialize with defaults
        init_telemetry()
    
    return _tracer


def get_current_span():
    """Get the current active span."""
    if not HAS_OTEL:
        return NoOpSpan()
    return trace.get_current_span()


def get_fdaa_exporter() -> Optional[FDAAExporter]:
    """Get the FDAA exporter instance."""
    return _fdaa_exporter


# Convenience decorators and context managers

@contextmanager
def trace_tier(tier_name: str, **attributes):
    """Context manager for tracing a verification tier."""
    tracer = get_tracer()
    with tracer.start_as_current_span(f"fdaa.{tier_name}") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span


def record_llm_call(
    span,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    cost_usd: float = None,
    prompt_preview: str = None,
    response_preview: str = None,
):
    """Record LLM call details on a span.
    
    This captures the "reasoning visibility" data that makes
    FDAA observability different from generic tracing.
    """
    span.set_attribute("llm.provider", provider)
    span.set_attribute("llm.model", model)
    span.set_attribute("llm.prompt_tokens", prompt_tokens)
    span.set_attribute("llm.completion_tokens", completion_tokens)
    span.set_attribute("llm.total_tokens", prompt_tokens + completion_tokens)
    span.set_attribute("llm.latency_ms", latency_ms)
    
    if cost_usd is not None:
        span.set_attribute("llm.cost_usd", cost_usd)
    
    # Store full prompt/response in FDAA-specific attributes
    # These won't show in Jaeger but will be captured by FDAAExporter
    if prompt_preview:
        span.set_attribute("fdaa.llm.prompt_preview", prompt_preview[:1000])
    if response_preview:
        span.set_attribute("fdaa.llm.response_preview", response_preview[:1000])


def record_verification_result(
    span,
    tier: int,
    passed: bool,
    recommendation: str = None,
    findings: list = None,
    confidence: float = None,
):
    """Record verification result on a span."""
    span.set_attribute("fdaa.tier", tier)
    span.set_attribute("fdaa.passed", passed)
    
    if recommendation:
        span.set_attribute("fdaa.recommendation", recommendation)
    if confidence is not None:
        span.set_attribute("fdaa.confidence", confidence)
    if findings:
        span.set_attribute("fdaa.findings_count", len(findings))
        span.set_attribute("fdaa.findings", str(findings[:5]))  # First 5


def record_sandbox_execution(
    span,
    container_id: str,
    exit_code: int,
    duration_ms: int,
    memory_mb: float = None,
    violations: list = None,
):
    """Record sandbox execution details."""
    span.set_attribute("sandbox.container_id", container_id)
    span.set_attribute("sandbox.exit_code", exit_code)
    span.set_attribute("sandbox.duration_ms", duration_ms)
    
    if memory_mb is not None:
        span.set_attribute("sandbox.memory_mb", memory_mb)
    if violations:
        span.set_attribute("sandbox.violations_count", len(violations))
        span.set_attribute("sandbox.violations", str(violations))


def record_signing(
    span,
    skill_id: str,
    content_hash: str,
    signer_id: str,
    signature: str,
):
    """Record signing operation details."""
    span.set_attribute("signing.skill_id", skill_id)
    span.set_attribute("signing.content_hash", content_hash)
    span.set_attribute("signing.signer_id", signer_id[:32])
    span.set_attribute("signing.signature", signature[:32])


# No-op implementations for when OpenTelemetry is not available

class NoOpSpan:
    """No-op span for when tracing is disabled."""
    
    def set_attribute(self, key, value):
        pass
    
    def add_event(self, name, attributes=None):
        pass
    
    def set_status(self, status):
        pass
    
    def record_exception(self, exception):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class NoOpTracer:
    """No-op tracer for when OpenTelemetry is not installed."""
    
    def start_as_current_span(self, name, **kwargs):
        return NoOpSpan()
    
    def start_span(self, name, **kwargs):
        return NoOpSpan()
