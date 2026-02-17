"""
FDAA Exporter - Custom OpenTelemetry exporter for agent-specific observability.

Stores traces with full reasoning context:
- LLM prompts and responses
- Verification decisions and confidence
- Sandbox execution details
- Cost attribution
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Any
import hashlib

try:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    ReadableSpan = Any
    SpanExporter = object
    SpanExportResult = None


@dataclass
class FDAASpan:
    """FDAA-specific span with reasoning visibility."""
    
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: str
    end_time: str
    duration_ms: float
    status: str
    
    # Standard attributes
    attributes: dict = field(default_factory=dict)
    
    # FDAA-specific enrichments
    tier: Optional[int] = None
    passed: Optional[bool] = None
    recommendation: Optional[str] = None
    
    # LLM reasoning (the differentiator)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_prompt_preview: Optional[str] = None
    llm_response_preview: Optional[str] = None
    llm_tokens_in: Optional[int] = None
    llm_tokens_out: Optional[int] = None
    llm_cost_usd: Optional[float] = None
    
    # Sandbox details
    sandbox_container_id: Optional[str] = None
    sandbox_exit_code: Optional[int] = None
    sandbox_violations: Optional[list] = None
    
    # Child spans
    children: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FDAATrace:
    """Complete FDAA trace with all spans."""
    
    trace_id: str
    skill_path: Optional[str] = None
    skill_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    
    # Overall result
    verdict: Optional[str] = None  # passed, failed, error
    recommendation: Optional[str] = None
    
    # Tier results
    tier1_passed: Optional[bool] = None
    tier2_passed: Optional[bool] = None
    tier3_passed: Optional[bool] = None
    tier4_signed: Optional[bool] = None
    
    # Cost tracking
    total_llm_tokens: int = 0
    total_llm_cost_usd: float = 0.0
    
    # All spans
    spans: list = field(default_factory=list)
    root_span: Optional[FDAASpan] = None
    
    def to_dict(self) -> dict:
        result = asdict(self)
        if self.root_span:
            result["root_span"] = self.root_span.to_dict()
        return result


class FDAAExporter(SpanExporter if HAS_OTEL else object):
    """Custom exporter that stores traces with reasoning context."""
    
    def __init__(
        self,
        backend_url: str = None,
        storage_path: str = None,
    ):
        """Initialize FDAA exporter.
        
        Args:
            backend_url: URL for remote FDAA Console backend (future)
            storage_path: Local path for trace storage
        """
        self.backend_url = backend_url
        self.storage_path = Path(storage_path or os.path.expanduser("~/.fdaa/traces"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory trace accumulator (keyed by trace_id)
        self._traces: dict[str, FDAATrace] = {}
        self._spans: dict[str, list[FDAASpan]] = {}  # trace_id -> spans
    
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to FDAA storage."""
        try:
            for span in spans:
                self._process_span(span)
            
            # Save completed traces
            self._save_completed_traces()
            
            return SpanExportResult.SUCCESS
        except Exception as e:
            print(f"FDAA export error: {e}")
            return SpanExportResult.FAILURE
    
    def _process_span(self, span: ReadableSpan):
        """Process a single span into FDAA format."""
        trace_id = format(span.context.trace_id, '032x')
        span_id = format(span.context.span_id, '016x')
        parent_span_id = None
        if span.parent and span.parent.span_id:
            parent_span_id = format(span.parent.span_id, '016x')
        
        # Calculate duration
        start_ns = span.start_time
        end_ns = span.end_time or time.time_ns()
        duration_ms = (end_ns - start_ns) / 1_000_000
        
        # Extract attributes
        attributes = dict(span.attributes) if span.attributes else {}
        
        # Create FDAA span
        fdaa_span = FDAASpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=span.name,
            start_time=datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat(),
            end_time=datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).isoformat(),
            duration_ms=duration_ms,
            status=span.status.status_code.name if span.status else "UNSET",
            attributes=attributes,
        )
        
        # Extract FDAA-specific attributes
        self._enrich_fdaa_span(fdaa_span, attributes)
        
        # Add to trace
        if trace_id not in self._spans:
            self._spans[trace_id] = []
        self._spans[trace_id].append(fdaa_span)
        
        # Create or update trace
        self._update_trace(trace_id, fdaa_span, attributes)
    
    def _enrich_fdaa_span(self, fdaa_span: FDAASpan, attributes: dict):
        """Extract FDAA-specific fields from attributes."""
        # Tier info
        if "fdaa.tier" in attributes:
            fdaa_span.tier = attributes["fdaa.tier"]
        if "fdaa.passed" in attributes:
            fdaa_span.passed = attributes["fdaa.passed"]
        if "fdaa.recommendation" in attributes:
            fdaa_span.recommendation = attributes["fdaa.recommendation"]
        
        # LLM info
        if "llm.provider" in attributes:
            fdaa_span.llm_provider = attributes["llm.provider"]
        if "llm.model" in attributes:
            fdaa_span.llm_model = attributes["llm.model"]
        if "llm.prompt_tokens" in attributes:
            fdaa_span.llm_tokens_in = attributes["llm.prompt_tokens"]
        if "llm.completion_tokens" in attributes:
            fdaa_span.llm_tokens_out = attributes["llm.completion_tokens"]
        if "llm.cost_usd" in attributes:
            fdaa_span.llm_cost_usd = attributes["llm.cost_usd"]
        
        # FDAA-specific LLM details (reasoning visibility)
        if "fdaa.llm.prompt_preview" in attributes:
            fdaa_span.llm_prompt_preview = attributes["fdaa.llm.prompt_preview"]
        if "fdaa.llm.response_preview" in attributes:
            fdaa_span.llm_response_preview = attributes["fdaa.llm.response_preview"]
        
        # Sandbox info
        if "sandbox.container_id" in attributes:
            fdaa_span.sandbox_container_id = attributes["sandbox.container_id"]
        if "sandbox.exit_code" in attributes:
            fdaa_span.sandbox_exit_code = attributes["sandbox.exit_code"]
        if "sandbox.violations" in attributes:
            fdaa_span.sandbox_violations = attributes["sandbox.violations"]
    
    def _update_trace(self, trace_id: str, span: FDAASpan, attributes: dict):
        """Update trace with span information."""
        if trace_id not in self._traces:
            self._traces[trace_id] = FDAATrace(
                trace_id=trace_id,
                skill_path=attributes.get("skill.path"),
                skill_id=attributes.get("skill.id"),
                started_at=span.start_time,
            )
        
        trace = self._traces[trace_id]
        
        # Update trace-level info from root span
        if span.parent_span_id is None:
            trace.root_span = span
            trace.completed_at = span.end_time
            trace.duration_ms = span.duration_ms
            
            if "fdaa.verdict" in attributes:
                trace.verdict = attributes["fdaa.verdict"]
            if "fdaa.recommendation" in attributes:
                trace.recommendation = attributes["fdaa.recommendation"]
        
        # Accumulate tier results
        if span.tier == 1:
            trace.tier1_passed = span.passed
        elif span.tier == 2:
            trace.tier2_passed = span.passed
        elif span.tier == 3:
            trace.tier3_passed = span.passed
        elif span.tier == 4:
            trace.tier4_signed = span.passed
        
        # Accumulate LLM costs
        if span.llm_tokens_in:
            trace.total_llm_tokens += span.llm_tokens_in
        if span.llm_tokens_out:
            trace.total_llm_tokens += span.llm_tokens_out
        if span.llm_cost_usd:
            trace.total_llm_cost_usd += span.llm_cost_usd
    
    def _save_completed_traces(self):
        """Save completed traces to storage."""
        completed_trace_ids = []
        
        for trace_id, trace in self._traces.items():
            # A trace is complete when its root span has ended
            if trace.root_span and trace.completed_at:
                # Build span tree
                trace.spans = self._spans.get(trace_id, [])
                
                # Save to file
                trace_file = self.storage_path / f"{trace_id}.json"
                trace_file.write_text(json.dumps(trace.to_dict(), indent=2, default=str))
                
                completed_trace_ids.append(trace_id)
        
        # Clean up completed traces from memory
        for trace_id in completed_trace_ids:
            del self._traces[trace_id]
            if trace_id in self._spans:
                del self._spans[trace_id]
    
    def shutdown(self):
        """Shutdown the exporter, saving any pending traces."""
        self._save_completed_traces()
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending traces."""
        self._save_completed_traces()
        return True
    
    # Query methods for the FDAA Console
    
    def list_traces(self, limit: int = 50) -> list[dict]:
        """List recent traces."""
        traces = []
        trace_files = sorted(
            self.storage_path.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )[:limit]
        
        for trace_file in trace_files:
            try:
                data = json.loads(trace_file.read_text())
                # Return summary only
                traces.append({
                    "trace_id": data["trace_id"],
                    "skill_path": data.get("skill_path"),
                    "skill_id": data.get("skill_id"),
                    "started_at": data.get("started_at"),
                    "duration_ms": data.get("duration_ms"),
                    "verdict": data.get("verdict"),
                    "total_llm_cost_usd": data.get("total_llm_cost_usd"),
                })
            except Exception:
                continue
        
        return traces
    
    def get_trace(self, trace_id: str) -> Optional[dict]:
        """Get a specific trace with full details."""
        trace_file = self.storage_path / f"{trace_id}.json"
        if trace_file.exists():
            return json.loads(trace_file.read_text())
        return None
    
    def get_trace_by_skill(self, skill_id: str) -> list[dict]:
        """Get all traces for a specific skill."""
        traces = []
        for trace_file in self.storage_path.glob("*.json"):
            try:
                data = json.loads(trace_file.read_text())
                if data.get("skill_id") == skill_id:
                    traces.append(data)
            except Exception:
                continue
        return sorted(traces, key=lambda t: t.get("started_at", ""), reverse=True)
