"""
FDAA Sandbox - Tier 3 Isolated Execution Environment

Provides containerized execution of skills with:
- Network isolation and monitoring
- Filesystem restrictions
- Resource limits (CPU, memory, time)
- Seccomp syscall filtering
- Result extraction
"""

from .executor import SandboxExecutor, SandboxConfig, ExecutionResult

__all__ = ["SandboxExecutor", "SandboxConfig", "ExecutionResult"]
