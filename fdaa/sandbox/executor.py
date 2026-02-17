"""
FDAA Sandbox Executor - Docker-based isolated skill execution

Executes skills in isolated containers with strict security controls.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
from enum import Enum


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    VIOLATION = "violation"


@dataclass
class Violation:
    """Security violation detected during execution."""
    type: str  # network_exfiltration, filesystem_violation, resource_abuse
    details: str
    severity: str = "high"


@dataclass
class NetworkRequest:
    """Logged network request from sandbox."""
    timestamp: float
    protocol: str
    destination: str
    port: int
    bytes_sent: int = 0
    blocked: bool = False


@dataclass
class ExecutionResult:
    """Result of sandbox execution."""
    status: ExecutionStatus
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    memory_peak_mb: float = 0.0
    network_requests: list[NetworkRequest] = field(default_factory=list)
    filesystem_writes: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    output_files: dict[str, str] = field(default_factory=dict)  # filename -> content
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    # Resource limits
    memory_limit_mb: int = 256
    cpu_limit: float = 1.0  # Number of CPUs
    timeout_seconds: int = 30
    
    # Network policy
    network_enabled: bool = True
    allowed_hosts: list[str] = field(default_factory=list)  # Empty = log only
    
    # Filesystem
    readonly_root: bool = True
    writable_paths: list[str] = field(default_factory=lambda: ["/tmp", "/output"])
    
    # Security
    no_new_privileges: bool = True
    drop_capabilities: bool = True
    seccomp_profile: Optional[str] = None  # Path to seccomp JSON
    
    # Execution
    working_dir: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)


class SandboxExecutor:
    """Execute skills in isolated Docker containers."""
    
    SANDBOX_IMAGE = "python:3.11-slim"
    NETWORK_NAME = "fdaa-sandbox-net"
    
    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self._ensure_prerequisites()
    
    def _ensure_prerequisites(self):
        """Ensure Docker and sandbox network exist."""
        # Check Docker
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError("Docker not available")
        
        # Create isolated network if needed
        result = subprocess.run(
            ["docker", "network", "ls", "--filter", f"name={self.NETWORK_NAME}", "-q"],
            capture_output=True,
            text=True
        )
        if not result.stdout.strip():
            subprocess.run(
                ["docker", "network", "create", "--internal", self.NETWORK_NAME],
                capture_output=True
            )
    
    def _build_docker_args(
        self,
        container_name: str,
        workspace_mount: str,
        output_mount: str
    ) -> list[str]:
        """Build Docker run arguments with security controls."""
        args = [
            "docker", "run",
            "--name", container_name,
            "--rm",  # Auto-remove on exit
            
            # Resource limits
            "--memory", f"{self.config.memory_limit_mb}m",
            "--cpus", str(self.config.cpu_limit),
            "--pids-limit", "100",
            
            # Security
            "--security-opt", "no-new-privileges:true",
            "--cap-drop", "ALL",
            
            # Filesystem mounts
            "-v", f"{workspace_mount}:/workspace:ro",
            "-v", f"{output_mount}:/output:rw",
            "--tmpfs", "/tmp:size=64m",
            
            # Working directory
            "-w", self.config.working_dir,
        ]
        
        # Network policy
        if not self.config.network_enabled:
            args.extend(["--network", "none"])
        else:
            args.extend(["--network", self.NETWORK_NAME])
        
        # Seccomp profile (disabled for now - Docker's default is already restrictive)
        # seccomp_path = self.config.seccomp_profile
        # if seccomp_path is None:
        #     seccomp_path = Path(__file__).parent / "seccomp_profile.json"
        # if seccomp_path and Path(seccomp_path).exists():
        #     args.extend(["--security-opt", f"seccomp={seccomp_path}"])
        
        # Environment variables
        for key, value in self.config.env.items():
            args.extend(["-e", f"{key}={value}"])
        
        return args
    
    def execute_script(
        self,
        script_content: str,
        input_files: dict[str, str] = None,
        interpreter: str = "python3"
    ) -> ExecutionResult:
        """Execute a script in the sandbox.
        
        Args:
            script_content: The script to execute
            input_files: Dict of filename -> content to include
            interpreter: Interpreter to use (python3, bash, node)
        
        Returns:
            ExecutionResult with output and security logs
        """
        container_name = f"fdaa-sandbox-{uuid.uuid4().hex[:8]}"
        input_files = input_files or {}
        
        # Create temp directories for mounts
        workspace_dir = tempfile.mkdtemp(prefix="fdaa-workspace-")
        output_dir = tempfile.mkdtemp(prefix="fdaa-output-")
        
        # Make directories accessible to container
        os.chmod(workspace_dir, 0o755)
        os.chmod(output_dir, 0o777)
        
        try:
            # Write script and input files to workspace
            script_ext = {"python3": ".py", "bash": ".sh", "node": ".js"}.get(interpreter, ".txt")
            script_path = Path(workspace_dir) / f"main{script_ext}"
            script_path.write_text(script_content)
            os.chmod(script_path, 0o644)
            
            for filename, content in input_files.items():
                file_path = Path(workspace_dir) / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
                os.chmod(file_path, 0o644)
            
            # Build Docker command
            docker_args = self._build_docker_args(
                container_name,
                workspace_dir,
                output_dir
            )
            
            # Add image and command
            docker_args.extend([
                self.SANDBOX_IMAGE,
                interpreter, f"/workspace/main{script_ext}"
            ])
            
            # Execute with timeout
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    docker_args,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds
                )
                status = ExecutionStatus.SUCCESS if result.returncode == 0 else ExecutionStatus.ERROR
                exit_code = result.returncode
                stdout = result.stdout
                stderr = result.stderr
                
            except subprocess.TimeoutExpired:
                # Kill the container
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                status = ExecutionStatus.TIMEOUT
                exit_code = -1
                stdout = ""
                stderr = f"Execution timeout after {self.config.timeout_seconds}s"
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Collect output files
            output_files = {}
            for output_file in Path(output_dir).rglob("*"):
                if output_file.is_file():
                    rel_path = output_file.relative_to(output_dir)
                    try:
                        output_files[str(rel_path)] = output_file.read_text()
                    except UnicodeDecodeError:
                        output_files[str(rel_path)] = f"<binary: {output_file.stat().st_size} bytes>"
            
            # Check for violations
            violations = self._analyze_violations(stdout, stderr)
            if violations:
                status = ExecutionStatus.VIOLATION
            
            return ExecutionResult(
                status=status,
                exit_code=exit_code,
                stdout=stdout[:10000],  # Limit output size
                stderr=stderr[:10000],
                duration_ms=duration_ms,
                violations=violations,
                output_files=output_files,
            )
            
        finally:
            # Cleanup temp directories
            shutil.rmtree(workspace_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            
            # Ensure container is removed
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True
            )
    
    def execute_skill(
        self,
        skill_path: Path,
        script_name: str = "main.py",
        args: list[str] = None
    ) -> ExecutionResult:
        """Execute a skill's script in the sandbox.
        
        Args:
            skill_path: Path to skill directory
            script_name: Script to execute from scripts/
            args: Command line arguments to pass
        
        Returns:
            ExecutionResult with output and security logs
        """
        skill_path = Path(skill_path)
        scripts_dir = skill_path / "scripts"
        
        if not scripts_dir.exists():
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                exit_code=1,
                stdout="",
                stderr="No scripts/ directory found",
                duration_ms=0,
            )
        
        script_path = scripts_dir / script_name
        if not script_path.exists():
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                exit_code=1,
                stdout="",
                stderr=f"Script not found: {script_name}",
                duration_ms=0,
            )
        
        # Determine interpreter
        if script_name.endswith(".py"):
            interpreter = "python3"
        elif script_name.endswith(".sh"):
            interpreter = "bash"
        elif script_name.endswith(".js"):
            interpreter = "node"
        else:
            interpreter = "python3"
        
        # Load all input files from skill
        input_files = {}
        for f in scripts_dir.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(scripts_dir)
                try:
                    input_files[str(rel_path)] = f.read_text()
                except UnicodeDecodeError:
                    pass  # Skip binary files
        
        # Load references if available
        refs_dir = skill_path / "references"
        if refs_dir.exists():
            for f in refs_dir.rglob("*"):
                if f.is_file():
                    rel_path = f.relative_to(skill_path)
                    try:
                        input_files[str(rel_path)] = f.read_text()
                    except UnicodeDecodeError:
                        pass
        
        # Execute main script
        script_content = script_path.read_text()
        return self.execute_script(
            script_content,
            input_files=input_files,
            interpreter=interpreter
        )
    
    def _analyze_violations(self, stdout: str, stderr: str) -> list[Violation]:
        """Analyze execution output for security violations."""
        violations = []
        combined = stdout + stderr
        
        # Check for exfiltration attempts
        exfil_patterns = [
            "curl",
            "wget",
            "nc ",
            "netcat",
            "requests.post",
            "requests.put",
            "urllib",
            "socket.connect",
        ]
        for pattern in exfil_patterns:
            if pattern in combined.lower():
                violations.append(Violation(
                    type="potential_exfiltration",
                    details=f"Pattern detected: {pattern}",
                    severity="medium"
                ))
        
        # Check for privilege escalation attempts
        privesc_patterns = [
            "sudo",
            "/etc/passwd",
            "/etc/shadow",
            "chmod +s",
            "setuid",
        ]
        for pattern in privesc_patterns:
            if pattern in combined:
                violations.append(Violation(
                    type="privilege_escalation_attempt",
                    details=f"Pattern detected: {pattern}",
                    severity="high"
                ))
        
        # Check for sensitive file access
        sensitive_paths = [
            "/.ssh/",
            "/.aws/",
            "/.env",
            "/secrets/",
            "/credentials",
        ]
        for pattern in sensitive_paths:
            if pattern in combined:
                violations.append(Violation(
                    type="sensitive_file_access",
                    details=f"Path detected: {pattern}",
                    severity="high"
                ))
        
        return violations


# ============================================================================
# CLI Integration
# ============================================================================

def sandbox_execute(
    script_or_skill: str,
    timeout: int = 30,
    memory_mb: int = 256,
    network: bool = False
) -> ExecutionResult:
    """High-level API for sandbox execution.
    
    Args:
        script_or_skill: Path to script file or skill directory
        timeout: Execution timeout in seconds
        memory_mb: Memory limit in MB
        network: Whether to allow network access
    
    Returns:
        ExecutionResult
    """
    config = SandboxConfig(
        timeout_seconds=timeout,
        memory_limit_mb=memory_mb,
        network_enabled=network,
    )
    
    executor = SandboxExecutor(config)
    path = Path(script_or_skill)
    
    if path.is_dir():
        # Skill directory
        return executor.execute_skill(path)
    elif path.is_file():
        # Single script
        return executor.execute_script(
            path.read_text(),
            interpreter="python3" if path.suffix == ".py" else "bash"
        )
    else:
        return ExecutionResult(
            status=ExecutionStatus.ERROR,
            exit_code=1,
            stdout="",
            stderr=f"Path not found: {script_or_skill}",
            duration_ms=0,
        )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m fdaa.sandbox.executor <script-or-skill>")
        sys.exit(1)
    
    result = sandbox_execute(sys.argv[1])
    print(json.dumps(result.to_dict(), indent=2))
    sys.exit(0 if result.status == ExecutionStatus.SUCCESS else 1)
