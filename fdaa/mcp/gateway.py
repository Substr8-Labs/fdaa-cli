"""
FDAA MCP Gateway

The governance middleware layer for MCP servers.

Features:
- Connect to upstream MCP servers
- Filter tools based on policy
- Enforce W^X (read vs write) separation
- Audit logging of all tool calls
- Rate limiting per tool
- Approval workflows for high-risk operations

Architecture:
    Agent → FDAA Gateway → Policy Check → Audit Log → Upstream MCP Server
"""

import json
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .client import MCPClient, MCPTool, MCPResult
from .policy import MCPPolicy, ToolCategory


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class AuditEntry:
    """Audit log entry for a tool call."""
    id: str
    timestamp: datetime
    server: str
    tool: str
    persona: Optional[str]
    role: Optional[str]
    
    # Policy check result
    allowed: bool
    policy_reason: str
    
    # Approval (if required)
    approval_required: bool = False
    approval_status: Optional[ApprovalStatus] = None
    approved_by: Optional[str] = None
    
    # Execution
    executed: bool = False
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "server": self.server,
            "tool": self.tool,
            "persona": self.persona,
            "role": self.role,
            "allowed": self.allowed,
            "policy_reason": self.policy_reason,
            "approval_required": self.approval_required,
            "approval_status": self.approval_status.value if self.approval_status else None,
            "approved_by": self.approved_by,
            "executed": self.executed,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class PendingApproval:
    """A tool call waiting for human approval."""
    id: str
    audit_entry: AuditEntry
    tool_name: str
    arguments: Dict[str, Any]
    approvers: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    callback: Optional[Callable] = None


class MCPGateway:
    """
    Governance gateway for MCP servers.
    
    Usage:
        # Create gateway with policy
        gateway = MCPGateway(
            server_command="npx",
            server_args=["-y", "@anthropic/mcp-server-github"],
            server_env={"GITHUB_TOKEN": "..."},
            policy=github_developer_policy()
        )
        
        # Connect to upstream server
        gateway.connect()
        
        # List governed tools (filtered by policy)
        tools = gateway.list_tools()
        
        # Call a tool (policy enforced)
        result = gateway.call_tool(
            "create_issue",
            {"repo": "org/repo", "title": "Bug fix"},
            persona="ada",
            role="developer"
        )
        
        # Disconnect
        gateway.disconnect()
    """
    
    def __init__(
        self,
        server_command: str,
        server_args: List[str] = None,
        server_env: Dict[str, str] = None,
        policy: MCPPolicy = None,
        audit_callback: Callable[[AuditEntry], None] = None,
        approval_callback: Callable[[PendingApproval], None] = None,
    ):
        self.server_command = server_command
        self.server_args = server_args or []
        self.server_env = server_env or {}
        self.policy = policy or MCPPolicy(server_name="unknown")
        
        self._client: Optional[MCPClient] = None
        self._all_tools: List[MCPTool] = []
        self._filtered_tools: List[MCPTool] = []
        
        # Callbacks
        self._audit_callback = audit_callback
        self._approval_callback = approval_callback
        
        # Audit log (in-memory, can be persisted via callback)
        self._audit_log: List[AuditEntry] = []
        
        # Pending approvals
        self._pending_approvals: Dict[str, PendingApproval] = {}
        
        # Request counter for IDs
        self._request_counter = 0
    
    def connect(self) -> Dict[str, Any]:
        """Connect to the upstream MCP server."""
        self._client = MCPClient(
            command=self.server_command,
            args=self.server_args,
            env=self.server_env,
            server_name=self.policy.server_name
        )
        
        result = self._client.connect()
        
        # Discover tools
        self._all_tools = self._client.list_tools()
        
        # Filter by policy
        self._filtered_tools = self.policy.get_filtered_tools(self._all_tools)
        
        return {
            "server_info": result,
            "total_tools": len(self._all_tools),
            "allowed_tools": len(self._filtered_tools),
            "blocked_tools": len(self._all_tools) - len(self._filtered_tools)
        }
    
    def disconnect(self):
        """Disconnect from the upstream MCP server."""
        if self._client:
            self._client.disconnect()
            self._client = None
    
    def list_tools(self) -> List[MCPTool]:
        """
        List tools available through the gateway.
        Returns only tools allowed by policy (Virtual MCP Server pattern).
        """
        return self._filtered_tools
    
    def list_all_tools(self) -> List[MCPTool]:
        """List all tools from upstream server (for admin/audit)."""
        return self._all_tools
    
    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        persona: str = None,
        role: str = None,
        skip_approval: bool = False,  # For pre-approved calls
    ) -> MCPResult:
        """
        Call a tool through the gateway.
        
        Enforces:
        - Policy check (allowed/blocked)
        - Approval workflow (if required)
        - Audit logging
        - Rate limiting (TODO)
        
        Returns MCPResult with success/error status.
        """
        arguments = arguments or {}
        start_time = datetime.now(timezone.utc)
        
        # Generate audit entry ID
        self._request_counter += 1
        audit_id = f"audit_{self._request_counter}_{start_time.strftime('%Y%m%d%H%M%S')}"
        
        # Check policy
        allowed, reason = self.policy.is_tool_allowed(tool_name, persona, role)
        
        # Create audit entry
        audit_entry = AuditEntry(
            id=audit_id,
            timestamp=start_time,
            server=self.policy.server_name,
            tool=tool_name,
            persona=persona,
            role=role,
            allowed=allowed,
            policy_reason=reason,
            arguments=arguments,
        )
        
        # If not allowed, return error
        if not allowed:
            audit_entry.error = f"Policy denied: {reason}"
            self._record_audit(audit_entry)
            return MCPResult(
                success=False,
                error=f"Policy denied: {reason}",
                is_error=True
            )
        
        # Check if approval required
        requires_approval, approvers = self.policy.requires_approval(tool_name)
        audit_entry.approval_required = requires_approval
        
        if requires_approval and not skip_approval:
            # Queue for approval
            pending = PendingApproval(
                id=audit_id,
                audit_entry=audit_entry,
                tool_name=tool_name,
                arguments=arguments,
                approvers=approvers,
                created_at=start_time,
            )
            self._pending_approvals[audit_id] = pending
            audit_entry.approval_status = ApprovalStatus.PENDING
            
            # Notify via callback
            if self._approval_callback:
                self._approval_callback(pending)
            
            self._record_audit(audit_entry)
            
            return MCPResult(
                success=False,
                error=f"Approval required. Request ID: {audit_id}",
                is_error=True,
                content=[{
                    "type": "text",
                    "text": f"This action requires approval. Request ID: {audit_id}"
                }]
            )
        
        # Execute the tool
        try:
            result = self._client.call_tool(tool_name, arguments)
            
            end_time = datetime.now(timezone.utc)
            audit_entry.executed = True
            audit_entry.result = result.content if result.success else None
            audit_entry.error = result.error if not result.success else None
            audit_entry.duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self._record_audit(audit_entry)
            return result
            
        except Exception as e:
            audit_entry.error = str(e)
            self._record_audit(audit_entry)
            return MCPResult(
                success=False,
                error=str(e),
                is_error=True
            )
    
    def approve_request(
        self,
        request_id: str,
        approved_by: str,
        approved: bool = True
    ) -> MCPResult:
        """
        Approve or deny a pending tool call.
        
        If approved, executes the tool call.
        """
        pending = self._pending_approvals.get(request_id)
        if not pending:
            return MCPResult(
                success=False,
                error=f"Request {request_id} not found or expired",
                is_error=True
            )
        
        audit_entry = pending.audit_entry
        
        if approved:
            audit_entry.approval_status = ApprovalStatus.APPROVED
            audit_entry.approved_by = approved_by
            
            # Execute the tool
            del self._pending_approvals[request_id]
            return self.call_tool(
                pending.tool_name,
                pending.arguments,
                persona=audit_entry.persona,
                role=audit_entry.role,
                skip_approval=True
            )
        else:
            audit_entry.approval_status = ApprovalStatus.DENIED
            audit_entry.approved_by = approved_by
            del self._pending_approvals[request_id]
            
            self._record_audit(audit_entry)
            return MCPResult(
                success=False,
                error="Request denied",
                is_error=True
            )
    
    def list_pending_approvals(self) -> List[PendingApproval]:
        """List all pending approval requests."""
        return list(self._pending_approvals.values())
    
    def get_audit_log(self, limit: int = 100) -> List[AuditEntry]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]
    
    def _record_audit(self, entry: AuditEntry):
        """Record an audit entry."""
        self._audit_log.append(entry)
        
        # Notify via callback
        if self._audit_callback:
            self._audit_callback(entry)
    
    @property
    def is_connected(self) -> bool:
        """Check if gateway is connected to upstream server."""
        return self._client is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gateway statistics."""
        return {
            "connected": self.is_connected,
            "server": self.policy.server_name,
            "total_tools": len(self._all_tools),
            "allowed_tools": len(self._filtered_tools),
            "blocked_tools": len(self._all_tools) - len(self._filtered_tools),
            "pending_approvals": len(self._pending_approvals),
            "total_requests": self._request_counter,
            "audit_log_size": len(self._audit_log),
        }
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


# =============================================================================
# Gateway Factory
# =============================================================================

def create_github_gateway(
    token: str,
    policy: MCPPolicy = None,
    audit_callback: Callable = None
) -> MCPGateway:
    """Create a governed GitHub MCP gateway."""
    from .policy import github_developer_policy
    
    return MCPGateway(
        server_command="npx",
        server_args=["-y", "@anthropic/mcp-server-github"],
        server_env={"GITHUB_TOKEN": token},
        policy=policy or github_developer_policy(),
        audit_callback=audit_callback,
    )


def create_slack_gateway(
    token: str,
    policy: MCPPolicy = None,
    audit_callback: Callable = None
) -> MCPGateway:
    """Create a governed Slack MCP gateway."""
    from .policy import slack_readonly_policy
    
    return MCPGateway(
        server_command="npx",
        server_args=["-y", "@anthropic/mcp-server-slack"],
        server_env={"SLACK_TOKEN": token},
        policy=policy or slack_readonly_policy(),
        audit_callback=audit_callback,
    )
