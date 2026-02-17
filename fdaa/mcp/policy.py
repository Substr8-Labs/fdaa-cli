"""
FDAA MCP Policy Engine

Defines governance policies for MCP tools:
- Tool filtering (allowlist/blocklist)
- W^X enforcement (read vs write separation)
- Rate limiting
- Approval requirements
- Audit logging

Policies are defined in .md files for human readability.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timezone
from enum import Enum


class ToolCategory(Enum):
    """Tool operation categories for W^X enforcement."""
    READ = "read"       # Safe read-only operations
    WRITE = "write"     # Mutating operations (needs governance)
    DELETE = "delete"   # Destructive operations (needs approval)
    ADMIN = "admin"     # Administrative operations (high risk)


@dataclass
class ToolPolicy:
    """Policy for a specific tool."""
    tool_name: str
    allowed: bool = True
    category: ToolCategory = ToolCategory.READ
    
    # Access control
    allowed_personas: List[str] = field(default_factory=lambda: ["*"])
    allowed_roles: List[str] = field(default_factory=lambda: ["*"])
    
    # Approval requirements
    requires_approval: bool = False
    approvers: List[str] = field(default_factory=list)
    
    # Rate limiting
    rate_limit: Optional[Dict[str, int]] = None  # {"per_minute": 10, "per_hour": 100}
    
    # Input constraints
    allowed_params: Optional[Dict[str, Any]] = None  # Parameter restrictions
    
    # Description override (for filtered view)
    description_override: Optional[str] = None


@dataclass
class MCPPolicy:
    """
    Policy configuration for an MCP server.
    
    Defines:
    - Which tools are exposed (allowlist/blocklist)
    - W^X categorization per tool
    - Approval requirements
    - Rate limits
    
    Example policy file (github-readonly.md):
    
    ```yaml
    server: @anthropic/mcp-server-github
    mode: allowlist
    
    tools:
      # Read operations - allowed freely
      - name: get_file_contents
        category: read
      - name: search_code
        category: read
      - name: list_issues
        category: read
      
      # Write operations - require approval
      - name: create_issue
        category: write
        requires_approval: true
        approvers: [security-team]
      
      # Blocked operations
      - name: delete_repo
        allowed: false
    ```
    """
    
    server_name: str
    mode: str = "allowlist"  # allowlist | blocklist
    
    # Tool policies
    tool_policies: Dict[str, ToolPolicy] = field(default_factory=dict)
    
    # Default policy for tools not explicitly listed
    default_allowed: bool = False  # allowlist mode: deny by default
    default_category: ToolCategory = ToolCategory.WRITE
    default_requires_approval: bool = True
    
    # Server-wide settings
    enabled: bool = True
    description: str = ""
    
    # W^X categories - which categories need human approval
    write_requires_approval: bool = True
    delete_requires_approval: bool = True
    admin_requires_approval: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPPolicy":
        """Create policy from dictionary (parsed from YAML/JSON)."""
        policy = cls(
            server_name=data.get("server", "unknown"),
            mode=data.get("mode", "allowlist"),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            default_allowed=data.get("default_allowed", False),
            write_requires_approval=data.get("write_requires_approval", True),
            delete_requires_approval=data.get("delete_requires_approval", True),
            admin_requires_approval=data.get("admin_requires_approval", True),
        )
        
        # Parse tool policies
        for tool_data in data.get("tools", []):
            name = tool_data.get("name")
            if not name:
                continue
            
            tool_policy = ToolPolicy(
                tool_name=name,
                allowed=tool_data.get("allowed", True),
                category=ToolCategory(tool_data.get("category", "read")),
                allowed_personas=tool_data.get("personas", ["*"]),
                allowed_roles=tool_data.get("roles", ["*"]),
                requires_approval=tool_data.get("requires_approval", False),
                approvers=tool_data.get("approvers", []),
                rate_limit=tool_data.get("rate_limit"),
                allowed_params=tool_data.get("params"),
                description_override=tool_data.get("description"),
            )
            policy.tool_policies[name] = tool_policy
        
        return policy
    
    @classmethod
    def from_markdown(cls, content: str) -> "MCPPolicy":
        """
        Parse policy from markdown file with YAML frontmatter.
        
        Format:
        ```
        ---
        server: @anthropic/mcp-server-github
        mode: allowlist
        ---
        
        # GitHub Read-Only Policy
        
        ## Allowed Tools
        
        - get_file_contents (read)
        - search_code (read)
        - list_issues (read)
        
        ## Blocked Tools
        
        - delete_repo
        - delete_branch
        ```
        """
        import yaml
        
        # Extract YAML frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        
        if frontmatter_match:
            frontmatter = yaml.safe_load(frontmatter_match.group(1))
        else:
            frontmatter = {}
        
        policy = cls.from_dict(frontmatter)
        
        # Parse tool lists from markdown body
        body = content[frontmatter_match.end():] if frontmatter_match else content
        
        # Simple parsing: look for list items
        # - tool_name (category)
        # - tool_name [blocked]
        
        current_section = None
        for line in body.split('\n'):
            line = line.strip()
            
            if line.startswith('## '):
                current_section = line[3:].lower()
                continue
            
            if line.startswith('- '):
                item = line[2:].strip()
                
                # Parse: tool_name (category) [blocked]
                tool_match = re.match(r'^(\w+)(?:\s*\((\w+)\))?(?:\s*\[(\w+)\])?', item)
                if tool_match:
                    tool_name = tool_match.group(1)
                    category = tool_match.group(2) or "read"
                    modifier = tool_match.group(3)
                    
                    allowed = modifier != "blocked" and current_section != "blocked tools"
                    
                    policy.tool_policies[tool_name] = ToolPolicy(
                        tool_name=tool_name,
                        allowed=allowed,
                        category=ToolCategory(category) if category in ["read", "write", "delete", "admin"] else ToolCategory.READ,
                    )
        
        return policy
    
    def is_tool_allowed(
        self,
        tool_name: str,
        persona: str = None,
        role: str = None
    ) -> tuple[bool, str]:
        """
        Check if a tool is allowed for the given persona/role.
        
        Returns: (allowed: bool, reason: str)
        """
        # Get tool policy
        tool_policy = self.tool_policies.get(tool_name)
        
        if tool_policy is None:
            # Tool not in policy - use defaults
            if self.mode == "allowlist":
                return self.default_allowed, "Not in allowlist"
            else:
                return True, "Not in blocklist"
        
        # Check if explicitly allowed/blocked
        if not tool_policy.allowed:
            return False, "Tool is blocked by policy"
        
        # Check persona access
        if persona and tool_policy.allowed_personas != ["*"]:
            if persona not in tool_policy.allowed_personas:
                return False, f"Persona '{persona}' not authorized"
        
        # Check role access
        if role and tool_policy.allowed_roles != ["*"]:
            if role not in tool_policy.allowed_roles:
                return False, f"Role '{role}' not authorized"
        
        return True, "Allowed"
    
    def requires_approval(self, tool_name: str) -> tuple[bool, List[str]]:
        """
        Check if a tool requires human approval.
        
        Returns: (requires_approval: bool, approvers: List[str])
        """
        tool_policy = self.tool_policies.get(tool_name)
        
        if tool_policy is None:
            # Use defaults based on category
            return self.default_requires_approval, []
        
        # Explicit approval requirement
        if tool_policy.requires_approval:
            return True, tool_policy.approvers
        
        # Category-based approval (W^X enforcement)
        if tool_policy.category == ToolCategory.WRITE and self.write_requires_approval:
            return True, []
        if tool_policy.category == ToolCategory.DELETE and self.delete_requires_approval:
            return True, []
        if tool_policy.category == ToolCategory.ADMIN and self.admin_requires_approval:
            return True, []
        
        return False, []
    
    def get_filtered_tools(self, all_tools: List[Any]) -> List[Any]:
        """
        Filter a list of tools based on policy.
        Returns only tools that are allowed.
        """
        filtered = []
        for tool in all_tools:
            name = tool.name if hasattr(tool, 'name') else tool.get('name', '')
            allowed, _ = self.is_tool_allowed(name)
            if allowed:
                filtered.append(tool)
        return filtered
    
    def get_allowed_tool_names(self) -> Set[str]:
        """Get set of allowed tool names."""
        return {
            name for name, policy in self.tool_policies.items()
            if policy.allowed
        }
    
    def get_blocked_tool_names(self) -> Set[str]:
        """Get set of blocked tool names."""
        return {
            name for name, policy in self.tool_policies.items()
            if not policy.allowed
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary."""
        return {
            "server": self.server_name,
            "mode": self.mode,
            "enabled": self.enabled,
            "description": self.description,
            "tools": [
                {
                    "name": policy.tool_name,
                    "allowed": policy.allowed,
                    "category": policy.category.value,
                    "requires_approval": policy.requires_approval,
                    "approvers": policy.approvers,
                }
                for policy in self.tool_policies.values()
            ]
        }


# =============================================================================
# Predefined Policies
# =============================================================================

def github_readonly_policy() -> MCPPolicy:
    """Pre-built policy for read-only GitHub access."""
    return MCPPolicy.from_dict({
        "server": "@anthropic/mcp-server-github",
        "mode": "allowlist",
        "description": "Read-only GitHub access for code review",
        "tools": [
            {"name": "get_file_contents", "category": "read"},
            {"name": "search_code", "category": "read"},
            {"name": "search_issues", "category": "read"},
            {"name": "search_users", "category": "read"},
            {"name": "list_issues", "category": "read"},
            {"name": "get_issue", "category": "read"},
            {"name": "list_commits", "category": "read"},
            {"name": "get_pull_request", "category": "read"},
            {"name": "list_pull_requests", "category": "read"},
        ]
    })


def github_developer_policy() -> MCPPolicy:
    """Pre-built policy for developer GitHub access (read + create issues/PRs)."""
    return MCPPolicy.from_dict({
        "server": "@anthropic/mcp-server-github",
        "mode": "allowlist",
        "description": "Developer GitHub access - create issues/PRs, no delete",
        "write_requires_approval": False,  # Developers can write freely
        "delete_requires_approval": True,  # Delete still needs approval
        "tools": [
            # Read operations
            {"name": "get_file_contents", "category": "read"},
            {"name": "search_code", "category": "read"},
            {"name": "search_issues", "category": "read"},
            {"name": "list_issues", "category": "read"},
            {"name": "get_issue", "category": "read"},
            {"name": "list_commits", "category": "read"},
            {"name": "get_pull_request", "category": "read"},
            {"name": "list_pull_requests", "category": "read"},
            # Write operations (allowed)
            {"name": "create_issue", "category": "write"},
            {"name": "update_issue", "category": "write"},
            {"name": "add_issue_comment", "category": "write"},
            {"name": "create_pull_request", "category": "write"},
            {"name": "create_branch", "category": "write"},
            {"name": "push_files", "category": "write"},
            # Delete operations (blocked or approval)
            {"name": "delete_branch", "category": "delete", "requires_approval": True},
            {"name": "delete_file", "category": "delete", "requires_approval": True},
        ]
    })


def slack_readonly_policy() -> MCPPolicy:
    """Pre-built policy for read-only Slack access."""
    return MCPPolicy.from_dict({
        "server": "@anthropic/mcp-server-slack",
        "mode": "allowlist",
        "description": "Read-only Slack access",
        "tools": [
            {"name": "list_channels", "category": "read"},
            {"name": "get_channel_history", "category": "read"},
            {"name": "search_messages", "category": "read"},
            {"name": "get_users", "category": "read"},
        ]
    })
