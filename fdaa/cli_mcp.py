#!/usr/bin/env python3
"""
FDAA MCP CLI Commands

Commands for managing MCP gateways from the command line.

Usage:
    fdaa mcp serve                     # Start gateway service
    fdaa mcp connect <gateway_id>      # Connect a gateway
    fdaa mcp list                      # List connected gateways
    fdaa mcp tools <gateway_id>        # List tools
    fdaa mcp call <gateway_id> <tool>  # Call a tool
    fdaa mcp audit <gateway_id>        # View audit log
"""

import os
import sys
import json
import click
import requests
from typing import Optional

# Default gateway service URL
GATEWAY_URL = os.environ.get("FDAA_GATEWAY_URL", "http://localhost:8766")


@click.group()
def mcp():
    """MCP Gateway management commands."""
    pass


@mcp.command()
@click.option("--port", default=8766, help="Port to run the service on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
def serve(port: int, host: str):
    """Start the MCP Gateway service."""
    os.environ["GATEWAY_PORT"] = str(port)
    
    from .gateway_service import main
    click.echo(f"Starting FDAA Gateway Service on {host}:{port}")
    main()


@mcp.command()
@click.argument("gateway_id")
@click.argument("server")
@click.option("--env", "-e", multiple=True, help="Environment variable (KEY=VALUE)")
@click.option("--policy-file", type=click.Path(exists=True), help="Policy file (YAML/JSON)")
@click.option("--readonly", is_flag=True, help="Use read-only policy")
def connect(gateway_id: str, server: str, env: tuple, policy_file: str, readonly: bool):
    """
    Connect an MCP server through the gateway.
    
    Example:
        fdaa mcp connect github @anthropic/mcp-server-github -e GITHUB_TOKEN=ghp_xxx
    """
    # Parse environment variables
    env_dict = {}
    for e in env:
        if "=" in e:
            k, v = e.split("=", 1)
            env_dict[k] = v
    
    # Load policy
    policy = {}
    if policy_file:
        with open(policy_file) as f:
            if policy_file.endswith(".json"):
                policy = json.load(f)
            else:
                import yaml
                policy = yaml.safe_load(f)
    elif readonly:
        policy = {"mode": "allowlist", "tools": []}  # Empty allowlist = block all writes
    
    # Register gateway
    try:
        response = requests.post(
            f"{GATEWAY_URL}/gateways",
            json={
                "gateway_id": gateway_id,
                "server": server,
                "env": env_dict,
                "policy": policy
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        click.echo(f"✓ Gateway '{gateway_id}' connected")
        click.echo(f"  Server: {server}")
        if "server_info" in data:
            info = data["server_info"]
            click.echo(f"  Tools: {info.get('total_tools', '?')} total, {info.get('allowed_tools', '?')} allowed")
    
    except requests.exceptions.ConnectionError:
        click.echo("✗ Gateway service not running. Start with: fdaa mcp serve", err=True)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        click.echo(f"✗ Error: {e.response.text}", err=True)
        sys.exit(1)


@mcp.command("list")
def list_gateways():
    """List connected gateways."""
    try:
        response = requests.get(f"{GATEWAY_URL}/gateways", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        gateways = data.get("gateways", [])
        if not gateways:
            click.echo("No gateways connected")
            return
        
        click.echo(f"Connected gateways ({len(gateways)}):\n")
        for gw in gateways:
            status = "✓" if gw.get("connected") else "✗"
            stats = gw.get("stats", {})
            click.echo(f"  {status} {gw['gateway_id']}")
            click.echo(f"    Tools: {stats.get('allowed_tools', '?')}/{stats.get('total_tools', '?')}")
            click.echo(f"    Requests: {stats.get('total_requests', 0)}")
            click.echo(f"    Pending: {stats.get('pending_approvals', 0)}")
            click.echo()
    
    except requests.exceptions.ConnectionError:
        click.echo("✗ Gateway service not running", err=True)
        sys.exit(1)


@mcp.command()
@click.argument("gateway_id")
@click.option("--all", "-a", "show_all", is_flag=True, help="Show all tools (including blocked)")
def tools(gateway_id: str, show_all: bool):
    """List available tools from a gateway."""
    try:
        response = requests.get(
            f"{GATEWAY_URL}/gateways/{gateway_id}/tools",
            params={"all": show_all},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        tools = data.get("tools", [])
        click.echo(f"Tools for '{gateway_id}' ({len(tools)}):\n")
        
        for tool in tools:
            click.echo(f"  • {tool['name']}")
            if tool.get("description"):
                # Truncate long descriptions
                desc = tool["description"][:80]
                if len(tool["description"]) > 80:
                    desc += "..."
                click.echo(f"    {desc}")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            click.echo(f"✗ Gateway '{gateway_id}' not found", err=True)
        else:
            click.echo(f"✗ Error: {e.response.text}", err=True)
        sys.exit(1)


@mcp.command()
@click.argument("gateway_id")
@click.argument("tool_name")
@click.option("--arg", "-a", multiple=True, help="Tool argument (key=value)")
@click.option("--persona", "-p", help="Persona making the call")
@click.option("--role", "-r", help="Role making the call")
@click.option("--json-args", type=str, help="Arguments as JSON string")
def call(gateway_id: str, tool_name: str, arg: tuple, persona: str, role: str, json_args: str):
    """
    Call a tool through the gateway.
    
    Example:
        fdaa mcp call github get_file_contents -a repo=org/repo -a path=README.md
    """
    # Parse arguments
    arguments = {}
    if json_args:
        arguments = json.loads(json_args)
    else:
        for a in arg:
            if "=" in a:
                k, v = a.split("=", 1)
                # Try to parse as JSON for nested values
                try:
                    arguments[k] = json.loads(v)
                except:
                    arguments[k] = v
    
    try:
        response = requests.post(
            f"{GATEWAY_URL}/gateways/{gateway_id}/call",
            json={
                "tool": tool_name,
                "arguments": arguments,
                "persona": persona,
                "role": role
            },
            timeout=60
        )
        
        if response.status_code == 202:
            # Pending approval
            data = response.json()
            click.echo(f"⏳ Approval required: {data.get('message')}")
            return
        
        if response.status_code == 403:
            click.echo(f"✗ Blocked: {response.json().get('detail')}", err=True)
            sys.exit(1)
        
        response.raise_for_status()
        data = response.json()
        
        click.echo(f"✓ {tool_name} executed successfully\n")
        
        # Pretty print result
        result = data.get("result", [])
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                click.echo(item.get("text", ""))
            else:
                click.echo(json.dumps(item, indent=2))
    
    except requests.exceptions.HTTPError as e:
        click.echo(f"✗ Error: {e.response.text}", err=True)
        sys.exit(1)


@mcp.command()
@click.argument("gateway_id")
@click.option("--limit", "-n", default=20, help="Number of entries to show")
def audit(gateway_id: str, limit: int):
    """View audit log for a gateway."""
    try:
        response = requests.get(
            f"{GATEWAY_URL}/gateways/{gateway_id}/audit",
            params={"limit": limit},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        entries = data.get("entries", [])
        click.echo(f"Audit log for '{gateway_id}' (last {len(entries)}):\n")
        
        for entry in entries:
            status = "✓" if entry.get("allowed") else "✗"
            ts = entry.get("timestamp", "")[:19]
            tool = entry.get("tool", "unknown")
            persona = entry.get("persona") or "-"
            
            click.echo(f"  {ts} {status} {tool} [{persona}]")
            if not entry.get("allowed"):
                click.echo(f"    Reason: {entry.get('policy_reason')}")
    
    except requests.exceptions.HTTPError as e:
        click.echo(f"✗ Error: {e.response.text}", err=True)
        sys.exit(1)


@mcp.command()
@click.argument("gateway_id")
def disconnect(gateway_id: str):
    """Disconnect a gateway."""
    try:
        response = requests.delete(f"{GATEWAY_URL}/gateways/{gateway_id}", timeout=10)
        response.raise_for_status()
        click.echo(f"✓ Gateway '{gateway_id}' disconnected")
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            click.echo(f"✗ Gateway '{gateway_id}' not found", err=True)
        else:
            click.echo(f"✗ Error: {e.response.text}", err=True)
        sys.exit(1)


@mcp.command()
@click.argument("gateway_id")
def pending(gateway_id: str):
    """List pending approval requests."""
    try:
        response = requests.get(f"{GATEWAY_URL}/gateways/{gateway_id}/pending", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pending = data.get("pending", [])
        if not pending:
            click.echo("No pending approvals")
            return
        
        click.echo(f"Pending approvals ({len(pending)}):\n")
        for p in pending:
            click.echo(f"  ID: {p['id']}")
            click.echo(f"  Tool: {p['tool']}")
            click.echo(f"  Created: {p['created_at']}")
            click.echo(f"  Args: {json.dumps(p['arguments'])}")
            click.echo()
    
    except requests.exceptions.HTTPError as e:
        click.echo(f"✗ Error: {e.response.text}", err=True)
        sys.exit(1)


@mcp.command()
@click.argument("gateway_id")
@click.argument("request_id")
@click.option("--approve/--deny", default=True, help="Approve or deny the request")
@click.option("--by", default="cli-user", help="Who is approving")
def approve(gateway_id: str, request_id: str, approve: bool, by: str):
    """Approve or deny a pending request."""
    try:
        response = requests.post(
            f"{GATEWAY_URL}/gateways/{gateway_id}/approve/{request_id}",
            json={"approved": approve, "approved_by": by},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        
        status = "approved" if approve else "denied"
        click.echo(f"✓ Request {request_id} {status}")
        
        if approve and data.get("result"):
            click.echo("\nResult:")
            click.echo(json.dumps(data["result"], indent=2))
    
    except requests.exceptions.HTTPError as e:
        click.echo(f"✗ Error: {e.response.text}", err=True)
        sys.exit(1)


# Add to main CLI if this file is used standalone
if __name__ == "__main__":
    mcp()
