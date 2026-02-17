#!/usr/bin/env python3
"""
FDAA Gateway Service

Persistent MCP Gateway for production deployment.
Bridges stdio-based MCP servers to HTTP/SSE API with full governance.

Features:
- Connection pooling for MCP servers
- W^X policy enforcement
- Audit logging to MongoDB
- HTTP/SSE API for tool execution
- Approval workflows

Run:
    python -m fdaa.gateway_service --port 8766

Or with uvicorn:
    uvicorn fdaa.gateway_service:app --host 0.0.0.0 --port 8766
"""

import os
import json
import asyncio
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient

from .mcp.client import MCPClient, MCPTool
from .mcp.policy import MCPPolicy, ToolCategory
from .mcp.gateway import MCPGateway, AuditEntry

# =============================================================================
# Configuration
# =============================================================================

MONGODB_URI = os.environ.get("MONGODB_URI", "")
SERVICE_PORT = int(os.environ.get("GATEWAY_PORT", "8766"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fdaa-gateway")

# =============================================================================
# Database
# =============================================================================

_db = None

def get_db():
    global _db
    if _db is None and MONGODB_URI:
        client = MongoClient(MONGODB_URI)
        _db = client.fdaa
    return _db

# =============================================================================
# Gateway Pool
# =============================================================================

class GatewayPool:
    """
    Manages persistent connections to MCP servers.
    
    Each gateway is kept alive and reused across requests.
    """
    
    def __init__(self):
        self._gateways: Dict[str, MCPGateway] = {}
        self._configs: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def register(
        self,
        gateway_id: str,
        server_command: str,
        server_args: List[str],
        server_env: Dict[str, str],
        policy: MCPPolicy
    ) -> Dict[str, Any]:
        """Register and connect a gateway."""
        async with self._lock:
            # Disconnect existing if any
            if gateway_id in self._gateways:
                try:
                    self._gateways[gateway_id].disconnect()
                except:
                    pass
            
            # Create new gateway
            gateway = MCPGateway(
                server_command=server_command,
                server_args=server_args,
                server_env=server_env,
                policy=policy,
                audit_callback=self._audit_callback
            )
            
            # Connect
            result = gateway.connect()
            
            self._gateways[gateway_id] = gateway
            self._configs[gateway_id] = {
                "server_command": server_command,
                "server_args": server_args,
                "policy": policy.to_dict(),
                "connected_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Gateway '{gateway_id}' connected: {result}")
            return result
    
    async def disconnect(self, gateway_id: str) -> bool:
        """Disconnect a gateway."""
        async with self._lock:
            if gateway_id not in self._gateways:
                return False
            
            try:
                self._gateways[gateway_id].disconnect()
            except:
                pass
            
            del self._gateways[gateway_id]
            if gateway_id in self._configs:
                del self._configs[gateway_id]
            
            logger.info(f"Gateway '{gateway_id}' disconnected")
            return True
    
    def get(self, gateway_id: str) -> Optional[MCPGateway]:
        """Get a connected gateway."""
        return self._gateways.get(gateway_id)
    
    def list(self) -> List[Dict[str, Any]]:
        """List all connected gateways."""
        result = []
        for gid, gateway in self._gateways.items():
            config = self._configs.get(gid, {})
            result.append({
                "gateway_id": gid,
                "connected": gateway.is_connected,
                "stats": gateway.get_stats(),
                "connected_at": config.get("connected_at")
            })
        return result
    
    def _audit_callback(self, entry: AuditEntry):
        """Log audit entry to MongoDB."""
        db = get_db()
        if db is not None:
            db.mcp_audit.insert_one(entry.to_dict())
    
    async def disconnect_all(self):
        """Disconnect all gateways (for shutdown)."""
        async with self._lock:
            for gid, gateway in list(self._gateways.items()):
                try:
                    gateway.disconnect()
                except:
                    pass
            self._gateways.clear()
            self._configs.clear()


# Global pool
pool = GatewayPool()

# =============================================================================
# Pydantic Models
# =============================================================================

class GatewayConfig(BaseModel):
    """Configuration for registering a gateway."""
    gateway_id: str
    server: str = Field(..., description="MCP server package, e.g. @anthropic/mcp-server-github")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables (tokens, etc)")
    policy: Dict[str, Any] = Field(default_factory=dict, description="Governance policy")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gateway_id": "github-dev",
                "server": "@anthropic/mcp-server-github",
                "env": {"GITHUB_TOKEN": "ghp_xxx"},
                "policy": {
                    "mode": "allowlist",
                    "tools": [
                        {"name": "get_file_contents", "category": "read"},
                        {"name": "create_issue", "category": "write"}
                    ]
                }
            }
        }


class ToolCallRequest(BaseModel):
    """Request to call a tool."""
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    persona: Optional[str] = None
    role: Optional[str] = None


class ApprovalRequest(BaseModel):
    """Request to approve/deny a pending call."""
    approved: bool
    approved_by: str


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("FDAA Gateway Service starting...")
    yield
    logger.info("FDAA Gateway Service shutting down...")
    await pool.disconnect_all()


app = FastAPI(
    title="FDAA Gateway Service",
    description="Persistent MCP Gateway with governance",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
async def root():
    """Health check and service info."""
    return {
        "service": "FDAA Gateway",
        "version": "0.1.0",
        "status": "running",
        "gateways": len(pool.list())
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


# -----------------------------------------------------------------------------
# Gateway Management
# -----------------------------------------------------------------------------

@app.get("/gateways")
async def list_gateways():
    """List all connected gateways."""
    return {
        "gateways": pool.list()
    }


@app.post("/gateways")
async def register_gateway(config: GatewayConfig):
    """
    Register and connect a new MCP gateway.
    
    The gateway process will be spawned and kept alive.
    """
    # Build policy
    policy = MCPPolicy.from_dict({
        "server": config.server,
        **config.policy
    })
    
    # Determine server command
    # Common pattern: npx -y @scope/mcp-server-name
    if config.server.startswith("@"):
        server_command = "npx"
        server_args = ["-y", config.server]
    else:
        # Assume it's a direct command
        parts = config.server.split()
        server_command = parts[0]
        server_args = parts[1:] if len(parts) > 1 else []
    
    try:
        result = await pool.register(
            gateway_id=config.gateway_id,
            server_command=server_command,
            server_args=server_args,
            server_env=config.env,
            policy=policy
        )
        
        return {
            "status": "connected",
            "gateway_id": config.gateway_id,
            "server_info": result
        }
    except Exception as e:
        logger.error(f"Failed to connect gateway: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/gateways/{gateway_id}")
async def disconnect_gateway(gateway_id: str):
    """Disconnect and remove a gateway."""
    if await pool.disconnect(gateway_id):
        return {"status": "disconnected", "gateway_id": gateway_id}
    raise HTTPException(status_code=404, detail=f"Gateway '{gateway_id}' not found")


@app.get("/gateways/{gateway_id}")
async def get_gateway(gateway_id: str):
    """Get gateway status and stats."""
    gateway = pool.get(gateway_id)
    if not gateway:
        raise HTTPException(status_code=404, detail=f"Gateway '{gateway_id}' not found")
    
    return {
        "gateway_id": gateway_id,
        "connected": gateway.is_connected,
        "stats": gateway.get_stats()
    }


# -----------------------------------------------------------------------------
# Tool Operations
# -----------------------------------------------------------------------------

@app.get("/gateways/{gateway_id}/tools")
async def list_tools(gateway_id: str, all: bool = False):
    """
    List available tools from a gateway.
    
    By default, returns only tools allowed by policy.
    Use ?all=true to see all upstream tools (for admin).
    """
    gateway = pool.get(gateway_id)
    if not gateway:
        raise HTTPException(status_code=404, detail=f"Gateway '{gateway_id}' not found")
    
    if all:
        tools = gateway.list_all_tools()
    else:
        tools = gateway.list_tools()
    
    return {
        "gateway_id": gateway_id,
        "tools": [t.to_dict() for t in tools],
        "count": len(tools),
        "filtered": not all
    }


@app.post("/gateways/{gateway_id}/call")
async def call_tool(gateway_id: str, request: ToolCallRequest):
    """
    Call a tool through the gateway.
    
    Policy is enforced:
    - Blocked tools return 403
    - Approval-required tools return 202 with request_id
    - Allowed tools execute immediately
    """
    gateway = pool.get(gateway_id)
    if not gateway:
        raise HTTPException(status_code=404, detail=f"Gateway '{gateway_id}' not found")
    
    result = gateway.call_tool(
        tool_name=request.tool,
        arguments=request.arguments,
        persona=request.persona,
        role=request.role
    )
    
    if not result.success:
        # Check if it's an approval request
        if "Approval required" in (result.error or ""):
            return JSONResponse(
                status_code=202,
                content={
                    "status": "pending_approval",
                    "message": result.error,
                    "tool": request.tool
                }
            )
        
        # Check if it's a policy denial
        if "Policy denied" in (result.error or ""):
            raise HTTPException(status_code=403, detail=result.error)
        
        raise HTTPException(status_code=500, detail=result.error)
    
    return {
        "status": "success",
        "tool": request.tool,
        "result": result.content
    }


# -----------------------------------------------------------------------------
# Approval Workflow
# -----------------------------------------------------------------------------

@app.get("/gateways/{gateway_id}/pending")
async def list_pending_approvals(gateway_id: str):
    """List pending approval requests."""
    gateway = pool.get(gateway_id)
    if not gateway:
        raise HTTPException(status_code=404, detail=f"Gateway '{gateway_id}' not found")
    
    pending = gateway.list_pending_approvals()
    return {
        "gateway_id": gateway_id,
        "pending": [
            {
                "id": p.id,
                "tool": p.tool_name,
                "arguments": p.arguments,
                "created_at": p.created_at.isoformat(),
                "approvers": p.approvers
            }
            for p in pending
        ]
    }


@app.post("/gateways/{gateway_id}/approve/{request_id}")
async def approve_request(gateway_id: str, request_id: str, request: ApprovalRequest):
    """Approve or deny a pending tool call."""
    gateway = pool.get(gateway_id)
    if not gateway:
        raise HTTPException(status_code=404, detail=f"Gateway '{gateway_id}' not found")
    
    result = gateway.approve_request(
        request_id=request_id,
        approved_by=request.approved_by,
        approved=request.approved
    )
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "status": "approved" if request.approved else "denied",
        "request_id": request_id,
        "result": result.content if result.success else None
    }


# -----------------------------------------------------------------------------
# Audit Log
# -----------------------------------------------------------------------------

@app.get("/gateways/{gateway_id}/audit")
async def get_audit_log(gateway_id: str, limit: int = 100):
    """Get audit log for a gateway."""
    gateway = pool.get(gateway_id)
    if not gateway:
        raise HTTPException(status_code=404, detail=f"Gateway '{gateway_id}' not found")
    
    entries = gateway.get_audit_log(limit=limit)
    return {
        "gateway_id": gateway_id,
        "entries": [e.to_dict() for e in entries]
    }


@app.get("/audit")
async def global_audit_log(
    gateway_id: Optional[str] = None,
    tool: Optional[str] = None,
    persona: Optional[str] = None,
    limit: int = 100
):
    """Query global audit log from MongoDB."""
    db = get_db()
    if not db:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    query = {}
    if gateway_id:
        query["server"] = gateway_id
    if tool:
        query["tool"] = tool
    if persona:
        query["persona"] = persona
    
    entries = list(
        db.mcp_audit.find(query, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    
    return {
        "count": len(entries),
        "entries": entries
    }


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the gateway service."""
    import uvicorn
    
    uvicorn.run(
        "fdaa.gateway_service:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
