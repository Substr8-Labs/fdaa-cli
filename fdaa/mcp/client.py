"""
FDAA MCP Client

Connects to MCP servers via stdio or SSE and provides
a Python interface for tool discovery and invocation.

MCP Protocol: JSON-RPC 2.0
Transport: stdio (subprocess) or SSE (HTTP)
"""

import json
import subprocess
import threading
import queue
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class MCPTool:
    """Represents an MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class MCPResult:
    """Result from an MCP tool call."""
    success: bool
    content: Any = None
    error: Optional[str] = None
    is_error: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        if self.is_error:
            return {"isError": True, "content": [{"type": "text", "text": self.error}]}
        return {"content": self.content}


class MCPClient:
    """
    Client for connecting to MCP servers.
    
    Supports:
    - stdio transport (subprocess)
    - Tool discovery (tools/list)
    - Tool invocation (tools/call)
    
    Usage:
        client = MCPClient("npx", ["-y", "@anthropic/mcp-server-github"])
        client.connect()
        tools = client.list_tools()
        result = client.call_tool("create_issue", {"repo": "...", "title": "..."})
        client.disconnect()
    """
    
    def __init__(
        self,
        command: str,
        args: List[str] = None,
        env: Dict[str, str] = None,
        server_name: str = "unknown"
    ):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.server_name = server_name
        
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending: Dict[int, queue.Queue] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._tools: List[MCPTool] = []
        self._server_info: Dict[str, Any] = {}
    
    def connect(self) -> Dict[str, Any]:
        """Start the MCP server process and initialize connection."""
        import os
        
        # Merge environment
        full_env = os.environ.copy()
        full_env.update(self.env)
        
        # Start subprocess
        self._process = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            bufsize=0
        )
        
        self._running = True
        
        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        
        # Send initialize request
        result = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "fdaa-mcp-client",
                "version": "0.1.0"
            }
        })
        
        self._server_info = result
        
        # Send initialized notification
        self._send_notification("notifications/initialized", {})
        
        return result
    
    def disconnect(self):
        """Stop the MCP server process."""
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
    
    def list_tools(self) -> List[MCPTool]:
        """Get list of available tools from the server."""
        result = self._send_request("tools/list", {})
        
        self._tools = []
        for tool_data in result.get("tools", []):
            tool = MCPTool(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {})
            )
            self._tools.append(tool)
        
        return self._tools
    
    def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> MCPResult:
        """Call a tool on the MCP server."""
        try:
            result = self._send_request("tools/call", {
                "name": name,
                "arguments": arguments or {}
            })
            
            return MCPResult(
                success=True,
                content=result.get("content", []),
                is_error=result.get("isError", False)
            )
        except Exception as e:
            return MCPResult(
                success=False,
                error=str(e),
                is_error=True
            )
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None
    
    @property
    def tools(self) -> List[MCPTool]:
        """List of discovered tools."""
        return self._tools
    
    @property
    def server_info(self) -> Dict[str, Any]:
        """Server information from initialize response."""
        return self._server_info
    
    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        self._request_id += 1
        request_id = self._request_id
        
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        # Create response queue
        response_queue = queue.Queue()
        self._pending[request_id] = response_queue
        
        # Send message
        self._write_message(message)
        
        # Wait for response (timeout: 30s)
        try:
            response = response_queue.get(timeout=30)
        except queue.Empty:
            del self._pending[request_id]
            raise TimeoutError(f"MCP request timed out: {method}")
        finally:
            if request_id in self._pending:
                del self._pending[request_id]
        
        # Check for error
        if "error" in response:
            error = response["error"]
            raise Exception(f"MCP error: {error.get('message', 'Unknown error')}")
        
        return response.get("result", {})
    
    def _send_notification(self, method: str, params: Dict[str, Any]):
        """Send a JSON-RPC notification (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        self._write_message(message)
    
    def _write_message(self, message: Dict[str, Any]):
        """Write a message to the MCP server."""
        if not self._process or not self._process.stdin:
            raise Exception("MCP server not connected")
        
        data = json.dumps(message)
        self._process.stdin.write(f"{data}\n".encode())
        self._process.stdin.flush()
    
    def _read_loop(self):
        """Read messages from the MCP server."""
        while self._running and self._process:
            try:
                line = self._process.stdout.readline()
                if not line:
                    break
                
                line = line.decode().strip()
                if not line:
                    continue
                
                message = json.loads(line)
                
                # Handle response
                if "id" in message:
                    request_id = message["id"]
                    if request_id in self._pending:
                        self._pending[request_id].put(message)
                
                # Handle notifications (no id)
                # Currently ignored, but could be logged
                
            except json.JSONDecodeError:
                continue
            except Exception:
                if self._running:
                    continue
                break
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
