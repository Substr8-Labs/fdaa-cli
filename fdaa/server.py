#!/usr/bin/env python3
"""
FDAA API Server - MongoDB Backend

FastAPI service that loads agent workspaces from MongoDB,
assembles prompts, calls LLMs, and persists memory updates.

Usage:
    cd fdaa-cli && source .venv/bin/activate
    uvicorn fdaa.server:app --reload --port 8000
"""

import os
import re
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient

# =============================================================================
# Configuration
# =============================================================================

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://rudiheydra_db_user:Ed0FeHVyJmq3SnNA@cluster0.ygfkd6s.mongodb.net/"
)
DATABASE_NAME = "fdaa"

# Load Anthropic API key
def load_anthropic_key():
    key_file = "/home/node/.openclaw/secrets/anthropic.env"
    if os.path.exists(key_file):
        with open(key_file) as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                    return True
    return False

load_anthropic_key()

# File injection order (FDAA spec)
INJECTION_ORDER = [
    "IDENTITY.md",
    "SOUL.md", 
    "CONTEXT.md",
    "MEMORY.md",
    "TOOLS.md",
]

# W^X Policy
WRITABLE_FILES = {"MEMORY.md", "CONTEXT.md"}


# =============================================================================
# Database
# =============================================================================

client: MongoClient = None
db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global client, db
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    print(f"Connected to MongoDB: {DATABASE_NAME}")
    yield
    client.close()
    print("MongoDB connection closed")


# =============================================================================
# Models
# =============================================================================

class ChatRequest(BaseModel):
    workspace_id: str
    persona: str  # e.g., "ada", "grace"
    message: str
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    response: str
    memory_updated: bool
    persona: str


class WorkspaceInfo(BaseModel):
    id: str
    name: str
    personas: List[str]
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Core Logic
# =============================================================================

def get_workspace(workspace_id: str) -> Dict:
    """Load workspace from MongoDB."""
    # Try by _id first (string), then by name
    workspace = db.workspaces.find_one({"_id": workspace_id})
    if not workspace:
        workspace = db.workspaces.find_one({"name": workspace_id})
    
    if not workspace:
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_id}")
    
    return workspace


def get_file_content(workspace: Dict, path: str) -> Optional[str]:
    """Get file content from workspace."""
    files = workspace.get("files", {})
    
    # Handle dict structure: {path: {content: "..."}}
    if isinstance(files, dict):
        file_data = files.get(path)
        if file_data and isinstance(file_data, dict):
            return file_data.get("content", "")
        return None
    
    # Handle list structure (legacy)
    for f in files:
        if isinstance(f, dict) and f.get("path") == path:
            return f.get("content", "")
    
    return None


def assemble_prompt(workspace: Dict, persona: str) -> str:
    """Assemble system prompt for a persona."""
    sections = []
    
    # Shared context first
    shared_context = get_file_content(workspace, "CONTEXT.md")
    if shared_context:
        sections.append(f"## Shared Context\n\n{shared_context}")
    
    # Persona files in injection order
    for filename in INJECTION_ORDER:
        path = f"personas/{persona}/{filename}"
        content = get_file_content(workspace, path)
        if content:
            sections.append(f"## {filename}\n\n{content}")
    
    # System instructions
    sections.append("""## System Instructions

You are an AI agent defined by the files above. Follow these rules:

1. **Stay in character** as defined by IDENTITY.md and SOUL.md
2. **Remember context** from MEMORY.md and CONTEXT.md  
3. **Use capabilities** listed in TOOLS.md (if present)

### Memory Updates

When you learn something important that should persist, include a memory update block:

```memory:MEMORY.md
[Your updated memory content here]
```

This will be saved to your MEMORY.md file. Only include what's worth remembering long-term.

### Boundaries

- You CANNOT modify IDENTITY.md or SOUL.md (these define who you are)
- You CAN update MEMORY.md and CONTEXT.md
- Be helpful, stay in character, and remember what matters.
""")
    
    return "\n\n---\n\n".join(sections)


def call_anthropic(system_prompt: str, history: List[Dict], message: str) -> str:
    """Call Anthropic API."""
    from anthropic import Anthropic
    
    client = Anthropic()
    
    messages = list(history) if history else []
    messages.append({"role": "user", "content": message})
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
    )
    
    return response.content[0].text


def process_memory_updates(workspace: Dict, persona: str, response: str) -> tuple[str, bool]:
    """Extract memory blocks and persist to MongoDB."""
    pattern = r"```memory:(\S+)\n(.*?)```"
    memory_updated = False
    
    def apply_update(match):
        nonlocal memory_updated
        filename = match.group(1)
        content = match.group(2).strip()
        
        # Enforce W^X policy
        if filename not in WRITABLE_FILES:
            return f"\n\n*[Blocked: Cannot write to {filename} - W^X policy]*\n\n"
        
        # Update in MongoDB
        path = f"personas/{persona}/{filename}"
        update_file_in_db(workspace["_id"], path, content)
        memory_updated = True
        
        return f"\n\n*[Memory updated: {filename}]*\n\n"
    
    clean_response = re.sub(pattern, apply_update, response, flags=re.DOTALL)
    return clean_response.strip(), memory_updated


def update_file_in_db(workspace_id: str, path: str, content: str):
    """Update a file in MongoDB workspace."""
    # Read-modify-write since paths contain slashes (can't use dot notation)
    workspace = db.workspaces.find_one({"_id": workspace_id})
    if not workspace:
        return
    
    files = workspace.get("files", {})
    if not isinstance(files, dict):
        files = {}
    
    # Update the file
    if path not in files:
        files[path] = {}
    files[path]["content"] = content
    
    # Write back
    db.workspaces.update_one(
        {"_id": workspace_id},
        {
            "$set": {
                "files": files,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )


# =============================================================================
# API Routes
# =============================================================================

app = FastAPI(
    title="FDAA API",
    description="File-Driven Agent Architecture - MongoDB Backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "fdaa-api", "database": DATABASE_NAME}


@app.get("/workspaces")
async def list_workspaces() -> List[WorkspaceInfo]:
    """List all workspaces."""
    workspaces = []
    for ws in db.workspaces.find():
        # Extract persona names from file paths
        personas = set()
        files = ws.get("files", {})
        
        if isinstance(files, dict):
            for path in files.keys():
                if path.startswith("personas/"):
                    parts = path.split("/")
                    if len(parts) >= 2:
                        personas.add(parts[1])
        
        workspaces.append(WorkspaceInfo(
            id=str(ws["_id"]),
            name=ws.get("name", "Unnamed"),
            personas=sorted(list(personas)),
            created_at=ws.get("created_at", datetime.now(timezone.utc)),
            updated_at=ws.get("updated_at", datetime.now(timezone.utc)),
        ))
    
    return workspaces


@app.get("/workspaces/{workspace_id}")
async def get_workspace_info(workspace_id: str) -> WorkspaceInfo:
    """Get workspace details."""
    ws = get_workspace(workspace_id)
    
    personas = set()
    files = ws.get("files", {})
    if isinstance(files, dict):
        for path in files.keys():
            if path.startswith("personas/"):
                parts = path.split("/")
                if len(parts) >= 2:
                    personas.add(parts[1])
    
    return WorkspaceInfo(
        id=str(ws["_id"]),
        name=ws.get("name", "Unnamed"),
        personas=sorted(list(personas)),
        created_at=ws.get("created_at", datetime.now(timezone.utc)),
        updated_at=ws.get("updated_at", datetime.now(timezone.utc)),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat with a persona."""
    workspace = get_workspace(request.workspace_id)
    
    # Assemble prompt
    system_prompt = assemble_prompt(workspace, request.persona)
    
    # Call LLM
    response = call_anthropic(
        system_prompt,
        request.history or [],
        request.message
    )
    
    # Process memory updates
    clean_response, memory_updated = process_memory_updates(
        workspace,
        request.persona,
        response
    )
    
    return ChatResponse(
        response=clean_response,
        memory_updated=memory_updated,
        persona=request.persona
    )


@app.get("/workspaces/{workspace_id}/files/{persona}/{filename}")
async def get_file(workspace_id: str, persona: str, filename: str):
    """Get a specific file from a persona."""
    workspace = get_workspace(workspace_id)
    path = f"personas/{persona}/{filename}"
    content = get_file_content(workspace, path)
    
    if content is None:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    
    return {"path": path, "content": content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
