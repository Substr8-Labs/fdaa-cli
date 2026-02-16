#!/usr/bin/env python3
"""
FDAA API Server

FastAPI service with MongoDB backend for file-driven agents.
Loads workspaces, assembles prompts, calls LLMs, persists memory.

Usage:
    cd fdaa-cli && source .venv/bin/activate
    uvicorn fdaa.server:app --host 0.0.0.0 --port 8000
"""

import os
import re
from contextlib import asynccontextmanager
from typing import Optional, Dict, List
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import database

# =============================================================================
# Configuration
# =============================================================================

# API keys should be set via environment variables:
# - ANTHROPIC_API_KEY (for Anthropic/Claude)
# - OPENAI_API_KEY (for OpenAI)
# - MONGODB_URI (for MongoDB connection)


# File injection order (FDAA spec)
INJECTION_ORDER = [
    "IDENTITY.md",
    "SOUL.md",
    "CONTEXT.md",
    "MEMORY.md",
    "TOOLS.md",
]

# W^X Policy: Files the agent CAN write to
WRITABLE_FILES = {"MEMORY.md", "CONTEXT.md"}


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    await database.connect_db()
    yield
    await database.close_db()


# =============================================================================
# Models
# =============================================================================

class CreateWorkspaceRequest(BaseModel):
    name: str
    files: Dict[str, str]


class UpdateFileRequest(BaseModel):
    content: str


class ChatRequest(BaseModel):
    message: str
    provider: str = "anthropic"
    model: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    response: str
    workspace_id: str
    persona: Optional[str] = None
    memory_updated: bool = False


class WorkspaceInfo(BaseModel):
    id: str
    name: str
    personas: List[str]
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Agent Logic
# =============================================================================

def _default_model(provider: str) -> str:
    return {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
    }.get(provider, "claude-sonnet-4-20250514")


def _system_instructions() -> str:
    return """## System Instructions

You are an AI agent defined by the files above. Follow these rules:

1. **Stay in character** as defined by IDENTITY.md and SOUL.md
2. **Remember context** from MEMORY.md and CONTEXT.md
3. **Use capabilities** listed in TOOLS.md (if present)

### Memory Updates

When you learn something important, include a memory update block:

```memory:MEMORY.md
[Your updated memory content here]
```

### Boundaries

- You CANNOT modify IDENTITY.md or SOUL.md
- You CAN update MEMORY.md and CONTEXT.md
"""


async def assemble_prompt(workspace_id: str, persona: Optional[str] = None) -> str:
    """Assemble system prompt from workspace files."""
    all_files = await database.get_files(workspace_id)
    
    # Filter files based on persona
    files = {}
    if persona:
        persona_prefix = f"personas/{persona}/"
        # Include shared files (no persona prefix)
        for path, content in all_files.items():
            if not path.startswith("personas/"):
                files[path] = content
        # Include persona-specific files (strip prefix)
        for path, content in all_files.items():
            if path.startswith(persona_prefix):
                filename = path.replace(persona_prefix, "")
                files[filename] = content
    else:
        files = all_files
    
    sections = []
    
    # Add files in defined order
    for filename in INJECTION_ORDER:
        if filename in files:
            sections.append(f"## {filename}\n\n{files[filename]}")
    
    # Add any additional files
    for filename, content in sorted(files.items()):
        if filename not in INJECTION_ORDER:
            sections.append(f"## {filename}\n\n{content}")
    
    # Add system instructions
    sections.append(_system_instructions())
    
    return "\n\n---\n\n".join(sections)


async def call_llm(
    system_prompt: str,
    history: List[Dict[str, str]],
    message: str,
    provider: str = "anthropic",
    model: Optional[str] = None
) -> str:
    """Call LLM with assembled prompt."""
    model = model or _default_model(provider)
    
    messages = list(history) if history else []
    messages.append({"role": "user", "content": message})
    
    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        return response.choices[0].message.content
    
    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def process_memory_updates(
    workspace_id: str,
    persona: Optional[str],
    response: str
) -> tuple[str, bool]:
    """Extract memory blocks and persist to MongoDB."""
    pattern = r"```memory:(\S+)\n(.*?)```"
    memory_updated = False
    
    matches = list(re.finditer(pattern, response, flags=re.DOTALL))
    clean_response = response
    
    for match in reversed(matches):
        filename = match.group(1)
        content = match.group(2).strip()
        
        if filename not in WRITABLE_FILES:
            replacement = f"\n\n*[Blocked: Cannot write to {filename}]*\n\n"
        else:
            # Determine full path
            if persona:
                path = f"personas/{persona}/{filename}"
            else:
                path = filename
            
            await database.update_file(workspace_id, path, content)
            memory_updated = True
            replacement = f"\n\n*[Memory updated: {filename}]*\n\n"
        
        clean_response = clean_response[:match.start()] + replacement + clean_response[match.end():]
    
    return clean_response.strip(), memory_updated


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="FDAA API",
    description="File-Driven Agent Architecture API Server",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health & Info

@app.get("/")
async def root():
    return {"name": "FDAA API", "version": "0.2.0", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# Workspace CRUD

@app.get("/workspaces")
async def list_workspaces() -> List[WorkspaceInfo]:
    """List all workspaces."""
    workspaces = []
    for ws in await database.list_workspaces():
        # Get full workspace to extract personas
        full_ws = await database.get_workspace(ws["_id"])
        personas = set()
        if full_ws:
            files = full_ws.get("files", {})
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
async def get_workspace(workspace_id: str):
    """Get a workspace."""
    workspace = await database.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@app.post("/workspaces/{workspace_id}")
async def create_workspace(workspace_id: str, request: CreateWorkspaceRequest):
    """Create a new workspace."""
    existing = await database.get_workspace(workspace_id)
    if existing:
        raise HTTPException(status_code=409, detail="Workspace already exists")
    
    await database.create_workspace(
        workspace_id=workspace_id,
        name=request.name,
        files=request.files
    )
    return {"status": "created", "workspace_id": workspace_id}


@app.delete("/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str):
    """Delete a workspace."""
    deleted = await database.delete_workspace(workspace_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"status": "deleted"}


# File Operations

@app.get("/workspaces/{workspace_id}/files")
async def list_files(workspace_id: str):
    """List files in a workspace."""
    files = await database.get_files(workspace_id)
    if not files:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"files": list(files.keys())}


@app.get("/workspaces/{workspace_id}/files/{path:path}")
async def get_file(workspace_id: str, path: str):
    """Get a file from a workspace."""
    content = await database.get_file(workspace_id, path)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")
    return {"path": path, "content": content}


@app.put("/workspaces/{workspace_id}/files/{path:path}")
async def update_file(workspace_id: str, path: str, request: UpdateFileRequest):
    """Update a file in a workspace."""
    updated = await database.update_file(workspace_id, path, request.content)
    if not updated:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"status": "updated", "path": path}


# Chat

@app.post("/workspaces/{workspace_id}/chat", response_model=ChatResponse)
async def chat(workspace_id: str, request: ChatRequest):
    """Chat with an agent (no persona - uses root files)."""
    workspace = await database.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    system_prompt = await assemble_prompt(workspace_id)
    response = await call_llm(
        system_prompt,
        request.history or [],
        request.message,
        request.provider,
        request.model
    )
    clean_response, memory_updated = await process_memory_updates(
        workspace_id, None, response
    )
    
    return ChatResponse(
        response=clean_response,
        workspace_id=workspace_id,
        memory_updated=memory_updated
    )


@app.post("/workspaces/{workspace_id}/personas/{persona}/chat", response_model=ChatResponse)
async def chat_with_persona(workspace_id: str, persona: str, request: ChatRequest):
    """Chat with a specific persona."""
    workspace = await database.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Verify persona exists
    files = workspace.get("files", {})
    persona_prefix = f"personas/{persona}/"
    has_persona = any(
        path.startswith(persona_prefix)
        for path in (files.keys() if isinstance(files, dict) else [])
    )
    
    if not has_persona:
        raise HTTPException(status_code=404, detail=f"Persona '{persona}' not found")
    
    system_prompt = await assemble_prompt(workspace_id, persona)
    response = await call_llm(
        system_prompt,
        request.history or [],
        request.message,
        request.provider,
        request.model
    )
    clean_response, memory_updated = await process_memory_updates(
        workspace_id, persona, response
    )
    
    return ChatResponse(
        response=clean_response,
        workspace_id=workspace_id,
        persona=persona,
        memory_updated=memory_updated
    )


# Legacy endpoint (for backwards compatibility with simple /chat)
@app.post("/chat", response_model=ChatResponse)
async def chat_simple(request: dict):
    """Simple chat endpoint (backwards compatible)."""
    workspace_id = request.get("workspace_id")
    persona = request.get("persona")
    message = request.get("message")
    history = request.get("history", [])
    
    if not workspace_id or not message:
        raise HTTPException(status_code=400, detail="workspace_id and message required")
    
    workspace = await database.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    system_prompt = await assemble_prompt(workspace_id, persona)
    response = await call_llm(system_prompt, history, message)
    clean_response, memory_updated = await process_memory_updates(
        workspace_id, persona, response
    )
    
    return ChatResponse(
        response=clean_response,
        workspace_id=workspace_id,
        persona=persona,
        memory_updated=memory_updated
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
