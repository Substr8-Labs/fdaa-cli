"""
FDAA API - Vercel Serverless Entry Point
"""

import os
import re
from datetime import datetime, timezone
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum

# =============================================================================
# Configuration
# =============================================================================

MONGODB_URI = os.environ.get("MONGODB_URI", "")

# File injection order (FDAA spec)
INJECTION_ORDER = ["IDENTITY.md", "SOUL.md", "CONTEXT.md", "MEMORY.md", "TOOLS.md"]

# W^X Policy
WRITABLE_FILES = {"MEMORY.md", "CONTEXT.md"}

# Lazy MongoDB connection
_client = None
_db = None

def get_db():
    global _client, _db
    if _db is None:
        if not MONGODB_URI:
            raise HTTPException(status_code=500, detail="MONGODB_URI not configured")
        from motor.motor_asyncio import AsyncIOMotorClient
        _client = AsyncIOMotorClient(MONGODB_URI)
        _db = _client.fdaa
    return _db

# =============================================================================
# Models
# =============================================================================

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


# =============================================================================
# Database Operations
# =============================================================================

async def get_workspace(workspace_id: str) -> Optional[Dict]:
    workspace = await get_db().workspaces.find_one({"_id": workspace_id})
    if not workspace:
        workspace = await get_db().workspaces.find_one({"name": workspace_id})
    return workspace


async def get_files(workspace_id: str) -> Dict[str, str]:
    workspace = await get_workspace(workspace_id)
    if workspace:
        files = workspace.get("files", {})
        if isinstance(files, dict):
            return {
                path: (f["content"] if isinstance(f, dict) else f)
                for path, f in files.items()
            }
    return {}


async def update_file(workspace_id: str, path: str, content: str) -> bool:
    workspace = await get_workspace(workspace_id)
    if not workspace:
        return False
    
    files = workspace.get("files", {})
    if not isinstance(files, dict):
        files = {}
    
    files[path] = {"content": content}
    
    result = await get_db().workspaces.update_one(
        {"_id": workspace["_id"]},
        {"$set": {"files": files, "updated_at": datetime.now(timezone.utc)}}
    )
    return result.modified_count > 0


# =============================================================================
# Agent Logic
# =============================================================================

def _default_model(provider: str) -> str:
    return {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-20250514"}.get(provider, "claude-sonnet-4-20250514")


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
    all_files = await get_files(workspace_id)
    
    files = {}
    if persona:
        persona_prefix = f"personas/{persona}/"
        for path, content in all_files.items():
            if not path.startswith("personas/"):
                files[path] = content
        for path, content in all_files.items():
            if path.startswith(persona_prefix):
                filename = path.replace(persona_prefix, "")
                files[filename] = content
    else:
        files = all_files
    
    sections = []
    for filename in INJECTION_ORDER:
        if filename in files:
            sections.append(f"## {filename}\n\n{files[filename]}")
    
    for filename, content in sorted(files.items()):
        if filename not in INJECTION_ORDER:
            sections.append(f"## {filename}\n\n{content}")
    
    sections.append(_system_instructions())
    return "\n\n---\n\n".join(sections)


async def call_llm(system_prompt: str, history: List[Dict], message: str, provider: str = "anthropic", model: Optional[str] = None) -> str:
    model = model or _default_model(provider)
    messages = list(history) if history else []
    messages.append({"role": "user", "content": message})
    
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text
    elif provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def process_memory_updates(workspace_id: str, persona: Optional[str], response: str) -> tuple[str, bool]:
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
            path = f"personas/{persona}/{filename}" if persona else filename
            await update_file(workspace_id, path, content)
            memory_updated = True
            replacement = f"\n\n*[Memory updated: {filename}]*\n\n"
        
        clean_response = clean_response[:match.start()] + replacement + clean_response[match.end():]
    
    return clean_response.strip(), memory_updated


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="FDAA API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"name": "FDAA API", "version": "0.2.0", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/workspaces")
async def list_workspaces() -> List[WorkspaceInfo]:
    workspaces = []
    async for ws in get_db().workspaces.find({}, {"_id": 1, "name": 1, "files": 1}):
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
        ))
    return workspaces


@app.get("/api/workspaces/{workspace_id}")
async def get_workspace_info(workspace_id: str):
    workspace = await get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@app.post("/api/workspaces/{workspace_id}/chat", response_model=ChatResponse)
async def chat(workspace_id: str, request: ChatRequest):
    workspace = await get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    system_prompt = await assemble_prompt(workspace_id)
    response = await call_llm(system_prompt, request.history or [], request.message, request.provider, request.model)
    clean_response, memory_updated = await process_memory_updates(workspace_id, None, response)
    
    return ChatResponse(response=clean_response, workspace_id=workspace_id, memory_updated=memory_updated)


@app.post("/api/workspaces/{workspace_id}/personas/{persona}/chat", response_model=ChatResponse)
async def chat_with_persona(workspace_id: str, persona: str, request: ChatRequest):
    workspace = await get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    files = workspace.get("files", {})
    persona_prefix = f"personas/{persona}/"
    has_persona = any(path.startswith(persona_prefix) for path in (files.keys() if isinstance(files, dict) else []))
    
    if not has_persona:
        raise HTTPException(status_code=404, detail=f"Persona '{persona}' not found")
    
    system_prompt = await assemble_prompt(workspace_id, persona)
    response = await call_llm(system_prompt, request.history or [], request.message, request.provider, request.model)
    clean_response, memory_updated = await process_memory_updates(workspace_id, persona, response)
    
    return ChatResponse(response=clean_response, workspace_id=workspace_id, persona=persona, memory_updated=memory_updated)


# Vercel serverless handler
handler = Mangum(app)
