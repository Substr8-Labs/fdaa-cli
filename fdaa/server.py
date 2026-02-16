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


def _system_instructions(skill_index: List[Dict] = None) -> str:
    base = """## System Instructions

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
    
    # Add skill index if present (Tier 1: Discovery)
    if skill_index:
        skills_section = "\n\n### Available Skills\n\n"
        skills_section += "You have access to the following skills. Use them when relevant:\n\n"
        for skill in skill_index:
            verified = " ✓" if skill.get("verified") else ""
            skills_section += f"- **{skill['name']}**{verified}: {skill['description']}\n"
        skills_section += "\nTo use a skill, mention it by name and the system will provide full instructions.\n"
        base += skills_section
    
    return base


def match_skills(user_message: str, skill_index: List[Dict]) -> List[str]:
    """
    Match user message to relevant skills (basic keyword matching).
    
    Returns list of skill_ids that should be activated.
    TODO: Replace with embedding-based semantic matching.
    """
    if not user_message or not skill_index:
        return []
    
    message_lower = user_message.lower()
    matched = []
    
    for skill in skill_index:
        description = skill.get("description", "").lower()
        name = skill.get("name", "").lower()
        
        # Check if skill name or key words from description appear in message
        # Extract trigger words (words in quotes or after "trigger on")
        if name in message_lower:
            matched.append(skill["skill_id"])
            continue
        
        # Simple keyword extraction from description
        # Look for quoted phrases or common trigger patterns
        words = description.split()
        for word in words:
            # Skip common words
            if word in {"use", "this", "when", "for", "the", "a", "an", "to", "on", "or"}:
                continue
            if len(word) > 3 and word in message_lower:
                matched.append(skill["skill_id"])
                break
    
    return matched


async def assemble_prompt(
    workspace_id: str, 
    persona: Optional[str] = None,
    user_message: Optional[str] = None
) -> str:
    """
    Assemble system prompt from workspace files.
    
    Implements progressive disclosure:
    - Tier 1: Always include skill index (name + description)
    - Tier 2: Include full instructions for activated skills
    """
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
    
    # Tier 1: Get skill index
    skill_index = await database.get_skill_index(workspace_id)
    
    # Tier 2: Activate skills based on user message
    activated_skills = []
    if user_message and skill_index:
        matched_ids = match_skills(user_message, skill_index)
        for skill_id in matched_ids:
            full_skill = await database.get_skill(workspace_id, skill_id)
            if full_skill:
                activated_skills.append(full_skill)
    
    # Add system instructions with skill index
    sections.append(_system_instructions(skill_index))
    
    # Add activated skill instructions (Tier 2)
    if activated_skills:
        skills_section = "\n\n---\n\n## Activated Skills\n\n"
        for skill in activated_skills:
            skills_section += f"### {skill['name']}\n\n{skill['instructions']}\n\n"
        sections.append(skills_section)
    
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
    """Extract memory blocks and persist to MongoDB with snapshots."""
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
                actor = f"agent:{persona}"
            else:
                path = filename
                actor = "agent"
            
            # Use snapshot-enabled update
            snapshot = await database.update_file_with_snapshot(
                workspace_id, path, content, actor=actor
            )
            memory_updated = True
            replacement = f"\n\n*[Memory updated: {filename} (v{snapshot['version']})*\n\n"
        
        clean_response = clean_response[:match.start()] + replacement + clean_response[match.end():]
    
    return clean_response.strip(), memory_updated


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="FDAA API",
    description="File-Driven Agent Architecture API Server with Progressive Skill Disclosure",
    version="0.3.0",
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
    return {
        "name": "FDAA API",
        "version": "0.3.0",
        "features": ["workspaces", "personas", "skills", "snapshotting"],
        "docs": "/docs"
    }


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
    
    # Pass user message for skill activation
    system_prompt = await assemble_prompt(workspace_id, user_message=request.message)
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
    
    # Pass user message for skill activation
    system_prompt = await assemble_prompt(workspace_id, persona, user_message=request.message)
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


# =============================================================================
# Skills API (Progressive Disclosure)
# =============================================================================

@app.get("/workspaces/{workspace_id}/skills")
async def list_skills(workspace_id: str, full: bool = False):
    """
    List skills in a workspace.
    
    Default: Returns Tier 1 index (name + description only) for prompt injection.
    With full=true: Returns full metadata (no content).
    """
    if full:
        skills = await database.list_skills(workspace_id)
        return {"workspace_id": workspace_id, "skills": skills}
    
    # Tier 1: Index only (~30 tokens per skill)
    index = await database.get_skill_index(workspace_id)
    return {
        "workspace_id": workspace_id,
        "skill_count": len(index),
        "skills": index
    }


@app.get("/workspaces/{workspace_id}/skills/{skill_id}")
async def get_skill(workspace_id: str, skill_id: str):
    """
    Get full skill details (Tier 2).
    
    Includes instructions. Called when skill is activated.
    """
    skill = await database.get_skill(workspace_id, skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    return skill


@app.get("/workspaces/{workspace_id}/skills/{skill_id}/scripts/{script_name}")
async def get_skill_script(workspace_id: str, skill_id: str, script_name: str):
    """
    Get a specific script from a skill (Tier 3).
    
    Called only when agent explicitly needs to execute a script.
    """
    content = await database.get_skill_script(workspace_id, skill_id, script_name)
    if content is None:
        raise HTTPException(status_code=404, detail=f"Script '{script_name}' not found")
    return {"skill_id": skill_id, "script": script_name, "content": content}


@app.get("/workspaces/{workspace_id}/skills/{skill_id}/references/{ref_name}")
async def get_skill_reference(workspace_id: str, skill_id: str, ref_name: str):
    """Get a specific reference document from a skill (Tier 3)."""
    content = await database.get_skill_reference(workspace_id, skill_id, ref_name)
    if content is None:
        raise HTTPException(status_code=404, detail=f"Reference '{ref_name}' not found")
    return {"skill_id": skill_id, "reference": ref_name, "content": content}


@app.post("/workspaces/{workspace_id}/skills")
async def install_skill(workspace_id: str, request: InstallSkillRequest):
    """
    Install a skill into a workspace.
    
    If skill_id already exists, it will be replaced (upgrade).
    """
    workspace = await database.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    skill = await database.install_skill(workspace_id, request.model_dump())
    return {
        "status": "installed",
        "workspace_id": workspace_id,
        "skill_id": request.skill_id,
        "version": skill.get("version", 1)
    }


@app.delete("/workspaces/{workspace_id}/skills/{skill_id}")
async def uninstall_skill(workspace_id: str, skill_id: str):
    """Uninstall a skill from a workspace."""
    deleted = await database.delete_skill(workspace_id, skill_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    return {"status": "uninstalled", "skill_id": skill_id}


# =============================================================================
# Snapshotting / History API
# =============================================================================

class RollbackRequest(BaseModel):
    target_version: int
    actor: Optional[str] = None


class SnapshotInfo(BaseModel):
    workspace_id: str
    path: str
    version: int
    content_hash: str
    parent_hash: str
    actor: str
    timestamp: datetime


class InstallSkillRequest(BaseModel):
    skill_id: str
    name: Optional[str] = None
    description: str
    instructions: str
    scripts: Optional[Dict[str, str]] = None
    references: Optional[Dict[str, str]] = None
    author: Optional[str] = None
    version: Optional[int] = 1
    signature: Optional[str] = None


class SkillIndexItem(BaseModel):
    skill_id: str
    name: str
    description: str
    verified: bool = False


class SkillInfo(BaseModel):
    skill_id: str
    name: str
    description: str
    instructions: str
    author: Optional[str]
    version: int
    verified: bool
    installed_at: datetime


@app.get("/workspaces/{workspace_id}/history/{path:path}")
async def get_file_history(workspace_id: str, path: str, limit: int = 50):
    """
    Get version history for a file.
    
    Returns list of snapshots with metadata (content excluded for performance).
    Use /workspaces/{id}/snapshots/{path}?version=N to get full content.
    """
    history = await database.get_file_history(workspace_id, path, limit)
    return {
        "workspace_id": workspace_id,
        "path": path,
        "total_versions": len(history),
        "history": history
    }


@app.get("/workspaces/{workspace_id}/snapshots/{path:path}")
async def get_snapshot(workspace_id: str, path: str, version: int):
    """
    Get a specific snapshot by version (includes full content).
    """
    snapshot = await database.get_snapshot(workspace_id, path, version)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Snapshot v{version} not found")
    return snapshot


@app.post("/workspaces/{workspace_id}/rollback/{path:path}")
async def rollback_file(workspace_id: str, path: str, request: RollbackRequest):
    """
    Rollback a file to a previous version.
    
    Creates a NEW snapshot with the old content (history is preserved).
    The rollback itself becomes a new entry in the chain.
    """
    snapshot = await database.rollback_to_version(
        workspace_id, 
        path, 
        request.target_version,
        request.actor
    )
    
    if not snapshot:
        raise HTTPException(
            status_code=404, 
            detail=f"Target version {request.target_version} not found"
        )
    
    return {
        "status": "rolled_back",
        "restored_from": request.target_version,
        "new_version": snapshot["version"],
        "snapshot": snapshot
    }


@app.get("/workspaces/{workspace_id}/verify/{path:path}")
async def verify_chain(workspace_id: str, path: str):
    """
    Verify the cryptographic integrity of a file's snapshot chain.
    
    Checks:
    1. Hash chain is unbroken (parent_hash → content_hash linkage)
    2. Content hashes match actual content (tamper detection)
    """
    result = await database.verify_snapshot_chain(workspace_id, path)
    return result


@app.get("/workspaces/{workspace_id}/personas/{persona}/history/{filename}")
async def get_persona_file_history(
    workspace_id: str, 
    persona: str, 
    filename: str,
    limit: int = 50
):
    """Get version history for a persona's file."""
    path = f"personas/{persona}/{filename}"
    history = await database.get_file_history(workspace_id, path, limit)
    return {
        "workspace_id": workspace_id,
        "persona": persona,
        "filename": filename,
        "path": path,
        "total_versions": len(history),
        "history": history
    }


@app.post("/workspaces/{workspace_id}/personas/{persona}/rollback/{filename}")
async def rollback_persona_file(
    workspace_id: str, 
    persona: str, 
    filename: str,
    request: RollbackRequest
):
    """Rollback a persona's file to a previous version."""
    path = f"personas/{persona}/{filename}"
    snapshot = await database.rollback_to_version(
        workspace_id, 
        path, 
        request.target_version,
        request.actor or f"user:rollback:{persona}"
    )
    
    if not snapshot:
        raise HTTPException(
            status_code=404, 
            detail=f"Target version {request.target_version} not found"
        )
    
    return {
        "status": "rolled_back",
        "persona": persona,
        "filename": filename,
        "restored_from": request.target_version,
        "new_version": snapshot["version"],
        "snapshot": snapshot
    }


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
    
    # Pass user message for skill activation
    system_prompt = await assemble_prompt(workspace_id, persona, user_message=message)
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
