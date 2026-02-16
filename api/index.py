"""
FDAA API - Vercel Serverless (Flask)
"""

import os
import re
import json
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from pymongo import MongoClient

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
            return None
        # Add TLS settings for Vercel compatibility
        _client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )
        _db = _client.fdaa
    return _db


# =============================================================================
# Database Operations
# =============================================================================

def get_workspace(workspace_id):
    db = get_db()
    if db is None:
        return None
    workspace = db.workspaces.find_one({"_id": workspace_id})
    if workspace is None:
        workspace = db.workspaces.find_one({"name": workspace_id})
    return workspace


def get_files(workspace_id):
    workspace = get_workspace(workspace_id)
    if workspace is not None:
        files = workspace.get("files", {})
        if isinstance(files, dict):
            result = {}
            for path, f in files.items():
                if isinstance(f, dict):
                    result[path] = f.get("content", "")
                else:
                    result[path] = f if f else ""
            return result
    return {}


def update_file(workspace_id, path, content):
    db = get_db()
    workspace = get_workspace(workspace_id)
    if workspace is None or db is None:
        return False
    
    files = workspace.get("files", {})
    if not isinstance(files, dict):
        files = {}
    
    files[path] = {"content": content}
    
    result = db.workspaces.update_one(
        {"_id": workspace["_id"]},
        {"$set": {"files": files, "updated_at": datetime.now(timezone.utc)}}
    )
    return result.modified_count > 0


# =============================================================================
# Agent Logic
# =============================================================================

def assemble_prompt(workspace_id, persona=None):
    all_files = get_files(workspace_id)
    
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
    
    sections.append("""## System Instructions

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
""")
    return "\n\n---\n\n".join(sections)


def call_llm(system_prompt, history, message, provider="anthropic", model=None):
    messages = list(history) if history else []
    messages.append({"role": "user", "content": message})
    
    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        model = model or "claude-sonnet-4-20250514"
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )
        # Handle response - could be list of content blocks
        if hasattr(response, 'content') and len(response.content) > 0:
            return response.content[0].text
        else:
            raise ValueError(f"Unexpected Anthropic response: {response}")
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = model or "gpt-4o"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unknown provider: {provider}")


def process_memory_updates(workspace_id, persona, response):
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
            update_file(workspace_id, path, content)
            memory_updated = True
            replacement = f"\n\n*[Memory updated: {filename}]*\n\n"
        
        clean_response = clean_response[:match.start()] + replacement + clean_response[match.end():]
    
    return clean_response.strip(), memory_updated


# =============================================================================
# Flask App
# =============================================================================

app = Flask(__name__)


@app.route("/")
def root():
    return jsonify({"name": "FDAA API", "version": "0.2.0", "docs": "/api/workspaces"})


@app.route("/api/health")
def health():
    import traceback
    try:
        db = get_db()
        if db is not None:
            # Test actual connection
            db.command("ping")
            return jsonify({"status": "healthy", "db": "connected"})
        else:
            return jsonify({"status": "no database", "mongodb_uri_set": bool(MONGODB_URI)})
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "type": type(e).__name__,
            "mongodb_uri_set": bool(MONGODB_URI),
            "mongodb_uri_prefix": MONGODB_URI[:30] + "..." if MONGODB_URI else "not set"
        }), 500


@app.route("/api/workspaces")
def list_workspaces():
    db = get_db()
    if db is None:
        return jsonify({"error": "Database not configured"}), 500
    
    workspaces = []
    for ws in db.workspaces.find({}, {"_id": 1, "name": 1, "files": 1}):
        personas = set()
        files = ws.get("files", {})
        if isinstance(files, dict):
            for path in files.keys():
                if path.startswith("personas/"):
                    parts = path.split("/")
                    if len(parts) >= 2:
                        personas.add(parts[1])
        workspaces.append({
            "id": str(ws["_id"]),
            "name": ws.get("name", "Unnamed"),
            "personas": sorted(list(personas)),
        })
    return jsonify(workspaces)


@app.route("/api/workspaces/<workspace_id>")
def get_workspace_info(workspace_id):
    workspace = get_workspace(workspace_id)
    if workspace is None:
        return jsonify({"error": "Workspace not found"}), 404
    
    # Convert ObjectId to string for JSON
    workspace["_id"] = str(workspace["_id"])
    return jsonify(workspace)


@app.route("/api/workspaces/<workspace_id>/chat", methods=["POST"])
def chat(workspace_id):
    workspace = get_workspace(workspace_id)
    if workspace is None:
        return jsonify({"error": "Workspace not found"}), 404
    
    data = request.get_json()
    message = data.get("message")
    provider = data.get("provider", "anthropic")
    model = data.get("model")
    history = data.get("history", [])
    
    if not message:
        return jsonify({"error": "message required"}), 400
    
    system_prompt = assemble_prompt(workspace_id)
    response = call_llm(system_prompt, history, message, provider, model)
    clean_response, memory_updated = process_memory_updates(workspace_id, None, response)
    
    return jsonify({
        "response": clean_response,
        "workspace_id": workspace_id,
        "memory_updated": memory_updated
    })


@app.route("/api/workspaces/<workspace_id>/personas/<persona>/context")
def get_persona_context(workspace_id, persona):
    """Get assembled prompt for a persona (no LLM call)"""
    try:
        workspace = get_workspace(workspace_id)
        if workspace is None:
            return jsonify({"error": "Workspace not found"}), 404
        
        files = workspace.get("files", {})
        persona_prefix = f"personas/{persona}/"
        has_persona = any(path.startswith(persona_prefix) for path in (files.keys() if isinstance(files, dict) else []))
        
        if not has_persona:
            return jsonify({"error": f"Persona '{persona}' not found"}), 404
        
        system_prompt = assemble_prompt(workspace_id, persona)
        
        return jsonify({
            "workspace_id": workspace_id,
            "persona": persona,
            "system_prompt": system_prompt
        })
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/api/workspaces/<workspace_id>/personas/<persona>/memory", methods=["POST"])
def update_persona_memory(workspace_id, persona):
    """Update persona memory directly"""
    try:
        workspace = get_workspace(workspace_id)
        if workspace is None:
            return jsonify({"error": "Workspace not found"}), 404
        
        data = request.get_json()
        content = data.get("content")
        filename = data.get("filename", "MEMORY.md")
        
        if not content:
            return jsonify({"error": "content required"}), 400
        
        if filename not in WRITABLE_FILES:
            return jsonify({"error": f"Cannot write to {filename}"}), 403
        
        path = f"personas/{persona}/{filename}"
        success = update_file(workspace_id, path, content)
        
        return jsonify({
            "success": success,
            "path": path
        })
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/api/workspaces/<workspace_id>/personas/<persona>/chat", methods=["POST"])
def chat_with_persona(workspace_id, persona):
    try:
        workspace = get_workspace(workspace_id)
        if workspace is None:
            return jsonify({"error": "Workspace not found"}), 404
        
        files = workspace.get("files", {})
        persona_prefix = f"personas/{persona}/"
        has_persona = any(path.startswith(persona_prefix) for path in (files.keys() if isinstance(files, dict) else []))
        
        if not has_persona:
            return jsonify({"error": f"Persona '{persona}' not found"}), 404
        
        data = request.get_json()
        message = data.get("message")
        provider = data.get("provider", "anthropic")
        model = data.get("model")
        history = data.get("history", [])
        
        if not message:
            return jsonify({"error": "message required"}), 400
        
        # Check API key
        if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            return jsonify({"error": "OPENAI_API_KEY not configured"}), 500
        
        system_prompt = assemble_prompt(workspace_id, persona)
        response = call_llm(system_prompt, history, message, provider, model)
        clean_response, memory_updated = process_memory_updates(workspace_id, persona, response)
        
        return jsonify({
            "response": clean_response,
            "workspace_id": workspace_id,
            "persona": persona,
            "memory_updated": memory_updated
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY"))
        }), 500
