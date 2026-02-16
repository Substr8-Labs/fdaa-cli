"""
FDAA API - Vercel Serverless (Flask)

Features:
- Workspace management
- Persona-based chat
- Cryptographic snapshotting (hash-chain versioning)
- Rollback support
"""

import os
import re
import json
import hashlib
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

# Genesis hash for snapshot chains
GENESIS_HASH = "sha256:" + "0" * 64


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
# Snapshotting System
# =============================================================================

def compute_content_hash(content):
    """Compute SHA256 hash of content."""
    hash_bytes = hashlib.sha256(content.encode('utf-8')).hexdigest()
    return f"sha256:{hash_bytes}"


def create_snapshot(workspace_id, path, content, actor=None):
    """
    Create a versioned snapshot for a file update.
    Implements hash-chain linking for cryptographic lineage.
    """
    db = get_db()
    if db is None:
        return None
    
    # Get the previous snapshot for this file
    previous = db.snapshots.find_one(
        {"workspace_id": workspace_id, "path": path},
        sort=[("version", -1)]
    )
    
    if previous:
        parent_hash = previous["content_hash"]
        version = previous["version"] + 1
    else:
        parent_hash = GENESIS_HASH
        version = 1
    
    content_hash = compute_content_hash(content)
    
    snapshot = {
        "workspace_id": workspace_id,
        "path": path,
        "content": content,
        "content_hash": content_hash,
        "parent_hash": parent_hash,
        "version": version,
        "actor": actor or "system",
        "timestamp": datetime.now(timezone.utc),
    }
    
    db.snapshots.insert_one(snapshot)
    snapshot.pop("_id", None)
    return snapshot


def get_file_history(workspace_id, path, limit=50):
    """Get version history for a file."""
    db = get_db()
    if db is None:
        return []
    
    snapshots = list(db.snapshots.find(
        {"workspace_id": workspace_id, "path": path},
        {"content": 0}  # Exclude content for performance
    ).sort("version", -1).limit(limit))
    
    for s in snapshots:
        s["_id"] = str(s["_id"])
    
    return snapshots


def get_snapshot(workspace_id, path, version):
    """Get a specific snapshot by version."""
    db = get_db()
    if db is None:
        return None
    
    snapshot = db.snapshots.find_one({
        "workspace_id": workspace_id,
        "path": path,
        "version": version
    })
    
    if snapshot:
        snapshot["_id"] = str(snapshot["_id"])
    
    return snapshot


def rollback_to_version(workspace_id, path, target_version, actor=None):
    """
    Rollback a file to a previous version.
    Creates a NEW snapshot with the old content (preserves history).
    """
    target = get_snapshot(workspace_id, path, target_version)
    if not target:
        return None
    
    # Create a new snapshot with the old content
    new_snapshot = create_snapshot(
        workspace_id, 
        path, 
        target["content"],
        actor=actor or f"rollback:v{target_version}"
    )
    
    # Also update the live workspace file
    update_file(workspace_id, path, target["content"])
    
    return new_snapshot


def verify_snapshot_chain(workspace_id, path):
    """Verify the integrity of a file's snapshot chain."""
    db = get_db()
    if db is None:
        return {"valid": False, "error": "Database not connected"}
    
    snapshots = list(db.snapshots.find(
        {"workspace_id": workspace_id, "path": path}
    ).sort("version", 1))
    
    if not snapshots:
        return {"valid": True, "message": "No snapshots found", "chain_length": 0}
    
    errors = []
    previous_hash = GENESIS_HASH
    
    for snapshot in snapshots:
        # Check parent hash
        if snapshot["parent_hash"] != previous_hash:
            errors.append({
                "version": snapshot["version"],
                "error": "broken_chain",
                "expected_parent": previous_hash,
                "actual_parent": snapshot["parent_hash"]
            })
        
        # Verify content hash
        computed_hash = compute_content_hash(snapshot["content"])
        if computed_hash != snapshot["content_hash"]:
            errors.append({
                "version": snapshot["version"],
                "error": "content_tampered",
                "expected_hash": computed_hash,
                "stored_hash": snapshot["content_hash"]
            })
        
        previous_hash = snapshot["content_hash"]
    
    return {
        "valid": len(errors) == 0,
        "chain_length": len(snapshots),
        "errors": errors if errors else None
    }


def update_file_with_snapshot(workspace_id, path, content, actor=None):
    """Update a file AND create a snapshot."""
    snapshot = create_snapshot(workspace_id, path, content, actor)
    update_file(workspace_id, path, content)
    return snapshot


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
            actor = f"agent:{persona}" if persona else "agent"
            
            # Use snapshot-enabled update
            snapshot = update_file_with_snapshot(workspace_id, path, content, actor)
            memory_updated = True
            
            if snapshot:
                replacement = f"\n\n*[Memory updated: {filename} (v{snapshot['version']})*\n\n"
            else:
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
    """Update persona memory directly (with snapshot)"""
    try:
        workspace = get_workspace(workspace_id)
        if workspace is None:
            return jsonify({"error": "Workspace not found"}), 404
        
        data = request.get_json()
        content = data.get("content")
        filename = data.get("filename", "MEMORY.md")
        actor = data.get("actor", "api")
        
        if not content:
            return jsonify({"error": "content required"}), 400
        
        if filename not in WRITABLE_FILES:
            return jsonify({"error": f"Cannot write to {filename}"}), 403
        
        path = f"personas/{persona}/{filename}"
        
        # Use snapshot-enabled update
        snapshot = update_file_with_snapshot(workspace_id, path, content, actor)
        
        return jsonify({
            "success": snapshot is not None,
            "path": path,
            "version": snapshot["version"] if snapshot else None,
            "content_hash": snapshot["content_hash"] if snapshot else None
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


# =============================================================================
# Snapshotting / History API
# =============================================================================

@app.route("/api/workspaces/<workspace_id>/history/<path:path>")
def api_get_file_history(workspace_id, path):
    """
    Get version history for a file.
    Returns list of snapshots with metadata (content excluded).
    """
    limit = request.args.get("limit", 50, type=int)
    history = get_file_history(workspace_id, path, limit)
    
    return jsonify({
        "workspace_id": workspace_id,
        "path": path,
        "total_versions": len(history),
        "history": history
    })


@app.route("/api/workspaces/<workspace_id>/snapshots/<path:path>")
def api_get_snapshot(workspace_id, path):
    """Get a specific snapshot by version (includes full content)."""
    version = request.args.get("version", type=int)
    if version is None:
        return jsonify({"error": "version parameter required"}), 400
    
    snapshot = get_snapshot(workspace_id, path, version)
    if not snapshot:
        return jsonify({"error": f"Snapshot v{version} not found"}), 404
    
    return jsonify(snapshot)


@app.route("/api/workspaces/<workspace_id>/rollback/<path:path>", methods=["POST"])
def api_rollback_file(workspace_id, path):
    """
    Rollback a file to a previous version.
    Creates a NEW snapshot with the old content (history preserved).
    """
    data = request.get_json()
    target_version = data.get("target_version")
    actor = data.get("actor")
    
    if target_version is None:
        return jsonify({"error": "target_version required"}), 400
    
    snapshot = rollback_to_version(workspace_id, path, target_version, actor)
    
    if not snapshot:
        return jsonify({"error": f"Target version {target_version} not found"}), 404
    
    return jsonify({
        "status": "rolled_back",
        "restored_from": target_version,
        "new_version": snapshot["version"],
        "snapshot": snapshot
    })


@app.route("/api/workspaces/<workspace_id>/verify/<path:path>")
def api_verify_chain(workspace_id, path):
    """
    Verify the cryptographic integrity of a file's snapshot chain.
    
    Checks:
    1. Hash chain is unbroken
    2. Content hashes match actual content
    """
    result = verify_snapshot_chain(workspace_id, path)
    return jsonify(result)


@app.route("/api/workspaces/<workspace_id>/personas/<persona>/history/<filename>")
def api_get_persona_history(workspace_id, persona, filename):
    """Get version history for a persona's file."""
    limit = request.args.get("limit", 50, type=int)
    path = f"personas/{persona}/{filename}"
    history = get_file_history(workspace_id, path, limit)
    
    return jsonify({
        "workspace_id": workspace_id,
        "persona": persona,
        "filename": filename,
        "path": path,
        "total_versions": len(history),
        "history": history
    })


@app.route("/api/workspaces/<workspace_id>/personas/<persona>/rollback/<filename>", methods=["POST"])
def api_rollback_persona_file(workspace_id, persona, filename):
    """Rollback a persona's file to a previous version."""
    data = request.get_json()
    target_version = data.get("target_version")
    actor = data.get("actor", f"user:rollback:{persona}")
    
    if target_version is None:
        return jsonify({"error": "target_version required"}), 400
    
    path = f"personas/{persona}/{filename}"
    snapshot = rollback_to_version(workspace_id, path, target_version, actor)
    
    if not snapshot:
        return jsonify({"error": f"Target version {target_version} not found"}), 404
    
    return jsonify({
        "status": "rolled_back",
        "persona": persona,
        "filename": filename,
        "restored_from": target_version,
        "new_version": snapshot["version"],
        "snapshot": snapshot
    })


@app.route("/api/workspaces/<workspace_id>/personas/<persona>/verify/<filename>")
def api_verify_persona_chain(workspace_id, persona, filename):
    """Verify the cryptographic integrity of a persona file's snapshot chain."""
    path = f"personas/{persona}/{filename}"
    result = verify_snapshot_chain(workspace_id, path)
    return jsonify(result)
