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
from datetime import datetime, timezone, timedelta
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
# Skills System (Progressive Disclosure)
# =============================================================================

def install_skill(workspace_id, skill):
    """
    Install a skill into a workspace.
    
    Skill schema:
    {
        "skill_id": "security-reviewer",
        "name": "Security Reviewer",
        "type": "knowledge",               # knowledge | tool | connector | composite
        "description": "OWASP-aligned reviews. Trigger on 'security scan'.",
        "instructions": "# Full SKILL.md content...",
        "scripts": {"scan.py": "..."},     # Optional
        "references": {"guide.md": "..."},  # Optional
        "author": "substr8-labs",           # Optional
        "version": 1,                       # Optional
        "permissions": {...},               # Optional - access control
        "dependencies": {...},              # Optional - connectors/skills required
    }
    """
    db = get_db()
    if db is None:
        return None
    
    now = datetime.now(timezone.utc)
    
    doc = {
        "workspace_id": workspace_id,
        "skill_id": skill["skill_id"],
        "name": skill.get("name", skill["skill_id"]),
        "type": skill.get("type", "knowledge"),
        "category": skill.get("category", "general"),
        "description": skill.get("description", ""),
        "instructions": skill.get("instructions", ""),
        "scripts": skill.get("scripts", {}),
        "references": skill.get("references", {}),
        "author": skill.get("author"),
        "version": skill.get("version", 1),
        "signature": skill.get("signature"),
        "verified": skill.get("verified", False),
        "trust_score": skill.get("trust_score"),
        "permissions": skill.get("permissions", {}),
        "dependencies": skill.get("dependencies", {}),
        "installed_at": now,
        "updated_at": now,
    }
    
    # Upsert (replace if exists)
    db.skills.replace_one(
        {"workspace_id": workspace_id, "skill_id": skill["skill_id"]},
        doc,
        upsert=True
    )
    
    return doc


def get_skill_index(workspace_id):
    """
    Get Tier 1 skill index (name + description only).
    Minimal payload for context-aware skill discovery.
    """
    db = get_db()
    if db is None:
        return []
    
    skills = list(db.skills.find(
        {"workspace_id": workspace_id},
        {"skill_id": 1, "name": 1, "description": 1, "verified": 1, "_id": 0}
    ))
    return skills


def get_skill(workspace_id, skill_id):
    """Get Tier 2 full skill (includes instructions)."""
    db = get_db()
    if db is None:
        return None
    
    skill = db.skills.find_one(
        {"workspace_id": workspace_id, "skill_id": skill_id}
    )
    if skill:
        skill["_id"] = str(skill["_id"])
    return skill


def get_skill_script(workspace_id, skill_id, script_name):
    """Get Tier 3 specific script content."""
    db = get_db()
    if db is None:
        return None
    
    skill = db.skills.find_one(
        {"workspace_id": workspace_id, "skill_id": skill_id},
        {"scripts": 1}
    )
    if skill and skill.get("scripts"):
        return skill["scripts"].get(script_name)
    return None


def get_skill_reference(workspace_id, skill_id, ref_name):
    """Get a specific reference document from a skill."""
    db = get_db()
    if db is None:
        return None
    
    skill = db.skills.find_one(
        {"workspace_id": workspace_id, "skill_id": skill_id},
        {"references": 1}
    )
    if skill and skill.get("references"):
        return skill["references"].get(ref_name)
    return None


def delete_skill(workspace_id, skill_id):
    """Uninstall a skill from a workspace."""
    db = get_db()
    if db is None:
        return False
    
    result = db.skills.delete_one({
        "workspace_id": workspace_id,
        "skill_id": skill_id
    })
    return result.deleted_count > 0


def update_skill_permissions(workspace_id, skill_id, permissions):
    """Update permissions for a skill."""
    db = get_db()
    if db is None:
        return None
    
    result = db.skills.update_one(
        {"workspace_id": workspace_id, "skill_id": skill_id},
        {"$set": {"permissions": permissions, "updated_at": datetime.now(timezone.utc)}}
    )
    
    if result.matched_count == 0:
        return None
    
    return get_skill(workspace_id, skill_id)


# =============================================================================
# Persona Permissions
# =============================================================================

def get_persona_permissions(workspace_id, persona):
    """Get permissions for a persona."""
    db = get_db()
    if db is None:
        return None
    
    perms = db.persona_permissions.find_one({
        "workspace_id": workspace_id,
        "persona": persona
    })
    
    if perms:
        perms["_id"] = str(perms["_id"])
    
    return perms


def set_persona_permissions(workspace_id, persona, permissions):
    """Set permissions for a persona."""
    db = get_db()
    if db is None:
        return None
    
    now = datetime.now(timezone.utc)
    
    doc = {
        "workspace_id": workspace_id,
        "persona": persona,
        "skill_access": permissions.get("skill_access", {"mode": "allowlist", "skills": [], "categories": []}),
        "connector_access": permissions.get("connector_access", {"allowed": [], "blocked": []}),
        "limits": permissions.get("limits", {"max_skills_per_message": 5, "max_chain_depth": 3}),
        "updated_at": now
    }
    
    db.persona_permissions.update_one(
        {"workspace_id": workspace_id, "persona": persona},
        {"$set": doc, "$setOnInsert": {"created_at": now}},
        upsert=True
    )
    
    return get_persona_permissions(workspace_id, persona)


def check_skill_access(skill, persona, persona_perms):
    """
    Check if a persona can use a skill.
    Returns {"allowed": bool, "reason": str}
    """
    skill_perms = skill.get("permissions", {})
    
    # Check skill's invoke_personas
    allowed_personas = skill_perms.get("invoke_personas", ["*"])
    if "*" not in allowed_personas and persona not in allowed_personas:
        return {"allowed": False, "reason": f"Skill not available to persona '{persona}'"}
    
    # If no persona permissions set, allow by default
    if not persona_perms:
        return {"allowed": True, "reason": "No restrictions"}
    
    # Check persona's skill access mode
    skill_access = persona_perms.get("skill_access", {})
    mode = skill_access.get("mode", "allowlist")
    skill_list = skill_access.get("skills", [])
    categories = skill_access.get("categories", [])
    
    skill_id = skill.get("skill_id")
    skill_category = skill.get("category", "general")
    
    if mode == "allowlist":
        # If allowlist is empty, allow all (no restrictions configured)
        if not skill_list and not categories:
            return {"allowed": True, "reason": "No allowlist restrictions"}
        
        # Check if skill is in allowlist
        if skill_id in skill_list:
            return {"allowed": True, "reason": "Skill in allowlist"}
        
        # Check if skill category is allowed
        if skill_category in categories:
            return {"allowed": True, "reason": f"Category '{skill_category}' allowed"}
        
        return {"allowed": False, "reason": "Skill not in persona's allowlist"}
    
    else:  # blocklist mode
        # Check if skill is blocked
        if skill_id in skill_list:
            return {"allowed": False, "reason": "Skill is blocked for this persona"}
        
        # Check if category is blocked
        blocked_categories = persona_perms.get("skill_access", {}).get("blocked_categories", [])
        if skill_category in blocked_categories:
            return {"allowed": False, "reason": f"Category '{skill_category}' is blocked"}
        
        return {"allowed": True, "reason": "Skill not in blocklist"}


# =============================================================================
# Skill Execution Audit
# =============================================================================

def log_skill_activation(workspace_id, skill_id, persona, trigger_message, status="activated"):
    """Log a skill activation for audit."""
    db = get_db()
    if db is None:
        return None
    
    doc = {
        "workspace_id": workspace_id,
        "skill_id": skill_id,
        "persona": persona,
        "triggered_by": "message",
        "trigger_message": trigger_message[:500] if trigger_message else None,  # Truncate
        "status": status,
        "timestamp": datetime.now(timezone.utc)
    }
    
    result = db.skill_executions.insert_one(doc)
    return str(result.inserted_id)


def check_rate_limit(workspace_id, skill_id, persona):
    """
    Check if skill activation is within rate limits.
    Returns {"allowed": bool, "reason": str, "usage": {...}}
    """
    db = get_db()
    if db is None:
        return {"allowed": True, "reason": "No database"}
    
    # Get skill rate limits
    skill = get_skill(workspace_id, skill_id)
    if not skill:
        return {"allowed": True, "reason": "Skill not found"}
    
    rate_limit = skill.get("permissions", {}).get("rate_limit", {})
    if not rate_limit:
        return {"allowed": True, "reason": "No rate limit configured"}
    
    now = datetime.now(timezone.utc)
    
    # Check hourly limit
    hourly_limit = rate_limit.get("requests_per_hour")
    if hourly_limit:
        hour_ago = now - timedelta(hours=1)
        hourly_count = db.skill_executions.count_documents({
            "workspace_id": workspace_id,
            "skill_id": skill_id,
            "persona": persona,
            "status": "activated",
            "timestamp": {"$gte": hour_ago}
        })
        
        if hourly_count >= hourly_limit:
            return {
                "allowed": False,
                "reason": f"Hourly limit exceeded ({hourly_count}/{hourly_limit})",
                "usage": {"hourly": hourly_count, "hourly_limit": hourly_limit}
            }
    
    # Check daily limit
    daily_limit = rate_limit.get("requests_per_day")
    if daily_limit:
        day_ago = now - timedelta(days=1)
        daily_count = db.skill_executions.count_documents({
            "workspace_id": workspace_id,
            "skill_id": skill_id,
            "persona": persona,
            "status": "activated",
            "timestamp": {"$gte": day_ago}
        })
        
        if daily_count >= daily_limit:
            return {
                "allowed": False,
                "reason": f"Daily limit exceeded ({daily_count}/{daily_limit})",
                "usage": {"daily": daily_count, "daily_limit": daily_limit}
            }
    
    return {"allowed": True, "reason": "Within limits"}


def get_skill_audit(workspace_id, skill_id=None, persona=None, limit=100):
    """Query skill execution audit log."""
    db = get_db()
    if db is None:
        return []
    
    query = {"workspace_id": workspace_id}
    if skill_id:
        query["skill_id"] = skill_id
    if persona:
        query["persona"] = persona
    
    cursor = db.skill_executions.find(query).sort("timestamp", -1).limit(limit)
    
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    
    return results


def list_skills(workspace_id):
    """List all skills in a workspace (full metadata, no content)."""
    db = get_db()
    if db is None:
        return []
    
    skills = list(db.skills.find(
        {"workspace_id": workspace_id},
        {"instructions": 0, "scripts": 0, "references": 0}
    ))
    for s in skills:
        s["_id"] = str(s["_id"])
    return skills


def match_skills(user_message, skill_index):
    """
    Match user message to relevant skills (basic keyword matching).
    Returns list of skill_ids that should be activated.
    """
    if not user_message or not skill_index:
        return []
    
    message_lower = user_message.lower()
    matched = []
    
    for skill in skill_index:
        description = skill.get("description", "").lower()
        name = skill.get("name", "").lower()
        
        # Check if skill name or key words from description appear
        if name in message_lower:
            matched.append(skill["skill_id"])
            continue
        
        # Simple keyword extraction from description
        words = description.split()
        for word in words:
            if word in {"use", "this", "when", "for", "the", "a", "an", "to", "on", "or"}:
                continue
            if len(word) > 3 and word in message_lower:
                matched.append(skill["skill_id"])
                break
    
    return matched


def activate_skills_with_permissions(workspace_id, persona, user_message):
    """
    Full skill activation with permission and rate limit checks.
    Returns {"activated": [...], "blocked": [...], "rate_limited": [...]}
    """
    # Get skill index
    skill_index = get_skill_index(workspace_id)
    if not skill_index:
        return {"activated": [], "blocked": [], "rate_limited": []}
    
    # Match skills to message
    matched_ids = match_skills(user_message, skill_index)
    if not matched_ids:
        return {"activated": [], "blocked": [], "rate_limited": []}
    
    # Get persona permissions
    persona_perms = get_persona_permissions(workspace_id, persona)
    
    activated = []
    blocked = []
    rate_limited = []
    
    for skill_id in matched_ids:
        skill = get_skill(workspace_id, skill_id)
        if not skill:
            continue
        
        # Check permission
        access = check_skill_access(skill, persona, persona_perms)
        
        if not access["allowed"]:
            blocked.append({
                "skill_id": skill_id,
                "name": skill.get("name", skill_id),
                "reason": access["reason"]
            })
            # Log blocked activation
            log_skill_activation(workspace_id, skill_id, persona, user_message, status="blocked")
            continue
        
        # Check rate limit
        rate_check = check_rate_limit(workspace_id, skill_id, persona)
        
        if not rate_check["allowed"]:
            rate_limited.append({
                "skill_id": skill_id,
                "name": skill.get("name", skill_id),
                "reason": rate_check["reason"],
                "usage": rate_check.get("usage", {})
            })
            # Log rate limited
            log_skill_activation(workspace_id, skill_id, persona, user_message, status="rate_limited")
            continue
        
        # Skill is activated
        activated.append({
            "skill_id": skill_id,
            "name": skill.get("name", skill_id),
            "description": skill.get("description", ""),
            "type": skill.get("type", "knowledge"),
            "instructions": skill.get("instructions", "")
        })
        
        # Log successful activation
        log_skill_activation(workspace_id, skill_id, persona, user_message, status="activated")
    
    return {"activated": activated, "blocked": blocked, "rate_limited": rate_limited}


# =============================================================================
# Agent Logic
# =============================================================================

def format_skill_index(skill_index):
    """Format skill index for system prompt injection."""
    if not skill_index:
        return ""
    
    lines = ["\n\n### Available Skills\n"]
    lines.append("You have access to the following skills. Use them when relevant:\n")
    for skill in skill_index:
        verified = " ✓" if skill.get("verified") else ""
        lines.append(f"- **{skill['name']}**{verified}: {skill['description']}")
    lines.append("\nTo use a skill, mention it by name and the system will provide full instructions.\n")
    return "\n".join(lines)


def assemble_prompt(workspace_id, persona=None, user_message=None):
    """
    Assemble system prompt from workspace files.
    
    Implements progressive disclosure:
    - Tier 1: Always include skill index (name + description)
    - Tier 2: Include full instructions for activated skills
    """
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
    
    # Tier 1: Get skill index
    skill_index = get_skill_index(workspace_id)
    skill_section = format_skill_index(skill_index)
    
    # Tier 2: Activate skills based on user message
    activated_skills = []
    if user_message and skill_index:
        matched_ids = match_skills(user_message, skill_index)
        for skill_id in matched_ids:
            full_skill = get_skill(workspace_id, skill_id)
            if full_skill:
                activated_skills.append(full_skill)
    
    system_instructions = """## System Instructions

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
    
    # Add skill index to system instructions
    if skill_section:
        system_instructions += skill_section
    
    sections.append(system_instructions)
    
    # Add activated skill instructions (Tier 2)
    if activated_skills:
        skills_content = "## Activated Skills\n\n"
        for skill in activated_skills:
            skills_content += f"### {skill['name']}\n\n{skill['instructions']}\n\n"
        sections.append(skills_content)
    
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
    return jsonify({
        "name": "FDAA API",
        "version": "0.4.0",
        "features": ["workspaces", "personas", "skills", "snapshotting", "permissions", "audit"],
        "docs": "/api/workspaces"
    })


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
    
    # Pass user message for skill activation
    system_prompt = assemble_prompt(workspace_id, user_message=message)
    response = call_llm(system_prompt, history, message, provider, model)
    clean_response, memory_updated = process_memory_updates(workspace_id, None, response)
    
    return jsonify({
        "response": clean_response,
        "workspace_id": workspace_id,
        "memory_updated": memory_updated
    })


@app.route("/api/workspaces/<workspace_id>/personas/<persona>/context")
def get_persona_context(workspace_id, persona):
    """
    Get assembled prompt for a persona (no LLM call).
    
    Optional query params:
    - message: User message to trigger skill activation
    
    Returns:
    - system_prompt: Assembled prompt with skill index
    - activated_skills: List of skills triggered by message (if provided)
    - blocked_skills: Skills that matched but were blocked by permissions
    - available_skills: Total skills available
    """
    try:
        workspace = get_workspace(workspace_id)
        if workspace is None:
            return jsonify({"error": "Workspace not found"}), 404
        
        files = workspace.get("files", {})
        persona_prefix = f"personas/{persona}/"
        has_persona = any(path.startswith(persona_prefix) for path in (files.keys() if isinstance(files, dict) else []))
        
        if not has_persona:
            return jsonify({"error": f"Persona '{persona}' not found"}), 404
        
        # Get optional message for skill activation
        user_message = request.args.get("message")
        
        # Assemble prompt (includes skill matching if message provided)
        system_prompt = assemble_prompt(workspace_id, persona, user_message=user_message)
        
        # Get skill info for response
        skill_index = get_skill_index(workspace_id)
        
        # Use permission-aware activation if message provided
        activated_skills = []
        blocked_skills = []
        rate_limited_skills = []
        
        if user_message and skill_index:
            activation = activate_skills_with_permissions(workspace_id, persona, user_message)
            activated_skills = [
                {"skill_id": s["skill_id"], "name": s["name"], "description": s["description"]}
                for s in activation["activated"]
            ]
            blocked_skills = activation["blocked"]
            rate_limited_skills = activation.get("rate_limited", [])
        
        return jsonify({
            "workspace_id": workspace_id,
            "persona": persona,
            "system_prompt": system_prompt,
            "available_skills": len(skill_index),
            "activated_skills": activated_skills,
            "blocked_skills": blocked_skills,
            "rate_limited_skills": rate_limited_skills
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
        
        # Pass user message for skill activation
        system_prompt = assemble_prompt(workspace_id, persona, user_message=message)
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
# Skills API (Progressive Disclosure)
# =============================================================================

@app.route("/api/workspaces/<workspace_id>/skills")
def api_list_skills(workspace_id):
    """
    List skills in a workspace.
    
    Default: Returns Tier 1 index (name + description only).
    With full=true: Returns full metadata (no content).
    """
    full = request.args.get("full", "false").lower() == "true"
    
    if full:
        skills = list_skills(workspace_id)
        return jsonify({"workspace_id": workspace_id, "skills": skills})
    
    # Tier 1: Index only (~30 tokens per skill)
    index = get_skill_index(workspace_id)
    return jsonify({
        "workspace_id": workspace_id,
        "skill_count": len(index),
        "skills": index
    })


@app.route("/api/workspaces/<workspace_id>/skills/<skill_id>")
def api_get_skill(workspace_id, skill_id):
    """Get full skill details (Tier 2). Includes instructions."""
    skill = get_skill(workspace_id, skill_id)
    if not skill:
        return jsonify({"error": f"Skill '{skill_id}' not found"}), 404
    return jsonify(skill)


@app.route("/api/workspaces/<workspace_id>/skills/<skill_id>/scripts/<script_name>")
def api_get_skill_script(workspace_id, skill_id, script_name):
    """Get a specific script from a skill (Tier 3)."""
    content = get_skill_script(workspace_id, skill_id, script_name)
    if content is None:
        return jsonify({"error": f"Script '{script_name}' not found"}), 404
    return jsonify({"skill_id": skill_id, "script": script_name, "content": content})


@app.route("/api/workspaces/<workspace_id>/skills/<skill_id>/references/<ref_name>")
def api_get_skill_reference(workspace_id, skill_id, ref_name):
    """Get a specific reference document from a skill (Tier 3)."""
    content = get_skill_reference(workspace_id, skill_id, ref_name)
    if content is None:
        return jsonify({"error": f"Reference '{ref_name}' not found"}), 404
    return jsonify({"skill_id": skill_id, "reference": ref_name, "content": content})


@app.route("/api/workspaces/<workspace_id>/skills", methods=["POST"])
def api_install_skill(workspace_id):
    """
    Install a skill into a workspace.
    If skill_id already exists, it will be replaced (upgrade).
    """
    workspace = get_workspace(workspace_id)
    if workspace is None:
        return jsonify({"error": "Workspace not found"}), 404
    
    data = request.get_json()
    
    if not data.get("skill_id"):
        return jsonify({"error": "skill_id required"}), 400
    if not data.get("description"):
        return jsonify({"error": "description required"}), 400
    if not data.get("instructions"):
        return jsonify({"error": "instructions required"}), 400
    
    skill = install_skill(workspace_id, data)
    
    if skill is None:
        return jsonify({"error": "Failed to install skill"}), 500
    
    return jsonify({
        "status": "installed",
        "workspace_id": workspace_id,
        "skill_id": data["skill_id"],
        "version": skill.get("version", 1)
    })


@app.route("/api/workspaces/<workspace_id>/skills/<skill_id>", methods=["DELETE"])
def api_uninstall_skill(workspace_id, skill_id):
    """Uninstall a skill from a workspace."""
    deleted = delete_skill(workspace_id, skill_id)
    if not deleted:
        return jsonify({"error": f"Skill '{skill_id}' not found"}), 404
    return jsonify({"status": "uninstalled", "skill_id": skill_id})


# =============================================================================
# Permissions API
# =============================================================================

@app.route("/api/workspaces/<workspace_id>/skills/<skill_id>/permissions", methods=["GET", "PUT"])
def api_skill_permissions(workspace_id, skill_id):
    """Get or update skill permissions."""
    skill = get_skill(workspace_id, skill_id)
    if not skill:
        return jsonify({"error": f"Skill '{skill_id}' not found"}), 404
    
    if request.method == "GET":
        return jsonify({
            "workspace_id": workspace_id,
            "skill_id": skill_id,
            "permissions": skill.get("permissions", {})
        })
    
    # PUT - update permissions
    data = request.get_json()
    permissions = data.get("permissions", {})
    
    # Validate permissions structure
    valid_keys = {"invoke_personas", "invoke_roles", "scopes", "rate_limit", "approval", "install_roles"}
    for key in permissions.keys():
        if key not in valid_keys:
            return jsonify({"error": f"Invalid permission key: {key}"}), 400
    
    updated = update_skill_permissions(workspace_id, skill_id, permissions)
    if not updated:
        return jsonify({"error": "Failed to update permissions"}), 500
    
    return jsonify({
        "status": "updated",
        "workspace_id": workspace_id,
        "skill_id": skill_id,
        "permissions": updated.get("permissions", {})
    })


@app.route("/api/workspaces/<workspace_id>/personas/<persona>/permissions", methods=["GET", "PUT"])
def api_persona_permissions(workspace_id, persona):
    """Get or update persona permissions."""
    workspace = get_workspace(workspace_id)
    if workspace is None:
        return jsonify({"error": "Workspace not found"}), 404
    
    if request.method == "GET":
        perms = get_persona_permissions(workspace_id, persona)
        return jsonify({
            "workspace_id": workspace_id,
            "persona": persona,
            "permissions": perms or {"skill_access": {"mode": "allowlist", "skills": [], "categories": []}}
        })
    
    # PUT - update permissions
    data = request.get_json()
    perms = set_persona_permissions(workspace_id, persona, data)
    
    return jsonify({
        "status": "updated",
        "workspace_id": workspace_id,
        "persona": persona,
        "permissions": perms
    })


@app.route("/api/workspaces/<workspace_id>/check-access", methods=["POST"])
def api_check_access(workspace_id):
    """
    Check if a persona can invoke a skill.
    Body: {"persona": "ada", "skill_id": "send-email"}
    """
    data = request.get_json()
    persona = data.get("persona")
    skill_id = data.get("skill_id")
    
    if not persona or not skill_id:
        return jsonify({"error": "persona and skill_id required"}), 400
    
    skill = get_skill(workspace_id, skill_id)
    if not skill:
        return jsonify({"error": f"Skill '{skill_id}' not found"}), 404
    
    persona_perms = get_persona_permissions(workspace_id, persona)
    access = check_skill_access(skill, persona, persona_perms)
    
    return jsonify({
        "workspace_id": workspace_id,
        "persona": persona,
        "skill_id": skill_id,
        "allowed": access["allowed"],
        "reason": access["reason"]
    })


@app.route("/api/workspaces/<workspace_id>/skills/audit")
def api_skill_audit(workspace_id):
    """
    Query skill execution audit log.
    Params: ?skill_id=&persona=&limit=
    """
    skill_id = request.args.get("skill_id")
    persona = request.args.get("persona")
    limit = request.args.get("limit", 100, type=int)
    
    audit = get_skill_audit(workspace_id, skill_id, persona, limit)
    
    return jsonify({
        "workspace_id": workspace_id,
        "filters": {"skill_id": skill_id, "persona": persona},
        "count": len(audit),
        "executions": audit
    })


# =============================================================================
# Mock Services (Downstream Simulation)
# =============================================================================

@app.route("/api/mock/slack/send", methods=["POST"])
def mock_slack_send():
    """
    Mock Slack message sending.
    Simulates: POST https://slack.com/api/chat.postMessage
    """
    data = request.get_json() or {}
    channel = data.get("channel", "#general")
    text = data.get("text", "")
    
    # Simulate response
    return jsonify({
        "ok": True,
        "channel": channel,
        "ts": f"1708123456.{hash(text) % 100000:06d}",
        "message": {
            "type": "message",
            "text": text,
            "user": "U_MOCK_BOT",
            "ts": f"1708123456.{hash(text) % 100000:06d}"
        },
        "_mock": True,
        "_service": "slack"
    })


@app.route("/api/mock/slack/channels", methods=["GET"])
def mock_slack_channels():
    """Mock Slack channel list."""
    return jsonify({
        "ok": True,
        "channels": [
            {"id": "C001", "name": "general", "is_member": True},
            {"id": "C002", "name": "engineering", "is_member": True},
            {"id": "C003", "name": "product", "is_member": True},
            {"id": "C004", "name": "random", "is_member": False},
        ],
        "_mock": True
    })


@app.route("/api/mock/email/send", methods=["POST"])
def mock_email_send():
    """
    Mock email sending.
    Simulates: Gmail/SendGrid API
    """
    data = request.get_json() or {}
    to = data.get("to", "unknown@example.com")
    subject = data.get("subject", "No Subject")
    body = data.get("body", "")
    
    return jsonify({
        "success": True,
        "message_id": f"mock_{hash(to + subject) % 1000000:06d}",
        "to": to,
        "subject": subject,
        "status": "sent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "_mock": True,
        "_service": "email"
    })


@app.route("/api/mock/calendar/events", methods=["GET", "POST"])
def mock_calendar_events():
    """Mock calendar events."""
    if request.method == "GET":
        return jsonify({
            "events": [
                {
                    "id": "evt_001",
                    "title": "Team Standup",
                    "start": "2026-02-17T09:00:00Z",
                    "end": "2026-02-17T09:30:00Z",
                    "attendees": ["ada@company.com", "grace@company.com"]
                },
                {
                    "id": "evt_002", 
                    "title": "Product Review",
                    "start": "2026-02-17T14:00:00Z",
                    "end": "2026-02-17T15:00:00Z",
                    "attendees": ["grace@company.com", "val@company.com"]
                }
            ],
            "_mock": True
        })
    
    # POST - create event
    data = request.get_json() or {}
    return jsonify({
        "success": True,
        "event_id": f"evt_{hash(str(data)) % 10000:04d}",
        "title": data.get("title", "New Event"),
        "created": True,
        "_mock": True
    })


@app.route("/api/mock/github/issues", methods=["GET", "POST"])
def mock_github_issues():
    """Mock GitHub issues."""
    if request.method == "GET":
        return jsonify({
            "issues": [
                {"number": 42, "title": "Fix login bug", "state": "open", "labels": ["bug"]},
                {"number": 41, "title": "Add dark mode", "state": "open", "labels": ["feature"]},
                {"number": 40, "title": "Update docs", "state": "closed", "labels": ["docs"]},
            ],
            "_mock": True
        })
    
    # POST - create issue
    data = request.get_json() or {}
    return jsonify({
        "success": True,
        "number": 43,
        "title": data.get("title", "New Issue"),
        "url": "https://github.com/mock/repo/issues/43",
        "_mock": True
    })


@app.route("/api/mock/notion/pages", methods=["GET", "POST"])
def mock_notion_pages():
    """Mock Notion pages."""
    if request.method == "GET":
        return jsonify({
            "pages": [
                {"id": "page_001", "title": "Project Roadmap", "last_edited": "2026-02-15"},
                {"id": "page_002", "title": "Meeting Notes", "last_edited": "2026-02-16"},
            ],
            "_mock": True
        })
    
    # POST - create page
    data = request.get_json() or {}
    return jsonify({
        "success": True,
        "page_id": f"page_{hash(str(data)) % 10000:04d}",
        "title": data.get("title", "New Page"),
        "url": "https://notion.so/mock-page",
        "_mock": True
    })


@app.route("/api/mock/linear/issues", methods=["GET", "POST"])
def mock_linear_issues():
    """Mock Linear issues."""
    if request.method == "GET":
        return jsonify({
            "issues": [
                {"id": "LIN-101", "title": "Implement auth flow", "status": "In Progress", "priority": 1},
                {"id": "LIN-102", "title": "Design review", "status": "Todo", "priority": 2},
            ],
            "_mock": True
        })
    
    data = request.get_json() or {}
    return jsonify({
        "success": True,
        "id": "LIN-103",
        "title": data.get("title", "New Issue"),
        "_mock": True
    })


@app.route("/api/mock/webhook", methods=["POST"])
def mock_webhook():
    """
    Generic webhook endpoint.
    Echoes back whatever you send + metadata.
    """
    data = request.get_json() or {}
    
    return jsonify({
        "received": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": data,
        "headers": {
            "content_type": request.content_type,
            "user_agent": request.headers.get("User-Agent", "unknown")
        },
        "_mock": True,
        "_service": "webhook"
    })


@app.route("/api/mock/execute", methods=["POST"])
def mock_execute():
    """
    Generic skill execution mock.
    Takes any action and returns success.
    """
    data = request.get_json() or {}
    action = data.get("action", "unknown")
    params = data.get("params", {})
    
    return jsonify({
        "success": True,
        "action": action,
        "params": params,
        "result": f"Mock executed: {action}",
        "execution_time_ms": 42,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "_mock": True
    })


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


# =============================================================================
# Waitlist API
# =============================================================================

@app.route("/api/waitlist", methods=["GET", "POST"])
def api_waitlist():
    """
    Waitlist signup endpoint.
    
    POST: Add email to waitlist
    GET: Get waitlist stats (admin)
    """
    db = get_db()
    if db is None:
        return jsonify({"error": "Database not configured"}), 500
    
    if request.method == "POST":
        data = request.get_json() or {}
        email = data.get("email", "").lower().strip()
        
        if not email:
            return jsonify({"error": "Email required"}), 400
        
        # Check if already exists
        existing = db.waitlist.find_one({"email": email})
        if existing:
            return jsonify({
                "success": True,
                "message": "You're already on the list!",
                "already_registered": True
            })
        
        # Insert new signup
        doc = {
            "email": email,
            "source": data.get("source", "unknown"),
            "ip": data.get("ip", "unknown"),
            "user_agent": data.get("userAgent", "unknown"),
            "timestamp": datetime.now(timezone.utc),
            "status": "pending",  # pending | invited | active
            "invited_at": None,
            "activated_at": None,
        }
        
        db.waitlist.insert_one(doc)
        
        return jsonify({
            "success": True,
            "message": "You're on the list!",
            "position": db.waitlist.count_documents({"status": "pending"})
        })
    
    # GET - admin stats
    total = db.waitlist.count_documents({})
    pending = db.waitlist.count_documents({"status": "pending"})
    invited = db.waitlist.count_documents({"status": "invited"})
    active = db.waitlist.count_documents({"status": "active"})
    
    # Recent signups
    recent = list(db.waitlist.find(
        {},
        {"email": 1, "source": 1, "timestamp": 1, "status": 1, "_id": 0}
    ).sort("timestamp", -1).limit(10))
    
    return jsonify({
        "total": total,
        "pending": pending,
        "invited": invited,
        "active": active,
        "recent": recent
    })
