"""MongoDB database connection and operations."""

import os
import hashlib
import json
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, List
from datetime import datetime, timezone

# MongoDB client (initialized on startup)
client: Optional[AsyncIOMotorClient] = None
db = None

# Genesis hash for snapshot chains
GENESIS_HASH = "sha256:" + "0" * 64

async def connect_db():
    """Connect to MongoDB Atlas."""
    global client, db
    
    mongodb_uri = os.environ.get("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI environment variable required")
    
    client = AsyncIOMotorClient(mongodb_uri)
    db = client.fdaa
    
    # Test connection
    await client.admin.command("ping")
    print("✓ Connected to MongoDB Atlas")


async def close_db():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        print("✓ Closed MongoDB connection")


# Workspace operations

async def create_workspace(workspace_id: str, name: str, files: Dict[str, str]) -> Dict:
    """Create a new workspace with files."""
    workspace = {
        "_id": workspace_id,
        "name": name,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "files": {
            path: {"content": content}
            for path, content in files.items()
        }
    }
    
    await db.workspaces.insert_one(workspace)
    return workspace


async def get_workspace(workspace_id: str) -> Optional[Dict]:
    """Get a workspace by ID or name."""
    # Try by _id first
    workspace = await db.workspaces.find_one({"_id": workspace_id})
    if not workspace:
        # Fallback to name
        workspace = await db.workspaces.find_one({"name": workspace_id})
    return workspace


async def list_workspaces() -> List[Dict]:
    """List all workspaces."""
    cursor = db.workspaces.find({}, {"_id": 1, "name": 1, "created_at": 1, "updated_at": 1})
    return await cursor.to_list(length=100)


async def get_file(workspace_id: str, path: str) -> Optional[str]:
    """Get a specific file from a workspace."""
    workspace = await get_workspace(workspace_id)
    if workspace:
        files = workspace.get("files", {})
        if isinstance(files, dict) and path in files:
            file_data = files[path]
            if isinstance(file_data, dict):
                return file_data.get("content", "")
            return file_data
    return None


async def get_files(workspace_id: str) -> Dict[str, str]:
    """Get all files from a workspace as {path: content}."""
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
    """Update a file in a workspace."""
    # Read-modify-write (paths contain slashes, can't use dot notation)
    workspace = await get_workspace(workspace_id)
    if not workspace:
        return False
    
    files = workspace.get("files", {})
    if not isinstance(files, dict):
        files = {}
    
    files[path] = {"content": content}
    
    result = await db.workspaces.update_one(
        {"_id": workspace["_id"]},
        {
            "$set": {
                "files": files,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )
    return result.modified_count > 0


async def delete_workspace(workspace_id: str) -> bool:
    """Delete a workspace."""
    result = await db.workspaces.delete_one({"_id": workspace_id})
    return result.deleted_count > 0


# =============================================================================
# Snapshotting System
# =============================================================================

def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    hash_bytes = hashlib.sha256(content.encode('utf-8')).hexdigest()
    return f"sha256:{hash_bytes}"


async def create_snapshot(
    workspace_id: str, 
    path: str, 
    content: str,
    actor: Optional[str] = None
) -> Dict:
    """
    Create a versioned snapshot for a file update.
    
    Implements hash-chain linking for cryptographic lineage.
    Every snapshot links to its parent via parent_hash.
    """
    # Get the previous snapshot for this file (if any)
    previous = await db.snapshots.find_one(
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
    
    await db.snapshots.insert_one(snapshot)
    
    # Return without MongoDB _id for cleaner response
    snapshot.pop("_id", None)
    return snapshot


async def get_file_history(
    workspace_id: str, 
    path: str,
    limit: int = 50
) -> List[Dict]:
    """
    Get version history for a file.
    
    Returns list of snapshots ordered by version (newest first).
    """
    cursor = db.snapshots.find(
        {"workspace_id": workspace_id, "path": path},
        {"content": 0}  # Exclude content for performance
    ).sort("version", -1).limit(limit)
    
    snapshots = await cursor.to_list(length=limit)
    
    # Clean up MongoDB _id
    for s in snapshots:
        s["_id"] = str(s["_id"])
    
    return snapshots


async def get_snapshot(
    workspace_id: str,
    path: str,
    version: int
) -> Optional[Dict]:
    """Get a specific snapshot by version."""
    snapshot = await db.snapshots.find_one({
        "workspace_id": workspace_id,
        "path": path,
        "version": version
    })
    
    if snapshot:
        snapshot["_id"] = str(snapshot["_id"])
    
    return snapshot


async def get_snapshot_by_hash(
    workspace_id: str,
    content_hash: str
) -> Optional[Dict]:
    """Get a snapshot by its content hash."""
    snapshot = await db.snapshots.find_one({
        "workspace_id": workspace_id,
        "content_hash": content_hash
    })
    
    if snapshot:
        snapshot["_id"] = str(snapshot["_id"])
    
    return snapshot


async def rollback_to_version(
    workspace_id: str,
    path: str,
    target_version: int,
    actor: Optional[str] = None
) -> Optional[Dict]:
    """
    Rollback a file to a previous version.
    
    Creates a NEW snapshot with the old content (preserves history).
    Does not delete any snapshots - history is immutable.
    """
    # Get the target snapshot
    target = await get_snapshot(workspace_id, path, target_version)
    if not target:
        return None
    
    # Create a new snapshot with the old content
    new_snapshot = await create_snapshot(
        workspace_id, 
        path, 
        target["content"],
        actor=actor or f"rollback:v{target_version}"
    )
    
    # Also update the live workspace file
    await update_file(workspace_id, path, target["content"])
    
    return new_snapshot


async def verify_snapshot_chain(
    workspace_id: str,
    path: str
) -> Dict:
    """
    Verify the integrity of a file's snapshot chain.
    
    Checks:
    1. Hash chain is unbroken (each parent_hash matches previous content_hash)
    2. Content hashes are correct (recomputed from content)
    """
    cursor = db.snapshots.find(
        {"workspace_id": workspace_id, "path": path}
    ).sort("version", 1)
    
    snapshots = await cursor.to_list(length=10000)
    
    if not snapshots:
        return {"valid": True, "message": "No snapshots found", "chain_length": 0}
    
    errors = []
    previous_hash = GENESIS_HASH
    
    for i, snapshot in enumerate(snapshots):
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


async def update_file_with_snapshot(
    workspace_id: str, 
    path: str, 
    content: str,
    actor: Optional[str] = None
) -> Dict:
    """
    Update a file AND create a snapshot.
    
    This is the preferred method for file updates - ensures history is preserved.
    """
    # Create snapshot first
    snapshot = await create_snapshot(workspace_id, path, content, actor)
    
    # Then update the live file
    await update_file(workspace_id, path, content)
    
    return snapshot
