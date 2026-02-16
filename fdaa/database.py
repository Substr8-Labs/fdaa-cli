"""MongoDB database connection and operations."""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, List
from datetime import datetime, timezone

# MongoDB client (initialized on startup)
client: Optional[AsyncIOMotorClient] = None
db = None

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
