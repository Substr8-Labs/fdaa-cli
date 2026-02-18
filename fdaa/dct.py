"""
DCT - Delegation Capability Tokens

Cryptographic tokens for delegating permissions between agents.
Implements monotonic attenuation: sub-agents can only receive
subsets of parent permissions.

Token Structure:
{
    "version": "1.0",
    "token_id": "uuid",
    "delegator": "agent_id or public_key",
    "delegate": "agent_id or public_key", 
    "permissions": ["file:read:/path/*", "api:call:weather"],
    "constraints": {
        "expires_at": "ISO8601",
        "max_delegations": 0,  # Can this be re-delegated?
        "allowed_hosts": ["localhost"],
    },
    "parent_token": "optional - for delegation chains",
    "issued_at": "ISO8601",
    "signature": "Ed25519 signature of canonical payload"
}
"""

import json
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

# Crypto
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


class PermissionType(Enum):
    """Standard permission types."""
    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    API_CALL = "api:call"
    EXEC = "exec"
    NETWORK = "network"
    SPAWN = "spawn"  # Can spawn sub-agents


@dataclass
class Permission:
    """A single permission grant."""
    type: str           # e.g., "file:read"
    resource: str       # e.g., "/home/user/docs/*"
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.type}:{self.resource}"
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "resource": self.resource,
            "conditions": self.conditions
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Permission":
        return cls(
            type=data["type"],
            resource=data["resource"],
            conditions=data.get("conditions", {})
        )
    
    @classmethod
    def from_string(cls, s: str) -> "Permission":
        """Parse permission from string like 'file:read:/path/*'"""
        parts = s.split(":", 2)
        if len(parts) == 3:
            return cls(type=f"{parts[0]}:{parts[1]}", resource=parts[2])
        elif len(parts) == 2:
            return cls(type=parts[0], resource=parts[1])
        else:
            return cls(type=s, resource="*")
    
    def is_subset_of(self, other: "Permission") -> bool:
        """Check if this permission is a subset of another."""
        # Same type required
        if self.type != other.type:
            return False
        
        # Check resource subset (simple glob matching)
        if other.resource == "*":
            return True
        if self.resource == other.resource:
            return True
        if other.resource.endswith("/*"):
            prefix = other.resource[:-1]
            return self.resource.startswith(prefix)
        
        return False


@dataclass
class Constraints:
    """Token constraints."""
    expires_at: Optional[str] = None
    max_delegations: int = 0  # 0 = cannot re-delegate
    allowed_hosts: List[str] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "expires_at": self.expires_at,
            "max_delegations": self.max_delegations,
            "allowed_hosts": self.allowed_hosts,
            "allowed_actions": self.allowed_actions,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Constraints":
        return cls(
            expires_at=data.get("expires_at"),
            max_delegations=data.get("max_delegations", 0),
            allowed_hosts=data.get("allowed_hosts", []),
            allowed_actions=data.get("allowed_actions", []),
        )
    
    def is_expired(self) -> bool:
        """Check if token has expired."""
        if not self.expires_at:
            return False
        expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > expiry


@dataclass 
class DCT:
    """Delegation Capability Token."""
    version: str
    token_id: str
    delegator: str          # Public key hex of delegator
    delegate: str           # Public key hex of delegate (or "*" for bearer)
    permissions: List[Permission]
    constraints: Constraints
    parent_token: Optional[str]  # Token ID of parent (for chains)
    issued_at: str
    signature: str = ""
    
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "token_id": self.token_id,
            "delegator": self.delegator,
            "delegate": self.delegate,
            "permissions": [p.to_dict() for p in self.permissions],
            "constraints": self.constraints.to_dict(),
            "parent_token": self.parent_token,
            "issued_at": self.issued_at,
            "signature": self.signature,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DCT":
        return cls(
            version=data["version"],
            token_id=data["token_id"],
            delegator=data["delegator"],
            delegate=data["delegate"],
            permissions=[Permission.from_dict(p) for p in data["permissions"]],
            constraints=Constraints.from_dict(data.get("constraints", {})),
            parent_token=data.get("parent_token"),
            issued_at=data["issued_at"],
            signature=data.get("signature", ""),
        )
    
    def canonical_payload(self) -> str:
        """Create canonical JSON for signing (excludes signature)."""
        payload = {
            "version": self.version,
            "token_id": self.token_id,
            "delegator": self.delegator,
            "delegate": self.delegate,
            "permissions": sorted([str(p) for p in self.permissions]),
            "constraints": self.constraints.to_dict(),
            "parent_token": self.parent_token,
            "issued_at": self.issued_at,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    
    def payload_hash(self) -> str:
        """SHA256 hash of canonical payload."""
        return hashlib.sha256(self.canonical_payload().encode()).hexdigest()


# ============================================================================
# Token Operations
# ============================================================================

def create_token(
    delegator_key_name: str,
    delegate_pubkey: str,
    permissions: List[str],
    expires_in_minutes: int = 60,
    max_delegations: int = 0,
    parent_token_id: Optional[str] = None,
) -> DCT:
    """Create a new DCT.
    
    Args:
        delegator_key_name: Name of the signing key (in ~/.fdaa/keys/)
        delegate_pubkey: Public key of the delegate (hex) or "*" for bearer
        permissions: List of permission strings (e.g., "file:read:/path/*")
        expires_in_minutes: Token lifetime
        max_delegations: How many times this can be re-delegated (0 = none)
        parent_token_id: If this is a delegation of another token
    
    Returns:
        Signed DCT
    """
    if not HAS_CRYPTO:
        raise ImportError("cryptography package required")
    
    # Load signing key
    keys_dir = Path.home() / ".fdaa" / "keys"
    key_path = keys_dir / f"{delegator_key_name}.pem"
    pub_path = keys_dir / f"{delegator_key_name}.pub"
    
    if not key_path.exists():
        raise FileNotFoundError(f"Key not found: {key_path}")
    
    with open(key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    
    delegator_pubkey = pub_path.read_text().strip() if pub_path.exists() else ""
    
    # Build token
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(minutes=expires_in_minutes)
    
    token = DCT(
        version="1.0",
        token_id=str(uuid.uuid4()),
        delegator=delegator_pubkey,
        delegate=delegate_pubkey,
        permissions=[Permission.from_string(p) for p in permissions],
        constraints=Constraints(
            expires_at=expiry.isoformat(),
            max_delegations=max_delegations,
        ),
        parent_token=parent_token_id,
        issued_at=now.isoformat(),
        signature="",
    )
    
    # Sign
    payload_hash = bytes.fromhex(token.payload_hash())
    signature = private_key.sign(payload_hash)
    token.signature = signature.hex()
    
    return token


def verify_token(token: DCT) -> tuple[bool, str]:
    """Verify a DCT's signature and constraints.
    
    Returns:
        (is_valid, reason)
    """
    if not HAS_CRYPTO:
        return False, "cryptography package required"
    
    # Check expiry
    if token.constraints.is_expired():
        return False, "Token has expired"
    
    # Verify signature
    try:
        pubkey_bytes = bytes.fromhex(token.delegator)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        
        payload_hash = bytes.fromhex(token.payload_hash())
        signature_bytes = bytes.fromhex(token.signature)
        
        public_key.verify(signature_bytes, payload_hash)
        return True, "Valid"
    except InvalidSignature:
        return False, "Invalid signature"
    except Exception as e:
        return False, f"Verification error: {e}"


def attenuate_token(
    parent_token: DCT,
    delegator_key_name: str,
    delegate_pubkey: str,
    permissions: List[str],
    expires_in_minutes: int = 60,
) -> DCT:
    """Create a new token that's a subset of an existing token.
    
    Implements monotonic attenuation: the new token can only have
    fewer permissions than the parent, not more.
    
    Args:
        parent_token: The token being attenuated
        delegator_key_name: Key of current holder (must match delegate in parent)
        delegate_pubkey: Who receives the attenuated token
        permissions: Requested permissions (must be subset of parent)
        expires_in_minutes: New expiry (cannot exceed parent)
    
    Returns:
        New attenuated DCT
    
    Raises:
        ValueError: If requested permissions exceed parent
    """
    # Verify parent token first
    valid, reason = verify_token(parent_token)
    if not valid:
        raise ValueError(f"Parent token invalid: {reason}")
    
    # Check delegation limit
    if parent_token.constraints.max_delegations <= 0:
        raise ValueError("Parent token does not allow re-delegation")
    
    # Check permission subset
    requested = [Permission.from_string(p) for p in permissions]
    for req in requested:
        allowed = any(req.is_subset_of(parent_perm) for parent_perm in parent_token.permissions)
        if not allowed:
            raise ValueError(f"Permission '{req}' not allowed by parent token")
    
    # Check expiry doesn't exceed parent
    parent_expiry = datetime.fromisoformat(
        parent_token.constraints.expires_at.replace("Z", "+00:00")
    )
    new_expiry = datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)
    if new_expiry > parent_expiry:
        expires_in_minutes = int((parent_expiry - datetime.now(timezone.utc)).total_seconds() / 60)
    
    # Create attenuated token
    return create_token(
        delegator_key_name=delegator_key_name,
        delegate_pubkey=delegate_pubkey,
        permissions=permissions,
        expires_in_minutes=expires_in_minutes,
        max_delegations=parent_token.constraints.max_delegations - 1,
        parent_token_id=parent_token.token_id,
    )


def check_permission(token: DCT, permission: str) -> bool:
    """Check if a token grants a specific permission.
    
    Args:
        token: The DCT to check
        permission: Permission string (e.g., "file:read:/home/user/doc.txt")
    
    Returns:
        True if permission is granted
    """
    # Verify token first
    valid, _ = verify_token(token)
    if not valid:
        return False
    
    requested = Permission.from_string(permission)
    return any(requested.is_subset_of(p) for p in token.permissions)


# ============================================================================
# Storage
# ============================================================================

def save_token(token: DCT, path: Optional[Path] = None) -> Path:
    """Save token to file."""
    if path is None:
        tokens_dir = Path.home() / ".fdaa" / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        path = tokens_dir / f"{token.token_id}.json"
    
    path.write_text(json.dumps(token.to_dict(), indent=2))
    return path


def load_token(token_id_or_path: str) -> DCT:
    """Load token from file or by ID."""
    path = Path(token_id_or_path)
    
    if not path.exists():
        # Try by ID
        tokens_dir = Path.home() / ".fdaa" / "tokens"
        path = tokens_dir / f"{token_id_or_path}.json"
    
    if not path.exists():
        raise FileNotFoundError(f"Token not found: {token_id_or_path}")
    
    data = json.loads(path.read_text())
    return DCT.from_dict(data)


def list_tokens() -> List[DCT]:
    """List all saved tokens."""
    tokens_dir = Path.home() / ".fdaa" / "tokens"
    if not tokens_dir.exists():
        return []
    
    tokens = []
    for path in tokens_dir.glob("*.json"):
        try:
            tokens.append(load_token(str(path)))
        except Exception:
            continue
    
    return tokens
