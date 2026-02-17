"""
FDAA Registry - Cryptographic Skill Signing and Verification

Provides:
- SHA256 content hashing
- Merkle tree computation for directories
- Ed25519 signature generation and verification
- Skill signature registry (local file-based)
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature
except ImportError:
    ed25519 = None


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class SkillSignature:
    """Cryptographic signature for a verified skill."""
    skill_id: str
    skill_path: str
    content_hash: str          # SHA256 of SKILL.md
    scripts_merkle_root: str   # Merkle root of scripts/ directory
    references_merkle_root: str # Merkle root of references/
    
    verification_timestamp: str  # ISO 8601
    verification_version: str    # Pipeline version used
    
    tier1_passed: bool
    tier2_passed: bool
    tier2_recommendation: str
    tier3_passed: Optional[bool]  # None if not run
    
    signer_id: str             # Verification service identity
    signature: str             # Hex-encoded Ed25519 signature
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SkillSignature":
        return cls(**data)


@dataclass
class VerificationResult:
    """Result of signature verification."""
    valid: bool
    skill_id: str
    error: Optional[str] = None
    content_match: bool = False
    scripts_match: bool = False
    references_match: bool = False
    signature_valid: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Hashing
# ============================================================================

def hash_content(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_merkle_root(directory: Path) -> str:
    """Compute Merkle root hash of a directory.
    
    Files are sorted by name and hashed individually,
    then combined in a binary tree structure.
    """
    if not directory.exists():
        return hash_content("")  # Empty directory
    
    # Get all files recursively, sorted
    files = sorted(directory.rglob("*"))
    files = [f for f in files if f.is_file()]
    
    if not files:
        return hash_content("")
    
    # Hash each file with its relative path
    hashes = []
    for file in files:
        rel_path = file.relative_to(directory)
        file_hash = hash_file(file)
        combined = hash_content(f"{rel_path}:{file_hash}")
        hashes.append(combined)
    
    # Build Merkle tree
    while len(hashes) > 1:
        new_hashes = []
        for i in range(0, len(hashes), 2):
            if i + 1 < len(hashes):
                combined = hash_content(hashes[i] + hashes[i + 1])
            else:
                combined = hashes[i]  # Odd one out
            new_hashes.append(combined)
        hashes = new_hashes
    
    return hashes[0]


# ============================================================================
# Ed25519 Key Management
# ============================================================================

REGISTRY_DIR = Path.home() / ".fdaa" / "registry"
KEYS_DIR = Path.home() / ".fdaa" / "keys"


def ensure_dirs():
    """Ensure registry directories exist."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    KEYS_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)


def generate_signing_key(key_name: str = "default") -> tuple[str, str]:
    """Generate a new Ed25519 signing key pair.
    
    Returns:
        (public_key_hex, private_key_path)
    """
    if ed25519 is None:
        raise ImportError("cryptography package required. Install with: pip install cryptography")
    
    ensure_dirs()
    
    # Generate key pair
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    # Serialize private key
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Save private key
    private_path = KEYS_DIR / f"{key_name}.pem"
    private_path.write_bytes(private_bytes)
    private_path.chmod(0o600)
    
    # Get public key hex
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    public_hex = public_bytes.hex()
    
    # Save public key for reference
    public_path = KEYS_DIR / f"{key_name}.pub"
    public_path.write_text(public_hex)
    
    return public_hex, str(private_path)


def load_signing_key(key_name: str = "default") -> "ed25519.Ed25519PrivateKey":
    """Load a signing key from disk."""
    if ed25519 is None:
        raise ImportError("cryptography package required")
    
    private_path = KEYS_DIR / f"{key_name}.pem"
    if not private_path.exists():
        raise FileNotFoundError(f"Signing key not found: {private_path}")
    
    private_bytes = private_path.read_bytes()
    return serialization.load_pem_private_key(private_bytes, password=None)


def load_public_key(public_hex: str) -> "ed25519.Ed25519PublicKey":
    """Load a public key from hex string."""
    if ed25519 is None:
        raise ImportError("cryptography package required")
    
    public_bytes = bytes.fromhex(public_hex)
    return ed25519.Ed25519PublicKey.from_public_bytes(public_bytes)


def get_signer_id(key_name: str = "default") -> str:
    """Get the signer ID (public key hex) for a key."""
    public_path = KEYS_DIR / f"{key_name}.pub"
    if public_path.exists():
        return public_path.read_text().strip()
    
    # Generate if doesn't exist
    public_hex, _ = generate_signing_key(key_name)
    return public_hex


# ============================================================================
# Signing
# ============================================================================

PIPELINE_VERSION = "1.0.0"


def create_signature_payload(sig: SkillSignature) -> str:
    """Create the canonical payload for signing."""
    # Exclude the signature field itself
    payload = {
        "skill_id": sig.skill_id,
        "skill_path": sig.skill_path,
        "content_hash": sig.content_hash,
        "scripts_merkle_root": sig.scripts_merkle_root,
        "references_merkle_root": sig.references_merkle_root,
        "verification_timestamp": sig.verification_timestamp,
        "verification_version": sig.verification_version,
        "tier1_passed": sig.tier1_passed,
        "tier2_passed": sig.tier2_passed,
        "tier2_recommendation": sig.tier2_recommendation,
        "tier3_passed": sig.tier3_passed,
        "signer_id": sig.signer_id,
    }
    # Canonical JSON (sorted keys, no whitespace)
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def sign_skill(
    skill_path: Path,
    tier1_passed: bool = True,
    tier2_passed: bool = True,
    tier2_recommendation: str = "approve",
    tier3_passed: Optional[bool] = None,
    key_name: str = "default"
) -> SkillSignature:
    """Sign a verified skill and return the signature.
    
    Args:
        skill_path: Path to skill directory or SKILL.md
        tier1_passed: Whether Tier 1 (regex) passed
        tier2_passed: Whether Tier 2 (guard model) passed
        tier2_recommendation: Guard model recommendation
        tier3_passed: Whether Tier 3 (sandbox) passed (None if not run)
        key_name: Name of the signing key to use
    
    Returns:
        SkillSignature with cryptographic signature
    """
    skill_path = Path(skill_path)
    
    # Resolve to directory
    if skill_path.name == "SKILL.md":
        skill_dir = skill_path.parent
        skill_md_path = skill_path
    else:
        skill_dir = skill_path
        skill_md_path = skill_dir / "SKILL.md"
    
    if not skill_md_path.exists():
        raise FileNotFoundError(f"SKILL.md not found: {skill_md_path}")
    
    # Compute hashes
    content_hash = hash_file(skill_md_path)
    scripts_merkle = compute_merkle_root(skill_dir / "scripts")
    references_merkle = compute_merkle_root(skill_dir / "references")
    
    # Generate skill ID from content hash (first 16 chars)
    skill_id = content_hash[:16]
    
    # Get signer identity
    signer_id = get_signer_id(key_name)
    
    # Create signature object (without signature field)
    sig = SkillSignature(
        skill_id=skill_id,
        skill_path=str(skill_dir.absolute()),
        content_hash=content_hash,
        scripts_merkle_root=scripts_merkle,
        references_merkle_root=references_merkle,
        verification_timestamp=datetime.now(timezone.utc).isoformat(),
        verification_version=PIPELINE_VERSION,
        tier1_passed=tier1_passed,
        tier2_passed=tier2_passed,
        tier2_recommendation=tier2_recommendation,
        tier3_passed=tier3_passed,
        signer_id=signer_id,
        signature="",  # Placeholder
    )
    
    # Create and sign payload
    payload = create_signature_payload(sig)
    private_key = load_signing_key(key_name)
    signature_bytes = private_key.sign(payload.encode('utf-8'))
    sig.signature = signature_bytes.hex()
    
    return sig


def verify_signature(sig: SkillSignature) -> bool:
    """Verify the cryptographic signature of a SkillSignature."""
    try:
        payload = create_signature_payload(sig)
        public_key = load_public_key(sig.signer_id)
        signature_bytes = bytes.fromhex(sig.signature)
        public_key.verify(signature_bytes, payload.encode('utf-8'))
        return True
    except (InvalidSignature, ValueError, Exception):
        return False


# ============================================================================
# Registry Operations
# ============================================================================

def save_signature(sig: SkillSignature):
    """Save a signature to the local registry."""
    ensure_dirs()
    
    sig_path = REGISTRY_DIR / f"{sig.skill_id}.json"
    sig_path.write_text(json.dumps(sig.to_dict(), indent=2))


def load_signature(skill_id: str) -> Optional[SkillSignature]:
    """Load a signature from the local registry."""
    sig_path = REGISTRY_DIR / f"{skill_id}.json"
    if not sig_path.exists():
        return None
    
    data = json.loads(sig_path.read_text())
    return SkillSignature.from_dict(data)


def list_signatures() -> list[SkillSignature]:
    """List all signatures in the local registry."""
    ensure_dirs()
    
    signatures = []
    for sig_path in REGISTRY_DIR.glob("*.json"):
        try:
            data = json.loads(sig_path.read_text())
            signatures.append(SkillSignature.from_dict(data))
        except Exception:
            continue
    
    return signatures


def verify_skill_integrity(skill_path: Path, sig: SkillSignature) -> VerificationResult:
    """Verify that a skill matches its signature.
    
    Checks:
    1. SKILL.md content hash
    2. scripts/ Merkle root
    3. references/ Merkle root
    4. Cryptographic signature
    """
    skill_path = Path(skill_path)
    
    # Resolve to directory
    if skill_path.name == "SKILL.md":
        skill_dir = skill_path.parent
        skill_md_path = skill_path
    else:
        skill_dir = skill_path
        skill_md_path = skill_dir / "SKILL.md"
    
    result = VerificationResult(
        valid=False,
        skill_id=sig.skill_id,
    )
    
    # Check SKILL.md hash
    if skill_md_path.exists():
        current_hash = hash_file(skill_md_path)
        result.content_match = (current_hash == sig.content_hash)
    
    # Check scripts Merkle root
    current_scripts = compute_merkle_root(skill_dir / "scripts")
    result.scripts_match = (current_scripts == sig.scripts_merkle_root)
    
    # Check references Merkle root
    current_refs = compute_merkle_root(skill_dir / "references")
    result.references_match = (current_refs == sig.references_merkle_root)
    
    # Verify cryptographic signature
    result.signature_valid = verify_signature(sig)
    
    # Overall validity
    result.valid = all([
        result.content_match,
        result.scripts_match,
        result.references_match,
        result.signature_valid,
    ])
    
    if not result.valid:
        errors = []
        if not result.content_match:
            errors.append("SKILL.md content modified")
        if not result.scripts_match:
            errors.append("scripts/ directory modified")
        if not result.references_match:
            errors.append("references/ directory modified")
        if not result.signature_valid:
            errors.append("Invalid cryptographic signature")
        result.error = "; ".join(errors)
    
    return result


# ============================================================================
# High-Level API
# ============================================================================

def sign_and_register(
    skill_path: str,
    tier1_passed: bool = True,
    tier2_passed: bool = True,
    tier2_recommendation: str = "approve",
    tier3_passed: Optional[bool] = None,
    key_name: str = "default"
) -> SkillSignature:
    """Sign a skill and save to the registry.
    
    Returns the signature.
    """
    sig = sign_skill(
        Path(skill_path),
        tier1_passed=tier1_passed,
        tier2_passed=tier2_passed,
        tier2_recommendation=tier2_recommendation,
        tier3_passed=tier3_passed,
        key_name=key_name,
    )
    save_signature(sig)
    return sig


def check_skill(skill_path: str) -> VerificationResult:
    """Check if a skill is registered and unmodified.
    
    Returns verification result.
    """
    skill_path = Path(skill_path)
    
    # Resolve to directory
    if skill_path.name == "SKILL.md":
        skill_dir = skill_path.parent
        skill_md_path = skill_path
    else:
        skill_dir = skill_path
        skill_md_path = skill_dir / "SKILL.md"
    
    if not skill_md_path.exists():
        return VerificationResult(
            valid=False,
            skill_id="unknown",
            error="SKILL.md not found"
        )
    
    # Get skill ID from content hash
    content_hash = hash_file(skill_md_path)
    skill_id = content_hash[:16]
    
    # Look up in registry
    sig = load_signature(skill_id)
    if sig is None:
        return VerificationResult(
            valid=False,
            skill_id=skill_id,
            error="Skill not found in registry (not verified)"
        )
    
    # Verify integrity
    return verify_skill_integrity(skill_path, sig)


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m fdaa.registry sign <skill-path>")
        print("  python -m fdaa.registry check <skill-path>")
        print("  python -m fdaa.registry list")
        print("  python -m fdaa.registry keygen [name]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "sign":
        path = sys.argv[2]
        sig = sign_and_register(path)
        print(json.dumps(sig.to_dict(), indent=2))
    
    elif command == "check":
        path = sys.argv[2]
        result = check_skill(path)
        print(json.dumps(result.to_dict(), indent=2))
        sys.exit(0 if result.valid else 1)
    
    elif command == "list":
        sigs = list_signatures()
        for sig in sigs:
            print(f"{sig.skill_id}: {sig.skill_path} ({sig.verification_timestamp})")
    
    elif command == "keygen":
        name = sys.argv[2] if len(sys.argv) > 2 else "default"
        pub, priv = generate_signing_key(name)
        print(f"Generated key pair: {name}")
        print(f"  Public key: {pub}")
        print(f"  Private key: {priv}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
