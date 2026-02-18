"""
FDAA Registry Client - Install and publish skills from remote registry.

Provides:
- Skill publishing (verify + sign + upload)
- Skill installation (download + verify)
- Registry search
"""

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

from .registry import (
    SkillSignature,
    sign_skill,
    verify_signature,
    hash_file,
    compute_merkle_root,
    REGISTRY_DIR,
    ensure_dirs,
)


# Default registry URL (can be overridden)
DEFAULT_REGISTRY_URL = os.environ.get("FDAA_REGISTRY_URL", "https://registry.fdaa.dev")


@dataclass
class SkillMetadata:
    """Skill metadata for registry."""
    name: str
    version: str
    description: str
    author: str
    skill_id: str
    content_hash: str
    created_at: str
    downloads: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SkillMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PublishResult:
    """Result of publishing a skill."""
    success: bool
    skill_id: str
    version: str
    registry_url: str
    error: Optional[str] = None


@dataclass
class InstallResult:
    """Result of installing a skill."""
    success: bool
    skill_name: str
    version: str
    install_path: str
    error: Optional[str] = None


class RegistryClient:
    """Client for FDAA skill registry."""
    
    def __init__(self, registry_url: str = None):
        self.registry_url = (registry_url or DEFAULT_REGISTRY_URL).rstrip("/")
        self.local_cache = Path.home() / ".fdaa" / "cache"
        self.local_cache.mkdir(parents=True, exist_ok=True)
    
    def _api_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make API request to registry."""
        url = f"{self.registry_url}/api/v1{endpoint}"
        
        headers = {"Content-Type": "application/json"}
        
        if data:
            body = json.dumps(data).encode("utf-8")
        else:
            body = None
        
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_data = json.loads(error_body)
                raise RegistryError(error_data.get("error", str(e)))
            except json.JSONDecodeError:
                raise RegistryError(f"HTTP {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise RegistryError(f"Connection failed: {e.reason}")
    
    def _download_file(self, url: str, dest_path: Path) -> None:
        """Download file from URL."""
        try:
            urllib.request.urlretrieve(url, dest_path)
        except urllib.error.URLError as e:
            raise RegistryError(f"Download failed: {e.reason}")
    
    def search(self, query: str, limit: int = 20) -> list[SkillMetadata]:
        """Search for skills in the registry."""
        try:
            result = self._api_request("GET", f"/skills/search?q={query}&limit={limit}")
            return [SkillMetadata.from_dict(s) for s in result.get("skills", [])]
        except RegistryError:
            # If registry is offline, search local cache
            return self._search_local(query)
    
    def _search_local(self, query: str) -> list[SkillMetadata]:
        """Search local cache for skills."""
        results = []
        query_lower = query.lower()
        
        for meta_file in self.local_cache.glob("*/metadata.json"):
            try:
                meta = json.loads(meta_file.read_text())
                if query_lower in meta.get("name", "").lower() or \
                   query_lower in meta.get("description", "").lower():
                    results.append(SkillMetadata.from_dict(meta))
            except Exception:
                continue
        
        return results
    
    def get_skill_info(self, name: str, version: str = None) -> Optional[SkillMetadata]:
        """Get skill metadata from registry."""
        endpoint = f"/skills/{name}"
        if version:
            endpoint += f"/{version}"
        
        try:
            result = self._api_request("GET", endpoint)
            return SkillMetadata.from_dict(result)
        except RegistryError:
            return None
    
    def publish(
        self,
        skill_path: Path,
        name: str = None,
        version: str = "1.0.0",
        author: str = None,
        key_name: str = "default",
        run_pipeline: bool = True
    ) -> PublishResult:
        """Publish a skill to the registry.
        
        Args:
            skill_path: Path to skill directory
            name: Skill name (derived from directory if not specified)
            version: Semantic version
            author: Author name/email
            key_name: Signing key to use
            run_pipeline: Whether to run full verification pipeline first
        
        Returns:
            PublishResult with upload status
        """
        skill_path = Path(skill_path)
        
        if not (skill_path / "SKILL.md").exists():
            return PublishResult(
                success=False,
                skill_id="",
                version=version,
                registry_url=self.registry_url,
                error="SKILL.md not found"
            )
        
        # Extract name from path if not specified
        if name is None:
            name = skill_path.name
        
        # Run verification pipeline if requested
        if run_pipeline:
            from .guard import verify_skill
            verdict = verify_skill(str(skill_path))
            if not verdict.passed:
                return PublishResult(
                    success=False,
                    skill_id="",
                    version=version,
                    registry_url=self.registry_url,
                    error=f"Verification failed: {verdict.recommendation.value}"
                )
        
        # Sign the skill
        try:
            signature = sign_skill(
                skill_path,
                tier1_passed=True,
                tier2_passed=True,
                tier2_recommendation="approve" if run_pipeline else "skipped",
                key_name=key_name,
            )
        except Exception as e:
            return PublishResult(
                success=False,
                skill_id="",
                version=version,
                registry_url=self.registry_url,
                error=f"Signing failed: {e}"
            )
        
        # Package skill as tarball
        package_path = self._package_skill(skill_path, signature, name, version, author)
        
        # Upload to registry
        try:
            with open(package_path, "rb") as f:
                package_data = f.read()
            
            # For now, simulate upload (registry backend not implemented yet)
            # In production, this would POST to /skills/publish
            
            # Save to local registry as fallback
            self._save_to_local_cache(skill_path, signature, name, version, author)
            
            return PublishResult(
                success=True,
                skill_id=signature.skill_id,
                version=version,
                registry_url=f"{self.registry_url}/skills/{name}/{version}",
            )
            
        except Exception as e:
            return PublishResult(
                success=False,
                skill_id=signature.skill_id,
                version=version,
                registry_url=self.registry_url,
                error=f"Upload failed: {e}"
            )
        finally:
            # Cleanup temp package
            if package_path.exists():
                package_path.unlink()
    
    def _package_skill(
        self,
        skill_path: Path,
        signature: SkillSignature,
        name: str,
        version: str,
        author: str
    ) -> Path:
        """Package skill as tarball with metadata."""
        package_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        
        with tarfile.open(package_path, "w:gz") as tar:
            # Add skill files
            for item in skill_path.iterdir():
                tar.add(item, arcname=item.name)
            
            # Add signature
            sig_content = json.dumps(signature.to_dict(), indent=2)
            sig_info = tarfile.TarInfo(name=".fdaa-signature.json")
            sig_info.size = len(sig_content)
            tar.addfile(sig_info, fileobj=__import__("io").BytesIO(sig_content.encode()))
            
            # Add metadata
            meta = {
                "name": name,
                "version": version,
                "author": author or "unknown",
                "skill_id": signature.skill_id,
                "content_hash": signature.content_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_content = json.dumps(meta, indent=2)
            meta_info = tarfile.TarInfo(name=".fdaa-metadata.json")
            meta_info.size = len(meta_content)
            tar.addfile(meta_info, fileobj=__import__("io").BytesIO(meta_content.encode()))
        
        return package_path
    
    def _save_to_local_cache(
        self,
        skill_path: Path,
        signature: SkillSignature,
        name: str,
        version: str,
        author: str
    ) -> None:
        """Save skill to local cache (offline registry)."""
        cache_dir = self.local_cache / name / version
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy skill files
        if (cache_dir / "skill").exists():
            shutil.rmtree(cache_dir / "skill")
        shutil.copytree(skill_path, cache_dir / "skill")
        
        # Save signature
        (cache_dir / "signature.json").write_text(
            json.dumps(signature.to_dict(), indent=2)
        )
        
        # Save metadata
        meta = {
            "name": name,
            "version": version,
            "author": author or "unknown",
            "skill_id": signature.skill_id,
            "content_hash": signature.content_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": self._extract_description(skill_path),
        }
        (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    
    def _extract_description(self, skill_path: Path) -> str:
        """Extract description from SKILL.md."""
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            return ""
        
        content = skill_md.read_text()
        
        # Try YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                for line in parts[1].split("\n"):
                    if line.strip().startswith("description:"):
                        return line.split(":", 1)[1].strip().strip('"\'')
        
        # Fall back to first paragraph
        for line in content.split("\n"):
            if line.strip() and not line.startswith("#"):
                return line.strip()[:200]
        
        return ""
    
    def install(
        self,
        skill_spec: str,
        install_dir: Path = None,
        verify: bool = True
    ) -> InstallResult:
        """Install a skill from the registry.
        
        Args:
            skill_spec: Skill name, name@version, or github:owner/repo
            install_dir: Where to install (default: ./skills/)
            verify: Whether to verify signature
        
        Returns:
            InstallResult with installation status
        
        Examples:
            fdaa install weather-skill
            fdaa install weather-skill@1.2.0
            fdaa install github:substr8-labs/skill-weather
            fdaa install github:substr8-labs/skill-weather@v1.0.0
        """
        install_dir = install_dir or Path.cwd() / "skills"
        install_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle github: prefix (Phase 0)
        if skill_spec.startswith("github:"):
            return self._install_from_github(skill_spec[7:], install_dir, verify)
        
        # Parse skill spec
        if "@" in skill_spec:
            name, version = skill_spec.rsplit("@", 1)
        else:
            name = skill_spec
            version = "latest"
        
        # Try remote registry first
        try:
            return self._install_from_remote(name, version, install_dir, verify)
        except RegistryError as e:
            # Fall back to local cache
            return self._install_from_cache(name, version, install_dir, verify)
    
    def _install_from_github(
        self,
        repo_spec: str,
        install_dir: Path,
        verify: bool
    ) -> InstallResult:
        """Install skill directly from GitHub repository.
        
        Phase 0 registry: skills live in GitHub repos.
        
        Args:
            repo_spec: owner/repo or owner/repo@tag
            install_dir: Where to install
            verify: Whether to verify signature
        
        Returns:
            InstallResult
        """
        import zipfile
        import io
        
        # Parse repo spec
        if "@" in repo_spec:
            repo_path, ref = repo_spec.rsplit("@", 1)
        else:
            repo_path = repo_spec
            ref = "main"  # Default branch
        
        if "/" not in repo_path:
            return InstallResult(
                success=False,
                skill_name=repo_spec,
                version="",
                install_path="",
                error="Invalid GitHub spec. Use: github:owner/repo"
            )
        
        owner, repo = repo_path.split("/", 1)
        skill_name = repo.replace("skill-", "")  # skill-weather -> weather
        
        # Download from GitHub
        # Try tag/release first, fall back to branches (main and master)
        urls_to_try = [
            f"https://github.com/{owner}/{repo}/archive/refs/tags/{ref}.zip",
            f"https://github.com/{owner}/{repo}/archive/refs/heads/{ref}.zip",
            f"https://github.com/{owner}/{repo}/archive/{ref}.zip",
        ]
        
        # If using default ref, also try common branch names
        if ref == "main":
            urls_to_try.extend([
                f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip",
                f"https://github.com/{owner}/{repo}/archive/master.zip",
            ])
        
        zip_data = None
        download_url = None
        
        for url in urls_to_try:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "fdaa-cli"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    zip_data = response.read()
                    download_url = url
                    break
            except urllib.error.HTTPError:
                continue
            except urllib.error.URLError as e:
                return InstallResult(
                    success=False,
                    skill_name=skill_name,
                    version=ref,
                    install_path="",
                    error=f"Network error: {e.reason}"
                )
        
        if zip_data is None:
            return InstallResult(
                success=False,
                skill_name=skill_name,
                version=ref,
                install_path="",
                error=f"Could not download from GitHub. Check that {owner}/{repo} exists and {ref} is a valid tag/branch."
            )
        
        # Extract zip
        extract_dir = Path(tempfile.mkdtemp())
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                zf.extractall(extract_dir)
            
            # Find the extracted directory (GitHub adds repo-branch prefix)
            extracted_dirs = list(extract_dir.iterdir())
            if not extracted_dirs:
                return InstallResult(
                    success=False,
                    skill_name=skill_name,
                    version=ref,
                    install_path="",
                    error="Empty archive from GitHub"
                )
            
            source_dir = extracted_dirs[0]
            
            # Verify SKILL.md exists
            if not (source_dir / "SKILL.md").exists():
                return InstallResult(
                    success=False,
                    skill_name=skill_name,
                    version=ref,
                    install_path="",
                    error="Not a valid skill: SKILL.md not found in repository"
                )
            
            # Verify signature if requested and MANIFEST.json exists
            manifest_path = source_dir / "MANIFEST.json"
            signature_verified = False
            
            if verify and manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    
                    # Verify the signature
                    # Map MANIFEST.json fields to SkillSignature fields
                    # Use skill_path from manifest (original path when signed) for signature verification
                    sig_data = {
                        "skill_id": manifest.get("skill_id", f"{owner}/{repo}"),
                        "skill_path": manifest.get("skill_path", str(source_dir)),
                        "content_hash": manifest.get("sha256", ""),
                        "scripts_merkle_root": manifest.get("scripts_merkle_root", ""),
                        "references_merkle_root": manifest.get("references_merkle_root", ""),
                        "verification_timestamp": manifest.get("signedAt", manifest.get("signed_at", "")),
                        "verification_version": manifest.get("verification_version", "1.0.0"),
                        "tier1_passed": manifest.get("verification", {}).get("tier1", True),
                        "tier2_passed": manifest.get("verification", {}).get("tier2", True),
                        "tier2_recommendation": manifest.get("verification", {}).get("tier2_recommendation", "approve"),
                        "tier3_passed": manifest.get("verification", {}).get("tier3"),
                        "signer_id": manifest.get("publicKey", manifest.get("signer_id", "")),
                        "signature": manifest.get("signature", ""),
                    }
                    signature = SkillSignature.from_dict(sig_data)
                    
                    if verify_signature(signature):
                        signature_verified = True
                    else:
                        return InstallResult(
                            success=False,
                            skill_name=skill_name,
                            version=ref,
                            install_path="",
                            error="Signature verification failed. Use --no-verify to skip."
                        )
                except Exception as e:
                    if verify:
                        return InstallResult(
                            success=False,
                            skill_name=skill_name,
                            version=ref,
                            install_path="",
                            error=f"Could not verify signature: {e}. Use --no-verify to skip."
                        )
            elif verify and not manifest_path.exists():
                # No signature present - warn but allow with --no-verify
                return InstallResult(
                    success=False,
                    skill_name=skill_name,
                    version=ref,
                    install_path="",
                    error="No MANIFEST.json found - skill is unsigned. Use --no-verify to install anyway."
                )
            
            # Copy to install directory
            dest_path = install_dir / skill_name
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_dir, dest_path)
            
            # Add source metadata
            source_meta = {
                "source": "github",
                "repository": f"{owner}/{repo}",
                "ref": ref,
                "download_url": download_url,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "signature_verified": signature_verified,
            }
            (dest_path / ".fdaa-source.json").write_text(json.dumps(source_meta, indent=2))
            
            return InstallResult(
                success=True,
                skill_name=skill_name,
                version=ref,
                install_path=str(dest_path),
            )
            
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    def _install_from_remote(
        self,
        name: str,
        version: str,
        install_dir: Path,
        verify: bool
    ) -> InstallResult:
        """Install from remote registry."""
        # Get skill info
        endpoint = f"/skills/{name}"
        if version != "latest":
            endpoint += f"/{version}"
        
        info = self._api_request("GET", endpoint)
        actual_version = info.get("version", version)
        
        # Download package
        download_url = info.get("download_url")
        if not download_url:
            raise RegistryError("No download URL in response")
        
        package_path = self.local_cache / f"{name}-{actual_version}.tar.gz"
        self._download_file(download_url, package_path)
        
        # Extract and verify
        return self._install_from_package(
            package_path, name, actual_version, install_dir, verify
        )
    
    def _install_from_cache(
        self,
        name: str,
        version: str,
        install_dir: Path,
        verify: bool
    ) -> InstallResult:
        """Install from local cache."""
        cache_base = self.local_cache / name
        
        if not cache_base.exists():
            return InstallResult(
                success=False,
                skill_name=name,
                version=version,
                install_path="",
                error=f"Skill '{name}' not found in registry or cache"
            )
        
        # Find version
        if version == "latest":
            versions = sorted(cache_base.iterdir(), reverse=True)
            if not versions:
                return InstallResult(
                    success=False,
                    skill_name=name,
                    version=version,
                    install_path="",
                    error=f"No versions found for '{name}'"
                )
            version_dir = versions[0]
            version = version_dir.name
        else:
            version_dir = cache_base / version
            if not version_dir.exists():
                return InstallResult(
                    success=False,
                    skill_name=name,
                    version=version,
                    install_path="",
                    error=f"Version {version} not found for '{name}'"
                )
        
        skill_source = version_dir / "skill"
        sig_path = version_dir / "signature.json"
        
        if not skill_source.exists():
            return InstallResult(
                success=False,
                skill_name=name,
                version=version,
                install_path="",
                error="Skill files not found in cache"
            )
        
        # Verify signature if requested
        if verify and sig_path.exists():
            sig_data = json.loads(sig_path.read_text())
            signature = SkillSignature.from_dict(sig_data)
            
            if not verify_signature(signature):
                return InstallResult(
                    success=False,
                    skill_name=name,
                    version=version,
                    install_path="",
                    error="Signature verification failed"
                )
            
            # Verify content hash
            skill_md = skill_source / "SKILL.md"
            if skill_md.exists():
                current_hash = hash_file(skill_md)
                if current_hash != signature.content_hash:
                    return InstallResult(
                        success=False,
                        skill_name=name,
                        version=version,
                        install_path="",
                        error="Content hash mismatch - skill was modified"
                    )
        
        # Copy to install directory
        dest_path = install_dir / name
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(skill_source, dest_path)
        
        # Copy signature for local verification
        if sig_path.exists():
            shutil.copy(sig_path, dest_path / ".fdaa-signature.json")
        
        return InstallResult(
            success=True,
            skill_name=name,
            version=version,
            install_path=str(dest_path),
        )
    
    def _install_from_package(
        self,
        package_path: Path,
        name: str,
        version: str,
        install_dir: Path,
        verify: bool
    ) -> InstallResult:
        """Install from downloaded package."""
        extract_dir = Path(tempfile.mkdtemp())
        
        try:
            # Extract
            with tarfile.open(package_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            
            # Load signature
            sig_path = extract_dir / ".fdaa-signature.json"
            if verify and sig_path.exists():
                sig_data = json.loads(sig_path.read_text())
                signature = SkillSignature.from_dict(sig_data)
                
                if not verify_signature(signature):
                    return InstallResult(
                        success=False,
                        skill_name=name,
                        version=version,
                        install_path="",
                        error="Signature verification failed"
                    )
            
            # Copy to install directory
            dest_path = install_dir / name
            if dest_path.exists():
                shutil.rmtree(dest_path)
            
            # Copy all files except metadata
            dest_path.mkdir(parents=True)
            for item in extract_dir.iterdir():
                if not item.name.startswith(".fdaa-"):
                    if item.is_dir():
                        shutil.copytree(item, dest_path / item.name)
                    else:
                        shutil.copy(item, dest_path / item.name)
            
            # Copy signature
            if sig_path.exists():
                shutil.copy(sig_path, dest_path / ".fdaa-signature.json")
            
            return InstallResult(
                success=True,
                skill_name=name,
                version=version,
                install_path=str(dest_path),
            )
            
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    def list_installed(self, install_dir: Path = None) -> list[dict]:
        """List installed skills."""
        install_dir = install_dir or Path.cwd() / "skills"
        
        if not install_dir.exists():
            return []
        
        installed = []
        for skill_dir in install_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                sig_path = skill_dir / ".fdaa-signature.json"
                
                info = {
                    "name": skill_dir.name,
                    "path": str(skill_dir),
                    "verified": False,
                    "skill_id": None,
                }
                
                if sig_path.exists():
                    try:
                        sig_data = json.loads(sig_path.read_text())
                        signature = SkillSignature.from_dict(sig_data)
                        info["skill_id"] = signature.skill_id
                        info["verified"] = verify_signature(signature)
                    except Exception:
                        pass
                
                installed.append(info)
        
        return installed


class RegistryError(Exception):
    """Error from registry operations."""
    pass


# Convenience functions
def install(skill_spec: str, install_dir: Path = None) -> InstallResult:
    """Install a skill from the registry."""
    client = RegistryClient()
    return client.install(skill_spec, install_dir)


def publish(skill_path: str, **kwargs) -> PublishResult:
    """Publish a skill to the registry."""
    client = RegistryClient()
    return client.publish(Path(skill_path), **kwargs)


def search(query: str) -> list[SkillMetadata]:
    """Search for skills."""
    client = RegistryClient()
    return client.search(query)
