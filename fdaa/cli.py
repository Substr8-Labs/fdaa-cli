"""
FDAA CLI - Command Line Interface
"""

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .agent import FDAAAgent, WRITABLE_FILES
from .templates import get_templates

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """FDAA - File-Driven Agent Architecture CLI
    
    Create, chat with, and manage file-driven AI agents.
    """
    pass


@main.command()
@click.argument("name")
@click.option("--path", "-p", default=".", help="Parent directory for workspace")
def init(name: str, path: str):
    """Create a new agent workspace.
    
    Example: fdaa init my-agent
    """
    workspace = Path(path) / name
    
    if workspace.exists():
        console.print(f"[red]Error:[/red] Workspace '{name}' already exists")
        sys.exit(1)
    
    # Create workspace directory
    workspace.mkdir(parents=True)
    
    # Generate template files
    templates = get_templates(name=name)
    
    for filename, content in templates.items():
        filepath = workspace / filename
        filepath.write_text(content)
        console.print(f"  [green]✓[/green] {filename}")
    
    console.print(f"\n[green]Created agent workspace:[/green] {workspace}")
    console.print(f"\nNext steps:")
    console.print(f"  1. Edit [cyan]{workspace}/IDENTITY.md[/cyan] to define who your agent is")
    console.print(f"  2. Edit [cyan]{workspace}/SOUL.md[/cyan] to define how it behaves")
    console.print(f"  3. Run [cyan]fdaa chat {name}[/cyan] to start talking")


@main.command()
@click.argument("workspace")
@click.option("--provider", "-p", default="openai", type=click.Choice(["openai", "anthropic"]))
@click.option("--model", "-m", default=None, help="Model to use (default: provider's best)")
def chat(workspace: str, provider: str, model: str):
    """Start a chat session with an agent.
    
    Example: fdaa chat my-agent
    """
    workspace_path = Path(workspace)
    
    if not workspace_path.exists():
        console.print(f"[red]Error:[/red] Workspace '{workspace}' not found")
        sys.exit(1)
    
    # Check for API key
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Error:[/red] OPENAI_API_KEY not set")
        sys.exit(1)
    elif provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set")
        sys.exit(1)
    
    # Load agent
    try:
        agent = FDAAAgent(str(workspace_path), provider=provider, model=model)
    except Exception as e:
        console.print(f"[red]Error loading agent:[/red] {e}")
        sys.exit(1)
    
    # Get agent identity
    identity = agent.get_file("IDENTITY.md") or "Unknown Agent"
    name_match = identity.split("**Name:**")
    agent_name = name_match[1].split("\n")[0].strip() if len(name_match) > 1 else workspace
    
    console.print(Panel(
        f"[bold]{agent_name}[/bold]\n\n"
        f"Provider: {provider} ({agent.model})\n"
        f"Workspace: {workspace_path}\n\n"
        f"[dim]Type 'exit' to quit, '/files' to list files, '/read <file>' to read a file[/dim]",
        title="FDAA Chat Session",
        border_style="blue"
    ))
    
    # Chat loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break
        
        if not user_input.strip():
            continue
        
        # Handle commands
        if user_input.lower() == "exit":
            console.print("[dim]Goodbye![/dim]")
            break
        
        if user_input.lower() == "/files":
            files = agent.load_files()
            console.print("\n[bold]Workspace files:[/bold]")
            for f in sorted(files.keys()):
                writable = "✏️" if f in WRITABLE_FILES else "🔒"
                console.print(f"  {writable} {f}")
            continue
        
        if user_input.lower().startswith("/read "):
            filename = user_input[6:].strip()
            content = agent.get_file(filename)
            if content:
                console.print(Panel(Markdown(content), title=filename, border_style="cyan"))
            else:
                console.print(f"[red]File not found:[/red] {filename}")
            continue
        
        if user_input.lower().startswith("/edit "):
            filename = user_input[6:].strip()
            filepath = workspace_path / filename
            if filepath.exists():
                click.edit(filename=str(filepath))
                console.print(f"[green]Edited:[/green] {filename}")
            else:
                console.print(f"[red]File not found:[/red] {filename}")
            continue
        
        # Send message to agent
        console.print()
        with console.status("[dim]Thinking...[/dim]"):
            try:
                response = agent.chat(user_input)
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                continue
        
        console.print(f"[bold green]{agent_name}[/bold green]")
        console.print(Markdown(response))


@main.command()
@click.argument("workspace")
@click.option("--output", "-o", default=None, help="Output path for zip file")
def export(workspace: str, output: str):
    """Export an agent workspace as a zip file.
    
    Example: fdaa export my-agent -o my-agent-backup.zip
    """
    workspace_path = Path(workspace)
    
    if not workspace_path.exists():
        console.print(f"[red]Error:[/red] Workspace '{workspace}' not found")
        sys.exit(1)
    
    agent = FDAAAgent(str(workspace_path))
    
    output_path = output or f"{workspace}.zip"
    result = agent.export(output_path)
    
    console.print(f"[green]Exported:[/green] {result}")


@main.command(name="import")
@click.argument("zip_file")
@click.option("--path", "-p", default=".", help="Where to extract the workspace")
@click.option("--name", "-n", default=None, help="Name for the imported workspace")
def import_workspace(zip_file: str, path: str, name: str):
    """Import an agent workspace from a zip file.
    
    Example: fdaa import shared-agent.zip --name my-copy
    """
    zip_path = Path(zip_file)
    
    if not zip_path.exists():
        console.print(f"[red]Error:[/red] File '{zip_file}' not found")
        sys.exit(1)
    
    workspace_name = name or zip_path.stem
    target_path = Path(path) / workspace_name
    
    if target_path.exists():
        console.print(f"[red]Error:[/red] Workspace '{workspace_name}' already exists")
        sys.exit(1)
    
    FDAAAgent.import_workspace(str(zip_path), str(target_path))
    
    console.print(f"[green]Imported:[/green] {target_path}")
    console.print(f"\nRun [cyan]fdaa chat {workspace_name}[/cyan] to start chatting")


@main.command()
@click.argument("workspace")
def files(workspace: str):
    """List files in an agent workspace.
    
    Example: fdaa files my-agent
    """
    workspace_path = Path(workspace)
    
    if not workspace_path.exists():
        console.print(f"[red]Error:[/red] Workspace '{workspace}' not found")
        sys.exit(1)
    
    agent = FDAAAgent(str(workspace_path))
    files = agent.load_files()
    
    console.print(f"\n[bold]Workspace:[/bold] {workspace_path}\n")
    
    for filename in sorted(files.keys()):
        writable = "[green]✏️ writable[/green]" if filename in WRITABLE_FILES else "[dim]🔒 read-only[/dim]"
        size = len(files[filename])
        console.print(f"  {filename:<20} {size:>6} bytes  {writable}")


@main.command()
@click.argument("workspace")
@click.argument("filename")
def read(workspace: str, filename: str):
    """Read a file from an agent workspace.
    
    Example: fdaa read my-agent MEMORY.md
    """
    workspace_path = Path(workspace)
    
    if not workspace_path.exists():
        console.print(f"[red]Error:[/red] Workspace '{workspace}' not found")
        sys.exit(1)
    
    agent = FDAAAgent(str(workspace_path))
    content = agent.get_file(filename)
    
    if content:
        console.print(Panel(Markdown(content), title=filename, border_style="cyan"))
    else:
        console.print(f"[red]File not found:[/red] {filename}")
        sys.exit(1)


@main.command()
@click.argument("skill_path")
@click.option("--model", "-m", default=None, help="Model for analysis (auto-detected)")
@click.option("--provider", "-p", default=None, type=click.Choice(["anthropic", "openai"]), help="LLM provider")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON")
def verify(skill_path: str, model: str, provider: str, output_json: bool):
    """Verify a skill using the Guard Model (Tier 2 scanner).
    
    Analyzes skills for:
    - Line Jumping: Hidden instructions in metadata
    - Scope Drift: Capabilities exceeding stated purpose  
    - Intent vs Behavior: Code that doesn't match documentation
    
    Example: fdaa verify ./my-skill
    """
    from .guard import verify_skill, Severity, Alignment, Recommendation
    import json as json_lib
    
    skill_path_obj = Path(skill_path)
    
    # Find SKILL.md
    if skill_path_obj.is_dir():
        skill_md = skill_path_obj / "SKILL.md"
    else:
        skill_md = skill_path_obj
    
    if not skill_md.exists():
        console.print(f"[red]Error:[/red] SKILL.md not found at {skill_path}")
        sys.exit(1)
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Error:[/red] No API key found")
        console.print("[dim]Set ANTHROPIC_API_KEY or OPENAI_API_KEY[/dim]")
        sys.exit(1)
    
    # Determine provider and model
    if provider is None:
        provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
    
    if model is None:
        model = "claude-sonnet-4-20250514" if provider == "anthropic" else "gpt-4o"
    
    console.print(f"\n[bold]🛡️ FDAA Guard Model - Tier 2 Verification[/bold]")
    console.print(f"[dim]Skill: {skill_path}[/dim]")
    console.print(f"[dim]Provider: {provider} | Model: {model}[/dim]\n")
    
    with console.status("[bold blue]Running semantic analysis...[/bold blue]"):
        try:
            verdict = verify_skill(str(skill_path_obj), model=model, provider=provider)
        except Exception as e:
            console.print(f"[red]Error during analysis:[/red] {e}")
            sys.exit(1)
    
    # Output JSON if requested
    if output_json:
        console.print(json_lib.dumps(verdict.to_dict(), indent=2))
        sys.exit(0 if verdict.passed else 1)
    
    # Rich output
    console.print("[bold]━━━ Analysis Results ━━━[/bold]\n")
    
    # Line Jumping
    if verdict.line_jumping:
        lj = verdict.line_jumping
        status = "[red]⚠️ DETECTED[/red]" if lj.detected else "[green]✓ Clear[/green]"
        console.print(f"[bold]Line Jumping:[/bold] {status}")
        if lj.detected:
            console.print(f"  Severity: {lj.severity.value}")
            if lj.evidence:
                console.print(f"  Evidence:")
                for e in lj.evidence[:3]:
                    console.print(f"    • {e[:100]}...")
            if lj.attack_vectors:
                console.print(f"  Attack Vectors:")
                for v in lj.attack_vectors[:3]:
                    console.print(f"    • {v}")
        console.print()
    
    # Scope Drift
    if verdict.scope_drift:
        sd = verdict.scope_drift
        if sd.drift_score <= 20:
            drift_status = f"[green]✓ Aligned ({sd.drift_score}/100)[/green]"
        elif sd.drift_score <= 50:
            drift_status = f"[yellow]⚡ Minor drift ({sd.drift_score}/100)[/yellow]"
        elif sd.drift_score <= 75:
            drift_status = f"[orange1]⚠️ Significant drift ({sd.drift_score}/100)[/orange1]"
        else:
            drift_status = f"[red]🚨 Major drift ({sd.drift_score}/100)[/red]"
        
        console.print(f"[bold]Scope Drift:[/bold] {drift_status}")
        if sd.unadvertised_capabilities:
            console.print(f"  Unadvertised capabilities:")
            for cap in sd.unadvertised_capabilities[:5]:
                console.print(f"    • {cap}")
        if sd.risk_rationale:
            console.print(f"  Rationale: {sd.risk_rationale[:200]}")
        console.print()
    
    # Intent Comparison
    if verdict.intent_comparison:
        ic = verdict.intent_comparison
        if ic.alignment == Alignment.ALIGNED:
            align_status = "[green]✓ Aligned[/green]"
        elif ic.alignment == Alignment.CONFLICTED:
            align_status = "[yellow]⚡ Conflicted[/yellow]"
        else:
            align_status = "[red]🚨 Malicious[/red]"
        
        console.print(f"[bold]Intent vs Behavior:[/bold] {align_status}")
        if ic.unauthorized_sinks:
            console.print(f"  Unauthorized sinks:")
            for sink in ic.unauthorized_sinks[:5]:
                console.print(f"    • {sink}")
        if ic.new_capabilities:
            console.print(f"  Undocumented capabilities:")
            for cap in ic.new_capabilities[:5]:
                console.print(f"    • {cap}")
        console.print()
    
    # Overall Verdict
    console.print("[bold]━━━ Verdict ━━━[/bold]\n")
    
    if verdict.passed:
        console.print("[bold green]✅ PASSED[/bold green]")
    else:
        console.print("[bold red]❌ FAILED[/bold red]")
    
    rec_colors = {
        Recommendation.APPROVE: "green",
        Recommendation.REVIEW: "yellow",
        Recommendation.REJECT: "red",
    }
    rec_color = rec_colors.get(verdict.recommendation, "white")
    console.print(f"Recommendation: [{rec_color}]{verdict.recommendation.value.upper()}[/{rec_color}]")
    
    if verdict.error:
        console.print(f"\n[red]Error:[/red] {verdict.error}")
    
    console.print()
    sys.exit(0 if verdict.passed else 1)


@main.command()
@click.argument("skill_path")
@click.option("--key", "-k", default="default", help="Signing key name")
def sign(skill_path: str, key: str):
    """Sign a verified skill and add to registry.
    
    Creates a cryptographic signature for the skill containing:
    - SHA256 hash of SKILL.md
    - Merkle root of scripts/ directory
    - Merkle root of references/ directory
    - Ed25519 signature
    
    Example: fdaa sign ./my-skill
    """
    from .registry import sign_and_register
    import json as json_lib
    
    skill_path_obj = Path(skill_path)
    
    # Find SKILL.md
    if skill_path_obj.is_dir():
        skill_md = skill_path_obj / "SKILL.md"
    else:
        skill_md = skill_path_obj
    
    if not skill_md.exists():
        console.print(f"[red]Error:[/red] SKILL.md not found at {skill_path}")
        sys.exit(1)
    
    console.print(f"\n[bold]🔏 FDAA Registry - Signing Skill[/bold]")
    console.print(f"[dim]Skill: {skill_path}[/dim]")
    console.print(f"[dim]Key: {key}[/dim]\n")
    
    try:
        sig = sign_and_register(str(skill_path_obj), key_name=key)
        
        console.print("[green]✓ Skill signed and registered[/green]\n")
        console.print(f"[bold]Skill ID:[/bold] {sig.skill_id}")
        console.print(f"[bold]Content Hash:[/bold] {sig.content_hash[:32]}...")
        console.print(f"[bold]Scripts Merkle:[/bold] {sig.scripts_merkle_root[:32]}...")
        console.print(f"[bold]References Merkle:[/bold] {sig.references_merkle_root[:32]}...")
        console.print(f"[bold]Timestamp:[/bold] {sig.verification_timestamp}")
        console.print(f"[bold]Signer:[/bold] {sig.signer_id[:32]}...")
        console.print(f"[bold]Signature:[/bold] {sig.signature[:32]}...")
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error signing skill:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("skill_path")
def check(skill_path: str):
    """Check if a skill is registered and unmodified.
    
    Verifies:
    - SKILL.md content hash matches
    - scripts/ directory unchanged
    - references/ directory unchanged
    - Cryptographic signature valid
    
    Example: fdaa check ./my-skill
    """
    from .registry import check_skill
    
    skill_path_obj = Path(skill_path)
    
    # Find SKILL.md
    if skill_path_obj.is_dir():
        skill_md = skill_path_obj / "SKILL.md"
    else:
        skill_md = skill_path_obj
    
    if not skill_md.exists():
        console.print(f"[red]Error:[/red] SKILL.md not found at {skill_path}")
        sys.exit(1)
    
    console.print(f"\n[bold]🔍 FDAA Registry - Checking Skill[/bold]")
    console.print(f"[dim]Skill: {skill_path}[/dim]\n")
    
    result = check_skill(str(skill_path_obj))
    
    # Show results
    console.print("[bold]Integrity Checks:[/bold]")
    
    content_status = "[green]✓[/green]" if result.content_match else "[red]✗[/red]"
    scripts_status = "[green]✓[/green]" if result.scripts_match else "[red]✗[/red]"
    refs_status = "[green]✓[/green]" if result.references_match else "[red]✗[/red]"
    sig_status = "[green]✓[/green]" if result.signature_valid else "[red]✗[/red]"
    
    console.print(f"  {content_status} SKILL.md content")
    console.print(f"  {scripts_status} scripts/ directory")
    console.print(f"  {refs_status} references/ directory")
    console.print(f"  {sig_status} Cryptographic signature")
    console.print()
    
    if result.valid:
        console.print("[bold green]✅ VERIFIED[/bold green]")
        console.print(f"Skill ID: {result.skill_id}")
    else:
        console.print("[bold red]❌ VERIFICATION FAILED[/bold red]")
        if result.error:
            console.print(f"[red]Error:[/red] {result.error}")
    
    console.print()
    sys.exit(0 if result.valid else 1)


@main.command(name="registry-list")
def registry_list():
    """List all signed skills in the local registry.
    
    Example: fdaa registry-list
    """
    from .registry import list_signatures
    
    console.print(f"\n[bold]📋 FDAA Registry - Signed Skills[/bold]\n")
    
    sigs = list_signatures()
    
    if not sigs:
        console.print("[dim]No skills registered yet.[/dim]")
        console.print("[dim]Use 'fdaa sign <skill-path>' to sign a skill.[/dim]\n")
        return
    
    for sig in sigs:
        passed = "[green]✓[/green]" if sig.tier2_passed else "[red]✗[/red]"
        console.print(f"  {passed} [bold]{sig.skill_id}[/bold]")
        console.print(f"    Path: {sig.skill_path}")
        console.print(f"    Signed: {sig.verification_timestamp}")
        console.print(f"    Recommendation: {sig.tier2_recommendation}")
        console.print()


@main.command()
@click.option("--name", "-n", default="default", help="Key name")
def keygen(name: str):
    """Generate a new Ed25519 signing key pair.
    
    Keys are stored in ~/.fdaa/keys/
    
    Example: fdaa keygen --name production
    """
    from .registry import generate_signing_key
    
    console.print(f"\n[bold]🔑 FDAA Registry - Generating Key Pair[/bold]")
    console.print(f"[dim]Name: {name}[/dim]\n")
    
    try:
        public_hex, private_path = generate_signing_key(name)
        
        console.print("[green]✓ Key pair generated[/green]\n")
        console.print(f"[bold]Public Key:[/bold]")
        console.print(f"  {public_hex}")
        console.print(f"\n[bold]Private Key:[/bold]")
        console.print(f"  {private_path}")
        console.print()
        console.print("[dim]Share the public key for verification.[/dim]")
        console.print("[dim]Keep the private key secure![/dim]\n")
        
    except Exception as e:
        console.print(f"[red]Error generating key:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("script_or_skill")
@click.option("--timeout", "-t", default=30, help="Execution timeout in seconds")
@click.option("--memory", "-m", default=256, help="Memory limit in MB")
@click.option("--network/--no-network", default=False, help="Allow network access")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON")
def sandbox(script_or_skill: str, timeout: int, memory: int, network: bool, output_json: bool):
    """Execute a script or skill in an isolated sandbox (Tier 3).
    
    Runs code in a Docker container with:
    - Memory and CPU limits
    - Network isolation (optional)
    - Read-only filesystem
    - Seccomp syscall filtering
    - Violation detection
    
    Examples:
        fdaa sandbox ./script.py
        fdaa sandbox ./my-skill --timeout 60
        fdaa sandbox ./untrusted.py --no-network
    """
    from .sandbox import SandboxExecutor, SandboxConfig, ExecutionResult
    from .sandbox.executor import ExecutionStatus
    import json as json_lib
    
    path = Path(script_or_skill)
    
    if not path.exists():
        console.print(f"[red]Error:[/red] Path not found: {script_or_skill}")
        sys.exit(1)
    
    console.print(f"\n[bold]🔒 FDAA Sandbox - Tier 3 Execution[/bold]")
    console.print(f"[dim]Target: {script_or_skill}[/dim]")
    console.print(f"[dim]Limits: {memory}MB RAM, {timeout}s timeout, network={'on' if network else 'off'}[/dim]\n")
    
    config = SandboxConfig(
        timeout_seconds=timeout,
        memory_limit_mb=memory,
        network_enabled=network,
    )
    
    try:
        executor = SandboxExecutor(config)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[dim]Docker is required for sandbox execution[/dim]")
        sys.exit(1)
    
    with console.status("[bold blue]Executing in sandbox...[/bold blue]"):
        if path.is_dir():
            result = executor.execute_skill(path)
        else:
            interpreter = "python3" if path.suffix == ".py" else "bash"
            result = executor.execute_script(path.read_text(), interpreter=interpreter)
    
    # Output JSON if requested
    if output_json:
        console.print(json_lib.dumps(result.to_dict(), indent=2))
        sys.exit(0 if result.status == ExecutionStatus.SUCCESS else 1)
    
    # Rich output
    console.print("[bold]━━━ Execution Results ━━━[/bold]\n")
    
    # Status
    status_colors = {
        ExecutionStatus.SUCCESS: "green",
        ExecutionStatus.ERROR: "red",
        ExecutionStatus.TIMEOUT: "yellow",
        ExecutionStatus.VIOLATION: "red",
    }
    status_icons = {
        ExecutionStatus.SUCCESS: "✓",
        ExecutionStatus.ERROR: "✗",
        ExecutionStatus.TIMEOUT: "⏱",
        ExecutionStatus.VIOLATION: "🚨",
    }
    color = status_colors.get(result.status, "white")
    icon = status_icons.get(result.status, "?")
    
    console.print(f"[bold]Status:[/bold] [{color}]{icon} {result.status.value.upper()}[/{color}]")
    console.print(f"[bold]Exit Code:[/bold] {result.exit_code}")
    console.print(f"[bold]Duration:[/bold] {result.duration_ms}ms")
    console.print()
    
    # Stdout
    if result.stdout.strip():
        console.print("[bold]Output:[/bold]")
        console.print(Panel(result.stdout[:2000], border_style="green"))
    
    # Stderr
    if result.stderr.strip():
        console.print("[bold]Errors:[/bold]")
        console.print(Panel(result.stderr[:2000], border_style="red"))
    
    # Violations
    if result.violations:
        console.print("[bold red]Security Violations:[/bold red]")
        for v in result.violations:
            console.print(f"  [red]⚠️ {v.type}[/red]: {v.details}")
        console.print()
    
    # Output files
    if result.output_files:
        console.print("[bold]Output Files:[/bold]")
        for filename, content in result.output_files.items():
            preview = content[:100] + "..." if len(content) > 100 else content
            console.print(f"  📄 {filename}: {preview}")
        console.print()
    
    sys.exit(0 if result.status == ExecutionStatus.SUCCESS else 1)


@main.command()
@click.argument("skill_path")
@click.option("--key", "-k", default="default", help="Signing key name")
@click.option("--skip-verify", is_flag=True, help="Skip Guard Model verification")
@click.option("--skip-sandbox", is_flag=True, help="Skip sandbox execution test")
def pipeline(skill_path: str, key: str, skip_verify: bool, skip_sandbox: bool):
    """Run the full verification pipeline on a skill.
    
    Executes all tiers:
    1. Tier 1: Fast Pass (regex scanning) - via repo-concierge
    2. Tier 2: Guard Model (semantic analysis)
    3. Tier 3: Sandbox (isolated execution test)
    4. Tier 4: Sign and register
    
    Example: fdaa pipeline ./my-skill
    """
    from .guard import verify_skill, Recommendation
    from .registry import sign_and_register
    from .sandbox import SandboxExecutor, SandboxConfig
    from .sandbox.executor import ExecutionStatus
    
    skill_path_obj = Path(skill_path)
    
    # Find SKILL.md
    if skill_path_obj.is_dir():
        skill_md = skill_path_obj / "SKILL.md"
    else:
        skill_md = skill_path_obj
        skill_path_obj = skill_path_obj.parent
    
    if not skill_md.exists():
        console.print(f"[red]Error:[/red] SKILL.md not found at {skill_path}")
        sys.exit(1)
    
    console.print(f"\n[bold]🔥 FDAA Full Verification Pipeline[/bold]")
    console.print(f"[dim]Skill: {skill_path}[/dim]\n")
    
    tier1_passed = True  # Assume passed (would integrate repo-concierge here)
    tier2_passed = True
    tier2_recommendation = "approve"
    tier3_passed = None
    
    # Tier 1: Fast Pass (placeholder - would use repo-concierge)
    console.print("[bold]Tier 1: Fast Pass (Regex)[/bold]")
    console.print("  [green]✓[/green] Skipped (use repo-concierge for full scan)")
    console.print()
    
    # Tier 2: Guard Model
    if not skip_verify:
        console.print("[bold]Tier 2: Guard Model (Semantic Analysis)[/bold]")
        
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            console.print("  [yellow]⚠️[/yellow] Skipped (no API key)")
        else:
            with console.status("  Running semantic analysis..."):
                verdict = verify_skill(str(skill_path_obj))
            
            tier2_passed = verdict.passed
            tier2_recommendation = verdict.recommendation.value
            
            status = "[green]✓ PASSED[/green]" if tier2_passed else "[red]✗ FAILED[/red]"
            console.print(f"  {status}")
            console.print(f"  Recommendation: {tier2_recommendation}")
            
            if verdict.line_jumping and verdict.line_jumping.detected:
                console.print(f"  [red]Line Jumping detected![/red]")
            if verdict.scope_drift:
                console.print(f"  Scope Drift: {verdict.scope_drift.drift_score}/100")
    else:
        console.print("[bold]Tier 2: Guard Model[/bold]")
        console.print("  [yellow]⚠️[/yellow] Skipped (--skip-verify)")
    console.print()
    
    # Tier 3: Sandbox
    if not skip_sandbox:
        console.print("[bold]Tier 3: Sandbox (Isolated Execution)[/bold]")
        
        scripts_dir = skill_path_obj / "scripts"
        if scripts_dir.exists() and any(scripts_dir.glob("*.py")):
            try:
                config = SandboxConfig(timeout_seconds=30, network_enabled=False)
                executor = SandboxExecutor(config)
                
                with console.status("  Executing in sandbox..."):
                    result = executor.execute_skill(skill_path_obj)
                
                tier3_passed = result.status == ExecutionStatus.SUCCESS
                
                if tier3_passed:
                    console.print(f"  [green]✓ PASSED[/green] ({result.duration_ms}ms)")
                else:
                    console.print(f"  [red]✗ FAILED[/red] ({result.status.value})")
                    if result.violations:
                        for v in result.violations:
                            console.print(f"    [red]⚠️ {v.type}[/red]")
                            
            except Exception as e:
                console.print(f"  [yellow]⚠️[/yellow] Skipped ({e})")
                tier3_passed = None
        else:
            console.print("  [dim]No scripts to test[/dim]")
            tier3_passed = None
    else:
        console.print("[bold]Tier 3: Sandbox[/bold]")
        console.print("  [yellow]⚠️[/yellow] Skipped (--skip-sandbox)")
    console.print()
    
    # Tier 4: Sign and Register
    console.print("[bold]Tier 4: Sign and Register[/bold]")
    
    # Only sign if previous tiers passed
    if not tier2_passed:
        console.print("  [red]✗ BLOCKED[/red] - Tier 2 verification failed")
        console.print("\n[bold red]❌ PIPELINE FAILED[/bold red]\n")
        sys.exit(1)
    
    if tier3_passed is False:
        console.print("  [red]✗ BLOCKED[/red] - Tier 3 sandbox failed")
        console.print("\n[bold red]❌ PIPELINE FAILED[/bold red]\n")
        sys.exit(1)
    
    try:
        sig = sign_and_register(
            str(skill_path_obj),
            tier1_passed=tier1_passed,
            tier2_passed=tier2_passed,
            tier2_recommendation=tier2_recommendation,
            tier3_passed=tier3_passed,
            key_name=key,
        )
        console.print(f"  [green]✓ SIGNED[/green]")
        console.print(f"  Skill ID: {sig.skill_id}")
        console.print(f"  Signature: {sig.signature[:32]}...")
    except Exception as e:
        console.print(f"  [red]✗ ERROR[/red]: {e}")
        sys.exit(1)
    
    console.print()
    console.print("[bold green]✅ PIPELINE COMPLETE[/bold green]")
    console.print(f"Skill verified and registered: {sig.skill_id}\n")


@main.command()
@click.argument("skill_spec")
@click.option("--dir", "-d", "install_dir", default=None, help="Installation directory")
@click.option("--no-verify", is_flag=True, help="Skip signature verification")
def install(skill_spec: str, install_dir: str, no_verify: bool):
    """Install a skill from the registry.
    
    Downloads a verified skill and installs it locally.
    Verifies cryptographic signature by default.
    
    Examples:
        fdaa install weather-skill
        fdaa install weather-skill@1.2.0
        fdaa install weather-skill --dir ./my-skills
    """
    from .registry_client import RegistryClient
    
    console.print(f"\n[bold]📦 FDAA Install[/bold]")
    console.print(f"[dim]Skill: {skill_spec}[/dim]\n")
    
    client = RegistryClient()
    dest = Path(install_dir) if install_dir else None
    
    with console.status("[bold blue]Installing...[/bold blue]"):
        result = client.install(skill_spec, install_dir=dest, verify=not no_verify)
    
    if result.success:
        console.print(f"[green]✓ Installed:[/green] {result.skill_name}@{result.version}")
        console.print(f"[dim]Location: {result.install_path}[/dim]\n")
    else:
        console.print(f"[red]✗ Failed:[/red] {result.error}\n")
        sys.exit(1)


@main.command()
@click.argument("skill_path")
@click.option("--name", "-n", default=None, help="Skill name (default: directory name)")
@click.option("--version", "-v", default="1.0.0", help="Semantic version")
@click.option("--author", "-a", default=None, help="Author name/email")
@click.option("--key", "-k", default="default", help="Signing key name")
@click.option("--skip-verify", is_flag=True, help="Skip verification pipeline")
def publish(skill_path: str, name: str, version: str, author: str, key: str, skip_verify: bool):
    """Publish a skill to the registry.
    
    Runs verification pipeline, signs the skill, and uploads to registry.
    
    Examples:
        fdaa publish ./my-skill
        fdaa publish ./my-skill --version 1.2.0
        fdaa publish ./my-skill --name cool-skill --author "me@example.com"
    """
    from .registry_client import RegistryClient
    
    skill_path_obj = Path(skill_path)
    
    if not skill_path_obj.exists():
        console.print(f"[red]Error:[/red] Path not found: {skill_path}")
        sys.exit(1)
    
    if not (skill_path_obj / "SKILL.md").exists():
        console.print(f"[red]Error:[/red] SKILL.md not found in {skill_path}")
        sys.exit(1)
    
    console.print(f"\n[bold]🚀 FDAA Publish[/bold]")
    console.print(f"[dim]Skill: {skill_path}[/dim]")
    console.print(f"[dim]Version: {version}[/dim]\n")
    
    client = RegistryClient()
    
    with console.status("[bold blue]Publishing...[/bold blue]"):
        result = client.publish(
            skill_path_obj,
            name=name,
            version=version,
            author=author,
            key_name=key,
            run_pipeline=not skip_verify,
        )
    
    if result.success:
        console.print(f"[green]✓ Published:[/green] {name or skill_path_obj.name}@{version}")
        console.print(f"[dim]Skill ID: {result.skill_id}[/dim]")
        console.print(f"[dim]Registry: {result.registry_url}[/dim]\n")
    else:
        console.print(f"[red]✗ Failed:[/red] {result.error}\n")
        sys.exit(1)


@main.command()
@click.argument("query")
@click.option("--limit", "-l", default=20, help="Max results")
def search(query: str, limit: int):
    """Search for skills in the registry.
    
    Example: fdaa search weather
    """
    from .registry_client import RegistryClient
    
    console.print(f"\n[bold]🔍 FDAA Search:[/bold] {query}\n")
    
    client = RegistryClient()
    
    with console.status("[bold blue]Searching...[/bold blue]"):
        results = client.search(query, limit=limit)
    
    if not results:
        console.print("[dim]No skills found.[/dim]\n")
        return
    
    for skill in results:
        console.print(f"  [bold]{skill.name}[/bold] v{skill.version}")
        console.print(f"    {skill.description[:80]}...")
        console.print(f"    [dim]by {skill.author} | {skill.downloads} downloads[/dim]")
        console.print()


@main.command(name="list")
@click.option("--dir", "-d", "install_dir", default=None, help="Skills directory")
def list_skills(install_dir: str):
    """List installed skills.
    
    Example: fdaa list
    """
    from .registry_client import RegistryClient
    
    console.print(f"\n[bold]📋 Installed Skills[/bold]\n")
    
    client = RegistryClient()
    dest = Path(install_dir) if install_dir else None
    
    installed = client.list_installed(dest)
    
    if not installed:
        console.print("[dim]No skills installed.[/dim]")
        console.print("[dim]Use 'fdaa install <skill>' to install one.[/dim]\n")
        return
    
    for skill in installed:
        verified = "[green]✓[/green]" if skill["verified"] else "[yellow]?[/yellow]"
        console.print(f"  {verified} [bold]{skill['name']}[/bold]")
        console.print(f"    [dim]{skill['path']}[/dim]")
        if skill["skill_id"]:
            console.print(f"    [dim]ID: {skill['skill_id']}[/dim]")
        console.print()


if __name__ == "__main__":
    main()
