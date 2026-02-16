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


if __name__ == "__main__":
    main()
