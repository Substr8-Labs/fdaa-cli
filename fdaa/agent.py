"""
FDAA Agent - Core implementation
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# File injection order (as specified in the whitepaper)
INJECTION_ORDER = [
    "IDENTITY.md",
    "SOUL.md",
    "CONTEXT.md",
    "MEMORY.md",
    "TOOLS.md",
]

# W^X Policy: Files the agent CAN write to
WRITABLE_FILES = {"MEMORY.md", "CONTEXT.md"}

# Files the agent can delete (bootstrap pattern)
DELETABLE_FILES = {"BOOTSTRAP.md"}


class FDAAAgent:
    """
    A file-driven agent whose identity, behavior, and memory
    are defined entirely through markdown files.
    """
    
    def __init__(self, workspace_path: str, provider: str = "openai", model: str = None):
        self.workspace = Path(workspace_path)
        self.provider = provider
        self.model = model or self._default_model()
        self.history: List[Dict[str, str]] = []
        
        if not self.workspace.exists():
            raise ValueError(f"Workspace not found: {workspace_path}")
    
    def _default_model(self) -> str:
        """Default model per provider."""
        return {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
        }.get(self.provider, "gpt-4o")
    
    def load_files(self) -> Dict[str, str]:
        """Load all markdown files from workspace."""
        files = {}
        for path in self.workspace.glob("*.md"):
            files[path.name] = path.read_text()
        return files
    
    def assemble_prompt(self) -> str:
        """Assemble system prompt from workspace files."""
        files = self.load_files()
        sections = []
        
        # Add files in defined order
        for filename in INJECTION_ORDER:
            if filename in files:
                content = files[filename]
                sections.append(f"## {filename}\n\n{content}")
        
        # Add any additional files not in standard order
        for filename, content in sorted(files.items()):
            if filename not in INJECTION_ORDER and filename != "BOOTSTRAP.md":
                sections.append(f"## {filename}\n\n{content}")
        
        # Add system instructions
        sections.append(self._system_instructions())
        
        return "\n\n---\n\n".join(sections)
    
    def _system_instructions(self) -> str:
        """Runtime instructions appended to every prompt."""
        return """## System Instructions

You are an AI agent defined by the files above. Follow these rules:

1. **Stay in character** as defined by IDENTITY.md and SOUL.md
2. **Remember context** from MEMORY.md and CONTEXT.md
3. **Use capabilities** listed in TOOLS.md (if present)

### Memory Updates

When you learn something important that should persist, include a memory update block:

```memory:MEMORY.md
[Your updated memory content here]
```

This will be saved to your MEMORY.md file. Only include what's worth remembering long-term.

### Boundaries

- You CANNOT modify IDENTITY.md or SOUL.md (these define who you are)
- You CAN update MEMORY.md and CONTEXT.md
- Be helpful, stay in character, and remember what matters.
"""
    
    def chat(self, message: str) -> str:
        """Send a message and get a response."""
        system_prompt = self.assemble_prompt()
        
        # Add user message to history
        self.history.append({"role": "user", "content": message})
        
        # Call LLM
        if self.provider == "openai":
            response = self._call_openai(system_prompt)
        elif self.provider == "anthropic":
            response = self._call_anthropic(system_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Process response (extract and apply memory updates)
        clean_response = self._process_memory_updates(response)
        
        # Add assistant response to history
        self.history.append({"role": "assistant", "content": clean_response})
        
        return clean_response
    
    def _call_openai(self, system_prompt: str) -> str:
        """Call OpenAI API."""
        from openai import OpenAI
        
        client = OpenAI()  # Uses OPENAI_API_KEY env var
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic(self, system_prompt: str) -> str:
        """Call Anthropic API."""
        from anthropic import Anthropic
        
        client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
        
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=self.history,
        )
        
        return response.content[0].text
    
    def _process_memory_updates(self, response: str) -> str:
        """Extract memory update blocks and persist them."""
        pattern = r"```memory:(\S+)\n(.*?)```"
        
        def apply_update(match):
            filename = match.group(1)
            content = match.group(2).strip()
            
            # Enforce W^X policy
            if filename not in WRITABLE_FILES:
                return f"\n\n*[Blocked: Cannot write to {filename} - W^X policy]*\n\n"
            
            # Write to file
            filepath = self.workspace / filename
            filepath.write_text(content)
            
            return f"\n\n*[Memory updated: {filename}]*\n\n"
        
        # Apply updates and clean response
        clean_response = re.sub(pattern, apply_update, response, flags=re.DOTALL)
        
        return clean_response.strip()
    
    def get_file(self, filename: str) -> Optional[str]:
        """Read a specific workspace file."""
        filepath = self.workspace / filename
        if filepath.exists():
            return filepath.read_text()
        return None
    
    def update_file(self, filename: str, content: str) -> bool:
        """Manually update a workspace file (respects W^X for agent, not user)."""
        filepath = self.workspace / filename
        filepath.write_text(content)
        return True
    
    def reset_history(self):
        """Clear conversation history (memory files persist)."""
        self.history = []
    
    def export(self, output_path: str) -> str:
        """Export workspace as a zip file."""
        import shutil
        
        output = Path(output_path)
        if output.suffix != ".zip":
            output = output.with_suffix(".zip")
        
        # Create zip
        shutil.make_archive(
            str(output.with_suffix("")),
            "zip",
            self.workspace
        )
        
        return str(output)
    
    @classmethod
    def import_workspace(cls, zip_path: str, target_path: str) -> "FDAAAgent":
        """Import a workspace from a zip file."""
        import shutil
        
        target = Path(target_path)
        target.mkdir(parents=True, exist_ok=True)
        
        shutil.unpack_archive(zip_path, target)
        
        return cls(str(target))
