# FDAA CLI

**File-Driven Agent Architecture — Reference Implementation**

A command-line tool for creating and interacting with file-driven AI agents.

## What is FDAA?

File-Driven Agent Architecture is a pattern where AI agents are defined entirely through human-readable markdown files. No code changes, no fine-tuning, no config UI — just files.

📄 [Read the whitepaper](https://github.com/Substr8-Labs/fdaa-spec)

## Installation

```bash
pip install fdaa
```

Or install from source:

```bash
git clone https://github.com/Substr8-Labs/fdaa-cli
cd fdaa-cli
pip install -e .
```

## Quick Start

### 1. Create an agent

```bash
fdaa init my-agent
```

This creates a workspace with template files:
```
my-agent/
├── IDENTITY.md   # Who the agent is
├── SOUL.md       # How it thinks and behaves
├── MEMORY.md     # What it remembers (writable)
├── CONTEXT.md    # Current state (writable)
└── TOOLS.md      # Capabilities
```

### 2. Customize your agent

Edit the files to define your agent's personality:

```bash
# Edit identity
nano my-agent/IDENTITY.md

# Edit behavior
nano my-agent/SOUL.md
```

### 3. Chat with your agent

```bash
export OPENAI_API_KEY=sk-...
fdaa chat my-agent
```

Or use Anthropic:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
fdaa chat my-agent --provider anthropic
```

### 4. Watch memory persist

After your conversation, check what the agent remembered:

```bash
fdaa read my-agent MEMORY.md
```

## Commands

| Command | Description |
|---------|-------------|
| `fdaa init <name>` | Create a new agent workspace |
| `fdaa chat <workspace>` | Start a chat session |
| `fdaa files <workspace>` | List workspace files |
| `fdaa read <workspace> <file>` | Read a specific file |
| `fdaa export <workspace>` | Export as zip |
| `fdaa import <zip>` | Import from zip |

## Chat Commands

While in a chat session:

| Command | Description |
|---------|-------------|
| `exit` | End the session |
| `/files` | List workspace files |
| `/read <file>` | Read a file |
| `/edit <file>` | Edit a file in $EDITOR |

## W^X Policy

The agent can only write to certain files:

| File | Agent Can Write |
|------|-----------------|
| `IDENTITY.md` | ❌ No |
| `SOUL.md` | ❌ No |
| `MEMORY.md` | ✅ Yes |
| `CONTEXT.md` | ✅ Yes |
| `TOOLS.md` | ❌ No |

This prevents the agent from modifying its own core identity.

## Providers

Supported LLM providers:

- **OpenAI** (default): GPT-4o
- **Anthropic**: Claude Sonnet

Set the appropriate API key:
```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

## Portability

Export your agent:
```bash
fdaa export my-agent -o my-agent.zip
```

Import on another machine:
```bash
fdaa import my-agent.zip
fdaa chat my-agent
```

The agent's identity, personality, and memories all travel with it.

## License

MIT

## About

Built by [Substr8 Labs](https://substr8labs.com) as a reference implementation for the [FDAA specification](https://github.com/Substr8-Labs/fdaa-spec).
