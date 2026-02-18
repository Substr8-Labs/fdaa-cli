# FDAA CLI

**File-Driven Agent Architecture — Reference Implementation**

Build, verify, sign, and publish AI agent skills with cryptographic proof.

📄 [Whitepaper](https://github.com/Substr8-Labs/fdaa-spec) | 🔐 [ACC Spec](https://github.com/Substr8-Labs/acc-spec) | 🏢 [Substr8 Labs](https://substr8labs.com)

## Installation

```bash
pip install fdaa
```

Requires Python 3.10+

## Quick Start

### For Skill Developers

**1. Create a signing key (one-time)**
```bash
fdaa keygen mykey
```

**2. Write your skill**
```
my-skill/
├── SKILL.md          # What it does
├── scripts/
│   └── run.py        # The code
└── references/       # Supporting docs (optional)
```

**3. Quick check (instant feedback)**
```bash
fdaa check ./my-skill
```

**4. Full verification + signing**
```bash
fdaa pipeline ./my-skill --key mykey
```

**5. Publish**
```bash
fdaa publish ./my-skill --name @you/my-skill --version 1.0.0
```

### For Skill Users

**Install a verified skill**
```bash
fdaa install @someone/cool-skill
```

Signature is verified automatically. Tampered skills are rejected.

---

## Commands

### Verification Pipeline

| Command | Description |
|---------|-------------|
| `fdaa check <path>` | Fast pattern check (~1s) |
| `fdaa verify <path>` | Guard Model security scan |
| `fdaa sandbox <path>` | Run in isolated container |
| `fdaa pipeline <path>` | Full Tier 1-4 verification |

### Signing & Keys

| Command | Description |
|---------|-------------|
| `fdaa keygen <name>` | Generate Ed25519 key pair |
| `fdaa sign <path>` | Sign a skill |

### Registry

| Command | Description |
|---------|-------------|
| `fdaa install <spec>` | Install a skill |
| `fdaa publish <path>` | Publish to registry |
| `fdaa search <query>` | Search skills |
| `fdaa list-skills` | List installed skills |

### Tracing (OpenTelemetry)

| Command | Description |
|---------|-------------|
| `fdaa traced-pipeline <path>` | Run pipeline with tracing |
| `fdaa trace <id>` | View a trace |
| `fdaa trace --list` | List recent traces |

### Agent Workspaces

| Command | Description |
|---------|-------------|
| `fdaa init <name>` | Create agent workspace |
| `fdaa chat <workspace>` | Chat with agent |
| `fdaa files <workspace>` | List files |
| `fdaa read <workspace> <file>` | Read file |
| `fdaa export <workspace>` | Export as zip |
| `fdaa import <zip>` | Import from zip |

---

## Verification Tiers

FDAA uses defense-in-depth with 4 verification tiers:

| Tier | What It Does | Speed |
|------|--------------|-------|
| **1. Fast Pass** | Pattern matching, known signatures | ~100ms |
| **2. Guard Model** | LLM semantic analysis | ~3-5s |
| **3. Sandbox** | Isolated execution, behavior monitoring | ~1-2s |
| **4. Registry** | Cryptographic signing, hash verification | ~100ms |

```bash
# Run all tiers
fdaa pipeline ./my-skill --key mykey

# Skip expensive steps during development
fdaa pipeline ./my-skill --skip-sandbox --skip-sign
```

---

## Tracing & Observability

Every pipeline run can be traced with OpenTelemetry:

```bash
fdaa traced-pipeline ./my-skill

# Output:
# Trace ID: f8112ae3...
# View with: fdaa trace f8112ae3
```

View trace details:
```bash
fdaa trace f8112ae3

# Shows:
# - Duration per tier
# - LLM tokens & cost
# - Sandbox metrics
# - Verification results
```

Export to Jaeger:
```bash
fdaa traced-pipeline ./my-skill --jaeger-host localhost
```

---

## Skill Format

A minimal skill:

```
my-skill/
├── SKILL.md
└── scripts/
    └── run.py
```

**SKILL.md**
```markdown
---
name: my-skill
description: Does something useful
version: 1.0.0
---

# My Skill

Use this skill to do X.

## Usage

\`\`\`bash
my-skill --input foo
\`\`\`
```

After signing, a `MANIFEST.json` is added:
```json
{
  "name": "@you/my-skill",
  "version": "1.0.0",
  "sha256": "7a3f2b...",
  "signature": "Kx8mQ2...",
  "publicKey": "a1b2c3...",
  "signedAt": "2026-02-18T00:00:00Z"
}
```

---

## W^X Security

FDAA enforces Write XOR Execute:

| File | Agent Can Modify |
|------|------------------|
| `SKILL.md` | ❌ No |
| `SOUL.md` | ❌ No |
| `IDENTITY.md` | ❌ No |
| `MEMORY.md` | ✅ Yes |
| `scripts/*` | ❌ No |

Agents cannot modify their own identity or capabilities.

---

## Environment Variables

```bash
# LLM Providers (one required for verify/pipeline)
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Optional
export FDAA_REGISTRY_URL=https://registry.fdaa.dev
export JAEGER_AGENT_HOST=localhost
```

---

## API Server

For web deployments:

```bash
pip install fdaa[server]

export MONGODB_URI="mongodb+srv://..."
export ANTHROPIC_API_KEY="sk-ant-..."

uvicorn fdaa.server:app --host 0.0.0.0 --port 8000
```

See [API docs](docs/API.md) for endpoints.

---

## Examples

**Verify a skill before installing:**
```bash
fdaa verify ./untrusted-skill --provider anthropic
```

**Create and publish a skill:**
```bash
fdaa keygen mykey
mkdir my-skill && cd my-skill
echo "# My Skill" > SKILL.md
fdaa pipeline . --key mykey
fdaa publish . --name @me/my-skill --version 1.0.0
```

**Debug a failing verification:**
```bash
fdaa traced-pipeline ./my-skill
fdaa trace <trace-id>
```

**Install from GitHub (Phase 0 registry):**
```bash
fdaa install github:substr8-labs/skill-code-review
```

---

## License

MIT

---

Built by [Substr8 Labs](https://substr8labs.com)

Research: [FDAA Whitepaper](https://doi.org/10.5281/zenodo.18675147) | [ACC Spec](https://doi.org/10.5281/zenodo.18675149)
