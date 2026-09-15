# Code Assistant Agent

A minimal, framework-free AI agent that writes, runs, and self-corrects Python code based on a plain-English task — built from scratch to understand exactly how agentic tool-calling loops work under the hood, without hiding the mechanics behind a framework like LangChain or CrewAI.

## What it does

Give it a task like *"write a program that prints the first 10 Fibonacci numbers"*, and the agent:

1. **Reasons** about how to solve it
2. **Calls a tool** to write a Python file to disk
3. **Calls a tool** to run that file **inside an isolated Docker container** — not directly on the host machine
4. **Reads the real stdout/stderr** back from the container
5. **Loops**: if it failed, it fixes the code and tries again; if it succeeded, it calls `task_complete` to finish

This is the same core pattern used by production coding agents (GitHub Copilot Workspace, Devin, Claude Code, etc.) — an LLM in a loop with tools, checking its own work against real execution results instead of just guessing.

## Why build it this way

Most agent tutorials wrap this loop inside a framework, which is great for production but hides how it actually works. This project implements the loop directly against the OpenAI-compatible `chat/completions` API so every part is visible and explainable:

- **Tool schema** — exactly 3 tools (`write_file`, `run_python`, `task_complete`), deliberately kept small so the model doesn't get lost picking between dozens of options
- **The loop** — send conversation → model picks a tool → we execute it for real → feed the result back → repeat
- **Prompt-based tool calling** — instead of relying on each provider's native `tools`/`tool_calls` API parameter (which turned out to have real, provider-specific bugs — see below), the system prompt tells the model the exact JSON shape to reply with, and the agent parses and normalizes that response itself. This makes the agent portable across providers with inconsistent native function-calling support.
- **Real sandboxing** — generated code runs in a throwaway Docker container (`agent-sandbox`) with `--network none` (no internet access from inside), a `--memory` cap, and a hard timeout, rather than trusting the host machine with arbitrary AI-generated code
- **Grounding** — the agent isn't allowed to claim success; it must actually see real `stdout`/`stderr` from the container before it's permitted to call `task_complete`

## Running it

Works with any OpenAI-compatible LLM endpoint — local (Ollama) or cloud (Groq, OpenAI, etc.). Requires Docker installed and running, since code execution happens inside a container.

### 1. Build the sandbox image (one-time setup)

```bash
docker build -t agent-sandbox .
```

The `Dockerfile` pre-installs common libraries (`pandas`, `numpy`, `requests`, `matplotlib`, `python-dateutil`) at build time, since the container has no network access at *run* time — anything the agent's generated code needs to import has to already be baked into the image.

### 2. Point the agent at an LLM

**Option A — Local, 100% free (Ollama)**
```bash
ollama pull qwen2.5-coder:7b
ollama serve
python agent.py "write a program that prints the first 10 Fibonacci numbers"
```

**Option B — Cloud (Groq, free tier)**
```bash
export AGENT_BASE_URL="https://api.groq.com/openai/v1"
export AGENT_MODEL="llama-3.3-70b-versatile"
export AGENT_API_KEY="your_groq_key"
python agent.py "write a program that reverses a string without using [::-1]"
```

## Example run

```
=== Assignment: write a program that prints the first 10 Fibonacci numbers ===

--- Turn 1 ---
-> model called: write_file({'filename': 'main.py', 'content': '...'})
   [Tool Output]: Wrote 210 characters to main.py

--- Turn 2 ---
-> model called: run_python({'filename': 'main.py'})
   [Tool Output]: exit_code: 0
stdout:
0 1 1 2 3 5 8 13 21 34

--- Turn 3 ---
-> model called: task_complete({'summary': 'Wrote and verified a Fibonacci program...'})

✅ Agent finished: Wrote and verified a Fibonacci program that correctly prints the first 10 numbers.
```

## Project structure

```
code-agent/
├── agent.py        # The full agent loop (~200 lines)
├── Dockerfile      # Defines the isolated execution sandbox
├── workspace/      # Where the agent writes generated code (mounted into the container)
└── README.md
```

## What I learned building this

- How LLM "function calling" actually works at the API level (tool schemas, `tool_calls`, feeding results back as `role: tool` messages) — and how to reimplement the same contract via prompting when a provider's native support is inconsistent
- Why smaller/local models struggle with large tool lists, and how narrowing the toolset improves reliability
- The importance of **grounding** an agent's self-reported success in real execution output, not just trusting the model's claim
- Why letting an LLM-driven agent execute arbitrary generated code directly on the host is a real risk, and how container isolation (`--network none`, memory limits, `--rm`) closes that gap
- Debugging real integration issues across multiple LLM providers (Gemini, Groq, local Ollama) — schema validation differences, deprecated models, and provider-specific tool-calling quirks

## Author

Heena Shaikh — [GitHub](https://github.com/HeenaShaikh09) · [LinkedIn](https://www.linkedin.com/in/heenashaikh077/)
