"""
Code Assistant Agent
---------------------
A minimal, self-contained AI agent that:
  1. Takes a plain-English coding assignment
  2. Writes a Python solution to disk
  3. Runs it inside an isolated Docker container (--network none, memory-capped)
  4. Reads the output/errors back
  5. Loops (fix -> re-run) until the code works, or gives up after N tries

Works with ANY OpenAI-compatible chat completions endpoint - by default it
points at a local Ollama server, but you can swap in Groq, OpenAI, etc. by
changing BASE_URL / MODEL / API_KEY below (or via environment variables).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration - defaults to local Ollama, override with env vars if needed
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-coder:7b")
API_KEY = os.environ.get("AGENT_API_KEY", "ollama")
WORKDIR = Path(os.environ.get("AGENT_WORKDIR", "./workspace"))
MAX_TURNS = 12

# FIX: clear any leftover files from a previous run before starting.
# Without this, a failed write_file this run can silently leave a STALE
# file from an earlier task in place - run_python then executes the old
# file and reports success/failure that has nothing to do with the
# current assignment. Always start from a clean workspace.
if WORKDIR.exists():
    shutil.rmtree(WORKDIR)
WORKDIR.mkdir(parents=True, exist_ok=True)


def write_file(filename: str, content: str) -> str:
    path = WORKDIR / filename
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {filename}"


def run_python(filename: str) -> str:
    path = WORKDIR / filename
    if not path.exists():
        return f"ERROR: {filename} does not exist yet. Write it first."

    absolute_workspace = str(WORKDIR.resolve())

    docker_command = [
        "docker", "run", "--rm",
        "-v", f"{absolute_workspace}:/workspace",
        "--memory", "256m",
        "--network", "none",
        "agent-sandbox",
        "python", filename,
    ]

    try:
        result = subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return f"exit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "ERROR: script timed out after 10 seconds. The sandboxed container was forced to terminate."
    except Exception as e:
        return f"ERROR spinning up Docker sandbox: {str(e)}"


TOOL_IMPLEMENTATIONS = {
    "write_file": write_file,
    "run_python": run_python,
}


def call_model(messages):
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def extract_fallback_tool_call(content: str, turn: int):
    """Parse a model's plain-text JSON tool call, tolerating markdown
    code fences (```json ... ```) that models sometimes add despite
    being told not to."""
    content = content.strip()

    # FIX: strip markdown code fences before attempting to parse.
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()

    parsed = None
    if content.startswith("{") and '"name"' in content and '"arguments"' in content:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    parsed = None

    if parsed and "name" in parsed and "arguments" in parsed:
        return {
            "id": f"fallback_{turn}",
            "type": "function",
            "function": {
                "name": parsed["name"],
                "arguments": (
                    json.dumps(parsed["arguments"])
                    if isinstance(parsed["arguments"], dict)
                    else parsed["arguments"]
                ),
            },
        }
    return None


def run_agent(assignment: str):
    print(f"\n=== Assignment: {assignment} ===\n")

    tool_instructions = (
        "You are a careful Python programmer working in a sandboxed workspace.\n"
        "You must respond ONLY with a raw JSON object matching one of these tool calls. "
        "Do not wrap your answer in markdown code blocks like ```json or add text around it.\n\n"
        "Available Tools:\n"
        "1. Write a file:\n"
        '{"name": "write_file", "arguments": {"filename": "main.py", "content": "YOUR CODE HERE"}}\n\n'
        "2. Run a Python script:\n"
        '{"name": "run_python", "arguments": {"filename": "main.py"}}\n\n'
        "3. Complete the assignment (Only call this after verifying the printed stdout output "
        "actually matches what the assignment asked for - not just that it ran without error):\n"
        '{"name": "task_complete", "arguments": {"summary": "A short recap verifying the code runs successfully."}}\n\n'
        "Workflow: Always write the code first, run it using run_python, and only call task_complete "
        "after you have read the stdout and confirmed it genuinely answers the assignment."
    )

    messages = [
        {"role": "system", "content": tool_instructions},
        {"role": "user", "content": assignment},
    ]

    last_run_stdout = None  # tracks the most recent run_python output for grounding checks

    for turn in range(1, MAX_TURNS + 1):
        print(f"--- Turn {turn} ---")
        message = call_model(messages)

        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            content = (message.get("content") or "").strip()
            fallback = extract_fallback_tool_call(content, turn)

            if fallback:
                tool_calls = [fallback]
                message["tool_calls"] = tool_calls
                message["content"] = None
            else:
                messages.append(message)
                print(f"(model said, no tool call): {content}")
                messages.append(
                    {
                        "role": "user",
                        "content": "Please respond by calling one of your tools (write_file, run_python, or task_complete) using proper tool formatting.",
                    }
                )
                continue

        messages.append(message)

        for call in tool_calls:
            name = call["function"]["name"]
            call_id = call.get("id", f"call_{turn}")
            try:
                args = (
                    json.loads(call["function"]["arguments"])
                    if isinstance(call["function"]["arguments"], str)
                    else call["function"]["arguments"]
                )
            except json.JSONDecodeError:
                args = {}

            print(f"-> model called: {name}({args})")

            if name == "task_complete":
                # FIX: don't just trust the model's claim of success - check
                # that the most recent run_python call actually produced
                # non-empty stdout. A model can (and did, in testing) say
                # "this didn't actually work" in its own summary and still
                # call task_complete anyway. Ground the decision in real
                # execution output instead of the model's self-report.
                if not last_run_stdout or not last_run_stdout.strip():
                    print(
                        "\n⚠️ Rejected task_complete: the last run_python call produced no "
                        "visible output. Forcing the agent to keep working."
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": (
                                "REJECTED: your last run_python call produced empty stdout, "
                                "meaning the assignment's output was not actually demonstrated. "
                                "Do NOT explain the fix in plain text. Call the write_file tool "
                                "right now with corrected code (make sure results are actually "
                                "printed using print(), not just returned or evaluated bare), "
                                "then call run_python again to verify before calling task_complete."
                            ),
                        }
                    )
                    continue
                print(f"\n✅ Agent finished: {args.get('summary')}")
                return True

            if name in TOOL_IMPLEMENTATIONS:
                impl = TOOL_IMPLEMENTATIONS[name]
                try:
                    result = impl(**args)
                except Exception as e:
                    result = f"ERROR executing tool: {str(e)}"
                print(f"   [Tool Output]: {result[:300]}")
                if name == "run_python":
                    # capture just the stdout portion for the grounding check above
                    stdout_marker = "stdout:\n"
                    stderr_marker = "\nstderr:"
                    if stdout_marker in result and stderr_marker in result:
                        last_run_stdout = result.split(stdout_marker, 1)[1].split(stderr_marker, 1)[0]
                    else:
                        last_run_stdout = None
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": result})
            else:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": f"ERROR: Tool '{name}' is not supported. Use write_file, run_python, or task_complete.",
                    }
                )

    print("\n⚠️ Reached max turns without the agent calling task_complete.")
    return False


if __name__ == "__main__":
    assignment = " ".join(sys.argv[1:]) or "Write a Python program that prints the first 10 Fibonacci numbers."
    run_agent(assignment)
