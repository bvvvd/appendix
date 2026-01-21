import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DIFF_PATH = Path("diff.txt")
GUIDELINES_PATH = Path("docs/review_guidelines.md")
OUT_MD = Path("review_comment.md")

MODEL = os.environ.get("MODEL", "qwen2.5:14b-instruct")
MAX_DIFF_CHARS = int(os.environ.get("MAX_DIFF_CHARS", "80000"))

# --- Utilities ---

def read_text(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""

def run_ollama(model: str, prompt: str) -> str:
    # Use stdin to avoid CLI arg limits on Windows.
    p = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama failed (code={p.returncode}): {err[:2000]}")
    return p.stdout.decode("utf-8", errors="ignore").strip()

def safe_json_loads(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)

def truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "\n\n[TRUNCATED]\n"

def md_escape(s: str) -> str:
    # Minimal escaping to avoid accidental markdown formatting explosions
    return s.replace("\r", "").strip()

# --- Agent definitions ---

@dataclass
class Agent:
    name: str
    focus: str

AGENTS: List[Agent] = [
    Agent(
        name="Correctness & Reliability",
        focus=(
            "Find correctness risks: idempotency, retries/backoff, timeouts, transactional boundaries, "
            "race conditions, partial failures, error handling, and data consistency."
        ),
    ),
    Agent(
        name="Architecture & Boundaries",
        focus=(
            "Review architecture: module boundaries, coupling, layering, dependency direction, "
            "API contracts, naming of abstractions, and maintainability trade-offs."
        ),
    ),
    Agent(
        name="Tests & Observability",
        focus=(
            "Suggest tests and observability: unit/integration tests worth adding, edge cases, "
            "logging/metrics/tracing, and how to reproduce failures."
        ),
    ),
    Agent(
        name="Cost & LLM Discipline",
        focus=(
            "Look for cost/perf traps: unnecessary LLM calls, large payloads, missing caching, "
            "missing state+delta pattern, and risk of GPU contention."
        ),
    ),
]

# JSON schema agent must output
JSON_SCHEMA = """
Return ONLY valid JSON (no markdown, no extra text).
Schema:
{
  "summary": "string (1-3 sentences)",
  "blocking": [{"issue":"string","evidence":"string","fix":"string"}],
  "non_blocking": [{"issue":"string","evidence":"string","fix":"string"}],
  "tests_to_add": ["string"],
  "questions": ["string"]
}
Rules:
- Use only the provided DIFF and GUIDELINES.
- Do not guess. If uncertain, put it into "questions".
- Keep evidence concrete (file/line hints if visible in diff).
""".strip()

def build_prompt(agent: Agent, guidelines: str, diff: str) -> str:
    return f"""
SYSTEM:
You are a strict senior software engineer acting as a pull request reviewer.

TASK:
{agent.focus}

GUIDELINES:
{guidelines if guidelines else "(none provided)"}

DIFF:
{diff}

OUTPUT INSTRUCTIONS:
{JSON_SCHEMA}
""".strip()

# --- Markdown report building ---

def fmt_issue_list(title: str, issues: List[Dict[str, Any]]) -> str:
    if not issues:
        return f"**{title}:** None ✅\n"
    out = f"**{title}:**\n"
    for it in issues:
        issue = md_escape(str(it.get("issue", "")))
        evidence = md_escape(str(it.get("evidence", "")))
        fix = md_escape(str(it.get("fix", "")))
        out += f"- **{issue}**\n  - Evidence: {evidence}\n  - Fix: {fix}\n"
    return out + "\n"

def fmt_list(title: str, items: List[str]) -> str:
    if not items:
        return f"**{title}:** None\n\n"
    out = f"**{title}:**\n"
    for x in items:
        out += f"- {md_escape(str(x))}\n"
    return out + "\n"

def main() -> None:
    diff = read_text(DIFF_PATH)
    guidelines = read_text(GUIDELINES_PATH)

    if not diff.strip():
        OUT_MD.write_text("### Local Multi-Agent AI Review\n\nNo diff found.\n", encoding="utf-8")
        return

    diff = truncate(diff, MAX_DIFF_CHARS)

    results: List[Tuple[Agent, Optional[Dict[str, Any]], Optional[str], str]] = []
    # tuple: (agent, json, json_error, raw)

    for agent in AGENTS:
        prompt = build_prompt(agent, guidelines, diff)
        raw = run_ollama(MODEL, prompt)
        data, err = safe_json_loads(raw)
        results.append((agent, data, err, raw))

    # Build consolidated markdown
    md = "### Local Multi-Agent AI Review\n\n"
    md += f"Model: `{MODEL}`\n\n"
    if guidelines.strip():
        md += "_Project guidelines were provided._\n\n"

    # High-level rollup
    md += "## Rollup\n\n"
    rollup_lines = []
    for agent, data, err, raw in results:
        if data:
            s = md_escape(str(data.get("summary", "")))
            rollup_lines.append(f"- **{agent.name}:** {s}")
        else:
            rollup_lines.append(f"- **{agent.name}:** ⚠️ invalid JSON output ({err})")
    md += "\n".join(rollup_lines) + "\n\n"

    # Detailed sections
    for agent, data, err, raw in results:
        md += f"---\n\n## {agent.name}\n\n"
        if not data:
            md += "**⚠️ Agent returned invalid JSON.**\n\n"
            md += "Raw output (truncated):\n\n```text\n"
            md += truncate(raw, 3000)
            md += "\n```\n\n"
            continue

        md += f"**Summary:** {md_escape(str(data.get('summary','')))}\n\n"
        md += fmt_issue_list("Blocking", data.get("blocking", []) or [])
        md += fmt_issue_list("Non-blocking", data.get("non_blocking", []) or [])
        md += fmt_list("Tests to add", data.get("tests_to_add", []) or [])
        md += fmt_list("Questions", data.get("questions", []) or [])

    OUT_MD.write_text(md, encoding="utf-8")

if __name__ == "__main__":
    main()
