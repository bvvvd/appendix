import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

DIFF_PATH = Path("diff.txt")
GUIDELINES_PATH = Path("docs/review_guidelines.md")
OUT_MD = Path("review_comment.md")

MODEL = os.environ.get("MODEL", "qwen2.5:14b-instruct")
# Optional: use a cheaper/faster model for the curator step
CURATOR_MODEL = os.environ.get("CURATOR_MODEL", MODEL)

MAX_DIFF_CHARS = int(os.environ.get("MAX_DIFF_CHARS", "80000"))
MAX_PREV_REVIEW_CHARS = int(os.environ.get("MAX_PREV_REVIEW_CHARS", "12000"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
PREV_SHA = (os.environ.get("PREV_SHA") or "").strip()
PREV_REVIEW_PATH = Path(os.environ.get("PREV_REVIEW_PATH", "prev_review.txt"))  # optional


# --- Utilities ---

def read_text(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def git(*args: str) -> str:
    p = subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"git {' '.join(args)} failed: {err[:2000]}")
    return p.stdout.decode("utf-8", errors="ignore").strip()


def get_head_sha() -> str:
    return git("rev-parse", "HEAD")


def truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "\n\n[TRUNCATED]\n"


def md_escape(s: str) -> str:
    # Minimal escaping to avoid accidental markdown formatting explosions
    return s.replace("\r", "").strip()


def run_ollama(model: str, prompt: str, timeout_s: int = 300) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # If you want to aggressively unload after each call:
        # "keep_alive": 0,
    }

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(body)
            return data.get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama HTTP call failed: {e}")


def safe_json_loads(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)


def cap_list(x: Any, max_items: int) -> List[Any]:
    if not isinstance(x, list):
        return []
    return x[:max_items]


def cap_issues(x: Any, max_items: int) -> List[Dict[str, Any]]:
    if not isinstance(x, list):
        return []
    out = []
    for it in x[:max_items]:
        if isinstance(it, dict):
            out.append(it)
    return out


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


# JSON schema for each agent (strict + short)
JSON_SCHEMA = """
Return ONLY valid JSON. No markdown. No explanations. No extra keys.

Schema:
{
  "summary": "string (max 2 sentences)",
  "blocking": [{"issue":"string","evidence":"string","fix":"string"}],
  "non_blocking": [{"issue":"string","evidence":"string","fix":"string"}],
  "tests_to_add": ["string"],
  "questions": ["string"]
}

Hard limits:
- blocking: max 2 items
- non_blocking: max 3 items
- tests_to_add: max 6 items
- questions: max 5 items

Rules:
- Use only the provided DIFF and GUIDELINES.
- Do not guess. If uncertain, put it into "questions" and set evidence to "unknown".
- Keep evidence concrete (file/line hints if visible in diff).
""".strip()


CURATOR_SCHEMA = """
Return ONLY valid JSON. No markdown. No explanations. No extra keys.

Schema:
{
  "short_summary": ["string (max 3 bullets)"],
  "top_actions": ["string (max 5 items)"],
  "resolved": ["string (max 5 items)"],
  "still_open": ["string (max 5 items)"],
  "new_risks": ["string (max 5 items)"]
}

Rules:
- Deduplicate and prioritize.
- Prefer concrete, actionable items.
- If evidence is weak/unknown, downgrade or omit it.
- If mode is INITIAL: resolved/still_open can be empty.
- If mode is FOLLOW_UP: classify items as resolved/still_open/new_risks when possible.
""".strip()


def build_prompt(agent: Agent, guidelines: str, diff: str, mode: str, prev_sha: str, head_sha: str) -> str:
    followup_hint = ""
    if mode == "FOLLOW_UP":
        followup_hint = f"""
FOLLOW-UP CONTEXT:
- This is a follow-up review after previous AI feedback.
- Previous reviewed HEAD SHA was: {prev_sha}
- Current HEAD SHA is: {head_sha}
- The DIFF is expected to contain ONLY changes since the previous review.
- Focus on what changed, what got resolved, and what new issues were introduced.
""".strip()

    return f"""
SYSTEM:
You are a strict senior/staff+ software engineer acting as a pull request reviewer.

MODE: {mode}
{followup_hint}

TASK:
{agent.focus}

GUIDELINES:
{guidelines if guidelines else "(none provided)"}

DIFF:
{diff}

OUTPUT INSTRUCTIONS:
{JSON_SCHEMA}
""".strip()


def build_curator_prompt(
        mode: str,
        prev_sha: str,
        head_sha: str,
        agent_json: List[Dict[str, Any]],
        prev_review_text: str,
) -> str:
    payload = {
        "mode": mode,
        "previous_reviewed_sha": prev_sha or None,
        "current_sha": head_sha,
        "agent_reviews": agent_json,
        "previous_review_text": truncate(prev_review_text, MAX_PREV_REVIEW_CHARS) if prev_review_text else "",
    }

    return f"""
SYSTEM:
You are a lead reviewer (staff+). Merge multiple agent reviews into a short, actionable PR comment.

INPUT (JSON):
{json.dumps(payload, ensure_ascii=False)}

OUTPUT INSTRUCTIONS:
{CURATOR_SCHEMA}
""".strip()


# --- Markdown building ---

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


def fmt_bullets(items: List[str]) -> str:
    if not items:
        return "- None\n"
    return "".join([f"- {md_escape(str(x))}\n" for x in items])


def main() -> None:
    diff = read_text(DIFF_PATH)
    guidelines = read_text(GUIDELINES_PATH)
    prev_review_text = read_text(PREV_REVIEW_PATH)

    if not diff.strip():
        OUT_MD.write_text("### Local Multi-Agent AI Review\n\nNo diff found.\n", encoding="utf-8")
        return

    diff = truncate(diff, MAX_DIFF_CHARS)

    head_sha = get_head_sha()
    mode = "FOLLOW_UP" if PREV_SHA else "INITIAL"

    # Run specialist agents
    results: List[Tuple[Agent, Optional[Dict[str, Any]], Optional[str], str]] = []
    agent_payloads: List[Dict[str, Any]] = []

    for agent in AGENTS:
        prompt = build_prompt(agent, guidelines, diff, mode, PREV_SHA, head_sha)
        raw = run_ollama(MODEL, prompt)
        data, err = safe_json_loads(raw)

        # Normalize & cap to keep things tight even if model violates limits
        if data:
            data["summary"] = str(data.get("summary", ""))[:500]
            data["blocking"] = cap_issues(data.get("blocking"), 2)
            data["non_blocking"] = cap_issues(data.get("non_blocking"), 3)
            data["tests_to_add"] = cap_list(data.get("tests_to_add"), 6)
            data["questions"] = cap_list(data.get("questions"), 5)

            agent_payloads.append({
                "agent": agent.name,
                "summary": data.get("summary", ""),
                "blocking": data.get("blocking", []),
                "non_blocking": data.get("non_blocking", []),
                "tests_to_add": data.get("tests_to_add", []),
                "questions": data.get("questions", []),
            })

        results.append((agent, data, err, raw))

    # Curator step
    curator_prompt = build_curator_prompt(mode, PREV_SHA, head_sha, agent_payloads, prev_review_text)
    curator_raw = run_ollama(CURATOR_MODEL, curator_prompt)
    curator, curator_err = safe_json_loads(curator_raw)

    if curator:
        curator["short_summary"] = cap_list(curator.get("short_summary"), 3)
        curator["top_actions"] = cap_list(curator.get("top_actions"), 5)
        curator["resolved"] = cap_list(curator.get("resolved"), 5)
        curator["still_open"] = cap_list(curator.get("still_open"), 5)
        curator["new_risks"] = cap_list(curator.get("new_risks"), 5)
    else:
        curator = {
            "short_summary": [f"⚠️ Curator returned invalid JSON: {curator_err}"],
            "top_actions": [],
            "resolved": [],
            "still_open": [],
            "new_risks": [],
        }

    # Build markdown
    title = "Local Multi-Agent AI Review"
    title += " (Follow-up)" if mode == "FOLLOW_UP" else " (Initial)"

    md = f"### {title}\n\n"
    md += f"Models: specialists=`{MODEL}`, curator=`{CURATOR_MODEL}`\n\n"
    md += f"Current HEAD: `{head_sha}`\n"
    if PREV_SHA:
        md += f"Previous reviewed HEAD: `{PREV_SHA}`\n"
    md += "\n"

    if guidelines.strip():
        md += "_Project guidelines were provided._\n\n"

    md += "## Curated summary\n\n"
    md += fmt_bullets(curator.get("short_summary", [])) + "\n"

    md += "### Top actions\n\n"
    md += fmt_bullets(curator.get("top_actions", [])) + "\n"

    if mode == "FOLLOW_UP":
        md += "### Resolved\n\n"
        md += fmt_bullets(curator.get("resolved", [])) + "\n"

        md += "### Still open\n\n"
        md += fmt_bullets(curator.get("still_open", [])) + "\n"

        md += "### New risks\n\n"
        md += fmt_bullets(curator.get("new_risks", [])) + "\n"

    # Keep details collapsible to reduce noise in PR
    md += "<details>\n<summary>Agent details</summary>\n\n"

    md += "## Rollup\n\n"
    rollup_lines = []
    for agent, data, err, raw in results:
        if data:
            s = md_escape(str(data.get("summary", "")))
            rollup_lines.append(f"- **{agent.name}:** {s}")
        else:
            rollup_lines.append(f"- **{agent.name}:** ⚠️ invalid JSON output ({err})")
    md += "\n".join(rollup_lines) + "\n\n"

    for agent, data, err, raw in results:
        md += f"---\n\n## {agent.name}\n\n"
        if not data:
            md += "**⚠️ Agent returned invalid JSON.**\n\n"
            md += "Raw output (truncated):\n\n```text\n"
            md += truncate(raw, 3000)
            md += "\n```\n\n"
            continue

        md += f"**Summary:** {md_escape(str(data.get('summary', '')))}\n\n"
        md += fmt_issue_list("Blocking", data.get("blocking", []) or [])
        md += fmt_issue_list("Non-blocking", data.get("non_blocking", []) or [])
        md += fmt_list("Tests to add", data.get("tests_to_add", []) or [])
        md += fmt_list("Questions", data.get("questions", []) or [])

    # Curator raw (optional debug)
    md += "\n---\n\n## Curator (debug)\n\n"
    if curator_raw:
        md += "```json\n" + truncate(curator_raw, 3000) + "\n```\n"

    md += "\n</details>\n\n"

    # Stable marker for next run to pick up
    md += "---\n\n"
    md += f"<!-- AI_REVIEW:HEAD_SHA={head_sha} -->\n"

    OUT_MD.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
