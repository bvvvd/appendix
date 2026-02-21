import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

DIFF_PATH = Path("diff.txt")
GUIDELINES_PATH = Path("docs/review_guidelines.md")
OUT_MD = Path("review_comment.md")

MODEL = os.environ.get("MODEL", "qwen2.5:14b-instruct")
CURATOR_MODEL = os.environ.get("CURATOR_MODEL", MODEL)

MAX_DIFF_CHARS = int(os.environ.get("MAX_DIFF_CHARS", "80000"))
MAX_PREV_REVIEW_CHARS = int(os.environ.get("MAX_PREV_REVIEW_CHARS", "12000"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
PREV_SHA = (os.environ.get("PREV_SHA") or "").strip()
PREV_REVIEW_PATH = Path(os.environ.get("PREV_REVIEW_PATH", "prev_review.txt"))  # optional

SHOW_DEBUG = (os.environ.get("SHOW_DEBUG") or "").lower() in ("1", "true", "yes")
SHOW_AGENT_TESTS_QUESTIONS = (os.environ.get("SHOW_AGENT_TESTS_QUESTIONS") or "").lower() in ("1", "true", "yes")
RETRY_ON_BAD_JSON = (os.environ.get("RETRY_ON_BAD_JSON") or "1").lower() in ("1", "true", "yes")  # Default to True

# Optional: aggressively prune vague summaries/top_actions
STRICT_FACT_GATING = (os.environ.get("STRICT_FACT_GATING") or "").lower() in ("1", "true", "yes")

def log(msg: str) -> None:
    """Print to stderr for debugging."""
    print(f"[REVIEW] {msg}", file=sys.stderr)


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
    return s.replace("\r", "").strip()


def run_ollama(model: str, prompt: str, timeout_s: int = 300) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # "keep_alive": 0,  # uncomment to unload model after each call
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
            response = data.get("response", "").strip()
            if not response:
                log(f"Warning: Empty response from Ollama model {model}")
            return response
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama connection failed (is Ollama running at {OLLAMA_URL}?): {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ollama returned invalid JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama HTTP call failed: {e}")


def sanitize_jsonish(s: str) -> str:
    """
    Normalize common model output issues:
    - smart quotes -> ascii quotes
    - odd apostrophes -> ascii apostrophes
    """
    if not s:
        return s
    s = s.replace("“", '"').replace("”", '"').replace("„", '"')
    s = s.replace("’", "'").replace("‘", "'")
    return s


def extract_first_json_object(s: str) -> Optional[str]:
    if not s:
        return None

    stripped = s.strip()

    # Strip a leading markdown fence if present
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            # Check for language identifier (json, jsonl, etc.)
            first_line = lines[0].lower()
            if "json" in first_line or first_line.strip() == "```":
                lines = lines[1:]  # drop first fence line
            else:
                # Not a JSON code block, might be text
                pass
        
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    # Try to find JSON object
    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start:i + 1]

    return None


def safe_json_loads(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not s or not s.strip():
        return None, "Empty response"
    
    try:
        s = sanitize_jsonish(s)
        js = extract_first_json_object(s)
        if not js:
            # Try parsing the whole string as JSON (maybe it's already clean)
            try:
                return json.loads(s.strip()), None
            except:
                return None, "No JSON object found in response"
        js = sanitize_jsonish(js)
        parsed = json.loads(js)
        if not isinstance(parsed, dict):
            return None, f"JSON is not an object (got {type(parsed).__name__})"
        return parsed, None
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {str(e)[:200]}"
    except Exception as e:
        return None, str(e)[:200]


def cap_list(x: Any, max_items: int) -> List[Any]:
    if not isinstance(x, list):
        return []
    return x[:max_items]


def cap_issues(x: Any, max_items: int) -> List[Dict[str, Any]]:
    if not isinstance(x, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in x[:max_items]:
        if isinstance(it, dict):
            out.append(it)
    return out


def extract_changed_files(diff: str) -> List[str]:
    files: List[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b = parts[3]  # b/...
                if b.startswith("b/"):
                    files.append(b[2:])

    seen = set()
    out: List[str] = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def filter_allowed_files(files: List[str]) -> List[str]:
    dropped_prefixes = ("docs/",)
    filtered = [f for f in files if not f.startswith(dropped_prefixes)]
    return filtered if filtered else files


def drop_weak_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in issues:
        ev = str(it.get("evidence", "")).strip().lower()
        if not ev or ev == "unknown":
            continue
        out.append(it)
    return out


def drop_issues_not_in_files(issues: List[Dict[str, Any]], allowed_files: List[str]) -> List[Dict[str, Any]]:
    if not allowed_files:
        return issues

    allowed_lower = [f.lower() for f in allowed_files]
    out: List[Dict[str, Any]] = []
    for it in issues:
        ev = str(it.get("evidence", "")).lower()
        if any(f in ev for f in allowed_lower):
            out.append(it)
    return out


def must_mention_file(line: str, allowed_files: List[str]) -> bool:
    s = (line or "").lower()
    return any(f.lower() in s for f in allowed_files)


def mentions_only_allowed_files(line: str, allowed_files: List[str]) -> bool:
    """
    Conservative: if line mentions a path with a known extension,
    ensure every mentioned path is within allowed_files.
    """
    import re
    paths = re.findall(r"[\w./-]+\.(?:py|yml|yaml|kt|java|md)", line or "")
    if not paths:
        return True
    allowed_set = set(allowed_files)
    return all(p in allowed_set for p in paths)


def diff_facts(diff: str, allowed_files: List[str], max_lines: int = 60) -> str:
    files = "\n".join(allowed_files) if allowed_files else "(none)"
    head = "\n".join(diff.splitlines()[:max_lines])
    return f"CHANGED_FILES:\n{files}\n\nDIFF_HEAD (first {max_lines} lines):\n{head}"


# --- Fact gating (reduce "smart generic" talk) ---

KEYWORD_GATES: Dict[str, List[str]] = {
    "idempot": ["idempot", "dedup", "at-least-once", "exactly-once"],
    "retry": ["retry", "backoff", "exponential"],
    "timeout": ["timeout", "time out", "deadline"],
    "logging": ["logger", "log.", "logging", "trace", "span", "metric"],
    "external": ["http", "request", "client", "api", "url", "fetch"],
    "concurrency": ["mutex", "lock", "synchronized", "race", "concurrent", "coroutine", "thread"],
}

def diff_contains_any(diff_lower: str, needles: List[str]) -> bool:
    return any(n in diff_lower for n in needles)

def gate_claims_to_diff(text: str, diff_lower: str) -> bool:
    """
    If the text talks about a gated topic, require the diff to contain related lexemes.
    """
    t = (text or "").lower()
    for topic, needles in KEYWORD_GATES.items():
        if topic in t or any(n in t for n in needles):
            if not diff_contains_any(diff_lower, needles):
                return False
    return True


def filter_bullets_by_fact_gate(items: List[str], diff_lower: str, allowed_files: List[str]) -> List[str]:
    out: List[str] = []
    for x in items:
        s = str(x).strip()
        if not s:
            continue
        
        # If no allowed files, skip file-based filtering
        if allowed_files:
            # Relaxed: allow items that mention files OR are general enough
            if not must_mention_file(s, allowed_files):
                # Still allow if it doesn't mention any specific files (general advice)
                if mentions_only_allowed_files(s, allowed_files):
                    # It mentions files but they're not in allowed_files - skip
                    continue
            elif not mentions_only_allowed_files(s, allowed_files):
                # Mentions allowed file but also non-allowed files - skip
                continue
        
        # Fact gating only applies if STRICT_FACT_GATING is enabled
        if STRICT_FACT_GATING and not gate_claims_to_diff(s, diff_lower):
            continue
        
        out.append(s)
    
    # If everything was filtered but we have items, return at least one
    if not out and items:
        # Return the first item that's not empty, even if it doesn't pass all filters
        for x in items:
            s = str(x).strip()
            if s:
                out.append(s)
                break
    
    return out


def agent_summary_ok(summary: str, diff_lower: str, allowed_files: List[str]) -> bool:
    s = (summary or "").strip()
    if not s:
        return False
    if STRICT_FACT_GATING and not gate_claims_to_diff(s, diff_lower):
        return False
    if allowed_files and not must_mention_file(s, allowed_files):
        return False
    if not mentions_only_allowed_files(s, allowed_files):
        return False
    return True


def rewrite_agent_summary_if_vague(summary: str, diff_lower: str, allowed_files: List[str]) -> str:
    s = (summary or "").strip()
    if not s:
        return "Reviewed changes but found no specific issues to report."
    
    # Only rewrite if STRICT_FACT_GATING is enabled
    if STRICT_FACT_GATING and not agent_summary_ok(s, diff_lower, allowed_files):
        if allowed_files:
            return f"Reviewed changes in {', '.join(allowed_files[:2])}{'...' if len(allowed_files) > 2 else ''}."
        return "Reviewed changes but found no specific issues to report."
    return s


def extract_relevant_guidelines(guidelines: str, diff: str, max_lines: int = 80) -> str:
    """
    Reduce guideline "prompt bleed" by only including relevant snippets.
    - If diff doesn't trigger any keywords, include only a tiny header (first ~20 lines).
    - Otherwise, include up to max_lines lines that match triggered keywords.
    """
    if not guidelines.strip():
        return ""
    diff_lower = diff.lower()
    triggers: List[str] = []
    for needles in KEYWORD_GATES.values():
        for n in needles:
            if n in diff_lower:
                triggers.append(n)
    triggers = list(dict.fromkeys(triggers))

    lines = guidelines.splitlines()

    if not triggers:
        return "\n".join(lines[:min(len(lines), 20)])

    picked: List[str] = []
    for ln in lines:
        l = ln.lower()
        if any(k in l for k in triggers):
            picked.append(ln)
            if len(picked) >= max_lines:
                break

    return "\n".join(picked) if picked else "\n".join(lines[:min(len(lines), 20)])


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
            "race conditions, partial failures, error handling, and data consistency. "
            "Focus ONLY on code/workflow changes in ALLOWED_FILES."
        ),
    ),
    Agent(
        name="Architecture & Boundaries",
        focus=(
            "Review architecture: module boundaries, coupling, layering, dependency direction, "
            "API contracts, naming of abstractions, and maintainability trade-offs. "
            "Focus ONLY on code/workflow changes in ALLOWED_FILES."
        ),
    ),
    Agent(
        name="Tests & Observability",
        focus=(
            "Suggest tests and observability: unit/integration tests worth adding, edge cases, "
            "logging/metrics/tracing, and how to reproduce failures. "
            "Do NOT review the guideline document itself; review only ALLOWED_FILES changes."
        ),
    ),
    Agent(
        name="Cost & LLM Discipline",
        focus=(
            "Look for cost/perf traps: unnecessary LLM calls, large payloads, missing caching, "
            "missing state+delta pattern, and risk of GPU contention. "
            "Focus ONLY on the actual automation code/workflow changes in ALLOWED_FILES."
        ),
    ),
]


JSON_SCHEMA = """
You MUST return valid JSON only. Start with { and end with }. No markdown code fences, no explanations before or after.

Required JSON schema:
{
  "summary": "string (1-2 sentences describing your review findings)",
  "blocking": [{"issue":"string","evidence":"string","fix":"string"}],
  "non_blocking": [{"issue":"string","evidence":"string","fix":"string"}],
  "tests_to_add": ["string"],
  "questions": ["string"]
}

Limits:
- blocking: maximum 2 items
- non_blocking: maximum 3 items  
- tests_to_add: maximum 6 items
- questions: maximum 5 items

Rules:
- DIFF contains the actual code changes. Use it as your primary source.
- GUIDELINES help you evaluate what you see in the DIFF.
- Only reference files from ALLOWED_FILES list.
- If you find no issues, return empty arrays but still provide a summary.
- Evidence should reference specific lines/files from the DIFF when possible.
- If evidence is unclear, use "unknown" for evidence field.

Example valid response:
{"summary":"Reviewed the changes. Found one potential issue with error handling.","blocking":[],"non_blocking":[{"issue":"Missing error handling","evidence":"EventStore.kt line 45","fix":"Add try-catch around database operation"}],"tests_to_add":[],"questions":[]}
""".strip()


CURATOR_SCHEMA = """
You MUST return valid JSON only. Start with { and end with }. No markdown code fences, no explanations.

Required JSON schema:
{
  "short_summary": ["string (max 2 bullets)"],
  "top_actions": ["string (max 3 items)"],
  "resolved": ["string (max 5 items)"],
  "still_open": ["string (max 5 items)"],
  "new_risks": ["string (max 5 items)"]
}

Rules:
- DIFF contains the actual code changes.
- Only mention files from the `allowed_files` array.
- short_summary should summarize what changed (mention files if possible).
- top_actions should be concrete, actionable items.
- If no issues found, still provide a summary like "Changed: <files> - looks good" or "No issues identified".
- For follow-up reviews, use resolved/still_open/new_risks to track changes.
- Each item should be specific and actionable, not generic.

Example valid response:
{"short_summary":["Changed: EventStore.kt - added error handling"],"top_actions":["Verify error handling covers all database operations"],"resolved":[],"still_open":[],"new_risks":[]}
""".strip()


def build_prompt(
        agent: Agent,
        guidelines: str,
        diff: str,
        mode: str,
        prev_sha: str,
        head_sha: str,
        allowed_files: List[str],
) -> str:
    followup_hint = ""
    if mode == "FOLLOW_UP":
        followup_hint = f"""
FOLLOW-UP CONTEXT:
- Previous reviewed HEAD SHA was: {prev_sha}
- Current HEAD SHA is: {head_sha}
- The DIFF is expected to contain ONLY changes since the previous review.
- Focus on what changed, what got resolved, and what new issues were introduced.
""".strip()

    allowed_files_text = "\n".join(allowed_files) if allowed_files else "(none)"
    facts = diff_facts(diff, allowed_files, max_lines=60)
    relevant_guidelines = extract_relevant_guidelines(guidelines, diff)

    return f"""
SYSTEM:
You are a strict senior/staff+ software engineer acting as a pull request reviewer.

MODE: {mode}
{followup_hint}

ALLOWED_FILES (critical):
{allowed_files_text}

FACTS_FROM_DIFF (use for factual claims):
{facts}

Critical rules:
- DIFF is the ONLY source of facts. GUIDELINES are for judging what you see in DIFF.
- You may ONLY reference file paths from ALLOWED_FILES.
- You MUST NOT mention technologies/files not present in DIFF.

TASK:
{agent.focus}

GUIDELINES (evaluation only, trimmed):
{relevant_guidelines if relevant_guidelines else "(none provided)"}

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
        allowed_files: List[str],
        diff: str,
) -> str:
    payload = {
        "mode": mode,
        "previous_reviewed_sha": prev_sha or None,
        "current_sha": head_sha,
        "allowed_files": allowed_files,
        "agent_reviews": agent_json,
        "previous_review_text": truncate(prev_review_text, MAX_PREV_REVIEW_CHARS) if prev_review_text else "",
        "facts_from_diff": diff_facts(diff, allowed_files, max_lines=60),
    }

    return f"""
SYSTEM:
You are a lead reviewer (staff+). Merge multiple agent reviews into a SHORT, actionable PR comment.

INPUT (JSON):
{json.dumps(payload, ensure_ascii=False)}

OUTPUT INSTRUCTIONS:
{CURATOR_SCHEMA}
""".strip()


def call_with_optional_retry(model: str, prompt: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        raw = run_ollama(model, prompt)
        if not raw:
            return "", None, "Empty response from model"
        
        data, err = safe_json_loads(raw)
        if data is not None:
            log(f"Successfully parsed JSON from {model}")
            return raw, data, None

        log(f"Failed to parse JSON from {model}: {err}")
        
        if not RETRY_ON_BAD_JSON:
            return raw, None, err

        log(f"Retrying {model} with stricter JSON instructions")
        retry_prompt = prompt + "\n\nCRITICAL: You MUST output ONLY valid JSON. Start with { and end with }. No markdown, no code fences, no explanations. Just the JSON object."
        raw2 = run_ollama(model, retry_prompt)
        if not raw2:
            return raw, None, f"Retry also returned empty response. Original error: {err}"
        
        data2, err2 = safe_json_loads(raw2)
        if data2 is not None:
            log(f"Successfully parsed JSON on retry from {model}")
            return raw2, data2, None
        return raw2, None, f"Retry failed: {err2} (original: {err})"
    except Exception as e:
        log(f"Exception calling {model}: {e}")
        return "", None, str(e)


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
    try:
        log("Starting review script")
        diff = read_text(DIFF_PATH)
        guidelines = read_text(GUIDELINES_PATH)
        prev_review_text = read_text(PREV_REVIEW_PATH)

        if not diff.strip():
            log("No diff found")
            OUT_MD.write_text("### Local Multi-Agent AI Review\n\nNo diff found.\n", encoding="utf-8")
            return

        log(f"Diff size: {len(diff)} chars")
        diff = truncate(diff, MAX_DIFF_CHARS)
        diff_lower = diff.lower()

        changed_files = extract_changed_files(diff)
        allowed_files = filter_allowed_files(changed_files)
        log(f"Changed files: {allowed_files}")

        try:
            head_sha = get_head_sha()
        except Exception as e:
            log(f"Warning: Could not get HEAD SHA: {e}")
            head_sha = "unknown"
        
        mode = "FOLLOW_UP" if PREV_SHA else "INITIAL"
        log(f"Mode: {mode}, Model: {MODEL}, Curator: {CURATOR_MODEL}")

        results: List[Tuple[Agent, Optional[Dict[str, Any]], Optional[str], str]] = []
        agent_payloads: List[Dict[str, Any]] = []

        for agent in AGENTS:
            log(f"Calling agent: {agent.name}")
            try:
                prompt = build_prompt(agent, guidelines, diff, mode, PREV_SHA, head_sha, allowed_files)
                raw, data, err = call_with_optional_retry(MODEL, prompt)

                if data:
                    data["summary"] = rewrite_agent_summary_if_vague(str(data.get("summary", ""))[:500], diff_lower, allowed_files)

                    blocking = drop_issues_not_in_files(
                        drop_weak_issues(cap_issues(data.get("blocking"), 2)),
                        allowed_files,
                    )
                    non_blocking = drop_issues_not_in_files(
                        drop_weak_issues(cap_issues(data.get("non_blocking"), 3)),
                        allowed_files,
                    )

                    data["blocking"] = blocking
                    data["non_blocking"] = non_blocking
                    data["tests_to_add"] = cap_list(data.get("tests_to_add"), 6)
                    data["questions"] = cap_list(data.get("questions"), 5)

                    agent_payloads.append({
                        "agent": agent.name,
                        "summary": data.get("summary", ""),
                        "blocking": blocking,
                        "non_blocking": non_blocking,
                        "tests_to_add": data.get("tests_to_add", []),
                        "questions": data.get("questions", []),
                    })
                    log(f"Agent {agent.name} succeeded")
                else:
                    log(f"Agent {agent.name} failed: {err}")
            except Exception as e:
                log(f"Exception calling agent {agent.name}: {e}")
                err = str(e)
                raw = ""
                data = None

            results.append((agent, data, err, raw))

        # Ensure we have at least some agent data, even if minimal
        if not agent_payloads:
            log("Warning: No agents succeeded, creating fallback payload")
            agent_payloads = [{
                "agent": "System",
                "summary": "Review agents encountered errors. See details below.",
                "blocking": [],
                "non_blocking": [],
                "tests_to_add": [],
                "questions": [],
            }]

        log(f"Calling curator with {len(agent_payloads)} agent results")
        curator_prompt = build_curator_prompt(
            mode=mode,
            prev_sha=PREV_SHA,
            head_sha=head_sha,
            agent_json=agent_payloads,
            prev_review_text=prev_review_text,
            allowed_files=allowed_files,
            diff=diff,
        )
        curator_raw, curator, curator_err = call_with_optional_retry(CURATOR_MODEL, curator_prompt)

        if curator:
            curator["short_summary"] = cap_list(curator.get("short_summary"), 2)
            curator["top_actions"] = cap_list(curator.get("top_actions"), 3)
            curator["resolved"] = cap_list(curator.get("resolved"), 5)
            curator["still_open"] = cap_list(curator.get("still_open"), 5)
            curator["new_risks"] = cap_list(curator.get("new_risks"), 5)

            curator["short_summary"] = filter_bullets_by_fact_gate(curator["short_summary"], diff_lower, allowed_files)[:2]
            curator["top_actions"] = filter_bullets_by_fact_gate(curator["top_actions"], diff_lower, allowed_files)[:3]

            if mode == "FOLLOW_UP":
                curator["resolved"] = filter_bullets_by_fact_gate(curator["resolved"], diff_lower, allowed_files)[:5]
                curator["still_open"] = filter_bullets_by_fact_gate(curator["still_open"], diff_lower, allowed_files)[:5]
                curator["new_risks"] = filter_bullets_by_fact_gate(curator["new_risks"], diff_lower, allowed_files)[:5]

            # Ensure we always have content
            if not curator["short_summary"]:
                if allowed_files:
                    curator["short_summary"] = [f"Changed: {', '.join(allowed_files[:3])}" + (" ..." if len(allowed_files) > 3 else "")]
                else:
                    curator["short_summary"] = ["Reviewed changes - no issues identified."]

            if not curator["top_actions"]:
                curator["top_actions"] = ["No high-priority actions needed."]

            log("Curator succeeded")
        else:
            log(f"Curator failed: {curator_err}")
            # Create fallback curator output
            curator = {
                "short_summary": [f"⚠️ Review system error: {curator_err[:100]}" if curator_err else "Review completed but curator had issues"],
                "top_actions": ["Check agent details below for review findings"] if agent_payloads else ["Review agents encountered errors"],
                "resolved": [],
                "still_open": [],
                "new_risks": [],
            }

        if SHOW_DEBUG and allowed_files and curator and isinstance(curator, dict):
            for section in ("short_summary", "top_actions"):
                for item in curator.get(section, []) or []:
                    if not must_mention_file(str(item), allowed_files):
                        log(f"[DEBUG] Curator {section} item missing allowed file mention: {item}")
                    if not mentions_only_allowed_files(str(item), allowed_files):
                        log(f"[DEBUG] Curator {section} item mentions non-allowed file(s): {item}")

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

        md += "## Top actions\n\n"
        md += fmt_bullets(curator.get("top_actions", [])) + "\n"

        if mode == "FOLLOW_UP":
            md += "## Resolved\n\n"
            md += fmt_bullets(curator.get("resolved", [])) + "\n"

            md += "## Still open\n\n"
            md += fmt_bullets(curator.get("still_open", [])) + "\n"

            md += "## New risks\n\n"
            md += fmt_bullets(curator.get("new_risks", [])) + "\n"

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

            if SHOW_AGENT_TESTS_QUESTIONS:
                md += fmt_list("Tests to add", data.get("tests_to_add", []) or [])
                md += fmt_list("Questions", data.get("questions", []) or [])

        if SHOW_DEBUG:
            md += "\n---\n\n## Curator (debug)\n\n"
            md += "```text\n" + truncate(curator_raw, 3000) + "\n```\n"

        md += "\n</details>\n\n"

        md += "---\n\n"
        md += f"<!-- AI_REVIEW:HEAD_SHA={head_sha} -->\n"

        OUT_MD.write_text(md, encoding="utf-8")
        log("Review comment written successfully")
        
    except Exception as e:
        log(f"Fatal error in main: {e}")
        import traceback
        log(traceback.format_exc())
        
        # Write error output so user knows what happened
        error_md = f"""### Local Multi-Agent AI Review

⚠️ **Error:** Review script encountered a fatal error.

```
{str(e)[:500]}
```

Please check:
1. Is Ollama running at {OLLAMA_URL}?
2. Is the model `{MODEL}` available?
3. Check workflow logs for details.

<!-- AI_REVIEW:HEAD_SHA=error -->
"""
        OUT_MD.write_text(error_md, encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
