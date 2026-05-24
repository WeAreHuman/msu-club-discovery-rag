"""
build_clubs_table.py
Builds data/clubs_table.parquet — one row per club, 28 structured columns + tags.

LLM backend options (pick one):
  --ollama-host 192.168.x.x:11434   local Ollama over WiFi  (recommended)
  --ollama-model qwen2.5:latest     model name on that Ollama server
  (default falls back to Claude Code CLI if ollama-host not given)

Resumes automatically from checkpoint on re-run.

Usage:
    python build_clubs_table.py --ollama-host 192.168.1.42:11434
    python build_clubs_table.py --ollama-host 192.168.1.42:11434 --club alphachimsu
    python build_clubs_table.py --ollama-host 192.168.1.42:11434 --clear
"""

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
import config

try:
    import pandas as pd
except ImportError:
    print("pandas not installed. Run: pip install pandas pyarrow")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────

CHECKPOINT_FILE = Path(__file__).parent / ".table_checkpoint.txt"
OUTPUT_DIR      = config.DATA_DIR
OUTPUT_PARQUET  = OUTPUT_DIR / "clubs_table.parquet"
OUTPUT_CSV      = OUTPUT_DIR / "clubs_table.csv"

# ── Tag taxonomy (fixed vocabulary — LLM picks 5-6 per club) ─────────────────
#
# 7 dimensions so tags stay interpretable and filterable.
# Extend here only — never let the LLM invent new tags.

TAG_TAXONOMY = {
    "activity":  [
        "sports", "music", "arts", "gaming", "outdoor", "dance",
        "media", "cooking", "theater", "film",
    ],
    "domain":    [
        "engineering", "business", "healthcare", "law", "stem",
        "humanities", "education", "finance", "journalism",
    ],
    "identity":  [
        "women", "lgbtq+", "cultural", "religious", "international",
        "greek", "veterans",
    ],
    "purpose":   [
        "volunteering", "philanthropy", "career", "research",
        "networking", "social", "competition", "entrepreneurship",
    ],
    "vibe":      [
        "competitive", "casual", "beginner-friendly", "leadership",
        "community", "professional", "high-commitment",
    ],
    "wellness":  ["fitness", "mental-health", "nutrition", "mindfulness"],
    "advocacy":  [
        "social-justice", "environmental", "political",
        "dei", "activism", "sustainability",
    ],
}

ALL_TAGS = sorted(tag for tags in TAG_TAXONOMY.values() for tag in tags)

# ── Constitution section extractor ────────────────────────────────────────────

_RELEVANT_RE = re.compile(
    r'(dues|fee|finance|meeting|membership|eligib|gpa|grade point|'
    r'national affiliation|gender|women only|men only|open to all)',
    re.IGNORECASE,
)
_ARTICLE_SPLIT_RE = re.compile(r'(?=Article\s+[IVXLCDM]+\b)', re.IGNORECASE)


def extract_relevant_sections(text: str, max_chars: int = 5000) -> str:
    """Keep only constitution articles that mention dues, meetings, membership, or GPA."""
    articles = _ARTICLE_SPLIT_RE.split(text)
    relevant = [a.strip() for a in articles if _RELEVANT_RE.search(a[:600])]
    joined = "\n\n---\n\n".join(relevant)
    return joined[:max_chars] if joined else text[:max_chars]


# ── Claude CLI extraction ─────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a structured data extractor for a university club database.
Read the club profile and constitution excerpts, then return ONE JSON object.
Output ONLY valid JSON — no prose, no markdown fences, nothing else.

Available tags — pick exactly 5 that best describe this club:
{all_tags}

--- PROFILE ---
Name: {name}
Categories: {categories}
Description: {description}

--- CONSTITUTION EXCERPTS ---
{constitution}

Return this exact JSON (use null for any unknown value):
{{
  "has_dues": <true|false>,
  "dues_amount": <"$X/semester" or null>,
  "membership_open": <true if any student can join freely, false if application/tryout/invite-only>,
  "is_greek": <true|false>,
  "national_affiliation": <"org name" or null>,
  "gender_restriction": <"open"|"women"|"men">,
  "gpa_requirement": <number or null>,
  "meeting_frequency": <"weekly"|"biweekly"|"monthly"|"irregular"|null>,
  "meeting_day": <"Monday"/"Tuesday"/etc. or null>,
  "has_service_requirement": <true|false>,
  "tags": ["tag1","tag2","tag3","tag4","tag5"]
}}"""


# ── LLM backend — configured via .env ────────────────────────────────────────
#
# Any OpenAI-compatible endpoint works (Ollama, Groq, LM Studio, etc.)
# Set in .env:
#   TABLE_LLM_BASE_URL=http://192.168.1.42:11434/v1   ← Ollama on another machine
#   TABLE_LLM_MODEL=qwen2.5:latest
#   TABLE_LLM_API_KEY=                                 ← leave empty for local Ollama
#
# Groq example:
#   TABLE_LLM_BASE_URL=https://api.groq.com/openai/v1
#   TABLE_LLM_MODEL=llama-3.1-8b-instant
#   TABLE_LLM_API_KEY=gsk_...
#
# Claude CLI fallback (no TABLE_LLM_BASE_URL set):
#   Uses `claude -p` subprocess — requires Claude Code installed locally.

_TABLE_LLM_BASE_URL: str | None = os.getenv("TABLE_LLM_BASE_URL")
_TABLE_LLM_MODEL:    str        = os.getenv("TABLE_LLM_MODEL", "qwen2.5:latest")
_TABLE_LLM_API_KEY:  str        = os.getenv("TABLE_LLM_API_KEY", "ollama")  # Ollama ignores the value


def call_openai_compat(prompt: str, timeout: int = 120) -> str | None:
    """Call any OpenAI-compatible /v1/chat/completions endpoint."""
    import requests as _req

    url = f"{_TABLE_LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {_TABLE_LLM_API_KEY}"}
    payload = {
        "model": _TABLE_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": False,
    }
    try:
        resp = _req.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"]["content"] or "").strip()
    except _req.exceptions.ConnectionError:
        print(f"\n    [llm] cannot reach {_TABLE_LLM_BASE_URL}")
        return None
    except _req.exceptions.Timeout:
        print("\n    [llm timeout]")
        return None
    except Exception as e:
        print(f"\n    [llm error] {e}")
        return None


def call_claude_cli(prompt: str, timeout: int = 90) -> str | None:
    """Fallback: Claude Code CLI via subprocess (requires claude installed locally)."""
    import shutil

    claude_path = shutil.which("claude")
    use_shell   = claude_path is None
    cmd         = [claude_path, "-p"] if claude_path else "claude -p"

    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True,
            text=True, shell=use_shell, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        print(f"\n    [claude-cli error] {result.stderr[:300]}")
        return None
    except subprocess.TimeoutExpired:
        print("\n    [claude-cli timeout]")
        return None
    except FileNotFoundError:
        raise RuntimeError(
            "No LLM backend configured.\n"
            "Set TABLE_LLM_BASE_URL in .env (Ollama/Groq) "
            "or install Claude Code CLI."
        )


def call_llm(prompt: str) -> str | None:
    """Route to the configured backend: OpenAI-compat endpoint or Claude CLI."""
    if _TABLE_LLM_BASE_URL:
        return call_openai_compat(prompt)
    return call_claude_cli(prompt)


def parse_json_response(text: str) -> dict | None:
    if not text:
        return None
    # Strip accidental markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


_LLM_FALLBACK: dict = {
    "has_dues": None,
    "dues_amount": None,
    "membership_open": None,
    "is_greek": None,
    "national_affiliation": None,
    "gender_restriction": "open",
    "gpa_requirement": None,
    "meeting_frequency": None,
    "meeting_day": None,
    "has_service_requirement": None,
    "tags": [],
}


def extract_via_llm(profile: dict, constitution_text: str | None) -> dict:
    desc = re.sub(r"<[^>]+>", " ", profile.get("description") or "")[:400].strip()
    constitution = (
        extract_relevant_sections(constitution_text)
        if constitution_text
        else "(no constitution available)"
    )

    prompt = _EXTRACTION_PROMPT.format(
        all_tags=", ".join(ALL_TAGS),
        name=profile.get("name", ""),
        categories=", ".join(profile.get("categories") or []),
        description=desc,
        constitution=constitution,
    )

    raw = call_llm(prompt)
    parsed = parse_json_response(raw)

    if not parsed:
        print("    [warn] LLM parse failed — filling with nulls")
        return dict(_LLM_FALLBACK)

    # Validate tags against known taxonomy
    raw_tags = parsed.get("tags") or []
    parsed["tags"] = [t for t in raw_tags if t in ALL_TAGS][:6]

    return {**_LLM_FALLBACK, **parsed}


# ── Profile field extraction (free — no LLM) ─────────────────────────────────

def extract_profile_fields(profile: dict) -> dict:
    contact = profile.get("contact") or {}
    social_platforms = {
        s.get("platform", "").lower()
        for s in (contact.get("social_links") or [])
    }
    events = profile.get("events") or []

    year_counts: Counter = Counter()
    for ev in events:
        yr = (ev.get("start_datetime") or "")[:4]
        if yr.isdigit():
            year_counts[yr] += 1

    desc_clean = re.sub(r"<[^>]+>", " ", profile.get("description") or "")

    return {
        "org_id":                 str(profile.get("org_id") or ""),
        "org_name":               profile.get("name") or "",
        "org_slug":               profile.get("slug") or "",
        "status":                 profile.get("status") or "",
        "categories":             profile.get("categories") or [],
        "contact_email":          contact.get("email") or "",
        "contact_website":        contact.get("website") or "",
        "has_instagram":          "instagram" in social_platforms,
        "has_website":            bool(contact.get("website")),
        "officer_count":          len(profile.get("officers") or []),
        "has_constitution":       bool(profile.get("documents")),
        "description_word_count": len(desc_clean.split()),
        "total_events":           len(events),
        "events_in_2023":         year_counts.get("2023", 0),
        "events_in_2024":         year_counts.get("2024", 0),
        "events_in_2025":         year_counts.get("2025", 0),
        "events_in_2026":         year_counts.get("2026", 0),
        "scraped_at":             profile.get("scraped_at") or "",
        "org_url":                profile.get("url") or "",
    }


# ── File discovery ────────────────────────────────────────────────────────────

def find_constitution_txt(club_dir: Path, profile: dict) -> Path | None:
    # 1. Match by doc_id listed in profile.documents
    for doc in (profile.get("documents") or []):
        doc_id = str(doc.get("doc_id") or "")
        if doc_id:
            for subdir in ["documents", "org"]:
                candidate = club_dir / subdir / f"{doc_id}.txt"
                if candidate.exists():
                    return candidate

    # 2. Fallback: first .txt in any known subdir
    for subdir in ["documents", "org"]:
        subdir_path = club_dir / subdir
        if subdir_path.exists():
            txts = sorted(subdir_path.glob("*.txt"))
            if txts:
                return txts[0]

    return None


# ── Per-club processing ───────────────────────────────────────────────────────

def process_club(club_dir: Path) -> dict | None:
    profile_path = club_dir / "profile.json"
    if not profile_path.exists():
        return None

    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [SKIP] {club_dir.name}: failed to load profile — {e}")
        return None

    row = extract_profile_fields(profile)

    txt_path = find_constitution_txt(club_dir, profile)
    constitution_text = None
    if txt_path:
        try:
            constitution_text = txt_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    row.update(extract_via_llm(profile, constitution_text))
    return row


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if not CHECKPOINT_FILE.exists():
        return set()
    return set(CHECKPOINT_FILE.read_text(encoding="utf-8").splitlines())


def save_checkpoint(slug: str):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(slug + "\n")


# ── Persistence ───────────────────────────────────────────────────────────────

def save_output(rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def build_table(orgs_dir: Path, club_slug: str = None, clear: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if clear:
        for f in [CHECKPOINT_FILE, OUTPUT_PARQUET, OUTPUT_CSV]:
            if f.exists():
                f.unlink()
        print("Checkpoint and output files cleared.")

    # ── Single-club test mode ─────────────────────────────────────────────────
    if club_slug:
        row = process_club(orgs_dir / club_slug)
        if row:
            # Pretty-print (skip long list fields)
            display = {k: v for k, v in row.items() if k != "categories"}
            print(json.dumps(display, indent=2, default=str))
            save_output([row])
            print(f"\nSaved to {OUTPUT_PARQUET}")
        else:
            print(f"No data for club '{club_slug}'")
        return

    # ── Full / resume mode ────────────────────────────────────────────────────
    done = load_checkpoint()
    club_dirs = sorted(
        d for d in orgs_dir.iterdir()
        if d.is_dir() and (d / "profile.json").exists()
    )
    remaining = [d for d in club_dirs if d.name not in done]

    print(f"Clubs : {len(club_dirs)} total | {len(done)} done | {len(remaining)} remaining")
    if not remaining:
        print("All clubs already processed. Use --clear to restart.")
        return

    # Load previously saved rows so we can append to them
    existing_rows: list[dict] = []
    if OUTPUT_PARQUET.exists() and done:
        try:
            existing_rows = pd.read_parquet(OUTPUT_PARQUET).to_dict("records")
        except Exception:
            pass

    new_rows: list[dict] = []
    errors = 0

    for idx, club_dir in enumerate(remaining, 1):
        print(f"  [{idx:>4}/{len(remaining)}] {club_dir.name:<40}", end="", flush=True)

        row = process_club(club_dir)
        if row:
            new_rows.append(row)
            save_checkpoint(club_dir.name)
            tags_str = ", ".join(row.get("tags") or [])
            print(f"  ✓  [{tags_str}]")
        else:
            errors += 1
            print("  ✗")

        # Incremental save every 25 clubs — safe against crashes
        if len(new_rows) % 25 == 0:
            save_output(existing_rows + new_rows)

        time.sleep(0.4)  # avoid hammering the CLI

    save_output(existing_rows + new_rows)

    total_saved = len(existing_rows) + len(new_rows)
    print(f"\n{'='*60}")
    print(f"Saved {total_saved} clubs → {OUTPUT_PARQUET}")
    print(f"Errors this run : {errors}")
    if errors:
        print("Re-run the same command — errored clubs were not checkpointed.")


def main():
    parser = argparse.ArgumentParser(
        description="Build clubs_table.parquet (one row per MSU club, structured fields + tags)"
    )
    parser.add_argument("--orgs-dir", type=str, default=None, help="Path to msu_scraper/data/orgs/")
    parser.add_argument("--club", type=str, default=None, metavar="SLUG", help="Test a single club by slug")
    parser.add_argument("--clear", action="store_true", help="Wipe checkpoint + output and restart from scratch")
    args = parser.parse_args()

    orgs_dir = Path(args.orgs_dir) if args.orgs_dir else config.SCRAPER_ORGS_DIR
    if not orgs_dir.exists():
        print(f"Orgs directory not found: {orgs_dir}")
        print("Set SCRAPER_ORGS_DIR in .env or pass --orgs-dir")
        sys.exit(1)

    build_table(orgs_dir=orgs_dir, club_slug=args.club, clear=args.clear)


if __name__ == "__main__":
    main()
