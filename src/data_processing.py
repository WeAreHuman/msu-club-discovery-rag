"""
Club Data Processor
Reads scraper output (profile.json + constitution .txt) and produces
typed chunks ready for Pinecone ingestion.

3 chunk types per club:
  profile      — club general info (name, description, contact, officers)
  event        — one chunk per event; aggregate stats in metadata
  constitution — article-level chunks, split further when > CONSTITUTION_MAX_TOKENS
"""

import re
import json
import html as html_lib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Split BEFORE each "Article X" heading (handles "ArticleI:", "Article II -", etc.)
_ARTICLE_SPLIT_RE = re.compile(r'(?=Article\s*[IVXLCDM]+\s*[:\-])', re.IGNORECASE)

# Parse article number and title from the header line
_ARTICLE_HEADER_RE = re.compile(r'Article\s*([IVXLCDM]+)\s*[:\-]?\s*(.*)', re.IGNORECASE)


class ClubDataProcessor:
    """
    Processes the msu_scraper output directory into typed Pinecone chunks.

    Input  : .../data/orgs/<slug>/  containing profile.json and
             documents/<id>.txt  (constitution/bylaws plain text)
    Output : list of {"text": str, "metadata": dict} ready for VectorStore.upsert_chunks()
    """

    CONSTITUTION_MAX_TOKENS = config.CONSTITUTION_MAX_TOKENS
    PROFILE_MAX_TOKENS = config.CHUNK_SIZE

    def __init__(self):
        self._enc = tiktoken.get_encoding("cl100k_base")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def process_orgs_dir(self, orgs_dir: Path) -> List[Dict]:
        """Walk all club directories under orgs_dir and return all chunks."""
        club_dirs = sorted(d for d in orgs_dir.iterdir() if d.is_dir())
        all_chunks: List[Dict] = []
        total = len(club_dirs)
        errors = 0

        for idx, club_dir in enumerate(club_dirs, 1):
            if not (club_dir / "profile.json").exists():
                continue
            try:
                chunks = self.process_club_dir(club_dir)
                all_chunks.extend(chunks)
            except Exception as e:
                errors += 1
                print(f"  [SKIP] {club_dir.name}: {e}")

            if idx % 200 == 0:
                print(f"  Progress: {idx}/{total} clubs — {len(all_chunks)} chunks ({errors} errors)")

        print(f"Done: {len(all_chunks)} chunks from {total} clubs ({errors} errors)")
        return all_chunks

    def process_club_dir(self, club_dir: Path) -> List[Dict]:
        """Process one club directory → typed chunks."""
        profile = self._load_json(club_dir / "profile.json")
        chunks: List[Dict] = []
        chunks.extend(self._profile_chunks(profile))
        chunks.extend(self._event_chunks(profile))
        chunks.extend(self._constitution_chunks(profile, club_dir))
        return chunks

    # -------------------------------------------------------------------------
    # Profile chunks
    # -------------------------------------------------------------------------

    def _profile_chunks(self, profile: Dict) -> List[Dict]:
        base = self._base_meta(profile)
        contact = profile.get("contact") or {}

        # Anchor is prepended to every sub-chunk so the club is always identifiable,
        # even when a long description or officer list forces a split.
        anchor = (
            f"Organization: {profile.get('name', '')}\n"
            f"Categories: {', '.join(profile.get('categories') or [])}\n"
            f"Status: {profile.get('status', '')}"
        )

        # Body holds the variable-length content that may need splitting
        body_lines: List[str] = []
        desc = self._strip_html(profile.get("description") or "")
        summary = (profile.get("summary") or "").strip()
        body_lines.append(desc or summary)

        if contact.get("email"):
            body_lines.append(f"Contact Email: {contact['email']}")
        if contact.get("website"):
            body_lines.append(f"Website: {contact['website']}")

        social = [
            f"{s.get('platform', '')}: {s.get('url', '')}"
            for s in (contact.get("social_links") or [])
        ]
        if social:
            body_lines.append(f"Social: {', '.join(social)}")

        officer_lines = []
        for o in (profile.get("officers") or []):
            role = o.get("role", "")
            role = re.sub(r'^[A-Z]\n', '', role).strip()
            role = re.sub(r'\nDocuments.*', '', role, flags=re.DOTALL).strip()
            role = role.replace('\n', ' ')
            officer_lines.append(f"  {role}: {o.get('name', '')}")
        if officer_lines:
            body_lines.append("Officers:")
            body_lines.extend(officer_lines)

        body = "\n".join(body_lines).strip()
        # Reserve tokens for the anchor so each assembled chunk stays within budget
        anchor_tokens = self._count_tokens(anchor + "\n\n")
        body_parts = self._split(body, self.PROFILE_MAX_TOKENS - anchor_tokens)

        return [
            {
                "text": anchor + "\n\n" + part,
                "metadata": {
                    **base,
                    "chunk_type": "profile",
                    "chunk_index": i,
                    "total_chunks": len(body_parts),
                },
            }
            for i, part in enumerate(body_parts)
        ]

    # -------------------------------------------------------------------------
    # Event chunks
    # -------------------------------------------------------------------------

    def _event_chunks(self, profile: Dict) -> List[Dict]:
        events = profile.get("events") or []
        if not events:
            return []

        base = self._base_meta(profile)
        total_events = len(events)

        # Aggregate: count events per calendar month
        month_counter: Counter = Counter()
        for ev in events:
            ym = (ev.get("start_datetime") or "")[:7]  # "YYYY-MM"
            if len(ym) == 7:
                month_counter[ym] += 1
        events_by_month = json.dumps(dict(sorted(month_counter.items())))

        chunks = []
        for ev in events:
            title = ev.get("title") or ""
            start = ev.get("start_datetime") or ""
            end = ev.get("end_datetime") or ""
            venue = ev.get("venue") or ""
            desc = (ev.get("description") or "").strip()
            ev_cats = ", ".join(ev.get("category_names") or [])
            url = ev.get("url") or ""

            lines = [
                f"Organization: {profile.get('name', '')}",
                f"Event: {title}",
            ]
            if start:
                date_line = f"Date: {start}"
                if end and end != start:
                    date_line += f" to {end}"
                lines.append(date_line)
            if venue:
                lines.append(f"Venue: {venue}")
            if ev_cats:
                lines.append(f"Categories: {ev_cats}")
            if desc:
                lines.append(f"\n{desc}")

            chunks.append({
                "text": "\n".join(lines).strip(),
                "metadata": {
                    **base,
                    "chunk_type": "event",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "event_id": str(ev.get("event_id") or ""),
                    "event_title": title,
                    "event_date": start,
                    "event_venue": venue,
                    "event_url": url,
                    "total_events": total_events,
                    "events_by_month": events_by_month,
                },
            })
        return chunks

    # -------------------------------------------------------------------------
    # Constitution chunks
    # -------------------------------------------------------------------------

    def _constitution_chunks(self, profile: Dict, club_dir: Path) -> List[Dict]:
        txt_path = self._find_constitution_txt(profile, club_dir)
        if txt_path is None:
            return []

        text = txt_path.read_text(encoding="utf-8", errors="replace")
        base = self._base_meta(profile)
        doc_id = txt_path.stem

        articles = self._parse_articles(text)
        chunks: List[Dict] = []

        for article_num, article_title, article_body in articles:
            prefix = (
                f"Organization: {profile.get('name', '')}\n"
                f"Constitution — Article {article_num}: {article_title}\n\n"
            )
            for sub_text in self._split(article_body.strip(), self.CONSTITUTION_MAX_TOKENS):
                chunks.append({
                    "text": prefix + sub_text,
                    "metadata": {
                        **base,
                        "chunk_type": "constitution",
                        "chunk_index": len(chunks),
                        "total_chunks": 0,  # patched below
                        "article_num": article_num,
                        "article_title": article_title,
                        "doc_id": doc_id,
                    },
                })

        for c in chunks:
            c["metadata"]["total_chunks"] = len(chunks)

        return chunks

    def _find_constitution_txt(self, profile: Dict, club_dir: Path) -> Optional[Path]:
        docs_dir = club_dir / "documents"
        if not docs_dir.exists():
            return None

        # Match by doc_id listed in profile (pick first with a .txt on disk)
        for doc in (profile.get("documents") or []):
            doc_id = str(doc.get("doc_id") or "")
            if doc_id:
                candidate = docs_dir / f"{doc_id}.txt"
                if candidate.exists():
                    return candidate

        # Fallback: first .txt in documents/
        txts = sorted(docs_dir.glob("*.txt"))
        return txts[0] if txts else None

    # -------------------------------------------------------------------------
    # Article parsing
    # -------------------------------------------------------------------------

    def _parse_articles(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Split constitution text on Article headings.
        Returns list of (article_num, article_title, body).
        Falls back to [("", "Constitution", full_text)] if no Articles found.
        """
        parts = _ARTICLE_SPLIT_RE.split(text)

        results: List[Tuple[str, str, str]] = []
        preamble_parts: List[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            m = _ARTICLE_HEADER_RE.match(part)
            if m:
                article_num = m.group(1).upper()
                first_line_rest = m.group(2).strip()
                body = part[m.end():].strip()
                article_title = first_line_rest or f"Article {article_num}"
                results.append((article_num, article_title, body or first_line_rest))
            else:
                preamble_parts.append(part)

        if not results:
            return [("", "Constitution", text.strip())]

        if preamble_parts:
            preamble_body = "\n\n".join(preamble_parts)
            results.insert(0, ("0", "Preamble", preamble_body))

        return results

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _base_meta(self, profile: Dict) -> Dict:
        contact = profile.get("contact") or {}
        return {
            "org_id": str(profile.get("org_id") or ""),
            "org_name": profile.get("name") or "",
            "org_slug": profile.get("slug") or "",
            # Stored as list[str] — Pinecone supports $in filtering on list metadata
            "categories": profile.get("categories") or [],
            "status": profile.get("status") or "",
            "contact_email": contact.get("email") or "",
            "contact_website": contact.get("website") or "",
            "org_url": profile.get("url") or "",
            "scraped_at": profile.get("scraped_at") or "",
        }

    @staticmethod
    def _strip_html(text: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html_lib.unescape(text)
        return re.sub(r' {2,}', ' ', text).strip()

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _split(self, text: str, max_tokens: int) -> List[str]:
        if self._count_tokens(text) <= max_tokens:
            return [text]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text) or [text]

    @staticmethod
    def _load_json(path: Path) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ============================================================================
# STANDALONE TEST
# ============================================================================
if __name__ == "__main__":
    processor = ClubDataProcessor()
    sample_dir = Path("D:/msu_scraper/data/orgs/-")
    if sample_dir.exists():
        chunks = processor.process_club_dir(sample_dir)
        print(f"\n{len(chunks)} chunks generated")
        for c in chunks:
            print(f"\n[{c['metadata']['chunk_type']}] {c['text'][:120]}...")
    else:
        print("Sample dir not found — set SCRAPER_ORGS_DIR in .env and run ingest_data.py")
