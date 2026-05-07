"""
Data Ingestion Script
Reads msu_scraper output and uploads club chunks to Pinecone.

Usage:
  python ingest_data.py                        # resume from checkpoint (skip already-done clubs)
  python ingest_data.py --scraper-orgs-dir D:/msu_scraper/data/orgs
  python ingest_data.py --club asa             # single club (by slug)
  python ingest_data.py --clear                # wipe namespace + checkpoint and restart
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data_processing import ClubDataProcessor
from src.vector_store import VectorStore
import config

CHECKPOINT_FILE = Path(__file__).parent / ".ingest_checkpoint.txt"


def load_checkpoint() -> set:
    """Return set of slugs already successfully upserted."""
    if not CHECKPOINT_FILE.exists():
        return set()
    return set(CHECKPOINT_FILE.read_text(encoding="utf-8").splitlines())


def save_checkpoint(slug: str):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(slug + "\n")


def ingest(
    orgs_dir: Path,
    club_slug: str = None,
    clear_existing: bool = False,
    batch_size: int = 96,
):
    print("=" * 70)
    print("MSU CLUB DISCOVERY — DATA INGESTION")
    print("=" * 70)

    try:
        config.validate_config()
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        print("Check your .env file.")
        return

    if not orgs_dir.exists():
        print(f"\nScraper orgs directory not found: {orgs_dir}")
        print("Set SCRAPER_ORGS_DIR in your .env or pass --scraper-orgs-dir")
        return

    vector_store = VectorStore(
        api_key=config.PINECONE_API_KEY,
        index_name=config.PINECONE_INDEX_NAME,
        namespace=config.PINECONE_NAMESPACE,
    )

    if clear_existing:
        print(f"\nClearing namespace '{config.PINECONE_NAMESPACE}'...")
        confirm = input("  This cannot be undone. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            vector_store.delete_namespace()
            if CHECKPOINT_FILE.exists():
                CHECKPOINT_FILE.unlink()
                print("  Checkpoint cleared.")
        else:
            print("  Skipped.")

    processor = ClubDataProcessor()
    print(f"\nSource : {orgs_dir}")

    # ── Single-club mode ──────────────────────────────────────────────────────
    if club_slug:
        club_dir = orgs_dir / club_slug
        if not club_dir.exists():
            print(f"Club directory not found: {club_dir}")
            return
        chunks = processor.process_club_dir(club_dir)
        if not chunks:
            print("No chunks generated.")
            return
        upserted = vector_store.upsert_chunks(chunks, batch_size=batch_size)
        if upserted:
            save_checkpoint(club_slug)
        print(f"\nDone: {upserted}/{len(chunks)} chunks upserted for '{club_slug}'")
        return

    # ── Full / resume mode ────────────────────────────────────────────────────
    done = load_checkpoint()
    club_dirs = sorted(d for d in orgs_dir.iterdir() if d.is_dir() and (d / "profile.json").exists())
    remaining = [d for d in club_dirs if d.name not in done]

    print(f"Clubs  : {len(club_dirs)} total  |  {len(done)} already done  |  {len(remaining)} remaining")

    if not remaining:
        print("\nAll clubs already ingested. Use --clear to restart from scratch.")
        return

    total_chunks = 0
    type_counts: Counter = Counter()
    errors = 0

    for idx, club_dir in enumerate(remaining, 1):
        try:
            chunks = processor.process_club_dir(club_dir)
            if not chunks:
                continue

            upserted = vector_store.upsert_chunks(chunks, batch_size=batch_size)
            if upserted == len(chunks):
                save_checkpoint(club_dir.name)
                total_chunks += upserted
                for c in chunks:
                    type_counts[c["metadata"].get("chunk_type", "?")] += 1
            else:
                # Partial upsert — token limit or other transient error
                print(f"\n[STOP] Partial upsert for '{club_dir.name}' ({upserted}/{len(chunks)})")
                print(f"       {len(done) + idx - 1} clubs done so far. Re-run to continue.")
                break

        except Exception as e:
            errors += 1
            print(f"  [SKIP] {club_dir.name}: {e}")

        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(remaining)} this run — {total_chunks} chunks uploaded")

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    done_now = load_checkpoint()
    print(f"Clubs done (total) : {len(done_now)}/{len(club_dirs)}")
    print(f"Chunks this run    : {total_chunks}")
    for chunk_type, count in sorted(type_counts.items()):
        print(f"  {chunk_type:<15} {count}")
    if errors:
        print(f"Errors             : {errors}")
    if len(done_now) < len(club_dirs):
        print(f"\nNot finished — re-run the same command to continue.")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest MSU club data into Pinecone (resumes automatically from checkpoint)"
    )
    parser.add_argument(
        "--scraper-orgs-dir",
        type=str,
        default=None,
        help=f"Path to msu_scraper/data/orgs/ (default: {config.SCRAPER_ORGS_DIR})",
    )
    parser.add_argument(
        "--club",
        type=str,
        default=None,
        metavar="SLUG",
        help="Process only one club by its slug (e.g. 'asa')",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe the Pinecone namespace AND the checkpoint, then restart",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Records per Pinecone upsert call (default: 96)",
    )

    args = parser.parse_args()
    orgs_dir = Path(args.scraper_orgs_dir) if args.scraper_orgs_dir else config.SCRAPER_ORGS_DIR

    ingest(
        orgs_dir=orgs_dir,
        club_slug=args.club,
        clear_existing=args.clear,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
