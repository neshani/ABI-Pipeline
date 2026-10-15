"""Align book transcripts against their source (EPUB) text.

Dry-run by default: writes ``transcript.aligned.txt``, ``transcript_timing.aligned.json``
and ``corrections.csv`` next to the originals, touching nothing else.  ``--commit``
overwrites ``transcript.txt`` / ``transcript_timing.json`` after backing the ASR text
up to ``transcript.asr.txt`` (once).

Run from the repo root (the DB and ``output/`` are resolved relative to CWD):

    python align_transcripts.py --project "The Wheel of Time Series by Robert Jordan" \
        --book "01 The Eye of the World" --epub "G:\\...\\The Eye Of The World (Retail).epub"

    python align_transcripts.py --project "..." --all --epub-dir "G:\\...\\ePubs" --commit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_settings():
    from database.connection import get_setting
    return Path(get_setting("output_dir", "./output")).resolve()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="Project name (output/<project>)")
    parser.add_argument("--book", help="Book folder name; required unless --all")
    parser.add_argument("--all", action="store_true",
                        help="Process every book in the project that has a transcript")
    parser.add_argument("--epub", help="Explicit EPUB path (only with --book)")
    parser.add_argument("--epub-dir", action="append", default=[],
                        help="Directory to search for EPUBs (repeatable)")
    parser.add_argument("--commit", action="store_true",
                        help="Overwrite transcript.txt / transcript_timing.json")
    args = parser.parse_args(argv)

    from services.aligner import (align_book, discover_source_epubs,
                                  match_source_epub, write_alignment)

    base = _load_settings()
    project_dir = base / args.project
    if not project_dir.is_dir():
        print(f"error: no such project directory: {project_dir}")
        return 2

    if args.all:
        books = [d.name for d in sorted(project_dir.iterdir())
                 if d.is_dir() and (d / "transcript.txt").exists()]
    elif args.book:
        books = [args.book]
    else:
        parser.error("pass --book or --all")
        return 2

    if args.epub and len(books) != 1:
        parser.error("--epub can only be used with a single --book")
        return 2

    candidates = discover_source_epubs([Path(d) for d in args.epub_dir])
    if args.all and not candidates and not args.epub:
        parser.error("--all needs --epub-dir (or per-book --epub runs)")
        return 2

    failures = 0
    for book in books:
        book_dir = project_dir / book
        transcript = book_dir / "transcript.txt"
        words = book_dir / "transcript.json"
        timing = book_dir / "transcript_timing.json"
        if not transcript.exists():
            print(f"[{book}] skip: no transcript.txt")
            continue
        if not words.exists():
            print(f"[{book}] skip: no transcript.json (word timings)")
            continue
        if not timing.exists():
            print(f"[{book}] skip: no transcript_timing.json (timings cannot be rebuilt)")
            continue


        epub = Path(args.epub) if args.epub else match_source_epub(book, candidates)
        if not epub or not Path(epub).exists():
            print(f"[{book}] skip: no EPUB source found")
            continue

        # Re-runs align the ORIGINAL ASR text (kept as transcript.asr.txt on the
        # first cleanup), not the already-cleaned transcript.txt.
        asr_transcript = book_dir / "transcript.asr.txt"
        asr_timing = book_dir / "transcript_timing.asr.json"
        source_transcript = asr_transcript if asr_transcript.exists() else transcript
        source_timing = asr_timing if (asr_transcript.exists() and asr_timing.exists()) else timing

        print(f"\n=== {book} ===")
        print(f"  source : {epub}")
        try:
            report = align_book(source_transcript, words, source_timing, str(epub), book_label=book)
        except Exception as exc:  # noqa: BLE001 - one bad book shouldn't stop the run
            print(f"  FAILED : {exc}")
            failures += 1
            continue

        written = write_alignment(report, book_dir, commit=args.commit)
        print(f"  chapters       : {report.chapters}")
        print(f"  asr words      : {report.asr_words:,}")
        print(f"  source words   : {report.source_words:,}")
        print(f"  emitted words  : {report.emitted_words:,}")
        print(f"  anchors        : {report.anchors:,}")
        print(f"  audio-only     : {report.audio_only_words:,} words kept (map-cleaned)")
        print(f"  source-only    : {report.source_only_words:,} words excluded")
        print(f"  timing chunks  : {report.timing_chunks:,}")
        print(f"  corrections    : {len(report.safe_corrections):,} safe, "
              f"{len(report.review_corrections):,} need review")
        for warning in report.warnings:
            print(f"  WARNING: {warning}")
        for label, path in written.items():
            print(f"  wrote {label:12s}: {path}")
        for wrong, right, count in report.safe_corrections[:8]:
            print(f"      {wrong:22s} -> {right:22s} ({count})")

    if failures:
        print(f"\n{failures} book(s) failed")
        return 1
    if not args.commit:
        print("\nDry run - nothing overwritten. Re-run with --commit to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
