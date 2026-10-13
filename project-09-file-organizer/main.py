"""
Project 09 - Duplicate Finder + Bulk File Organizer  (automation, no ML)
Author : Sanchit Kumar Mishra

What it does
------------
1. Finds DUPLICATE files by their CONTENT (SHA-256 hash), not by name.
2. Sorts every file into folders by type (Images, Documents, Audio, ...),
   optionally into sub-folders by month (2024-03, ...).
3. Moves duplicates into a "_duplicates" folder instead of deleting them.
4. Writes an undo log so the whole operation can be reversed with --undo.

Safe by default: without --apply nothing is moved (it only prints a plan).
Uses only the Python standard library - nothing to install.

Usage
-----
    python main.py                          # self-contained DEMO in outputs/demo_folder (safe!)
    python main.py "C:/Users/me/Downloads"  # dry run: show what WOULD happen
    python main.py ~/Downloads --apply      # really move the files
    python main.py ~/Downloads --apply --by-date
    python main.py ~/Downloads --undo       # put everything back
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE / "outputs" / "demo_folder"
LOG_NAME = "organizer_log.json"
DUPLICATES_DIR = "_duplicates"

CATEGORIES = {
    "Images": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".heic"},
    "Documents": {".pdf", ".doc", ".docx", ".txt", ".md", ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".odt"},
    "Audio": {".mp3", ".wav", ".flac", ".aac", ".m4a"},
    "Video": {".mp4", ".mkv", ".avi", ".mov", ".webm"},
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz"},
    "Code": {".py", ".js", ".html", ".css", ".java", ".c", ".cpp", ".ipynb", ".json"},
}


# --------------------------------------------------------------------------- #
# 1. Scanning and duplicate detection
# --------------------------------------------------------------------------- #
def category_of(path: Path) -> str:
    """Folder name for a file, based on its extension ('Other' if unknown)."""
    extension = path.suffix.lower()
    for category, extensions in CATEGORIES.items():
        if extension in extensions:
            return category
    return "Other"


def list_files(root: Path) -> list[Path]:
    """Files directly inside `root` (not in sub-folders); skips hidden files and our own log."""
    return sorted(p for p in root.iterdir() if p.is_file() and not p.name.startswith(".") and p.name != LOG_NAME)


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of the file content, read in chunks so huge files do not fill the memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicates(files: list[Path]) -> tuple[list[list[Path]], dict]:
    """Group files with identical content.

    Speed trick: two files can only be identical if they have the SAME SIZE, so we
    first group by size (very cheap) and only hash files that share a size with
    another file.

    Returns (list of duplicate groups, statistics).
    """
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in files:
        by_size[path.stat().st_size].append(path)

    candidates = [p for group in by_size.values() if len(group) > 1 for p in group]
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in candidates:
        by_hash[file_hash(path)].append(path)

    groups = [sorted(g, key=lambda p: (p.stat().st_mtime, p.name)) for g in by_hash.values() if len(g) > 1]
    same_size_groups = sum(1 for g in by_size.values() if len(g) > 1)
    stats = {
        "files": len(files),
        "hashed": len(candidates),
        "duplicate_groups": len(groups),
        "duplicate_files": sum(len(g) - 1 for g in groups),
        "same_size_groups": same_size_groups,
        "same_size_but_different_content": same_size_groups - len(groups),
    }
    return groups, stats


# --------------------------------------------------------------------------- #
# 2. Planning and executing moves
# --------------------------------------------------------------------------- #
def unique_destination(destination: Path, reserved: set[Path]) -> Path:
    """If `destination` already exists (or is already planned), add _1, _2 ... before the extension."""
    candidate, counter = destination, 1
    while candidate.exists() or candidate in reserved:
        candidate = destination.with_name(f"{destination.stem}_{counter}{destination.suffix}")
        counter += 1
    return candidate


def plan_moves(root: Path, files: list[Path], groups: list[list[Path]], by_date: bool) -> list[dict]:
    """Decide where every file goes. Of each duplicate group the OLDEST file is kept in place
    (organised normally); the others go to _duplicates/."""
    duplicates = {p for group in groups for p in group[1:]}
    moves, reserved = [], set()
    for path in files:
        if path in duplicates:
            folder = root / DUPLICATES_DIR
            reason = "duplicate"
        else:
            folder = root / category_of(path)
            if by_date:
                folder = folder / datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m")
            reason = "organise"
        destination = unique_destination(folder / path.name, reserved)
        reserved.add(destination)
        moves.append({"src": path, "dst": destination, "reason": reason})
    return moves


def execute_moves(root: Path, moves: list[dict]) -> None:
    """Move the files and save an undo log (appended to any earlier log, so several runs can all be undone)."""
    log_path = root / LOG_NAME
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    for move in moves:
        move["dst"].parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(move["src"]), str(move["dst"]))
        log.append({"from": str(move["src"]), "to": str(move["dst"])})
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")


def undo(root: Path) -> int:
    """Reverse every move recorded in the log (newest first). Returns the number of files restored."""
    log_path = root / LOG_NAME
    if not log_path.exists():
        print(f"No {LOG_NAME} in {root} - nothing to undo.")
        return 0
    restored = 0
    for entry in reversed(json.loads(log_path.read_text(encoding="utf-8"))):
        source, target = Path(entry["to"]), Path(entry["from"])
        if source.exists() and not target.exists():
            shutil.move(str(source), str(target))
            restored += 1
        else:
            print(f"  skipped (missing or already occupied): {target.name}")
    for folder in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(folder.iterdir()):
            folder.rmdir()  # remove folders that became empty
    log_path.unlink()
    return restored


# --------------------------------------------------------------------------- #
# 3. Pretty printing
# --------------------------------------------------------------------------- #
def print_plan(root: Path, moves: list[dict], stats: dict, groups: list[list[Path]]) -> None:
    print(f"Scanned {stats['files']} files in {root}")
    print(f"Hashed only {stats['hashed']} of them (the rest have a unique size, so they cannot be duplicates).")
    for number, group in enumerate(groups, 1):
        print(f"  duplicate group {number}: keep '{group[0].name}', duplicates -> " + ", ".join(f"'{p.name}'" for p in group[1:]))
    print()
    for move in moves:
        label = "DUP " if move["reason"] == "duplicate" else "MOVE"
        print(f"  [{label}] {move['src'].name:28} -> {move['dst'].relative_to(root)}")
    print()


def print_tree(root: Path) -> None:
    """Simple text tree of a folder."""
    for path in sorted(root.rglob("*")):
        if path.name == LOG_NAME:
            continue
        depth = len(path.relative_to(root).parts) - 1
        print("   " + "    " * depth + ("[+] " if path.is_dir() else "    ") + path.name)


# --------------------------------------------------------------------------- #
# 4. Demo (a messy folder created from scratch, so the demo never touches real files)
# --------------------------------------------------------------------------- #
def build_demo_folder(folder: Path) -> None:
    """Create ~15 messy files, including renamed copies and a same-size-but-different pair."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    def make(name: str, content: bytes, year: int, month: int) -> None:
        path = folder / name
        path.write_bytes(content)
        stamp = datetime(year, month, 15, 12, 0).timestamp()
        os.utime(path, (stamp, stamp))  # pretend the file was last changed in that month

    photo = b"\xff\xd8\xff-pretend-this-is-a-holiday-photo-" * 50
    notes = b"Meeting notes: finish the ML lab report by Friday.\n" * 20
    make("holiday.jpg", photo, 2024, 3)
    make("IMG_0042 (1).jpg", photo, 2024, 5)          # same content, different name
    make("holiday_backup.jpg", photo, 2024, 6)        # a third copy
    make("notes.txt", notes, 2024, 1)
    make("notes_final_FINAL.txt", notes, 2024, 2)     # renamed copy
    make("report.pdf", b"%PDF-1.4 report " * 100, 2024, 4)
    make("budget.xlsx", b"PK-budget-sheet " * 80, 2024, 4)
    make("song.mp3", b"ID3-song-data " * 300, 2023, 11)
    make("lecture.mp4", b"video-frames " * 400, 2023, 12)
    make("project.zip", b"PK-zip-archive " * 120, 2024, 2)
    make("script.py", b"print('hello')\n" * 10, 2024, 5)
    make("data_a.bin", b"A" * 1000, 2024, 5)          # same SIZE as data_b ...
    make("data_b.bin", b"B" * 1000, 2024, 5)          # ... but DIFFERENT content -> not duplicates
    make("readme", b"a file without extension\n", 2024, 5)


def run_demo() -> dict:
    root = DEMO_DIR
    build_demo_folder(root)
    original = {p.name: file_hash(p) for p in list_files(root)}

    print("=== DEMO: a messy folder with renamed copies ===\n")
    files = list_files(root)
    groups, stats = find_duplicates(files)
    moves = plan_moves(root, files, groups, by_date=True)
    print_plan(root, moves, stats, groups)

    print("--- applying the plan (with --by-date) ---")
    execute_moves(root, moves)
    print_tree(root)

    print("\n--- undoing everything ---")
    restored = undo(root)
    after_undo = {p.name: file_hash(p) for p in list_files(root)}
    undo_ok = after_undo == original and not any(p.is_dir() for p in root.iterdir())
    print(f"Restored {restored} files. Folder identical to the original: {undo_ok}\n")

    insight = (
        f"Duplicates are found by content, not by name: {stats['duplicate_files']} redundant files were caught even "
        f"though they were renamed ('IMG_0042 (1).jpg', 'notes_final_FINAL.txt'). Only {stats['hashed']} of "
        f"{stats['files']} files needed hashing, because grouping by size first rules out the rest, and "
        f"{stats['same_size_but_different_content']} same-size pair (data_a.bin / data_b.bin) was correctly NOT flagged "
        f"since its content differs - size alone would have produced a false positive."
    )
    print("KEY INSIGHT:", insight)
    return {"stats": stats, "undo_ok": undo_ok, "insight": insight}


# --------------------------------------------------------------------------- #
# 5. Command line
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project 09 - duplicate finder and file organizer")
    parser.add_argument("folder", nargs="?", help="folder to organise (omit to run the safe demo)")
    parser.add_argument("--apply", action="store_true", help="actually move files (default is a dry run)")
    parser.add_argument("--by-date", action="store_true", help="also create YYYY-MM sub-folders from the modified date")
    parser.add_argument("--undo", action="store_true", help="reverse all --apply runs recorded in this folder's log")
    parser.add_argument("--demo", action="store_true", help="run the self-contained demo")
    parser.add_argument("--offline", action="store_true", help="(no effect) this project never uses the network")
    args = parser.parse_args(argv)

    if args.demo or not args.folder:
        run_demo()
        return 0

    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a folder: {root}")
        return 1
    if args.undo:
        print(f"Restored {undo(root)} files.")
        return 0

    files = list_files(root)
    groups, stats = find_duplicates(files)
    moves = plan_moves(root, files, groups, args.by_date)
    print_plan(root, moves, stats, groups)
    if args.apply:
        execute_moves(root, moves)
        print(f"Done. {len(moves)} files moved. Undo with:  python main.py \"{root}\" --undo")
    else:
        print("Dry run only - nothing was moved. Add --apply to do it for real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
