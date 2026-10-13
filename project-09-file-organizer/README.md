# Project 09 - Duplicate Finder and Bulk File Organizer (automation, no ML)
**Author:** Sanchit Kumar Mishra

### Problem Statement
A Downloads folder fills up with renamed copies and mixed file types. Automate cleaning it: find duplicates, sort files by type (and optionally month), and make it all reversible.

### Dataset
None - it works on any local folder. A `--demo` mode (the default) builds a messy 14-file folder from scratch in `outputs/demo_folder`, including renamed copies and a same-size-but-different pair, so it can be shown safely in class.

### Approach
1. **Duplicates by content:** group files by size first (cheap), then hash only the files that share a size (SHA-256, read in chunks). Same hash = same content, whatever the name.
2. **Organise:** move files into `Images / Documents / Audio / Video / Archives / Code / Other`, optionally into `YYYY-MM` sub-folders from the modified date. The oldest copy of a duplicate group stays; the rest go to `_duplicates/` (nothing is ever deleted).
3. **Safety:** dry run by default (`--apply` is needed to move anything), name collisions get `_1`, `_2` suffixes, hidden files are skipped, and an undo log (`organizer_log.json`) lets `--undo` restore everything.
4. Standard library only.

### Result
Demo run: 14 files scanned -> only 8 hashed -> 2 duplicate groups (3 redundant files) moved to `_duplicates/`; files sorted into 7 category folders (incl. `Other`) with month sub-folders; `--undo` restored all 14 files and the folder had the same file names and contents as before (checked by hash in the script).

### Key Insight
Duplicates must be found by content, not name: grouping by size first meant only 8 of 14 files were hashed, and a same-size pair with different content (`data_a.bin` / `data_b.bin`) was correctly *not* flagged.

### How to Run
```bash
python main.py                              # safe demo
python main.py ~/Downloads                  # dry run: shows the plan only
python main.py ~/Downloads --apply --by-date
python main.py ~/Downloads --undo           # put everything back
```
No `pip install` needed.
