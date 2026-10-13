"""
run_all.py - run all 10 mini-projects and report PASS / FAIL.
Author : Sanchit Kumar Mishra

Usage (from the repository root):
    python run_all.py            # offline mode: bundled data, no internet needed (reproducible)
    python run_all.py --live     # live mode: try the real APIs / downloads first (needs internet)
    python run_all.py --only 03 07   # run just some projects

What it checks: every project's main.py exits with code 0 and prints a "KEY INSIGHT:" line.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all mini-projects")
    parser.add_argument("--live", action="store_true", help="do NOT pass --offline (use real data sources if reachable)")
    parser.add_argument("--only", nargs="*", help="project numbers to run, e.g. --only 01 05")
    args = parser.parse_args()

    projects = sorted(p for p in ROOT.glob("project-*") if (p / "main.py").exists())
    if args.only:
        projects = [p for p in projects if any(p.name.startswith(f"project-{n.zfill(2)}") for n in args.only)]

    mode = "LIVE (real sources first, fallback if unreachable)" if args.live else "OFFLINE (bundled data)"
    print(f"Running {len(projects)} projects in {mode} mode with {sys.executable}\n")

    results = []
    for project in projects:
        command = [sys.executable, "main.py"] + ([] if args.live else ["--offline"])
        start = time.time()
        completed = subprocess.run(command, cwd=project, capture_output=True, text=True)
        seconds = time.time() - start

        has_insight = "KEY INSIGHT:" in completed.stdout
        ok = completed.returncode == 0 and has_insight
        data_line = next((l for l in completed.stdout.splitlines() if l.startswith("[data] source")), "")
        results.append((project.name, ok, seconds))
        print(f"[{'PASS' if ok else 'FAIL'}] {project.name:42} {seconds:5.1f}s  {data_line[:70]}")
        if not ok:
            print("   --- stderr (last 15 lines) ---")
            print("   " + "\n   ".join(completed.stderr.strip().splitlines()[-15:]))
            if completed.returncode == 0:
                print("   (script ran but printed no 'KEY INSIGHT:' line)")

    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed}/{len(results)} projects passed.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
