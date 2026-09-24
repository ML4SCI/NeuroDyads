#!/usr/bin/env python
"""
preprocessing/organize_edf_by_dyad.py
======================================
Stage a FLAT folder of converted EDF files into per-dyad subfolders WITHOUT
moving or modifying the originals (copy by default; --hardlink to save space).

This is optional -- several downstream scripts in this pipeline (e.g.
datasets/batch_speaker_first_stack.py) can scan a flat folder directly. This
tool is useful for tidy shared-drive uploads and for any tool that expects
dyad subfolders.

Parses filenames like:
    dyad20_60_speak.edf
    cut_dyad20_60_speak.edf      (cut_ prefix preserved in the copy)
    cut_dyad_20_60_listen.edf    (tolerated)
rest files are NOT copied for CEBRA input, but they ARE listed in the report.

Example
-------
  python organize_edf_by_dyad.py \
      --input-dir "/path/to/EDF Files cut" \
      --output-dir "/path/to/EDF Files cut organized"
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

FILE_RE = re.compile(
    r"^(?:cut_)?dyad_?(?P<dyad>\d+)_(?P<pid>\d+)_(?P<role>speak|listen|rest)\.edf$",
    re.IGNORECASE,
)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Stage flat EDF files into per-dyad subfolders (copy/hardlink).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dyad-min", type=int, default=None,
                   help="Optional lower bound on dyad id (inclusive).")
    p.add_argument("--dyad-max", type=int, default=None,
                   help="Optional upper bound on dyad id (inclusive).")
    p.add_argument("--include-rest", action="store_true", default=False,
                   help="Also stage rest files (default: skip, but report them).")
    p.add_argument("--hardlink", action="store_true", default=False,
                   help="Hardlink instead of copy (same filesystem only).")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Overwrite existing files in the output tree.")
    p.add_argument("--report-csv", default=None,
                   help="Report path. Default: <output-dir>/organize_report.csv")
    args = p.parse_args(argv)

    in_dir = Path(args.input_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    if not in_dir.is_dir():
        sys.exit(f"ERROR: --input-dir does not exist: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_csv = Path(args.report_csv).expanduser() if args.report_csv \
        else out_dir / "organize_report.csv"

    copied, skipped_rest, skipped_range, unknown, existed = [], [], [], [], []

    for name in sorted(os.listdir(in_dir)):
        src = in_dir / name
        if not src.is_file() or not name.lower().endswith(".edf"):
            continue
        m = FILE_RE.match(name)
        if not m:
            unknown.append(name)
            continue
        dyad = int(m.group("dyad"))
        role = m.group("role").lower()

        if args.dyad_min is not None and dyad < args.dyad_min:
            skipped_range.append(name); continue
        if args.dyad_max is not None and dyad > args.dyad_max:
            skipped_range.append(name); continue
        if role == "rest" and not args.include_rest:
            skipped_rest.append(name); continue

        dst_dir = out_dir / f"dyad{dyad}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / name  # preserve original filename (incl. any cut_ prefix)

        if dst.exists() and not args.overwrite:
            existed.append(str(dst))
            continue

        if args.hardlink:
            try:
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
            except OSError as exc:
                print(f"  hardlink failed ({exc}); copying instead: {name}")
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        copied.append((name, str(dst)))

    # ---- report ----
    with open(report_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["status", "filename", "destination_or_reason"])
        for n, d in copied:
            w.writerow(["copied", n, d])
        for n in skipped_rest:
            w.writerow(["skipped_rest", n, "rest file (not used for CEBRA input)"])
        for n in skipped_range:
            w.writerow(["skipped_range", n, "outside --dyad-min/--dyad-max"])
        for d in existed:
            w.writerow(["exists", Path(d).name, d + " (use --overwrite)"])
        for n in unknown:
            w.writerow(["unknown_pattern", n, "did not match dyad filename regex"])

    print(f"Organized into {out_dir}")
    print(f"  copied/linked   : {len(copied)}")
    print(f"  skipped rest    : {len(skipped_rest)}  {skipped_rest if skipped_rest else ''}")
    print(f"  skipped (range) : {len(skipped_range)}")
    print(f"  already existed : {len(existed)}")
    print(f"  unknown pattern : {len(unknown)}  {unknown if unknown else ''}")
    print(f"  report          : {report_csv}")


if __name__ == "__main__":
    main()
