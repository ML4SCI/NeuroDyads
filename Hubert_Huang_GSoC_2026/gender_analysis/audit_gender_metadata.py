#!/usr/bin/env python
"""
gender_analysis/audit_gender_metadata.py
=========================================
Audit explicit gender metadata and intersect it with the finalized 250 Hz
speaker-first dataset.

Gender is read ONLY from the demographic spreadsheet's 'Full-Info' sheet,
columns 'Speaker Gender' / 'Listener Gender'. Nothing is inferred from names,
participant IDs, or file order. The output JSON
(`gender_metadata_audit.json`) is consumed by every other script in this
gender_analysis/ folder, so run this one first.

Usage
-----
  python audit_gender_metadata.py \
      --demographics-xlsx /path/to/dyad_demographics.xlsx \
      --speakerfirst-manifest /path/to/fullband250_manifest.csv \
      --resampled-dir "/path/to/Resampled250 (33dyad)" \
      --raw-dir "/path/to/EDF Files cut" \
      --out-dir /path/to/output

Note: the demographic spreadsheet contains participant-identifiable
information and is intentionally NOT included in this submission (see the
top-level README's "Dataset description" section). Point --demographics-xlsx
at your own copy of that file to reproduce this audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import collections
import re
from pathlib import Path

import pandas as pd


def scan_roles(root: Path, rx: str):
    """Which speak/listen/rest EDFs exist per participant under `root`."""
    got = collections.defaultdict(set)
    for p in Path(root).rglob("*.edf"):
        m = re.search(rx, p.name, re.I)
        if m:
            got[int(m.group(2))].add(m.group(3).lower())
    return got


def main():
    ap = argparse.ArgumentParser(
        description="Audit explicit participant/dyad gender labels and "
                    "intersect them with the finalized 250 Hz dataset.")
    ap.add_argument("--demographics-xlsx", required=True, type=Path,
                    help="Path to the dyad/participant demographics "
                         "spreadsheet (must have a 'Full-Info' sheet with "
                         "'Speaker Gender' / 'Listener Gender' columns). Not "
                         "included in this repository -- see README.")
    ap.add_argument("--speakerfirst-manifest", required=True, type=Path,
                    help="Manifest CSV produced by "
                         "datasets/batch_speaker_first_stack.py for the "
                         "finalized 250 Hz full-band speaker-first dataset.")
    ap.add_argument("--resampled-dir", type=Path, default=None,
                    help="Optional: folder of uniform-250 Hz role EDFs "
                         "(speak/listen/rest), to report role/state "
                         "availability per participant.")
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="Optional: a second EDF folder to check role/state "
                         "availability against (e.g. the original cut EDFs).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Directory to write gender_metadata_audit.json into.")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.demographics_xlsx, sheet_name="Full-Info")
    d = df[df["dyad_id"].notna()].copy()
    d["__d"] = d["dyad_id"].astype(str).str.extract(r"dyad0*(\d+)")[0].astype(float)

    # ---- participant -> gender, from explicit labels only ----
    pg, conflicts, missing = {}, [], []
    for _, r in d.iterrows():
        for idc, gc in (("Speaker ID", "Speaker Gender"), ("Listener ID", "Listener Gender")):
            try:
                pid = int(r[idc])
            except Exception:
                continue
            g = r[gc]
            if not isinstance(g, str) or g.strip().upper() not in ("M", "F"):
                missing.append((pid, str(g))); continue
            g = g.strip().upper()
            if pid in pg and pg[pid] != g:
                conflicts.append((pid, pg[pid], g))
            pg[pid] = g

    # ---- dyad -> type, from the two participants' explicit genders ----
    dyad_members = collections.defaultdict(set)
    for _, r in d.iterrows():
        dd = r["__d"]
        if dd != dd:
            continue
        for idc in ("Speaker ID", "Listener ID"):
            try:
                dyad_members[int(dd)].add(int(r[idc]))
            except Exception:
                pass

    dyad_type, dyad_bad = {}, []
    for dd, mem in dyad_members.items():
        gs = sorted(pg.get(p) for p in mem)
        if any(g is None for g in gs) or len(mem) != 2:
            dyad_bad.append((dd, sorted(mem), gs)); continue
        dyad_type[dd] = "MM" if gs == ["M", "M"] else ("FF" if gs == ["F", "F"] else "MF")

    # ---- cross-check against the sheet's own congruence column, if present ----
    sheet_ct = {}
    if "Gender Congruence" in d.columns:
        for _, r in d.iterrows():
            dd = r["__d"]
            if dd == dd:
                sheet_ct[int(dd)] = str(r["Gender Congruence"])
    mismatch = []
    for dd, t in dyad_type.items():
        exp = {"MM": "M-Matched", "FF": "F-Matched", "MF": "Mixed"}[t]
        if dd in sheet_ct and sheet_ct.get(dd) != exp:
            mismatch.append((dd, t, sheet_ct.get(dd)))

    print("=== 1. GENDER METADATA AUDIT ===")
    print(f"source: {args.demographics_xlsx.name} sheet 'Full-Info' (explicit columns only)")
    print(f"participants with an explicit gender label : {len(pg)}")
    print(f"  M = {sum(1 for v in pg.values() if v=='M')}   F = {sum(1 for v in pg.values() if v=='F')}")
    print(f"label conflicts across rows               : {len(conflicts)} {conflicts[:5]}")
    print(f"rows with missing/ambiguous gender        : {len(missing)} {missing[:5]}")
    print(f"dyads with a derivable type               : {len(dyad_type)}")
    print(f"dyads NOT derivable                       : {len(dyad_bad)} {dyad_bad[:5]}")
    print(f"disagreements vs sheet 'Gender Congruence': {len(mismatch)} {mismatch[:5]}")
    print(f"ALL dyads in sheet, by type: {dict(collections.Counter(dyad_type.values()))}")

    # ---- intersect with the finalized 250 Hz speaker-first dataset ----
    rows = list(csv.DictReader(open(args.speakerfirst_manifest)))
    kept = [r for r in rows if str(r.get("flagged", "")).lower() != "true"]
    used_dyads = sorted({int(r["dyad_id"]) for r in kept})
    print("\n=== FINALIZED 250 Hz SPEAKER-FIRST DATASET ===")
    print(f"files (unflagged) {len(kept)}   dyads {len(used_dyads)}")
    have = [x for x in used_dyads if x in dyad_type]
    lack = [x for x in used_dyads if x not in dyad_type]
    print(f"dyads with gender type: {len(have)}   without: {len(lack)} {lack}")
    ct = collections.Counter(dyad_type[x] for x in have)
    print(f"DYAD TYPE COUNTS (analysis set): {dict(ct)}")
    for t in ("MM", "FF", "MF"):
        ds = sorted(x for x in have if dyad_type[x] == t)
        nf = sum(1 for r in kept if int(r["dyad_id"]) in ds)
        print(f"  {t}: {len(ds)} dyads, {nf} files -> {ds}")

    # individual counts within the analysis set
    ppl = set()
    for r in kept:
        ppl.add(int(r["speaker_id"])); ppl.add(int(r["listener_id"]))
    print(f"\nparticipants in analysis set: {len(ppl)}  "
          f"M={sum(1 for p in ppl if pg.get(p)=='M')} F={sum(1 for p in ppl if pg.get(p)=='F')} "
          f"unlabelled={sum(1 for p in ppl if p not in pg)}")

    # ---- which participants have speak / listen / rest EDFs ----
    print("\n=== ROLE/STATE AVAILABILITY ===")
    for lab, root in [("resampled-dir", args.resampled_dir), ("raw-dir", args.raw_dir)]:
        if root is None:
            continue
        if not Path(root).exists():
            print(f"  {lab} ({root}): MISSING"); continue
        got = scan_roles(root, r"dyad_?(\d+)_(\d+)_(speak|listen|rest)\.edf$")
        c = collections.Counter(frozenset(v) for v in got.values())
        print(f"  {lab} ({root}): {len(got)} participants")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"      {sorted(k)}: {v} participants")
        nrest = sum(1 for v in got.values() if "rest" in v)
        print(f"      participants WITH rest: {nrest}")

    json.dump({"participant_gender": {str(k): v for k, v in sorted(pg.items())},
               "dyad_type_all": {str(k): v for k, v in sorted(dyad_type.items())},
               "dyad_type_analysis_set": {str(k): dyad_type[k] for k in have},
               "analysis_dyad_counts": dict(ct),
               "conflicts": conflicts, "missing_rows": missing,
               "dyads_not_derivable": dyad_bad, "sheet_mismatches": mismatch},
              open(out / "gender_metadata_audit.json", "w"), indent=2)
    print(f"\nwrote {out / 'gender_metadata_audit.json'}")


if __name__ == "__main__":
    main()
