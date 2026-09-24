#!/usr/bin/env python
"""
scripts/audit_edf_sampling_rates.py
===================================
TASK 1 -- sampling-rate / integrity audit of every source EDF.

Recursively inspects one or more EDF source roots and records, per file:
filename, path, source_root, dyad, participant, role, sfreq, n_channels,
n_samples, duration, SHA-256, VREF present, read errors.

Read-only. Never modifies originals.
"""
from __future__ import annotations

import argparse, csv, hashlib, re, sys, warnings
from pathlib import Path

import numpy as np

try:
    import mne
except ImportError:
    sys.exit("ERROR: mne required")

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

COLS = ["source_root", "filename", "rel_path", "full_path", "dyad", "participant", "role",
        "sfreq", "n_channels", "n_samples", "duration_sec", "sha256", "has_vref",
        "n_eeg_channels", "read_error"]


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_ids(path: Path):
    """Extract dyad / participant / role from filename or parent folder."""
    name = path.name
    dyad = participant = role = None
    m = re.search(r"dyad0*(\d+)", name, re.I) or re.search(r"dyad0*(\d+)", str(path.parent), re.I)
    if m:
        dyad = int(m.group(1))
    m = re.search(r"dyad0*\d+[_-](\d+)", name, re.I)
    if m:
        participant = int(m.group(1))
    else:
        m = re.search(r"_(\d+)_(speak|listen|rest)", name, re.I)
        if m:
            participant = int(m.group(1))
    m = re.search(r"(speak|listen|rest)", name, re.I)
    if m:
        role = m.group(1).lower()
    return dyad, participant, role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 (faster)")
    args = ap.parse_args()

    outc = Path(args.out_csv); outc.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for root in args.roots:
        rp = Path(root)
        if not rp.exists():
            print(f"!! missing root: {rp}")
            continue
        edfs = sorted(rp.rglob("*.edf"))
        print(f"[{rp.name}] {len(edfs)} EDF files", flush=True)
        for i, p in enumerate(edfs, 1):
            d, part, role = parse_ids(p)
            rec = {"source_root": rp.name, "filename": p.name,
                   "rel_path": str(p.relative_to(rp)), "full_path": str(p),
                   "dyad": d, "participant": part, "role": role, "read_error": ""}
            try:
                raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
                rec["sfreq"] = float(raw.info["sfreq"])
                rec["n_channels"] = len(raw.ch_names)
                rec["n_samples"] = int(raw.n_times)
                rec["duration_sec"] = round(raw.n_times / raw.info["sfreq"], 4)
                names_up = [c.upper() for c in raw.ch_names]
                rec["has_vref"] = any("VREF" in c for c in names_up)
                rec["n_eeg_channels"] = sum(1 for c in names_up if "VREF" not in c)
                raw.close()
            except Exception as ex:
                rec["read_error"] = f"{type(ex).__name__}: {ex}"[:300]
            if not args.no_hash:
                try:
                    rec["sha256"] = sha256(p)
                except Exception as ex:
                    rec["sha256"] = ""; rec["read_error"] += f" | hash: {ex}"
            recs.append(rec)
            if i % 25 == 0 or i == len(edfs):
                print(f"  [{i}/{len(edfs)}]", flush=True)

    with open(outc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(recs)
    print(f"wrote {outc} ({len(recs)} rows)")

    # ---------- summary ----------
    import collections
    L = ["# EDF sampling-rate audit", "",
         f"Total EDF files scanned: **{len(recs)}**", ""]
    by_root = collections.defaultdict(list)
    for r in recs:
        by_root[r["source_root"]].append(r)

    L += ["## Per source root", "",
          "| source root | files | sfreq values | n_channels | duration (s) min/max | VREF present | read errors |",
          "|---|---|---|---|---|---|---|"]
    for root, rs in by_root.items():
        ok = [r for r in rs if not r["read_error"]]
        sf = sorted(set(r.get("sfreq") for r in ok))
        ch = sorted(set(r.get("n_channels") for r in ok))
        du = [r.get("duration_sec") for r in ok if r.get("duration_sec") is not None]
        vr = sum(1 for r in ok if r.get("has_vref"))
        errs = sum(1 for r in rs if r["read_error"])
        L.append(f"| {root} | {len(rs)} | {sf} | {ch} | "
                 f"{min(du) if du else '-'} / {max(du) if du else '-'} | {vr}/{len(ok)} | {errs} |")

    L += ["", "## Resampling decision (target 250 Hz)", "",
          "| source root | sfreq | files | action |", "|---|---|---|---|"]
    for root, rs in by_root.items():
        c = collections.Counter(r.get("sfreq") for r in rs if not r["read_error"])
        for sf, n in sorted(c.items(), key=lambda kv: -kv[1]):
            if sf is None:
                act = "UNREADABLE - excluded"
            elif sf > 250:
                act = f"RESAMPLE {sf:g} -> 250 Hz (MNE anti-aliased)"
            elif sf == 250:
                act = "KEEP as-is (already 250 Hz)"
            else:
                act = f"**FLAG** {sf:g} Hz < 250 Hz - do NOT upsample; excluded from matched analyses"
            L.append(f"| {root} | {sf} | {n} | {act} |")

    dups = collections.defaultdict(list)
    for r in recs:
        if r.get("sha256"):
            dups[r["sha256"]].append(f"{r['source_root']}/{r['rel_path']}")
    cross = {h: v for h, v in dups.items() if len(v) > 1}
    L += ["", "## Byte-identical files (SHA-256 collisions)", ""]
    if cross:
        L.append(f"**{len(cross)} hash groups contain >1 file.** This is how we detect "
                 "'filtered' uploads that are actually copies of the unfiltered data.")
        L.append("")
        L.append("| sha256 (first 12) | n | files |")
        L.append("|---|---|---|")
        for h, v in sorted(cross.items(), key=lambda kv: -len(kv[1]))[:60]:
            L.append(f"| `{h[:12]}` | {len(v)} | {'<br>'.join(v[:6])}{' …' if len(v) > 6 else ''} |")
        if len(cross) > 60:
            L.append(f"\n… and {len(cross)-60} further groups (see CSV).")
    else:
        L.append("None - every scanned EDF is byte-unique.")

    errs = [r for r in recs if r["read_error"]]
    L += ["", "## Read errors", ""]
    if errs:
        L.append("| file | error |"); L.append("|---|---|")
        for r in errs[:40]:
            L.append(f"| {r['source_root']}/{r['rel_path']} | {r['read_error']} |")
    else:
        L.append("None.")

    Path(args.out_md).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out_md}")
    print("EDF_AUDIT_DONE")


if __name__ == "__main__":
    main()
