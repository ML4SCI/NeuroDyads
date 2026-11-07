#!/usr/bin/env python
"""
scripts/resample_33dyad_to_250.py
=================================
Downsample the 33-dyad speaker-first SOURCE recordings to a uniform 250 Hz.

Why: the audit showed this dataset mixes sampling rates -- 14 of the 33 included
dyads were recorded at 1000 Hz and 19 at 250 Hz. Because the stacker cropped to a
fixed SAMPLE COUNT (180,500), the two groups contributed very different amounts
of real time (90.2 s vs 361 s per role), and the fixed `time_offsets=10` spanned
10 ms for one group and 40 ms for the other. Resampling everything to 250 Hz
removes both problems at once: once the rate is uniform, a sample-count crop is
automatically a duration crop.

Source note: these EDFs live under "Alpha-Removed EEG Datafiles" but were shown
to be UNFILTERED full-band recordings (numerically identical to the Alpha-Only,
Beta-Removed and Gamma-Removed uploads, max abs difference 0.000e+00). The output
prefix is therefore `fullband250`, which is what the data actually is.

Resampling uses MNE's `Raw.resample`, which applies an anti-aliasing filter
before decimation. Files already at 250 Hz are re-exported unchanged so that
every downstream file has gone through an identical read/write path.
"""
from __future__ import annotations

import argparse, csv, re, sys, time, warnings
from pathlib import Path

import numpy as np

try:
    import mne
except ImportError:
    sys.exit("ERROR: mne required")

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SRC_RE = re.compile(r"^(?:minusAlpha|noAlpha|alphaRemoved)_dyad_?(\d+)_(\d+)_"
                    r"(speak|listen|rest)\.edf$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--qc-dir", required=True)
    ap.add_argument("--prefix", default="fullband250")
    ap.add_argument("--source-prefix", default="minusAlpha",
                    help="which of the duplicate upload prefixes to read")
    ap.add_argument("--target-sfreq", type=float, default=250.0)
    ap.add_argument("--skip-rest", action="store_true", default=True)
    ap.add_argument("--include-rest", action="store_true", help="also process rest recordings")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    qcd = Path(args.qc_dir); qcd.mkdir(parents=True, exist_ok=True)

    jobs = []
    for p in sorted(Path(args.input_root).rglob("*.edf")):
        m = SRC_RE.match(p.name)
        if not m:
            continue
        if not p.name.lower().startswith(args.source_prefix.lower()):
            continue
        d, pid, role = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        if args.skip_rest and (not args.include_rest) and role == "rest":
            continue
        jobs.append((p, d, pid, role))
    if args.shard_count > 1:
        jobs = jobs[args.shard_index::args.shard_count]
    print(f"{len(jobs)} recordings (prefix '{args.source_prefix}', shard "
          f"{args.shard_index+1}/{args.shard_count}) -> {args.target_sfreq:g} Hz", flush=True)

    rows = []
    for i, (src, d, pid, role) in enumerate(jobs, 1):
        dst = out / f"{args.prefix}_dyad{d}_{pid}_{role}.edf"
        if dst.exists() and not args.force:
            print(f"[{i}/{len(jobs)}] {dst.name} exists, skip", flush=True)
            continue
        t0 = time.time()
        try:
            raw = mne.io.read_raw_edf(str(src), preload=True, verbose="ERROR")
            sf0 = float(raw.info["sfreq"]); n0 = int(raw.n_times)
            dur0 = n0 / sf0
            did = False
            if abs(sf0 - args.target_sfreq) > 1e-6:
                raw.resample(args.target_sfreq, verbose="ERROR")
                did = True
            sf1 = float(raw.info["sfreq"]); n1 = int(raw.n_times)
            if dst.exists():
                dst.unlink()
            raw.export(str(dst), fmt="edf", physical_range="auto",
                       overwrite=True, verbose="ERROR")
            chk = mne.io.read_raw_edf(str(dst), preload=False, verbose="ERROR")
            ok = (abs(float(chk.info["sfreq"]) - args.target_sfreq) < 1e-6
                  and len(chk.ch_names) == len(raw.ch_names))
            rows.append({"source": src.name, "output": dst.name, "dyad": d,
                         "participant": pid, "role": role,
                         "sfreq_in": sf0, "sfreq_out": sf1,
                         "n_samples_in": n0, "n_samples_out": n1,
                         "duration_in_s": round(dur0, 3),
                         "duration_out_s": round(n1 / sf1, 3),
                         "resampled": did, "n_channels": len(raw.ch_names),
                         "verify_ok": bool(ok), "seconds": round(time.time() - t0, 1)})
            chk.close(); raw.close()
            print(f"[{i}/{len(jobs)}] {src.name} {sf0:g}->{sf1:g} Hz  "
                  f"{dur0:.1f}s  verify={'ok' if ok else 'FAIL'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as ex:
            print(f"[{i}/{len(jobs)}] {src.name} ERROR: {ex}", flush=True)
            rows.append({"source": src.name, "output": "", "dyad": d, "participant": pid,
                         "role": role, "verify_ok": False, "error": str(ex)[:200]})

    if rows:
        f = qcd / f"resample250_qc_shard{args.shard_index}.csv"
        cols = sorted({c for r in rows for c in r})
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
        good = [r for r in rows if r.get("verify_ok")]
        rs = sum(1 for r in good if r.get("resampled"))
        print(f"\nverified {len(good)}/{len(rows)}; {rs} actually resampled")
        if good:
            du = [r["duration_out_s"] for r in good]
            print(f"output durations {min(du):.1f}-{max(du):.1f} s "
                  f"(global min -> crop = {int(min(du)*args.target_sfreq)} samples/role)")
    print("RESAMPLE250_DONE")


if __name__ == "__main__":
    main()
