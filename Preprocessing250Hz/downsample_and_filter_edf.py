#!/usr/bin/env python
"""
scripts/downsample_and_filter_edf.py
====================================
Downsample EDFs to 250 Hz (if needed) and generate band-removed versions with
our OWN filters (Michelle's uploaded filtered EDFs were identical to raw, so we
filter locally).

Filter spec (per Michelle):
  design            : FIR, firwin
  window            : Hamming
  phase             : zero-phase (phase='zero')
  transition band   : 1.0 Hz each edge (2.0 Hz for the gamma low-pass edge)

Bands to remove (each produces one output dataset):
  minusDelta  : high-pass 4 Hz            (removes 0-4 Hz)
  minusTheta  : band-stop 4-8 Hz
  minusAlpha  : band-stop 8-12 Hz
  minusBeta   : band-stop 12-30 Hz
  minusGamma  : low-pass 30 Hz            (removes >30 Hz)
  fullband    : no filtering (reference; re-exported identically)

MNE convention: raw.filter(l_freq, h_freq) with l_freq > h_freq = BAND-STOP.
Output: <out-root>/<band>/<tag>_dyad{N}_{P}_{role}.edf  (flat, for stacking).
"""
from __future__ import annotations

import argparse, re, sys, warnings
from pathlib import Path

try:
    import mne
except ImportError:
    sys.exit("ERROR: mne required")

FILE_RE = re.compile(r"^(?:cut_)?dyad_?(\d+)_(\d+)_(speak|listen|rest)\.edf$", re.I)

# band tag -> (l_freq, h_freq, l_trans, h_trans)  [MNE: l>h => band-stop]
BANDS = {
    "minusDelta": (4.0,  None, 1.0, None),   # high-pass 4
    "minusTheta": (8.0,  4.0,  1.0, 1.0),    # band-stop 4-8
    "minusAlpha": (12.0, 8.0,  1.0, 1.0),    # band-stop 8-12
    "minusBeta":  (30.0, 12.0, 1.0, 1.0),    # band-stop 12-30
    "minusGamma": (None, 30.0, None, 2.0),   # low-pass 30 (2.0 Hz edge)
    "fullband":   (None, None, None, None),  # reference (no filter)
}


def apply_band(raw, l, h, lt, ht):
    if l is None and h is None:
        return raw  # fullband
    kw = dict(fir_design="firwin", fir_window="hamming", phase="zero",
              method="fir", verbose="ERROR")
    if lt is not None:
        kw["l_trans_bandwidth"] = lt
    if ht is not None:
        kw["h_trans_bandwidth"] = ht
    raw.filter(l_freq=l, h_freq=h, **kw)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="folder of source EDFs")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--target-sfreq", type=float, default=250.0)
    ap.add_argument("--bands", default=",".join(BANDS), help="comma list of band tags")
    ap.add_argument("--only-file", default=None, help="process just this basename (QC)")
    args = ap.parse_args()

    in_dir, out_root = Path(args.input_dir), Path(args.out_root)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    edfs = sorted(in_dir.glob("*.edf"))
    if args.only_file:
        edfs = [f for f in edfs if f.name == args.only_file]
    print(f"source EDFs: {len(edfs)} | bands: {bands} | target {args.target_sfreq} Hz")

    for band in bands:
        (out_root / band).mkdir(parents=True, exist_ok=True)

    for i, edf in enumerate(edfs, 1):
        m = FILE_RE.match(edf.name)
        if not m:
            print(f"  skip (name) {edf.name}"); continue
        dyad, pid, role = m.group(1), m.group(2), m.group(3).lower()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
        if abs(base.info["sfreq"] - args.target_sfreq) > 1e-6:
            base.resample(args.target_sfreq, verbose="ERROR")
        for band in bands:
            l, h, lt, ht = BANDS[band]
            raw = apply_band(base.copy(), l, h, lt, ht)
            out = out_root / band / f"{band}_dyad{dyad}_{pid}_{role}.edf"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.export(str(out), fmt="edf", overwrite=True)
        if i % 10 == 0 or i == len(edfs):
            print(f"  [{i}/{len(edfs)}] {edf.name} -> {len(bands)} bands")
    print("DONE")


if __name__ == "__main__":
    main()
