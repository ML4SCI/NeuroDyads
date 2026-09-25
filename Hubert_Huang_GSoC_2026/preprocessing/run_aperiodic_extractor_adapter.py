#!/usr/bin/env python
"""
preprocessing/run_aperiodic_extractor_adapter.py
==================================================
Run a collaborator-authored aperiodic (non-oscillatory) extractor on THIS
project's EDFs.

NOTE ON PROVENANCE: the underlying extraction script (the "source script" you
point --source-script at below) was written by another NeuroDyads collaborator
(Michelle), not by this contributor, and is therefore NOT bundled in this
submission -- request it directly from that collaborator, or write your own
module exposing the same four names described below. This file is only an
ADAPTER around it: it is imported unmodified at runtime and its
`reconstruct_aperiodic_channel()` does all the science, so the method here is
verbatim: Welch PSD -> specparam/FOOOF fit -> replace FFT magnitude inside
the fit range with the fitted aperiodic curve (phase preserved) -> inverse FFT.

WHY AN ADAPTER IS NEEDED
------------------------
Her script selects EEG channels with

    EEG_CHANNEL_PATTERN = r'^(E\\d+|Cz)$'        # EGI/Netstation naming: E1, E2, Cz

but the EDFs in this project name their channels

    'EEG 1', 'EEG 2', ... 'EEG 64', 'EEG VREF', 'Status'

so the pattern matches **zero** channels. Every channel then hits
`if not is_eeg_channel(label): continue` and passes through untouched, and the
output file is just the input re-exported. That is precisely what we measured in
the uploaded Non-Oscillatory dataset: identical spectra to the raw recordings
(alpha prominence -0.89 dB in both) and a max absolute difference of 1.24e-07,
which is EDF write quantisation, not filtering.

This adapter changes exactly two things and nothing else:
  1. the channel pattern, to match this project's naming
  2. input/output paths and output filenames (so the existing speaker-first
     stacker can index them)

All fit settings are taken from her file, not re-chosen here.

QC: for every file we measure alpha prominence (8-12 Hz power relative to a 1/f
line fitted on 2-40 Hz excluding 7-14 Hz) before and after. A genuine aperiodic
reconstruction must come out near 0 dB. This is the check the previous upload
failed.

Expected interface of the source script (four names, all module-level):
    is_eeg_channel(label, pattern) -> bool
    reconstruct_aperiodic_channel(x, sf, fit_range, aperiodic_mode, label=...) -> np.ndarray
    FIT_RANGE, APERIODIC_MODE, PEAK_WIDTH_LIMITS, MAX_N_PEAKS,
    PEAK_THRESHOLD, MIN_PEAK_HEIGHT, WELCH_WINDOW_SEC, WELCH_OVERLAP_FRAC,
    _SPECPARAM_PACKAGE
"""
from __future__ import annotations

import argparse, csv, importlib.util, re, sys, warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch

import mne
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# this project's channel naming; VREF and Status deliberately excluded
OUR_EEG_PATTERN = r"^(E\d+|Cz|EEG\s*\d+)$"
SRC_RE = re.compile(r"^(?:cut_)?dyad_?(\d+)_(\d+)_(speak|listen|rest)\.edf$", re.I)


def load_source_module(source_script: Path):
    """Import the collaborator's extraction script as a module, unmodified."""
    if not source_script.exists():
        raise FileNotFoundError(
            f"--source-script not found: {source_script}\n"
            "This adapter requires the aperiodic-extraction script written by "
            "the NeuroDyads collaborator who provided it (see module docstring "
            "for the expected interface). It is not bundled in this submission.")
    spec = importlib.util.spec_from_file_location("aperiodic_source", source_script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aperiodic_source"] = mod
    spec.loader.exec_module(mod)
    return mod


def alpha_prominence(x, sf, secs=60):
    """dB of 8-12 Hz power above a 1/f line fitted on 2-40 Hz excluding 7-14 Hz."""
    n = int(min(x.shape[1], secs * sf))
    f, P = welch(x[:, :n], fs=sf, nperseg=int(4 * sf), axis=-1)
    Pm = P.mean(0)
    m = (f >= 2) & (f <= 40) & (Pm > 0)
    fitm = m & ~((f >= 7) & (f <= 14))
    if fitm.sum() < 5:
        return None, None
    co = np.polyfit(np.log10(f[fitm]), np.log10(Pm[fitm]), 1)
    am = (f >= 8) & (f <= 12)
    pred = 10 ** np.polyval(co, np.log10(f[am]))
    return float(10 * np.log10(Pm[am].mean() / pred.mean())), float(co[0])


def main():
    ap = argparse.ArgumentParser(
        description="Adapter that runs a collaborator-provided aperiodic "
                    "(non-oscillatory) EEG extractor on this project's EDFs, "
                    "fixing only the EEG-channel-name pattern (see module "
                    "docstring).")
    ap.add_argument("--source-script", required=True, type=Path,
                    help="Path to the collaborator's aperiodic-extraction "
                         ".py file (not bundled in this submission -- request "
                         "it from the collaborator who wrote it).")
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="Folder of source EDFs (uniform 250 Hz).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Folder to write aperiodic-only EDFs into.")
    ap.add_argument("--qc-dir", required=True, type=Path,
                    help="Folder to write the per-file alpha-prominence QC CSV.")
    ap.add_argument("--prefix", default="nonOsci",
                    help="Filename prefix for output EDFs (default: nonOsci).")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="This shard's index, for splitting work across "
                         "parallel invocations (default: 0).")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="Total number of shards (default: 1, i.e. no sharding).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N matching files (for testing).")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess files even if the output already exists.")
    args = ap.parse_args()

    mnos = load_source_module(args.source_script)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    qcd = Path(args.qc_dir); qcd.mkdir(parents=True, exist_ok=True)

    print(f"using source module: {args.source_script}")
    print(f"  package          : {mnos._SPECPARAM_PACKAGE}")
    print(f"  FIT_RANGE        : {mnos.FIT_RANGE}")
    print(f"  APERIODIC_MODE   : {mnos.APERIODIC_MODE}")
    print(f"  PEAK_WIDTH_LIMITS: {mnos.PEAK_WIDTH_LIMITS}   MAX_N_PEAKS: {mnos.MAX_N_PEAKS}")
    print(f"  PEAK_THRESHOLD   : {mnos.PEAK_THRESHOLD}   MIN_PEAK_HEIGHT: {mnos.MIN_PEAK_HEIGHT}")
    print(f"  WELCH            : {mnos.WELCH_WINDOW_SEC}s / overlap {mnos.WELCH_OVERLAP_FRAC}")
    print(f"  channel pattern  : {OUR_EEG_PATTERN}   (adapter override)")

    edfs = sorted(Path(args.input_dir).glob("*.edf"))
    if args.limit:
        edfs = edfs[:args.limit]
    jobs = []
    for p in edfs:
        m = SRC_RE.match(p.name)
        if not m:
            print(f"  skip (name) {p.name}"); continue
        d, pid, role = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        if role == "rest":
            continue
        jobs.append((p, d, pid, role))
    if args.shard_count > 1:
        jobs = jobs[args.shard_index::args.shard_count]
    print(f"{len(jobs)} recordings to process "
          f"(shard {args.shard_index+1}/{args.shard_count})\n", flush=True)

    rows = []
    for i, (src, d, pid, role) in enumerate(jobs, 1):
        dst = out / f"{args.prefix}_dyad{d}_{pid}_{role}.edf"
        if dst.exists() and not args.force:
            print(f"[{i}/{len(jobs)}] {src.name} -> exists, skip", flush=True)
            continue
        t0 = time.time()
        try:
            raw = mne.io.read_raw_edf(str(src), preload=True, verbose="ERROR")
            sf = float(raw.info["sfreq"])
            data = raw.get_data()
            names = raw.ch_names
            eeg_idx = [k for k, c in enumerate(names)
                       if mnos.is_eeg_channel(c, OUR_EEG_PATTERN)]
            before, slope_b = alpha_prominence(data[eeg_idx], sf)

            new = data.copy()
            n_fit = 0
            for k in eeg_idx:
                new[k, :] = mnos.reconstruct_aperiodic_channel(
                    data[k, :], sf, mnos.FIT_RANGE, mnos.APERIODIC_MODE, label=names[k])
                n_fit += 1
            after, slope_a = alpha_prominence(new[eeg_idx], sf)

            # how much did the signal actually change?
            rel = float(np.abs(new[eeg_idx] - data[eeg_idx]).max() /
                        max(np.abs(data[eeg_idx]).max(), 1e-30))

            raw_out = mne.io.RawArray(new, raw.info.copy(), verbose="ERROR")
            try:
                raw_out.set_annotations(raw.annotations)
            except Exception:
                pass
            if dst.exists():
                dst.unlink()
            raw_out.export(str(dst), fmt="edf", physical_range="auto",
                           overwrite=True, verbose="ERROR")
            raw.close()

            rows.append({"source": src.name, "output": dst.name, "dyad": d,
                         "participant": pid, "role": role, "sfreq": sf,
                         "n_channels_total": len(names), "n_eeg_channels_fitted": n_fit,
                         "alpha_prominence_before_db": round(before, 3) if before else None,
                         "alpha_prominence_after_db": round(after, 3) if after else None,
                         "slope_before": round(slope_b, 3) if slope_b else None,
                         "slope_after": round(slope_a, 3) if slope_a else None,
                         "rel_max_change": f"{rel:.3e}",
                         "changed": bool(rel > 1e-4),
                         "seconds": round(time.time() - t0, 1)})
            print(f"[{i}/{len(jobs)}] {src.name} -> {n_fit} ch fitted | "
                  f"alpha {before:+.2f} -> {after:+.2f} dB | rel change {rel:.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as ex:
            print(f"[{i}/{len(jobs)}] {src.name} ERROR: {ex}", flush=True)
            rows.append({"source": src.name, "output": "", "dyad": d, "participant": pid,
                         "role": role, "error": str(ex)[:200]})

    if rows:
        qf = qcd / f"nonosc_michelle_qc_shard{args.shard_index}.csv"
        cols = sorted({c for r in rows for c in r})
        with open(qf, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        ok = [r for r in rows if r.get("alpha_prominence_after_db") is not None]
        if ok:
            b = np.mean([r["alpha_prominence_before_db"] for r in ok])
            a = np.mean([r["alpha_prominence_after_db"] for r in ok])
            print(f"\nmean alpha prominence: {b:+.3f} dB -> {a:+.3f} dB "
                  f"| files actually changed: {sum(r['changed'] for r in ok)}/{len(ok)}")
    print("MICHELLE_NONOSC_DONE")


if __name__ == "__main__":
    main()
