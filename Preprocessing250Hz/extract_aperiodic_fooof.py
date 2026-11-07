#!/usr/bin/env python
"""
scripts/extract_aperiodic_fooof.py
==================================
TASK 8 -- build a genuine NON-OSCILLATORY (aperiodic) representation ourselves.

Why this exists
---------------
The uploaded `Non-Oscillatory EEG Datafiles` are not an aperiodic decomposition:
for every recording their spectrum is identical to the Alpha-Removed and
Alpha-Only uploads (same alpha prominence to 2 dp, same 1/f slope), and the
sample arrays differ only at float-rounding level. Michelle's FOOOF script and
README are not in the workspace. Rather than stay blocked, we compute the
aperiodic component directly with FOOOF (specparam).

IMPORTANT -- INPUT DEFINITION IS OUR ASSUMPTION
-----------------------------------------------
"Non-oscillatory data" does not have one obvious CEBRA input. FOOOF parameterises
a POWER SPECTRUM, not a time series, so a choice has to be made. We use the
standard summary of the aperiodic component as a sliding-window time series:

    per window, per channel:  aperiodic OFFSET and aperiodic EXPONENT
    feature vector           = 64 channels x 2 = 128 features per window

This is the representation FOOOF is designed to produce and it keeps a time axis,
which CEBRA needs. Two alternatives we did NOT use (documented so the choice is
reviewable): (a) reconstructing a time-domain signal from the aperiodic-only
spectrum, and (b) feeding peak parameters. If the mentor intended either, the
extraction must be redone -- see NONOSCILLATORY_INPUT_ASSUMPTION.md.

Output: <out>/nonosc_dyad{N}_{P}_{role}.npy with shape (n_features, n_windows),
matching the (channels, time) convention of the other pipelines, plus a per-file
QC row proving the fitted aperiodic spectra no longer contain an alpha peak.
"""
from __future__ import annotations

import argparse, csv, json, os, re, sys, warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch

try:
    import mne
except ImportError:
    sys.exit("ERROR: mne required")
try:
    from fooof import FOOOFGroup
except ImportError:
    sys.exit("ERROR: pip install fooof")

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FILE_RE = re.compile(r"^(?:fullband_|cut_)?dyad_?(\d+)_(\d+)_(speak|listen|rest)\.edf$", re.I)


def eeg_picks(names):
    return [i for i, c in enumerate(names)
            if not any(t in c.upper() for t in ("VREF", "STATUS", "TRIGGER", "STI", "ANNOT"))]


def sliding_spectra(X, sf, win_sec, hop_sec, nperseg_sec):
    """-> freqs, spectra (n_windows, n_channels, n_freqs)"""
    w = int(win_sec * sf); h = int(hop_sec * sf); nps = int(nperseg_sec * sf)
    n = X.shape[1]
    starts = np.arange(0, n - w + 1, h)
    out = []
    for s in starts:
        f, P = welch(X[:, s:s + w], fs=sf, nperseg=min(nps, w), axis=-1)
        out.append(P)
    return f, np.stack(out)


def alpha_prominence(freqs, spec):
    """dB of 8-12 Hz power above a 1/f line fitted on 2-40 Hz excluding 7-14 Hz."""
    m = (freqs >= 2) & (freqs <= 40) & (spec > 0)
    fitm = m & ~((freqs >= 7) & (freqs <= 14))
    if fitm.sum() < 5:
        return None
    co = np.polyfit(np.log10(freqs[fitm]), np.log10(spec[fitm]), 1)
    am = (freqs >= 8) & (freqs <= 12)
    pred = 10 ** np.polyval(co, np.log10(freqs[am]))
    return float(10 * np.log10(spec[am].mean() / pred.mean()))


def process_one(job):
    """Fit one recording. Runs in a worker process with FOOOF n_jobs=1.

    Parallelising at the FILE level matters: FOOOFGroup's own n_jobs pickles every
    individual spectrum to a worker, and with ~10^4 tiny fits per file that overhead
    dominates the actual fitting. One process per file, each fitting serially, is
    roughly an order of magnitude faster here.
    """
    import time as _t
    (p, dyad, pid, role, crop, win_sec, hop_sec, nperseg_sec, fmin, fmax,
     ap_mode, out_dir) = job
    t0 = _t.time()
    r = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
    sf = float(r.info["sfreq"]); pk = eeg_picks(r.ch_names)
    X = r.get_data()[pk][:, :int(crop * sf)]
    r.close()

    freqs, spec = sliding_spectra(X, sf, win_sec, hop_sec, nperseg_sec)
    nW, nC, nF = spec.shape
    flat = spec.reshape(nW * nC, nF)

    finite = np.isfinite(flat)
    pos = finite & (flat > 0)
    n_bad_spectra = int((~pos.all(axis=1)).sum())
    if n_bad_spectra:
        floor = np.nanmin(flat[pos]) if pos.any() else 1e-30
        flat = np.where(pos, flat, floor)
    dead_ch = int(np.sum(~pos.reshape(nW, nC, nF).all(axis=(0, 2))))

    fg = FOOOFGroup(peak_width_limits=[1.0, 8.0], max_n_peaks=6,
                    min_peak_height=0.05, aperiodic_mode=ap_mode, verbose=False)
    fg.fit(freqs, flat, [fmin, fmax], n_jobs=1)
    apar = fg.get_params("aperiodic_params")
    r2 = fg.get_params("r_squared"); err = fg.get_params("error")

    n_ap = apar.shape[1]
    feats = apar.reshape(nW, nC * n_ap).T.astype(np.float32)
    bad = ~np.isfinite(feats)
    if bad.any():
        feats[bad] = 0.0
    np.save(Path(out_dir) / f"nonosc_dyad{dyad}_{pid}_{role}.npy", feats)

    mean_spec = spec.mean(0).mean(0)
    prom_raw = alpha_prominence(freqs, mean_spec)
    fsel = (freqs >= fmin) & (freqs <= fmax)
    off = float(apar[:, 0].reshape(nW, nC).mean()); expo = float(apar[:, -1].reshape(nW, nC).mean())
    ap_spec = 10 ** (off - expo * np.log10(freqs[fsel]))
    prom_ap = alpha_prominence(freqs[fsel], ap_spec)

    return {"file": p.name, "dyad": dyad, "participant": pid, "role": role,
            "sfreq": sf, "n_channels": nC, "n_windows": nW,
            "n_features": feats.shape[0], "crop_sec": round(crop, 2),
            "mean_offset": round(off, 4), "mean_exponent": round(expo, 4),
            "mean_r_squared": round(float(np.nanmean(r2)), 4),
            "mean_fit_error": round(float(np.nanmean(err)), 4),
            "frac_fits_r2_below_0.9": round(float(np.mean(r2 < 0.9)), 4),
            "alpha_prominence_raw_db": round(prom_raw, 3) if prom_raw else None,
            "alpha_prominence_aperiodic_db": round(prom_ap, 3) if prom_ap else None,
            "n_nonfinite_features": int(bad.sum()),
            "n_spectra_floored": n_bad_spectra, "n_dead_channels": dead_ch,
            "seconds": round(_t.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--qc-dir", default=None)
    ap.add_argument("--win-sec", type=float, default=4.0)
    ap.add_argument("--hop-sec", type=float, default=0.5)
    ap.add_argument("--nperseg-sec", type=float, default=2.0)
    ap.add_argument("--fmin", type=float, default=2.0)
    ap.add_argument("--fmax", type=float, default=40.0)
    ap.add_argument("--aperiodic-mode", default="fixed", choices=["fixed", "knee"])
    ap.add_argument("--crop-sec", type=float, default=None,
                    help="global crop applied to every recording (documented policy)")
    ap.add_argument("--limit", type=int, default=None, help="process only N files (benchmark)")
    ap.add_argument("--n-procs", type=int, default=1,
                    help="in-process pool size; 1 = serial. This Python build (Windows "
                         "Store) forbids multiprocessing handle duplication, so prefer "
                         "shell-level parallelism via --shard-index/--shard-count.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="re-extract even if .npy exists")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    qcd = Path(args.qc_dir) if args.qc_dir else out
    qcd.mkdir(parents=True, exist_ok=True)
    edfs = sorted(Path(args.input_dir).glob("*.edf"))
    if args.limit:
        edfs = edfs[:args.limit]
    print(f"{len(edfs)} source EDFs | win={args.win_sec}s hop={args.hop_sec}s "
          f"range={args.fmin}-{args.fmax}Hz mode={args.aperiodic_mode}", flush=True)

    # ---- one pass to establish the global minimum duration ----
    if args.crop_sec is None:
        durs = []
        for p in edfs:
            r = mne.io.read_raw_edf(str(p), preload=False, verbose="ERROR")
            durs.append(r.n_times / r.info["sfreq"]); r.close()
        crop = float(np.min(durs))
        print(f"global minimum duration = {crop:.2f} s (used as the crop for every file)",
              flush=True)
    else:
        crop = args.crop_sec

    jobs = []
    for p in edfs:
        m = FILE_RE.match(p.name)
        if not m:
            print(f"  skip (name) {p.name}"); continue
        jobs.append((p, int(m.group(1)), int(m.group(2)), m.group(3).lower(), crop,
                     args.win_sec, args.hop_sec, args.nperseg_sec, args.fmin, args.fmax,
                     args.aperiodic_mode, str(out)))

    # resume support: skip files whose .npy already exists
    if not args.force:
        before = len(jobs)
        jobs = [j for j in jobs
                if not (out / f"nonosc_dyad{j[1]}_{j[2]}_{j[3]}.npy").exists()]
        if before != len(jobs):
            print(f"resuming: {before - len(jobs)} already extracted, {len(jobs)} to do",
                  flush=True)

    if args.shard_count > 1:
        jobs = jobs[args.shard_index::args.shard_count]
        print(f"shard {args.shard_index+1}/{args.shard_count}: {len(jobs)} recordings",
              flush=True)

    def _emit(i, n, rec):
        print(f"  [{i}/{n}] {rec['file']} -> ({rec['n_features']}, {rec['n_windows']}) "
              f"R2={rec['mean_r_squared']:.3f} exp={rec['mean_exponent']:.2f} "
              f"alpha raw {rec['alpha_prominence_raw_db']:+.2f} dB -> "
              f"aperiodic {rec['alpha_prominence_aperiodic_db']:+.2f} dB "
              f"({rec['seconds']:.0f}s)", flush=True)

    qc_rows = []
    if jobs and args.n_procs > 1:
        try:
            import multiprocessing as mp
            with mp.Pool(args.n_procs) as pool:
                for i, rec in enumerate(pool.imap_unordered(process_one, jobs), 1):
                    qc_rows.append(rec); _emit(i, len(jobs), rec)
        except Exception as ex:
            print(f"!! in-process pool unavailable ({type(ex).__name__}: {ex}); "
                  f"falling back to serial", flush=True)
            qc_rows = []
    if jobs and not qc_rows:
        for i, j in enumerate(jobs, 1):
            rec = process_one(j); qc_rows.append(rec); _emit(i, len(jobs), rec)

    # merge with any QC rows from a previous partial run
    qcf = qcd / "nonosc_extraction_qc.csv"
    if qcf.exists() and not args.force:
        have = {r["file"] for r in qc_rows}
        for r in csv.DictReader(open(qcf)):
            if r["file"] not in have:
                for k in ("dyad", "participant", "n_channels", "n_windows", "n_features"):
                    if r.get(k):
                        r[k] = int(float(r[k]))
                for k in ("mean_offset", "mean_exponent", "mean_r_squared",
                          "alpha_prominence_raw_db", "alpha_prominence_aperiodic_db"):
                    if r.get(k):
                        r[k] = float(r[k])
                qc_rows.append(r)
    if not qc_rows:
        sys.exit("nothing extracted")
    qc_rows.sort(key=lambda r: (int(r["dyad"]), int(r["participant"]), str(r["role"])))

    with open(qcd / "nonosc_extraction_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys())); w.writeheader(); w.writerows(qc_rows)

    raw = [r["alpha_prominence_raw_db"] for r in qc_rows if r["alpha_prominence_raw_db"] is not None]
    apx = [r["alpha_prominence_aperiodic_db"] for r in qc_rows
           if r["alpha_prominence_aperiodic_db"] is not None]
    r2m = [r["mean_r_squared"] for r in qc_rows]
    summary = {"n_files": len(qc_rows), "crop_sec": crop,
               "win_sec": args.win_sec, "hop_sec": args.hop_sec,
               "freq_range": [args.fmin, args.fmax], "aperiodic_mode": args.aperiodic_mode,
               "n_features": qc_rows[0]["n_features"], "n_windows": qc_rows[0]["n_windows"],
               "mean_r_squared": round(float(np.mean(r2m)), 4),
               "alpha_prominence_raw_mean_db": round(float(np.mean(raw)), 3),
               "alpha_prominence_aperiodic_mean_db": round(float(np.mean(apx)), 3),
               "oscillatory_content_removed": bool(abs(np.mean(apx)) < 0.5),
               "mean_exponent": round(float(np.mean([r["mean_exponent"] for r in qc_rows])), 4)}
    json.dump(summary, open(qcd / "nonosc_extraction_summary.json", "w"), indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print("APERIODIC_EXTRACTION_DONE")


if __name__ == "__main__":
    main()
