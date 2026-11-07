#!/usr/bin/env python
"""
scripts/qc_filtered_edfs.py
===========================
TASK 2 QC -- prove that each "filtered" EDF really is filtered.

Project rule: never call a dataset 'filtered' until PSD-based QC verifies actual
attenuation in the intended band. Michelle's uploaded band files turned out to be
byte-identical to the unfiltered recordings, which is exactly the failure mode
this script is built to catch.

For every (source recording, band condition) pair we:
  * reopen the output, compare sfreq / channels / n_samples against the source
  * compare SHA-256 (byte-identical output = filter silently did nothing)
  * compute Welch PSD before and after
  * measure mean attenuation INSIDE the removed band (dB, negative = attenuated)
  * measure preservation OUTSIDE the removed band (dB, want ~0)
  * compute waveform correlation (reported, but never used alone as proof)
  * PASS/FAIL each file against explicit thresholds

Outputs per-file CSV, per-condition summary CSV, a Markdown report and
representative PSD figures.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, re, sys, warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch

try:
    import mne
except ImportError:
    sys.exit("ERROR: mne required")

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# band -> (removed_lo, removed_hi) in Hz ; None,None = no removal expected
REMOVED = {
    "fullband":   (None, None),
    "minusDelta": (0.5, 4.0),
    "minusTheta": (4.0, 8.0),
    "minusAlpha": (8.0, 12.0),
    "minusBeta":  (12.0, 30.0),
    "minusGamma": (30.0, 45.0),
}
# how much attenuation we demand inside the removed band, and how much
# deviation we tolerate outside it (dB)
ATTEN_REQ_DB = -12.0
PRESERVE_TOL_DB = 3.0
GUARD = 1.5          # Hz kept away from each edge when scoring (transition band)

FILE_RE = re.compile(r"^(?:cut_)?dyad_?(\d+)_(\d+)_(speak|listen|rest)\.edf$", re.I)


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def eeg_picks(ch_names):
    """The 64 real EEG channels: drop VREF and any non-EEG channel (Status/trigger).

    This matters: a trigger channel carries enormous values and, if included in a
    channel-averaged PSD, can supply >99% of the total power and completely mask
    the filter's effect on the actual EEG. The CEBRA inputs use exactly these
    64 channels, so QC must score exactly these 64 channels too.
    """
    keep = []
    for i, c in enumerate(ch_names):
        u = c.upper()
        if "VREF" in u or "STATUS" in u or "TRIGGER" in u or "STI" in u or "ANNOT" in u:
            continue
        keep.append(i)
    return keep


def psd_mean(data, sf, nper=2048):
    f, P = welch(data, fs=sf, nperseg=min(nper, data.shape[-1]), axis=-1)
    return f, P.mean(0)


def band_db(f, Pa, Pb, lo, hi):
    """10*log10(mean power after / mean power before) inside [lo,hi]."""
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return None
    a = Pa[m].mean(); b = Pb[m].mean()
    if b <= 0:
        return None
    return float(10 * np.log10(max(a, 1e-30) / b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, help="original role-specific EDFs")
    ap.add_argument("--filtered-root", required=True, help="root holding <band>/ folders")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-sfreq", type=float, default=250.0)
    ap.add_argument("--bands", default=",".join(REMOVED))
    ap.add_argument("--max-files", type=int, default=None, help="limit source files (QC speed)")
    ap.add_argument("--crop-sec", type=float, default=120.0, help="seconds used for PSD")
    ap.add_argument("--n-figures", type=int, default=1, help="representative PSD figs per band")
    args = ap.parse_args()

    out = Path(args.out_dir); (out / "figures").mkdir(parents=True, exist_ok=True)
    src_dir = Path(args.source_dir); froot = Path(args.filtered_root)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]

    srcs = sorted(src_dir.glob("*.edf"))
    if args.max_files:
        srcs = srcs[:args.max_files]
    print(f"QC: {len(srcs)} source recordings x {len(bands)} bands", flush=True)

    rows = []
    figs_done = {b: 0 for b in bands}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    for i, sp in enumerate(srcs, 1):
        m = FILE_RE.match(sp.name)
        if not m:
            continue
        dyad, pid, role = m.group(1), m.group(2), m.group(3).lower()
        try:
            base = mne.io.read_raw_edf(str(sp), preload=True, verbose="ERROR")
            src_sf_orig = float(base.info["sfreq"])
            if abs(base.info["sfreq"] - args.target_sfreq) > 1e-6:
                base.resample(args.target_sfreq, verbose="ERROR")
            sf = float(base.info["sfreq"])
            nkeep = int(min(base.n_times, args.crop_sec * sf))
            picks = eeg_picks(base.ch_names)
            Xs = base.get_data()[picks][:, :nkeep]
            src_hash = sha256(sp)
            src_ch = list(base.ch_names); src_n = int(base.n_times)
            src_pick_names = [base.ch_names[i] for i in picks]
            base.close()
        except Exception as ex:
            print(f"  !! source unreadable {sp.name}: {ex}"); continue
        fs_, Ps = psd_mean(Xs, sf)

        for band in bands:
            fp = froot / band / f"{band}_dyad{dyad}_{pid}_{role}.edf"
            rec = {"band": band, "source_file": sp.name, "output_file": fp.name,
                   "dyad": int(dyad), "participant": int(pid), "role": role,
                   "source_sfreq_orig": src_sf_orig, "source_sfreq_used": sf,
                   "status": "", "notes": ""}
            if not fp.exists():
                rec["status"] = "MISSING"; rows.append(rec); continue
            try:
                r = mne.io.read_raw_edf(str(fp), preload=True, verbose="ERROR")
                out_sf = float(r.info["sfreq"])
                opicks = eeg_picks(r.ch_names)
                Xo = r.get_data()[opicks][:, :nkeep]
                rec["n_eeg_channels_scored"] = len(opicks)
                rec.update({"out_sfreq": out_sf, "out_n_channels": len(r.ch_names),
                            "out_n_samples": int(r.n_times),
                            "sfreq_match": bool(abs(out_sf - sf) < 1e-6),
                            "channels_match": bool(list(r.ch_names) == src_ch),
                            "n_samples_match": bool(int(r.n_times) == src_n)})
                r.close()
            except Exception as ex:
                rec["status"] = "READ_ERROR"; rec["notes"] = str(ex)[:200]
                rows.append(rec); continue

            o_hash = sha256(fp)
            rec["output_sha256"] = o_hash[:16]
            rec["byte_identical_to_source"] = bool(o_hash == src_hash)

            fo_, Po = psd_mean(Xo, out_sf)
            lo, hi = REMOVED[band]
            nyq = out_sf / 2
            if lo is None:
                rec["atten_in_removed_band_db"] = None
                # fullband: expect near-perfect preservation 1..45 Hz
                rec["preserve_out_of_band_db"] = band_db(fo_, Po, Ps, 1.0, min(45.0, nyq - 1))
            else:
                a_lo, a_hi = lo + (GUARD if lo > 0.5 else 0.0), min(hi - GUARD, nyq - 1)
                if a_hi <= a_lo:
                    a_lo, a_hi = lo, min(hi, nyq - 1)
                rec["atten_in_removed_band_db"] = band_db(fo_, Po, Ps, a_lo, a_hi)
                # preserved region = everything 1..45 Hz outside [lo-guard, hi+guard]
                msk = ((fo_ >= 1.0) & (fo_ <= min(45.0, nyq - 1)) &
                       ~((fo_ >= lo - GUARD) & (fo_ <= hi + GUARD)))
                if msk.any():
                    rec["preserve_out_of_band_db"] = float(
                        10 * np.log10(max(Po[msk].mean(), 1e-30) / max(Ps[msk].mean(), 1e-30)))
                else:
                    rec["preserve_out_of_band_db"] = None
            with np.errstate(invalid="ignore"):
                c = np.corrcoef(Xs.ravel(), Xo.ravel())[0, 1]
            rec["waveform_corr"] = float(c) if c == c else None

            # verdict
            if rec["byte_identical_to_source"] and band != "fullband":
                rec["status"] = "FAIL_IDENTICAL"
                rec["notes"] = "output is byte-identical to source -- no filtering applied"
            elif not rec.get("sfreq_match") or not rec.get("n_samples_match"):
                rec["status"] = "FAIL_SHAPE"
            elif band == "fullband":
                p = rec["preserve_out_of_band_db"]
                rec["status"] = "PASS" if (p is not None and abs(p) <= PRESERVE_TOL_DB) else "WARN"
            else:
                a = rec["atten_in_removed_band_db"]; p = rec["preserve_out_of_band_db"]
                if a is None:
                    rec["status"] = "WARN"
                elif a <= ATTEN_REQ_DB and (p is None or abs(p) <= PRESERVE_TOL_DB):
                    rec["status"] = "PASS"
                elif a <= ATTEN_REQ_DB:
                    rec["status"] = "WARN"; rec["notes"] = f"out-of-band drift {p:.1f} dB"
                else:
                    rec["status"] = "FAIL_NO_ATTENUATION"
                    rec["notes"] = f"only {a:.1f} dB inside removed band"
            rows.append(rec)

            # representative figure
            if figs_done[band] < args.n_figures:
                figs_done[band] += 1
                fig, ax = plt.subplots(figsize=(7.2, 4.2))
                sel = fs_ <= min(60, nyq)
                ax.semilogy(fs_[sel], Ps[sel], color="#999999", lw=1.6, label="source (250 Hz)")
                ax.semilogy(fo_[fo_ <= min(60, nyq)], Po[fo_ <= min(60, nyq)],
                            color="#0173B2", lw=1.6, label=f"{band}")
                if lo is not None:
                    ax.axvspan(lo, hi, color="#DE8F05", alpha=0.18,
                               label=f"target removed {lo:g}-{hi:g} Hz")
                ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (V²/Hz)")
                t = (f"{band}: dyad{dyad} p{pid} {role}")
                if rec["atten_in_removed_band_db"] is not None:
                    t += f"  |  in-band {rec['atten_in_removed_band_db']:.1f} dB"
                ax.set_title(t, fontsize=10)
                ax.legend(fontsize=8); fig.tight_layout()
                fig.savefig(out / "figures" / f"psd_{band}.png", dpi=300,
                            facecolor="white", bbox_inches="tight")
                plt.close(fig)

        if i % 5 == 0 or i == len(srcs):
            print(f"  [{i}/{len(srcs)}]", flush=True)

    # ---------------- write CSVs ----------------
    cols = ["band", "source_file", "output_file", "dyad", "participant", "role",
            "source_sfreq_orig", "source_sfreq_used", "out_sfreq", "out_n_channels",
            "n_eeg_channels_scored",
            "out_n_samples", "sfreq_match", "channels_match", "n_samples_match",
            "output_sha256", "byte_identical_to_source", "atten_in_removed_band_db",
            "preserve_out_of_band_db", "waveform_corr", "status", "notes"]
    with open(out / "filter_qc_per_file.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    summ = []
    for band in bands:
        rs = [r for r in rows if r["band"] == band]
        at = [r["atten_in_removed_band_db"] for r in rs if r.get("atten_in_removed_band_db") is not None]
        pr = [r["preserve_out_of_band_db"] for r in rs if r.get("preserve_out_of_band_db") is not None]
        npass = sum(1 for r in rs if r["status"] == "PASS")
        summ.append({"band": band, "n_files": len(rs), "n_pass": npass,
                     "n_warn": sum(1 for r in rs if r["status"] == "WARN"),
                     "n_fail": sum(1 for r in rs if r["status"].startswith("FAIL")),
                     "n_missing": sum(1 for r in rs if r["status"] == "MISSING"),
                     "n_byte_identical": sum(1 for r in rs if r.get("byte_identical_to_source")),
                     "removed_band_hz": (f"{REMOVED[band][0]:g}-{REMOVED[band][1]:g}"
                                         if REMOVED[band][0] is not None else "none"),
                     "atten_mean_db": round(float(np.mean(at)), 2) if at else None,
                     "atten_min_db": round(float(np.min(at)), 2) if at else None,
                     "atten_max_db": round(float(np.max(at)), 2) if at else None,
                     "preserve_mean_db": round(float(np.mean(pr)), 2) if pr else None,
                     "qc_verdict": ("PASS" if rs and npass == len(rs) else
                                    "PARTIAL" if npass else "FAIL")})
    with open(out / "filter_qc_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

    L = ["# Filter QC report", "",
         f"Source: `{src_dir}`  →  filtered root: `{froot}`", "",
         f"{len(srcs)} source recordings × {len(bands)} conditions. PSD via Welch on the first "
         f"{args.crop_sec:g} s, averaged over channels. Attenuation is scored inside the removed "
         f"band with a {GUARD:g} Hz guard away from each edge (the FIR transition band). "
         f"Pass requires ≤ {ATTEN_REQ_DB:g} dB inside the removed band and "
         f"≤ {PRESERVE_TOL_DB:g} dB drift outside it.", "",
         "## Per-condition summary", "",
         "| condition | removed (Hz) | files | PASS | WARN | FAIL | byte-identical | "
         "mean atten (dB) | worst atten (dB) | out-of-band drift (dB) | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summ:
        L.append(f"| {s['band']} | {s['removed_band_hz']} | {s['n_files']} | {s['n_pass']} | "
                 f"{s['n_warn']} | {s['n_fail']} | {s['n_byte_identical']} | "
                 f"{s['atten_mean_db']} | {s['atten_max_db']} | {s['preserve_mean_db']} | "
                 f"**{s['qc_verdict']}** |")
    bad = [r for r in rows if r["status"].startswith("FAIL") or r["status"] == "MISSING"]
    L += ["", "## Failures / missing", ""]
    if bad:
        L.append("| band | file | status | notes |"); L.append("|---|---|---|---|")
        for r in bad[:60]:
            L.append(f"| {r['band']} | {r.get('output_file','')} | {r['status']} | {r.get('notes','')} |")
        if len(bad) > 60:
            L.append(f"\n… {len(bad)-60} more (see CSV).")
    else:
        L.append("None — every generated file passed or warned only.")
    L += ["", "## Gate decision", "",
          "Conditions cleared for CEBRA (QC verdict PASS):", ""]
    okb = [s["band"] for s in summ if s["qc_verdict"] == "PASS"]
    for b in bands:
        v = next(s["qc_verdict"] for s in summ if s["band"] == b)
        L.append(f"- **{b}** — {v}" + ("" if v == "PASS" else "  ← blocked / needs review"))
    (out / "filter_qc_report.md").write_text("\n".join(L), encoding="utf-8")
    json.dump({"passing_bands": okb, "summary": summ}, open(out / "filter_qc_gate.json", "w"), indent=2)
    print("\n".join(L[:30]))
    print("FILTER_QC_DONE")


if __name__ == "__main__":
    main()
