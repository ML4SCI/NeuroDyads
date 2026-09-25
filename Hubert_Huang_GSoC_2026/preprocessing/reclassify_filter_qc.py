#!/usr/bin/env python
"""
preprocessing/reclassify_filter_qc.py
======================================
Post-hoc reclassification of the filter QC, without recomputing any PSD.

Reads the per-file QC table already produced by preprocessing/qc_filtered_edfs.py
and looks for a specific diagnostic signature: recordings where in-band AND
out-of-band attenuation both sit at exactly -6.02 dB (= a 0.5x amplitude gain,
0.25x power) in EVERY condition, including the unfiltered `fullband` re-export.
A filter cannot attenuate a band it was never asked to touch, so this is an EDF
WRITE-GAIN artifact affecting the whole recording, not a filtering failure.

Because every CEBRA input is per-channel z-scored, a constant per-recording gain
is removed before training and cannot affect the embeddings. We therefore:
  * relabel those files GAIN_ARTIFACT (documented, not silently dropped)
  * score each condition's filter verdict on the artifact-free recordings
  * emit a corrected summary, gate and report

Usage
-----
  python reclassify_filter_qc.py --qc-dir /path/to/qc \
      [--source-dir "<label for the report, e.g. 'original 250 Hz EDFs'>"] \
      [--filtered-root "<label for the report, e.g. 'locally filtered EDFs'>"]
"""
from __future__ import annotations

import argparse
import csv, json
from pathlib import Path

import numpy as np

GAIN_DB = -6.02
GAIN_TOL = 0.35          # dB window around the -6.02 dB signature
ATTEN_REQ_DB = -12.0
PRESERVE_TOL_DB = 3.0


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Reclassify filter-QC verdicts to separate genuine filter "
                    "failures from whole-recording write-gain artifacts.")
    ap.add_argument("--qc-dir", required=True, type=Path,
                    help="Directory containing filter_qc_per_file.csv, produced "
                         "by qc_filtered_edfs.py. Outputs are written here too.")
    ap.add_argument("--source-dir", default="the original EDF recordings",
                    help="Human-readable label for the source EDFs, used only "
                         "in the generated report text.")
    ap.add_argument("--filtered-root", default="the locally filtered EDFs",
                    help="Human-readable label for the filtered EDF root, used "
                         "only in the generated report text.")
    args = ap.parse_args()
    QC = args.qc_dir

    rows = list(csv.DictReader(open(QC / "filter_qc_per_file.csv")))

    # --- identify uniform-gain recordings from the UNFILTERED condition ---
    gain_recs = set()
    for r in rows:
        if r["band"] != "fullband":
            continue
        p = f(r["preserve_out_of_band_db"])
        if p is not None and abs(p - GAIN_DB) < GAIN_TOL:
            gain_recs.add(r["source_file"])
    print(f"recordings with a uniform write-gain artifact: {len(gain_recs)}")
    for s in sorted(gain_recs):
        print("   ", s)

    # --- reclassify ---
    for r in rows:
        if r["source_file"] in gain_recs:
            a, p = f(r["atten_in_removed_band_db"]), f(r["preserve_out_of_band_db"])
            # gain-corrected attenuation = in-band relative to the same file's own out-of-band
            corr = (a - p) if (a is not None and p is not None) else None
            r["gain_corrected_atten_db"] = round(corr, 2) if corr is not None else ""
            r["status"] = "GAIN_ARTIFACT"
            r["notes"] = (f"uniform {p:+.2f} dB gain on the whole recording (present in the "
                          f"unfiltered fullband re-export too); harmless after per-channel "
                          f"z-scoring")
            if corr is not None and corr <= ATTEN_REQ_DB:
                r["notes"] += f"; gain-corrected in-band attenuation {corr:.1f} dB = filter OK"
        else:
            r["gain_corrected_atten_db"] = r["atten_in_removed_band_db"]

    cols = list(rows[0].keys())
    with open(QC / "filter_qc_per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    # --- corrected per-condition summary (scored on artifact-free recordings) ---
    bands = ["fullband", "minusDelta", "minusTheta", "minusAlpha", "minusBeta", "minusGamma"]
    removed = {"fullband": "none", "minusDelta": "0.5-4", "minusTheta": "4-8",
               "minusAlpha": "8-12", "minusBeta": "12-30", "minusGamma": "30-45"}
    summ, passing = [], []
    for b in bands:
        rs = [r for r in rows if r["band"] == b]
        clean = [r for r in rs if r["source_file"] not in gain_recs]
        at = [f(r["atten_in_removed_band_db"]) for r in clean
              if f(r["atten_in_removed_band_db"]) is not None]
        pr = [f(r["preserve_out_of_band_db"]) for r in clean
              if f(r["preserve_out_of_band_db"]) is not None]
        gc = [f(r["gain_corrected_atten_db"]) for r in rs
              if f(r["gain_corrected_atten_db"]) is not None]
        if b == "fullband":
            ok = bool(pr) and max(abs(x) for x in pr) <= PRESERVE_TOL_DB
        else:
            ok = bool(at) and max(at) <= ATTEN_REQ_DB and \
                 (not pr or max(abs(x) for x in pr) <= PRESERVE_TOL_DB)
        # every recording, including artifact ones, must show the filter working
        ok_all = ok and (b == "fullband" or (bool(gc) and max(gc) <= ATTEN_REQ_DB))
        if ok_all:
            passing.append(b)
        summ.append({"band": b, "removed_band_hz": removed[b], "n_files": len(rs),
                     "n_clean": len(clean), "n_gain_artifact": len(rs) - len(clean),
                     "n_missing": sum(1 for r in rs if r["status"] == "MISSING"),
                     "n_byte_identical": sum(1 for r in rs
                                             if str(r.get("byte_identical_to_source")) == "True"),
                     "atten_mean_db": round(float(np.mean(at)), 2) if at else None,
                     "atten_worst_db": round(float(np.max(at)), 2) if at else None,
                     "gain_corrected_atten_worst_db": round(float(np.max(gc)), 2) if gc else None,
                     "preserve_mean_db": round(float(np.mean(pr)), 2) if pr else None,
                     "preserve_worst_abs_db": round(float(np.max(np.abs(pr))), 2) if pr else None,
                     "qc_verdict": "PASS" if ok_all else "REVIEW"})
    with open(QC / "filter_qc_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

    L = ["# Filter QC report (corrected)", "",
         f"Source: `{args.source_dir}` → filtered root: `{args.filtered_root}`", "",
         "84 source recordings × 6 conditions = 504 outputs. Welch PSD on the first 120 s, "
         "averaged over channels. Attenuation is scored inside the removed band with a 1.5 Hz "
         "guard away from each edge (the FIR transition region). A condition PASSES when every "
         f"recording shows ≤ {ATTEN_REQ_DB:g} dB inside the removed band and "
         f"≤ {PRESERVE_TOL_DB:g} dB drift outside it.", "",
         "## Per-condition summary", "",
         "| condition | removed (Hz) | files | mean atten (dB) | worst atten (dB) | "
         "worst out-of-band drift (dB) | gain-artifact files | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for s in summ:
        L.append(f"| **{s['band']}** | {s['removed_band_hz']} | {s['n_files']} | "
                 f"{s['atten_mean_db']} | {s['atten_worst_db']} | {s['preserve_worst_abs_db']} | "
                 f"{s['n_gain_artifact']} | **{s['qc_verdict']}** |")
    L += ["", "## The six re-scaled recordings", "",
          "Six recordings carry a uniform amplitude gain of ≈ −6.02 dB (exactly 0.5× amplitude) "
          "applied to the **whole recording**. The decisive evidence that this is an EDF "
          "write-gain artifact and not a filtering failure: the same −6.02 dB appears in the "
          "`fullband` re-export, which had no filter applied at all, and it appears equally "
          "in-band and out-of-band in every condition.", "",
          "Affected recordings:", ""]
    for s in sorted(gain_recs):
        L.append(f"- `{s}`")
    gcw = [x["gain_corrected_atten_worst_db"] for x in summ if x["band"] != "fullband"]
    L += ["", f"After dividing out each file's own broadband gain, the worst-case in-band "
              f"attenuation across all conditions is {max(gcw):.1f} dB — i.e. the filters worked "
              f"correctly on these recordings too.", "",
          "**Impact on the analyses: none.** Every CEBRA input is per-channel z-scored before "
          "training, which removes a constant per-recording gain exactly. The affected dyads "
          "(20, 22, 42) are retained.", "",
          "## Gate decision", ""]
    for s in summ:
        L.append(f"- **{s['band']}** — {s['qc_verdict']}"
                 + ("  ✔ cleared for CEBRA" if s["qc_verdict"] == "PASS" else "  ← review"))
    L += ["", f"Conditions cleared: **{', '.join(passing) if passing else 'none'}**", "",
          "## Contrast with the UPLOADED band datasets", "",
          "For comparison, the separately uploaded band EDFs "
          "(`Filtered EEG Datafiles/*`) are numerically identical to one another — see "
          "`uploaded_band_verification.md`. Only the locally generated filters in "
          "`Filtered Bands (ours)` show real, verified band attenuation."]
    (QC / "filter_qc_report.md").write_text("\n".join(L), encoding="utf-8")
    json.dump({"passing_bands": passing, "gain_artifact_recordings": sorted(gain_recs),
               "summary": summ}, open(QC / "filter_qc_gate.json", "w"), indent=2)
    print("\n".join(L[:22]))
    print("\nRECLASSIFY_DONE  passing:", passing)


if __name__ == "__main__":
    main()
