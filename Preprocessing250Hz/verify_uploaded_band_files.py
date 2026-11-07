#!/usr/bin/env python
"""
scripts/verify_uploaded_band_files.py
=====================================
Verify whether the UPLOADED band-specific EDF datasets are genuinely filtered.

The SHA-256 audit showed no two files are byte-identical -- but EDF headers carry
timestamps and patient fields, so byte-difference proves nothing about the signal.
This script compares the actual SAMPLE ARRAYS and the PSDs of the same recording
across upload conditions.

For each matched recording we report, for every pair of conditions:
  * max |difference| and RMS difference of the sample arrays
  * whether the arrays are numerically identical (within float tolerance)
  * the band-power ratio in each canonical band (delta..gamma)
and for each condition we report the attenuation in the band its NAME claims to
have removed.
"""
from __future__ import annotations

import argparse, csv, itertools, json, re, sys, warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch

try:
    import mne
except ImportError:
    sys.exit("ERROR: mne required")

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
# condition folder -> band its name claims to remove ("only" = keeps just that band)
CLAIM = {
    "Alpha-Only EEG Datafiles": ("alpha", "only"),
    "Alpha-Removed EEG Datafiles": ("alpha", "removed"),
    "Beta-Removed EEG Datafiles": ("beta", "removed"),
    "Gamma-Removed EEG Datafiles": ("gamma", "removed"),
    "Non-Oscillatory EEG Datafiles": (None, "aperiodic"),
}
KEY_RE = re.compile(r"dyad0*(\d+)_(\d+)_(speak|listen|rest)", re.I)


def keyof(p: Path):
    m = KEY_RE.search(p.name)
    return (int(m.group(1)), int(m.group(2)), m.group(3).lower()) if m else None


def bandpowers(x, sf):
    f, P = welch(x, fs=sf, nperseg=min(2048, x.shape[-1]), axis=-1)
    Pm = P.mean(0)
    return {b: float(Pm[(f >= lo) & (f < hi)].mean()) for b, (lo, hi) in BANDS.items()}, f, Pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-recordings", type=int, default=6)
    ap.add_argument("--crop-sec", type=float, default=60.0)
    args = ap.parse_args()

    out = Path(args.out_dir); (out / "figures").mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    conds = [d.name for d in sorted(root.iterdir()) if d.is_dir() and d.name in CLAIM]
    print("conditions:", conds, flush=True)

    # index each condition by (dyad, participant, role)
    index = {}
    for c in conds:
        idx = {}
        for p in (root / c).rglob("*.edf"):
            k = keyof(p)
            if k and k not in idx:
                idx[k] = p
        index[c] = idx
        print(f"  {c}: {len(idx)} unique recordings", flush=True)

    common = sorted(set.intersection(*[set(v) for v in index.values()]))
    print(f"recordings present in ALL conditions: {len(common)}", flush=True)
    picks = common[:: max(1, len(common) // args.n_recordings)][:args.n_recordings]

    pair_rows, band_rows = [], []
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    for n, k in enumerate(picks, 1):
        arrs, sfs, psds = {}, {}, {}
        for c in conds:
            try:
                r = mne.io.read_raw_edf(str(index[c][k]), preload=True, verbose="ERROR")
                sf = float(r.info["sfreq"])
                nk = int(min(r.n_times, args.crop_sec * sf))
                arrs[c] = r.get_data()[:, :nk]; sfs[c] = sf
                r.close()
            except Exception as ex:
                print(f"  !! {c} {k}: {ex}")
        if len(arrs) < 2:
            continue
        for c, a in arrs.items():
            bp, f, Pm = bandpowers(a, sfs[c])
            psds[c] = (f, Pm)
            band_rows.append({"dyad": k[0], "participant": k[1], "role": k[2],
                              "condition": c, "sfreq": sfs[c], "n_channels": a.shape[0],
                              **{f"power_{b}": v for b, v in bp.items()}})
        # pairwise numerical comparison (only where shapes+sfreq match)
        for c1, c2 in itertools.combinations(conds, 2):
            if c1 not in arrs or c2 not in arrs:
                continue
            a, b = arrs[c1], arrs[c2]
            row = {"dyad": k[0], "participant": k[1], "role": k[2],
                   "cond_a": c1, "cond_b": c2, "sfreq_a": sfs[c1], "sfreq_b": sfs[c2],
                   "shape_a": str(a.shape), "shape_b": str(b.shape)}
            if a.shape == b.shape and abs(sfs[c1] - sfs[c2]) < 1e-9:
                d = a - b
                scale = max(float(np.abs(a).max()), 1e-30)
                row["max_abs_diff"] = float(np.abs(d).max())
                row["rms_diff"] = float(np.sqrt((d ** 2).mean()))
                row["rel_max_diff"] = float(np.abs(d).max() / scale)
                row["numerically_identical"] = bool(row["rel_max_diff"] < 1e-6)
                with np.errstate(invalid="ignore"):
                    cc = np.corrcoef(a.ravel(), b.ravel())[0, 1]
                row["corr"] = float(cc) if cc == cc else None
            else:
                row["numerically_identical"] = False
                row["note"] = "shape/sfreq mismatch - not comparable"
            pair_rows.append(row)

        if n == 1:
            fig, ax = plt.subplots(figsize=(8, 4.6))
            for c in conds:
                if c in psds:
                    f, Pm = psds[c]
                    s = f <= 60
                    ax.semilogy(f[s], Pm[s], lw=1.5, label=c.replace(" EEG Datafiles", ""))
            for b, (lo, hi) in BANDS.items():
                ax.axvline(lo, color="#cccccc", lw=0.6)
            ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (V²/Hz)")
            ax.set_title(f"Uploaded band conditions, same recording "
                         f"(dyad{k[0]} p{k[1]} {k[2]})", fontsize=10)
            ax.legend(fontsize=8); fig.tight_layout()
            fig.savefig(out / "figures" / "uploaded_conditions_psd.png", dpi=300,
                        facecolor="white", bbox_inches="tight")
            plt.close(fig)
        print(f"  [{n}/{len(picks)}] dyad{k[0]} p{k[1]} {k[2]}", flush=True)

    with open(out / "uploaded_band_pairwise.csv", "w", newline="") as f:
        cols = sorted({c for r in pair_rows for c in r})
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(pair_rows)
    with open(out / "uploaded_band_powers.csv", "w", newline="") as f:
        cols = sorted({c for r in band_rows for c in r})
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(band_rows)

    # ---- verdicts ----
    L = ["# Verification of the UPLOADED band EDF datasets", "",
         f"Sampled {len(picks)} recordings that exist in every uploaded condition; "
         f"first {args.crop_sec:g} s compared.", "",
         "## Does each condition attenuate the band its name claims?", "",
         "| condition | claimed effect | band power ratio vs Alpha-Removed reference | verdict |",
         "|---|---|---|---|"]
    import collections
    bycond = collections.defaultdict(list)
    for r in band_rows:
        bycond[r["condition"]].append(r)
    ref = "Alpha-Removed EEG Datafiles"
    verdicts = {}
    for c in conds:
        claim_band, kind = CLAIM[c]
        rows = bycond.get(c, [])
        if not rows:
            continue
        if claim_band:
            ratios = []
            for r in rows:
                m = [x for x in bycond[ref] if (x["dyad"], x["participant"], x["role"])
                     == (r["dyad"], r["participant"], r["role"])]
                if m and m[0][f"power_{claim_band}"] > 0:
                    ratios.append(r[f"power_{claim_band}"] / m[0][f"power_{claim_band}"])
            db = 10 * np.log10(np.mean(ratios)) if ratios else float("nan")
            if kind == "removed" and c != ref:
                ok = db < -6
                v = "FILTERED" if ok else "**NOT FILTERED**"
            else:
                v = "reference / n-a"
            L.append(f"| {c} | {kind} {claim_band} | {db:+.2f} dB | {v} |")
            verdicts[c] = {"claimed": f"{kind} {claim_band}", "db_vs_reference": float(db)}
        else:
            L.append(f"| {c} | aperiodic (FOOOF) | - | separate representation |")

    ident = [r for r in pair_rows if r.get("numerically_identical")]
    comparable = [r for r in pair_rows if "max_abs_diff" in r]
    L += ["", "## Are different 'conditions' numerically the same signal?", "",
          f"Comparable pairs (matching shape and sampling rate): **{len(comparable)}**  ",
          f"Numerically identical pairs (relative max diff < 1e-6): **{len(ident)}**", ""]
    if comparable:
        L += ["| condition A | condition B | max abs diff | rel max diff | corr | identical |",
              "|---|---|---|---|---|---|"]
        seen = set()
        for r in comparable:
            kk = (r["cond_a"], r["cond_b"])
            if kk in seen:
                continue
            seen.add(kk)
            L.append(f"| {r['cond_a'].replace(' EEG Datafiles','')} | "
                     f"{r['cond_b'].replace(' EEG Datafiles','')} | "
                     f"{r['max_abs_diff']:.3e} | {r['rel_max_diff']:.3e} | "
                     f"{r['corr']:.6f} | {'YES' if r['numerically_identical'] else 'no'} |")
    json.dump(verdicts, open(out / "uploaded_band_verdicts.json", "w"), indent=2)
    (out / "uploaded_band_verification.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("UPLOADED_VERIFY_DONE")


if __name__ == "__main__":
    main()
