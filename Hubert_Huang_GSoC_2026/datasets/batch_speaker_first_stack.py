#!/usr/bin/env python
"""
scripts/batch_speaker_first_stack.py
====================================
STEP 1 of the filtered-band workflow (Michelle's new instructions).

Build SPEAKER-FIRST time-stacked CEBRA inputs from a filtered EEG band, labelled
by AQ MAGNITUDE (Low/High). "Speaker always first" removes the speaker/listener
ROLE as a variable: every stacked file is [speaker_data ; listener_data] in time.

For each dyad (two participants), BOTH real interactions are kept, each
speaker-first:
    interaction A:  [ A_speak ; B_listen ]   speaker=A, listener=B
    interaction B:  [ B_speak ; A_listen ]   speaker=B, listener=A
Both get the SAME AQ-magnitude label because |dAQ| is symmetric (Michelle's
"buckets stay the same" correction).

Label:  Low  = |dAQ| in {0,1}   (aq_code 0/1) -> 0
        High = |dAQ| >= 2       (aq_code 2/3) -> 1
(read from the AQ sheet via pair_id = "<speaker>_<listener>")

Processing per file:
  read EDF -> pick EEG -> drop VREF (->64 ch) -> crop to GLOBAL min samples ->
  time-concatenate speaker then listener -> save .npy (channels, time).

Band-agnostic: pass --input-dir (a band folder), --prefix (e.g. minusAlpha),
--aq-sheet, --output-dir. Flagged participants (dyad38/p98, dyad41/p104) are
generated but marked flagged in the manifest.
"""
from __future__ import annotations

import argparse, csv, json, re, sys, warnings
from pathlib import Path

import numpy as np

try:
    import mne
except ImportError:
    sys.exit("ERROR: mne required.")

FAULTY = {(38, 98): "dyad38 p98 artifact", (41, 104): "dyad41 p104 artifact"}
MANIFEST_COLS = ["output_npy", "dyad_id", "speaker_id", "listener_id", "pair_id",
                 "speaker_aq", "listener_aq", "abs_daq", "aq_magnitude", "aq_mag_label",
                 "n_channels", "n_times", "sfreq", "duration_sec", "flagged", "notes"]


def load_aq(xlsx, sheet="Main"):
    import pandas as pd
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df = df[df["dyad_id"].notna()].copy()
    df["__d"] = df["dyad_id"].astype(str).str.extract(r"dyad0*(\d+)")[0]
    lut = {}
    for _, r in df.iterrows():
        if r["__d"] != r["__d"]:
            continue
        key = (int(r["__d"]), str(r["pair_id"]).strip())
        code = r.get("Δ AQ Coding (0=LU, 1=LD, 2=HU, 3=HD)")
        lut[key] = {"aq_code": int(code) if code == code else None,
                    "spk_aq": _f(r.get("Speaker AQ")), "lst_aq": _f(r.get("Listener AQ"))}
    return lut


def _f(v):
    try:
        return float(v) if v == v else None
    except Exception:
        return None


def scan(input_dir: Path, prefix: str, dmin, dmax):
    """Return {dyad:{pid:{role:Path}}} for prefix_dyad{N}_{P}_{role}.edf (skip rest)."""
    rx = re.compile(rf"^{re.escape(prefix)}_dyad_?(\d+)_(\d+)_(speak|listen|rest)\.edf$", re.I)
    found = {}
    for edf in input_dir.rglob("*.edf"):
        m = rx.match(edf.name)
        if not m:
            continue
        dyad, pid, role = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        if role == "rest" or not (dmin <= dyad <= dmax):
            continue
        found.setdefault(dyad, {}).setdefault(pid, {})[role] = edf
    return found


def n_times(edf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.io.read_raw_edf(str(edf), preload=False, verbose="ERROR").n_times


def load_eeg(edf, crop, drop_vref=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
    raw.pick("eeg")
    if drop_vref:
        v = [c for c in raw.ch_names if "VREF" in c.upper() or c.upper() == "CZ"]
        if v:
            raw.drop_channels(v)
    d = raw.get_data(picks="eeg")
    return d[:, :crop], list(raw.ch_names), float(raw.info["sfreq"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--prefix", required=True, help="filename prefix e.g. minusAlpha")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--aq-sheet", required=True)
    ap.add_argument("--dyad-min", type=int, default=1)
    ap.add_argument("--dyad-max", type=int, default=45)
    ap.add_argument("--band-name", default=None, help="short tag for output filenames")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    band = args.band_name or args.prefix
    aq = load_aq(args.aq_sheet)

    found = scan(in_dir, args.prefix, args.dyad_min, args.dyad_max)
    good = {d: sorted(p) for d, p in found.items()
            if len(p) == 2 and all("speak" in p[x] and "listen" in p[x] for x in p)}
    print(f"[{band}] complete dyads: {len(good)} -> {sorted(good)}")

    # global-min crop across all included speak+listen files
    files = [found[d][pid][r] for d in good for pid in good[d] for r in ("speak", "listen")]
    gmin = min(n_times(f) for f in files)
    print(f"[{band}] GLOBAL MIN crop = {gmin} samples ({gmin/250:.1f}s @250Hz); "
          f"stacked length = {2*gmin}")

    rows, n_ok, n_nolabel = [], 0, 0
    for dyad in sorted(good):
        p_lo, p_hi = good[dyad]
        for spk, lst in [(p_lo, p_hi), (p_hi, p_lo)]:
            pair = f"{spk}_{lst}"
            rec = aq.get((dyad, pair), {})
            code = rec.get("aq_code")
            if code is None:
                n_nolabel += 1
                print(f"  no AQ label for dyad{dyad} {pair}; skip")
                continue
            mag = 0 if code in (0, 1) else 1
            flagged = any(dd == dyad and pp in (spk, lst) for (dd, pp) in FAULTY)
            try:
                s_d, s_names, sf = load_eeg(found[dyad][spk]["speak"], gmin)
                l_d, _, _ = load_eeg(found[dyad][lst]["listen"], gmin)
            except Exception as exc:
                print(f"  ERROR dyad{dyad} {pair}: {exc}")
                continue
            stacked = np.concatenate([s_d, l_d], axis=1)  # speaker-first, time-concat
            name = f"{band}_dyad{dyad}_spk{spk}-lst{lst}_mag{mag}.npy"
            np.save(out_dir / name, stacked)
            daq = None
            if rec.get("spk_aq") is not None and rec.get("lst_aq") is not None:
                daq = abs(rec["spk_aq"] - rec["lst_aq"])
            side = {"output_npy": name, "dyad_id": dyad, "speaker_id": spk, "listener_id": lst,
                    "pair_id": pair, "speaker_aq": rec.get("spk_aq"), "listener_aq": rec.get("lst_aq"),
                    "abs_daq": daq, "aq_code": code, "aq_magnitude": mag,
                    "aq_mag_label": "High" if mag else "Low", "shape": list(stacked.shape),
                    "n_channels": stacked.shape[0], "n_times": stacked.shape[1], "sfreq": sf,
                    "duration_sec": stacked.shape[1] / sf, "stacking": "speaker-first time-concat",
                    "band": band, "flagged": flagged, "channel_names": s_names}
            json.dump(side, open((out_dir / name).with_suffix(".json"), "w"), indent=2, default=str)
            rows.append({k: side.get(k, "") for k in MANIFEST_COLS} | {"notes": ""})
            n_ok += 1
            print(f"  {name:44s} shape={stacked.shape} mag={mag}({side['aq_mag_label']})"
                  f"{'  [FLAGGED]' if flagged else ''}")

    with open(out_dir / f"{band}_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"\n[{band}] wrote {n_ok} files, {n_nolabel} skipped(no label)")
    print(f"[{band}] magnitude balance: {dict(Counter(r['aq_magnitude'] for r in rows))}")
    print(f"[{band}] flagged: {[r['output_npy'] for r in rows if r['flagged']]}")
    print(f"[{band}] manifest -> {out_dir/f'{band}_manifest.csv'}")


if __name__ == "__main__":
    main()
