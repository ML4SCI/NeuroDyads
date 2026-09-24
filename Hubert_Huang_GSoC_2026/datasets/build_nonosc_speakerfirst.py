#!/usr/bin/env python
"""
scripts/build_nonosc_speakerfirst.py
====================================
Speaker-first stacking for the NON-OSCILLATORY (aperiodic) representation.

Mirrors batch_speaker_first_stack.py exactly -- same dyad handling, same AQ
magnitude labelling, same flagged exclusions -- but the per-recording arrays are
(128 aperiodic features, n_windows) instead of (64 channels, n_samples).

For each dyad both interactions are kept, each ordered speaker-first:
    [A_speak ; B_listen]   and   [B_speak ; A_listen]
Both carry the SAME AQ-magnitude label because |dAQ| is symmetric.
"""
from __future__ import annotations

import argparse, csv, re, sys
from pathlib import Path

import numpy as np

FAULTY = {(38, 98): "dyad38 p98 artifact", (41, 104): "dyad41 p104 artifact"}
NPY_RE = re.compile(r"^nonosc_dyad(\d+)_(\d+)_(speak|listen|rest)\.npy$", re.I)
COLS = ["output_npy", "dyad_id", "speaker_id", "listener_id", "pair_id", "speaker_aq",
        "listener_aq", "abs_daq", "aq_magnitude", "aq_mag_label", "n_features",
        "n_times", "flagged", "notes"]


def _f(v):
    try:
        return float(v) if v == v else None
    except Exception:
        return None


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", required=True)
    ap.add_argument("--aq-sheet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="nonosc")
    args = ap.parse_args()

    fd = Path(args.feature_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lut = load_aq(args.aq_sheet)

    # index available feature files: (dyad, participant, role) -> path
    idx = {}
    for p in sorted(fd.glob("*.npy")):
        m = NPY_RE.match(p.name)
        if m:
            idx[(int(m.group(1)), int(m.group(2)), m.group(3).lower())] = p
    dyads = sorted({k[0] for k in idx})
    print(f"{len(idx)} feature files over {len(dyads)} dyads", flush=True)

    # global minimum window count, so every stacked file has the same length
    nmin = min(np.load(p, mmap_mode="r").shape[1] for p in idx.values())
    print(f"global minimum window count = {nmin}", flush=True)

    rows, skipped = [], []
    for d in dyads:
        parts = sorted({k[1] for k in idx if k[0] == d})
        if len(parts) != 2:
            skipped.append((d, f"expected 2 participants, found {len(parts)}")); continue
        for spk, lst in ((parts[0], parts[1]), (parts[1], parts[0])):
            ks, kl = (d, spk, "speak"), (d, lst, "listen")
            if ks not in idx or kl not in idx:
                skipped.append((d, f"missing {ks if ks not in idx else kl}")); continue
            pair = f"{spk}_{lst}"
            info = lut.get((d, pair))
            if info is None or info["aq_code"] is None:
                skipped.append((d, f"no AQ label for pair {pair}")); continue
            code = info["aq_code"]
            spk_aq, lst_aq = info["spk_aq"], info["lst_aq"]
            daq = abs(spk_aq - lst_aq) if (spk_aq is not None and lst_aq is not None) else None
            # Low = |dAQ| in {0,1} (aq_code 0/1); High = |dAQ| >= 2 (aq_code 2/3)
            mag = 0 if code in (0, 1) else 1
            A = np.load(idx[ks])[:, :nmin]
            B = np.load(idx[kl])[:, :nmin]
            if A.shape[0] != B.shape[0]:
                skipped.append((d, "feature-count mismatch")); continue
            stacked = np.concatenate([A, B], axis=1).astype(np.float32)   # speaker THEN listener
            flagged = (d, spk) in FAULTY or (d, lst) in FAULTY
            name = f"{args.prefix}_dyad{d}_spk{spk}-lst{lst}_mag{mag}.npy"
            np.save(out / name, stacked)
            rows.append({"output_npy": name, "dyad_id": d, "speaker_id": spk,
                         "listener_id": lst, "pair_id": pair, "speaker_aq": spk_aq,
                         "listener_aq": lst_aq, "abs_daq": daq, "aq_magnitude": mag,
                         "aq_mag_label": "High" if mag else "Low",
                         "n_features": stacked.shape[0], "n_times": stacked.shape[1],
                         "flagged": flagged,
                         "notes": FAULTY.get((d, spk), FAULTY.get((d, lst), ""))})
            print(f"  {name:52s} shape={stacked.shape} mag={mag}"
                  f"{'  [FLAGGED]' if flagged else ''}", flush=True)

    man = Path(args.out_dir) / f"{args.prefix}_manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)

    keep = [r for r in rows if not r["flagged"]]
    import collections
    print(f"\n[{args.prefix}] wrote {len(rows)} files ({len(keep)} unflagged over "
          f"{len({r['dyad_id'] for r in keep})} dyads)")
    print(f"[{args.prefix}] magnitude balance (unflagged): "
          f"{dict(collections.Counter(r['aq_magnitude'] for r in keep))}")
    print(f"[{args.prefix}] |dAQ| distribution (unflagged dyads): "
          f"{dict(collections.Counter(int(r['abs_daq']) for r in keep))}")
    if skipped:
        print(f"[{args.prefix}] skipped: {skipped}")
    print(f"[{args.prefix}] manifest -> {man}")
    print("NONOSC_STACK_DONE")


if __name__ == "__main__":
    main()
