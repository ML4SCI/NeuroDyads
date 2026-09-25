#!/usr/bin/env python
"""
scripts/relabel_manifest_magnitude.py
=====================================
Rewrite a speaker-first manifest's `aq_magnitude` column under a different
|dAQ| -> Low/High rule, without touching the .npy data.

Mentor's new rule for the non-oscillatory analysis:
    Low  = |dAQ| in {0,1,2}
    High = |dAQ| in {3,4,5}

This is NOT the rule used by every previously reported result, which was:
    Low  = |dAQ| in {0,1}
    High = |dAQ| >= 2

|dAQ| = 2 therefore moves from High to Low, which changes the dyad split and the
majority-class chance level. Any comparison against the earlier full-band numbers
has to account for that, so the script prints both groupings side by side and
writes the old label to `aq_magnitude_oldrule` for traceability.

The original manifest is copied to `<name>.oldrule.bak` before being rewritten.
"""
from __future__ import annotations

import argparse, collections, csv, json, shutil
from pathlib import Path


def parse_rule(s):
    """'0,1,2:0;3,4,5:1' -> {0:0,1:0,2:0,3:1,4:1,5:1}"""
    out = {}
    for part in s.split(";"):
        vals, lab = part.split(":")
        for v in vals.split(","):
            out[int(v)] = int(lab)
    return out


def summarise(rows, key):
    dy = {}
    for r in rows:
        dy[int(r["dyad_id"])] = int(r[key])
    c = collections.Counter(dy.values())
    files = collections.Counter(int(r[key]) for r in rows)
    return dy, c, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rule", default="0,1,2:0;3,4,5:1")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    man = Path(args.manifest)
    rule = parse_rule(args.rule)
    rows = list(csv.DictReader(open(man)))
    if not rows:
        raise SystemExit("empty manifest")

    shutil.copy2(man, man.with_suffix(man.suffix + ".oldrule.bak"))

    unknown = set()
    for r in rows:
        d = int(float(r["abs_daq"]))
        r["aq_magnitude_oldrule"] = r["aq_magnitude"]
        if d in rule:
            r["aq_magnitude"] = str(rule[d])
            r["aq_mag_label"] = "High" if rule[d] else "Low"
        else:
            unknown.add(d)

    kept = [r for r in rows if str(r.get("flagged", "")).lower() != "true"]
    if unknown:
        print(f"!! |dAQ| values with no rule entry: {sorted(unknown)} — rows left unchanged")

    cols = list(rows[0].keys())
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    dy_new, c_new, f_new = summarise(kept, "aq_magnitude")
    dy_old, c_old, f_old = summarise(kept, "aq_magnitude_oldrule")
    daq = collections.Counter(int(float(r["abs_daq"])) for r in kept)
    daq_dy = collections.Counter()
    seen = set()
    for r in kept:
        d = int(r["dyad_id"])
        if d not in seen:
            seen.add(d); daq_dy[int(float(r["abs_daq"]))] += 1

    n_files = len(kept)
    maj_new = max(f_new.values()) / n_files
    maj_old = max(f_old.values()) / n_files

    print(f"manifest: {man}")
    print(f"unflagged files: {n_files}   dyads: {len(dy_new)}")
    print(f"\n|dAQ| distribution (dyads): {dict(sorted(daq_dy.items()))}")
    print(f"\n{'rule':<28s} {'Low dyads':>10s} {'High dyads':>11s} {'Low files':>10s} "
          f"{'High files':>11s} {'chance':>8s}")
    print(f"{'OLD  0,1=Low / >=2=High':<28s} {c_old.get(0,0):>10d} {c_old.get(1,0):>11d} "
          f"{f_old.get(0,0):>10d} {f_old.get(1,0):>11d} {maj_old:>8.4f}")
    print(f"{'NEW  0-2=Low / 3-5=High':<28s} {c_new.get(0,0):>10d} {c_new.get(1,0):>11d} "
          f"{f_new.get(0,0):>10d} {f_new.get(1,0):>11d} {maj_new:>8.4f}")
    moved = [d for d in dy_new if dy_new[d] != dy_old.get(d)]
    print(f"\ndyads whose label changed: {len(moved)} -> {sorted(moved)}")

    info = {"manifest": str(man), "rule": args.rule,
            "n_files_unflagged": n_files, "n_dyads": len(dy_new),
            "daq_distribution_dyads": {str(k): v for k, v in sorted(daq_dy.items())},
            "daq_distribution_files": {str(k): v for k, v in sorted(daq.items())},
            "new_rule": {"low_dyads": c_new.get(0, 0), "high_dyads": c_new.get(1, 0),
                         "low_files": f_new.get(0, 0), "high_files": f_new.get(1, 0),
                         "majority_chance": round(maj_new, 4)},
            "old_rule": {"low_dyads": c_old.get(0, 0), "high_dyads": c_old.get(1, 0),
                         "low_files": f_old.get(0, 0), "high_files": f_old.get(1, 0),
                         "majority_chance": round(maj_old, 4)},
            "dyads_relabelled": sorted(moved),
            "backup": str(man.with_suffix(man.suffix + ".oldrule.bak"))}
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(info, open(args.out_json, "w"), indent=2)
    print("\nRELABEL_DONE")


if __name__ == "__main__":
    main()
