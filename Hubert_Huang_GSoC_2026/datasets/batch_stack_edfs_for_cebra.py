#!/usr/bin/env python
"""
scripts/batch_stack_edfs_for_cebra.py
=====================================
Batch wrapper around the official `stacking_edfs.py` workflow.

IMPORTANT — what "stacking" means here
--------------------------------------
`stacking_edfs.py` uses `mne.io.concatenate_raws([raw1, raw2])`, which
concatenates the two recordings along **TIME** (not channels). Both inputs must
share channel names/order and sampling rate; the output keeps the same channels
and has length T1 + T2. This wrapper follows that same time-concatenation,
adding only the cropping the meeting note requires ("cut all to the same
length") plus batch discovery, sidecars, and a manifest.

Per Michelle's stacking guide, for each dyad (participants sorted -> P1 < P2):
    Direction 0 : P1_speak  + P2_listen  -> 0-stacked_dyadXX.edf   (label 0)
    Direction 1 : P1_listen + P2_speak   -> 1-stacked_dyadXX.edf   (label 1)

Cropping
--------
--crop-mode global_min : find the min sample count across ALL included
    speak/listen EDFs first, crop every file to it before concatenating. Each
    stacked output then has exactly 2 * global_min samples. (default; matches
    "cut all to the same length")
--crop-mode pair_min   : crop only the two files in each stack to their pairwise
    minimum.

Originals are never modified. Output EDFs are written via MNE's EDF exporter
(needs `edfio`).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import mne
    from mne.io import concatenate_raws
except ImportError:
    sys.exit("ERROR: mne not installed. pip install 'mne>=1.10.0'")

FILE_RE = re.compile(
    r"^(?:cut_)?dyad_?(?P<dyad>\d+)_(?P<pid>\d+)_(?P<role>speak|listen|rest)\.edf$",
    re.IGNORECASE,
)

FAULTY = {
    (38, 98): "dyad38 participant 98: persistent high-frequency/repetitive "
              "artifact across most/all recording",
    (41, 104): "dyad41 participant 104: persistent abnormal "
               "signal/saturation/repetitive artifacts across large portions",
}

MANIFEST_COLUMNS = [
    "dyad_id", "direction_label", "output_edf", "file1", "file2",
    "p1_id", "p2_id", "file1_role", "file2_role", "crop_samples", "sfreq",
    "output_samples", "output_duration_sec", "flagged", "notes",
]


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def scan(input_dir: Path, dyad_min, dyad_max):
    """Return ({dyad:{pid:{role:Path}}}, rest_files, unknown_files)."""
    found, rest, unknown = {}, [], []
    for edf in sorted(input_dir.rglob("*.edf")):
        m = FILE_RE.match(edf.name)
        if not m:
            unknown.append(edf.name)
            continue
        dyad, pid, role = int(m.group("dyad")), int(m.group("pid")), m.group("role").lower()
        if not (dyad_min <= dyad <= dyad_max):
            continue
        if role == "rest":
            rest.append(edf.name)
            continue
        found.setdefault(dyad, {}).setdefault(pid, {})[role] = edf
    return found, rest, unknown


def complete_dyads(found):
    """Yield (dyad, p1, p2, parts) only for dyads with 2 participants x 2 roles."""
    good, problems = [], []
    for dyad in sorted(found):
        parts = found[dyad]
        pids = sorted(parts)
        if len(pids) != 2:
            problems.append((dyad, f"expected 2 participants, found {len(pids)}: {pids}"))
            continue
        miss = [f"p{pid} missing {role}" for pid in pids
                for role in ("speak", "listen") if role not in parts[pid]]
        if miss:
            problems.append((dyad, "; ".join(miss)))
            continue
        good.append((dyad, pids[0], pids[1], parts))
    return good, problems


def n_times(edf_path: Path) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    return raw.n_times


def load_cropped(edf_path: Path, crop_samples: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    if raw.n_times > crop_samples:
        sfreq = raw.info["sfreq"]
        raw.crop(tmax=(crop_samples - 1) / sfreq)
    return raw


def stack_pair(file1: Path, file2: Path, crop_samples: int, out_path: Path):
    """Crop both to crop_samples, validate, concatenate along time, export EDF.

    Returns (output_samples, sfreq, orig1, orig2).
    """
    raw1 = load_cropped(file1, crop_samples)
    raw2 = load_cropped(file2, crop_samples)
    if raw1.info["ch_names"] != raw2.info["ch_names"]:
        raise ValueError(f"channel name/order mismatch: {file1.name} vs {file2.name}")
    if raw1.info["sfreq"] != raw2.info["sfreq"]:
        raise ValueError(f"sfreq mismatch: {file1.name} vs {file2.name}")
    # capture lengths BEFORE concatenate_raws (it mutates raw1 in place)
    n1, n2 = raw1.n_times, raw2.n_times
    cat = concatenate_raws([raw1, raw2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat.export(str(out_path), fmt="edf", overwrite=True)
    return cat.n_times, float(cat.info["sfreq"]), n1, n2


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Batch time-stack cross-role EDF pairs (stacking_edfs.py workflow).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dyad-min", type=int, default=20)
    p.add_argument("--dyad-max", type=int, default=45)
    p.add_argument("--crop-mode", choices=["global_min", "pair_min"], default="global_min")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--manifest-csv", default=None,
                   help="Default: <output-dir>/stacked_edf_manifest.csv")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    in_dir = Path(args.input_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    if not in_dir.is_dir():
        sys.exit(f"ERROR: --input-dir does not exist: {in_dir}")
    manifest_csv = Path(args.manifest_csv).expanduser() if args.manifest_csv \
        else out_dir / "stacked_edf_manifest.csv"

    print(f"Scanning {in_dir} for dyads {args.dyad_min}-{args.dyad_max} ...")
    found, rest, unknown = scan(in_dir, args.dyad_min, args.dyad_max)
    if unknown:
        print(f"  WARNING: {len(unknown)} unrecognized .edf file(s): {unknown}")
    if rest:
        print(f"  Skipping {len(rest)} rest file(s): {rest}")
    if not found:
        sys.exit("No matching cut_dyad*_*_{speak,listen}.edf found in range.")

    good, problems = complete_dyads(found)
    print(f"Complete dyads: {[d for d, *_ in good]}")
    for dyad, why in problems:
        print(f"  INCOMPLETE dyad{dyad}: {why}")

    # ---- global min sample count (across all included speak/listen files) ----
    all_files = []
    for dyad, p1, p2, parts in good:
        for pid in (p1, p2):
            all_files += [parts[pid]["speak"], parts[pid]["listen"]]
    sample_counts = {f: n_times(f) for f in all_files}
    global_min = min(sample_counts.values()) if sample_counts else 0
    print(f"\nFile count for crop calc: {len(sample_counts)}")
    if args.crop_mode == "global_min":
        print(f"GLOBAL MIN sample count = {global_min} "
              f"({global_min/250.0:.2f}s @250Hz assumed)")

    rows = []
    for dyad, p1, p2, parts in good:
        flagged_notes = []
        for (dy, pid), note in FAULTY.items():
            if dy == dyad and pid in (p1, p2):
                flagged_notes.append(note)
        flagged = bool(flagged_notes)

        # Direction 0: P1_speak + P2_listen ; Direction 1: P1_listen + P2_speak
        directions = [
            (0, parts[p1]["speak"], parts[p2]["listen"], "speak", "listen"),
            (1, parts[p1]["listen"], parts[p2]["speak"], "listen", "speak"),
        ]
        for label, f1, f2, role1, role2 in directions:
            out_name = f"{label}-stacked_dyad{dyad}.edf"
            out_path = out_dir / out_name
            if out_path.exists() and not args.overwrite:
                print(f"  exists (skip, use --overwrite): {out_name}")
                continue

            if args.crop_mode == "global_min":
                crop = global_min
            else:  # pair_min
                crop = min(sample_counts[f1], sample_counts[f2])

            try:
                osamp, sfreq, o1, o2 = stack_pair(f1, f2, crop, out_path)
            except Exception as exc:
                print(f"  ERROR {out_name}: {exc}")
                rows.append({
                    "dyad_id": dyad, "direction_label": label, "output_edf": out_name,
                    "file1": str(f1), "file2": str(f2), "p1_id": p1, "p2_id": p2,
                    "file1_role": role1, "file2_role": role2, "crop_samples": crop,
                    "sfreq": "", "output_samples": "", "output_duration_sec": "",
                    "flagged": flagged, "notes": f"ERROR: {exc}",
                })
                continue

            dur = osamp / sfreq
            sidecar = {
                "dyad_id": dyad, "direction_label": label,
                "file1": str(f1), "file2": str(f2),
                "file1_participant": p1, "file1_role": role1,
                "file2_participant": p2, "file2_role": role2,
                "crop_mode": args.crop_mode, "crop_samples": crop,
                "original_samples": {"file1": o1, "file2": o2,
                                     "file1_raw": sample_counts[f1],
                                     "file2_raw": sample_counts[f2]},
                "sfreq": sfreq, "output_edf": str(out_path),
                "output_samples": osamp, "output_duration_sec": dur,
                "stacking": "time-concatenation (mne.concatenate_raws)",
                "flagged": flagged,
                "notes": " | ".join(flagged_notes),
            }
            with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as fp:
                json.dump(sidecar, fp, indent=2, default=_json_default)

            rows.append({
                "dyad_id": dyad, "direction_label": label, "output_edf": out_name,
                "file1": str(f1), "file2": str(f2), "p1_id": p1, "p2_id": p2,
                "file1_role": role1, "file2_role": role2, "crop_samples": crop,
                "sfreq": sfreq, "output_samples": osamp,
                "output_duration_sec": round(dur, 3), "flagged": flagged,
                "notes": " | ".join(flagged_notes),
            })
            flag = "  [FLAGGED]" if flagged else ""
            print(f"  wrote {out_name:26s} samples={osamp} dur={dur:6.1f}s{flag}")

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    missing = sorted(set(range(args.dyad_min, args.dyad_max + 1)) - set(found))
    n_ok = sum(1 for r in rows if not str(r["notes"]).startswith("ERROR") and r["output_samples"] != "")
    print("\nSummary")
    print(f"  EDFs discovered (in range) : {sum(len(p) for d in found.values() for p in d.values()) if False else len(all_files)}")
    print(f"  complete dyads             : {len(good)}")
    print(f"  stacked EDFs written       : {n_ok}")
    print(f"  incomplete dyads           : {len(problems)}")
    print(f"  missing dyads (no source)  : {missing}")
    print(f"  crop mode / global_min     : {args.crop_mode} / {global_min} samples")
    print(f"  manifest                   : {manifest_csv}")
    print(f"  output dir                 : {out_dir}")


if __name__ == "__main__":
    main()
