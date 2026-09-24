#!/usr/bin/env python
"""
run_250hz_rerun.py
==================
Re-run every analysis that was built on the mixed-rate (250/1000 Hz) 33-dyad
dataset, now on the uniformly-resampled 250 Hz version.

Requested by EM: the source recordings mixed 1000 Hz (14 of the 33 included
dyads) and 250 Hz (19 dyads). Because the stacker cropped to a fixed SAMPLE
COUNT, the two groups contributed 90.2 s vs 361 s of real time per role, and the
fixed time_offsets=10 spanned 10 ms vs 40 ms. All sources are now 250 Hz, so a
sample-count crop is automatically a duration crop.

Not re-run (already uniform 250 Hz, verified 504/504):
  * the band-removal comparison
  * the non-oscillatory analysis
  * the individual-participant analysis (built from `EDF Files cut`, 250 Hz)

Idempotent and non-fatal, same as the other drivers.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUG = ROOT / "results" / "aug4_pipeline"
LOGS = AUG / "logs"
# --data-root (see argparse below) is expected to contain, at minimum:
#   <data-root>/Filtered CEBRA Inputs/fullband250/fullband250_manifest.csv
# i.e. the finalized uniform-250 Hz speaker-first dataset produced by
# preprocessing/resample_33dyad_to_250.py + datasets/batch_speaker_first_stack.py.
BASE = AUG / "rerun250" / "fullband_baseline"
PERM = AUG / "rerun250" / "dyad_permutation"
D6 = AUG / "rerun250" / "aq_delta6"
GMM = AUG / "rerun250" / "gmm_identity_check"
SVM = AUG / "rerun250" / "aq_delta6_svm"
PY = sys.executable


def log(m):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "rerun250.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, logfile):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(logfile, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    tail = ""
    try:
        t = Path(logfile).read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(l for l in t.replace("\r", "\n").splitlines()
                         if l.strip() and "it/s]" not in l)[-800:]
    except Exception:
        pass
    return p.returncode == 0, tail


def stages(data_root: Path):
    SF = data_root / "Filtered CEBRA Inputs" / "fullband250"
    MAN = SF / "fullband250_manifest.csv"
    return [
        dict(id="B1", name="Full-band speaker-first baseline @250 Hz",
             done=lambda: (BASE / "metrics.json").exists(),
             blocked=lambda: None if MAN.exists() else "250 Hz manifest missing",
             cmd=[PY, "-u", "scripts/run_band_cebra_analysis.py",
                  "--data-dir", str(SF), "--manifest", str(MAN),
                  "--band", "fullband250",
                  "--out-dir", str(BASE.relative_to(ROOT)),
                  "--max-iterations", "5000", "--seed", "0"],
             logf="rerun250_baseline.log"),

        dict(id="P1", name="Dyad permutation control @250 Hz (5 retrainings)",
             done=lambda: len(list(PERM.glob("perm*/metrics.json"))) >= 5,
             blocked=lambda: None if (BASE / "embedding.npy").exists()
             else "baseline embedding missing",
             cmd=[PY, "-u", "scripts/run_dyad_aq_permutation_control.py",
                  "--data-dir", str(SF), "--manifest", str(MAN),
                  "--out-dir", str(PERM.relative_to(ROOT)),
                  "--condition", "fullband250_speakerfirst",
                  "--max-iterations", "5000", "--model-seed", "0",
                  "--perm-seeds", "0", "1", "2", "3", "4",
                  "--real-embedding", str((BASE / "embedding.npy").relative_to(ROOT))],
             logf="rerun250_permutation.log"),

        dict(id="P2", name="Permutation summary @250 Hz",
             done=lambda: (PERM / "permutation_summary.md").exists(),
             blocked=lambda: None if list(PERM.glob("perm*/metrics.json"))
             else "no permutations completed",
             cmd=[PY, "-u", "scripts/summarize_permutation.py",
                  "--perm-dir", str(PERM.relative_to(ROOT)),
                  "--real-metrics", str((BASE / "metrics.json").relative_to(ROOT)),
                  "--real-rescored",
                  str((PERM / "real_rescored_metrics.json").relative_to(ROOT)),
                  "--out-md", str((PERM / "permutation_summary.md").relative_to(ROOT))],
             logf="rerun250_perm_summary.log"),

        dict(id="D6", name="Six-class |dAQ| @250 Hz",
             done=lambda: (D6 / "metrics.json").exists(),
             blocked=lambda: None if MAN.exists() else "250 Hz manifest missing",
             cmd=[PY, "-u", "scripts/run_aq_delta6_pipeline.py",
                  "--data-dir", str(SF), "--manifest", str(MAN),
                  "--out-dir", str(D6.relative_to(ROOT)),
                  "--max-iterations", "5000", "--seed", "0", "--drop-daq", "7",
                  "--kmax", "20", "--label", "fullband250_speakerfirst"],
             logf="rerun250_delta6.log"),

        dict(id="S1", name="Pairwise SVM + six-panel figure @250 Hz",
             done=lambda: (SVM / "svm_metrics.json").exists(),
             blocked=lambda: None if (D6 / "embedding.npy").exists()
             else "six-class embedding missing",
             cmd=[PY, "-u", "scripts/aq_delta6_svm_and_panels.py",
                  "--run-dir", str(D6.relative_to(ROOT)),
                  "--out-dir", str(SVM.relative_to(ROOT)),
                  "--n-samples", "24000", "--seed", "0"],
             logf="rerun250_svm.log"),

        dict(id="G1", name="GMM K-sweep / dyad identity @250 Hz",
             done=lambda: (GMM / "identity_answers.json").exists(),
             blocked=lambda: None if (BASE / "embedding.npy").exists()
             else "baseline embedding missing",
             cmd=[PY, "-u", "scripts/analyze_gmm_component_identity.py",
                  "--embedding", str((BASE / "embedding.npy").relative_to(ROOT)),
                  "--sample-metadata", str((BASE / "sample_metadata.csv").relative_to(ROOT)),
                  "--manifest", str(MAN),
                  "--out-dir", str(GMM.relative_to(ROOT)),
                  "--kmax", "20", "--n-init", "10", "--per-dyad", "2000", "--seed", "0",
                  "--label", "fullband250_speakerfirst"],
             logf="rerun250_gmm.log"),
    ]


def main():
    ap = argparse.ArgumentParser(
        description="Idempotent orchestrator for the validated uniform-250 Hz "
                    "pipeline: full-band baseline -> dyad permutation control "
                    "-> six-class |dAQ| -> SVM comparisons -> GMM K-sweep. "
                    "Each stage is skipped if its output already exists.")
    ap.add_argument("--data-root", required=True, type=Path,
                    help="Root folder containing 'Filtered CEBRA Inputs/"
                         "fullband250/fullband250_manifest.csv' (the "
                         "finalized 250 Hz speaker-first dataset).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these stage IDs (e.g. --only B1 P1).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the execution plan without running anything.")
    args = ap.parse_args()
    plan = stages(args.data_root)
    if args.only:
        plan = [s for s in plan if s["id"] in args.only]

    log("=" * 70)
    log(f"250 Hz RE-RUN START  stages: {[s['id'] for s in plan]}")
    log("=" * 70)
    if args.dry_run:
        for s in plan:
            b = s["blocked"]()
            log(f"  {s['id']:3s} {s['name'][:52]:52s} "
                f"{'BLOCKED:' + b if b else ('done' if s['done']() else 'WILL RUN')}")
        return

    res, t0 = [], time.time()
    for s in plan:
        b = s["blocked"]()
        if b:
            log(f"[{s['id']}] BLOCKED - {b}"); res.append((s["id"], "blocked", 0)); continue
        try:
            if s["done"]():
                log(f"[{s['id']}] skip - already complete")
                res.append((s["id"], "skipped", 0)); continue
        except Exception:
            pass
        log(f"[{s['id']}] RUN  {s['name']}")
        t = time.time()
        ok, tail = run(s["cmd"], LOGS / s["logf"])
        try:
            v = s["done"]()
        except Exception:
            v = False
        st = "completed" if (ok and v) else ("failed" if not ok else "incomplete")
        log(f"[{s['id']}] {st}  ({(time.time()-t)/60:.1f} min)")
        if st != "completed":
            log(f"      tail: {tail[-400:]}")
        res.append((s["id"], st, round((time.time() - t) / 60, 1)))

    log("=" * 70)
    log(f"250 Hz RE-RUN DONE in {(time.time()-t0)/60:.1f} min")
    for i, st, m in res:
        log(f"  {i:3s} {st:11s} {m} min")
    json.dump([{"id": i, "status": s, "minutes": m} for i, s, m in res],
              open(AUG / "rerun250_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
