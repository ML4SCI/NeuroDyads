#!/usr/bin/env python
"""
PermutationControls/run_individual_aq_permutation_control.py
===================================================
PARTICIPANT-LEVEL AQ PERMUTATION CONTROL, with CEBRA retrained per permutation.

Written for the October 9 2026 mentor audit. Before this script existed, the
ONLY participant-level AQ permutation test in the project was the inline
decoder-level check inside CEBRATraining/run_individual_aq_role_mapping.py
(see its --n-perms flag) and PermutationControls/followup_individual_aq_checks.py's
"dyad_permutation" block -- BOTH of those load the single already-trained
embedding ONCE and only re-run k-NN decoding under permuted labels. CEBRA
itself is never retrained in either of them. That is exactly the weaker
control the project's own dyad-level finding (see the 4 August 2026
"Report for Evie", section 1) showed can look misleadingly strong: a frozen
encoder cannot reveal label information the encoder itself absorbed during
training.

This script closes that gap for the participant-level result by mirroring
PermutationControls/run_dyad_aq_permutation_control.py's design, one level down:

  1. Load the validated individual-AQ dataset (EDF Files cut, both-role
     participants only, flagged participants excluded), identically to
     CEBRATraining/run_individual_aq_role_mapping.py.
  2. For permutation seed s: permute the PARTICIPANT -> AQ-score assignment.
     * both of a participant's recordings (speak + listen) get the SAME
       permuted AQ score
     * the multiset of AQ scores across participants is preserved exactly
     * sample ordering and file structure are untouched
  3. Retrain CEBRA from scratch on the permuted label with the SAME config
     and model-init seed as the real run -- only the label changes.
  4. Evaluate with the SAME grouped leave-one-DYAD-out 5-NN used for the
     project's headline participant-level number (the stricter holdout --
     leave-one-participant-out alone does not remove session leakage,
     since a participant's dyad partner stays in the training set).
  5. Also report leave-one-participant-out, for direct comparison against
     the existing frozen-encoder numbers in followup_individual_aq_checks.py.

Checkpointed: each perm{seed}/metrics.json is written only once; reruns skip
seeds that already have a metrics.json, so this is safe to resume after a
crash or an early stop.
"""
from __future__ import annotations

import argparse, csv, json, re, sys, time, warnings
from pathlib import Path

import numpy as np

try:
    import torch, cebra
    from cebra import CEBRA
    from cebra.integrations.sklearn import metrics as cmetrics
except ImportError:
    sys.exit("ERROR: need torch + cebra")
import mne
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

mne.set_log_level("ERROR"); warnings.filterwarnings("ignore")

# ---- CEBRA config: MUST match CEBRATraining/run_individual_aq_role_mapping.py ----
CEBRA_KW = dict(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                temperature=1.12, conditional="time_delta", output_dimension=3,
                distance="cosine", device="cuda_if_available", verbose=True, time_offsets=10)
FILE_RE = re.compile(r"^(?:cut_)?dyad_?(\d+)_(\d+)_(speak|listen)\.edf$", re.I)
FAULTY = {(38, 98), (41, 104)}
SUBSAMPLE = 60000


def eeg_picks(names):
    return [i for i, c in enumerate(names)
            if not any(t in c.upper() for t in ("VREF", "STATUS", "TRIGGER", "STI"))]


def load_participant_aq(xlsx, sheet="Main"):
    import pandas as pd
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df = df[df["dyad_id"].notna()]
    aq = {}
    for _, r in df.iterrows():
        for idc, aqc in (("Speaker ID", "Speaker AQ"), ("Listener ID", "Listener AQ")):
            try:
                pid = int(r[idc]); val = float(r[aqc])
            except Exception:
                continue
            if val == val:
                aq.setdefault(pid, val)
    return aq


def grouped_knn(feat, lab, grp, k=5):
    yt, yp = [], []
    for q in sorted(set(grp.tolist())):
        te = grp == q; tr = ~te
        if te.sum() == 0 or len(set(lab[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(feat[tr])
        m = KNeighborsClassifier(k).fit(sc.transform(feat[tr]), lab[tr])
        yp.append(m.predict(sc.transform(feat[te]))); yt.append(lab[te])
    return np.concatenate(yt), np.concatenate(yp)


def evaluate(emb, cls, dyad_g, pid_g, seed):
    """Grouped decode by dyad (primary, stricter) and by participant
    (secondary, matches the existing frozen-encoder numbers)."""
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(emb), min(SUBSAMPLE, len(emb)), replace=False); sub.sort()
    E, C, Dg, Pg = emb[sub], cls[sub], dyad_g[sub], pid_g[sub]
    out = {"majority_chance": float(np.bincount(C).max() / len(C))}
    yt, yp = grouped_knn(E, C, Dg)
    out["knn5_leave_one_dyad_out"] = float((yt == yp).mean())
    yt2, yp2 = grouped_knn(E, C, Pg)
    out["knn5_leave_one_participant_out"] = float((yt2 == yp2).mean())
    return out, sub


def permute_participant_labels(pid_to_cls, seed):
    """Permute WHICH participant carries which AQ class. Preserves the
    multiset of classes exactly, so per-class sample counts are unchanged."""
    people = sorted(pid_to_cls)
    labels = np.array([pid_to_cls[p] for p in people])
    rng = np.random.RandomState(6000 + seed)
    perm = rng.permutation(len(labels))
    return {p: int(labels[perm[i]]) for i, p in enumerate(people)}


def main():
    ap = argparse.ArgumentParser(
        description="Participant-level AQ permutation control with full CEBRA "
                    "retraining per permutation (see module docstring).")
    ap.add_argument("--input-dir", required=True, help="Folder of cut per-role EDFs "
                    "(same as CEBRATraining/run_individual_aq_role_mapping.py --input-dir).")
    ap.add_argument("--aq-sheet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--perm-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--real-embedding", default=None,
                    help="existing real embedding.npy (results/aug4_pipeline/"
                         "individual_aq_role/embedding.npy) -- re-scored with "
                         "identical evaluation code for a fair comparison")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "permutation_mappings").mkdir(exist_ok=True)
    aq_map = load_participant_aq(args.aq_sheet)

    files = []
    for p in sorted(Path(args.input_dir).glob("*.edf")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        d, pid, role = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        if (d, pid) in FAULTY:
            continue
        if pid not in aq_map:
            continue
        files.append({"path": p, "dyad": d, "pid": pid, "role": role, "aq": aq_map[pid]})

    have = {}
    for f in files:
        have.setdefault(f["pid"], set()).add(f["role"])
    both = {p for p, r in have.items() if {"speak", "listen"} <= r}
    files = [f for f in files if f["pid"] in both]
    print(f"[perm-control-indiv] {len(files)} recordings, {len(both)} participants "
          f"with both roles", flush=True)

    crop = min(mne.io.read_raw_edf(str(f["path"]), preload=False, verbose="ERROR").n_times
               for f in files)
    print(f"[perm-control-indiv] global minimum crop = {crop} samples", flush=True)

    blocks, meta, cur = [], [], 0
    for f in files:
        raw = mne.io.read_raw_edf(str(f["path"]), preload=True, verbose="ERROR")
        X = raw.get_data()[eeg_picks(raw.ch_names)][:, :crop].T.astype(np.float32)
        raw.close()
        blocks.append(X)
        meta.append({**{k: f[k] for k in ("dyad", "pid", "role", "aq")},
                     "file": f["path"].name, "start": cur, "end": cur + X.shape[0]})
        cur += X.shape[0]
    X = np.concatenate(blocks); del blocks
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1
    X -= mu; X /= sd; X = np.ascontiguousarray(X, np.float32)
    T = X.shape[0]

    aq_vals = sorted({m["aq"] for m in meta})
    aq_to_cls = {v: i for i, v in enumerate(aq_vals)}
    pidv = np.zeros(T, np.int64); dyadv = np.zeros(T, np.int64)
    cls_real = np.zeros(T, np.int64)
    for m in meta:
        s, e = m["start"], m["end"]
        pidv[s:e] = m["pid"]; dyadv[s:e] = m["dyad"]; cls_real[s:e] = aq_to_cls[m["aq"]]
    pid_to_cls = {m["pid"]: aq_to_cls[m["aq"]] for m in meta}
    n_people = len(both)
    print(f"[perm-control-indiv] X={X.shape} participants={n_people} "
          f"classes={len(aq_vals)} chance={np.bincount(cls_real).max()/T:.4f}", flush=True)

    with open(out / "dataset_provenance.json", "w") as f:
        json.dump({"n_recordings": len(meta), "n_participants": n_people,
                   "crop_samples": int(crop), "n_samples": int(T),
                   "aq_levels_present": aq_vals,
                   "real_participant_to_aq_class": {str(k): v for k, v in sorted(pid_to_cls.items())},
                   "cebra_config": {**CEBRA_KW, "max_iterations": args.max_iterations,
                                    "model_seed": args.model_seed},
                   "files": [m["file"] for m in meta]}, f, indent=2)

    records = []

    if args.real_embedding and Path(args.real_embedding).exists():
        print("[perm-control-indiv] scoring REAL embedding with identical evaluation code", flush=True)
        emb_real = np.load(args.real_embedding).astype(np.float32)
        if len(emb_real) == T:
            m, sub = evaluate(emb_real, cls_real, dyadv, pidv, args.model_seed)
            m.update({"run": "real", "perm_seed": None})
            json.dump(m, open(out / "real_rescored_metrics.json", "w"), indent=2)
            records.append(m)
            print(f"  REAL  LODO(dyad) 5-NN={m['knn5_leave_one_dyad_out']:.4f}  "
                  f"LOPO(participant) 5-NN={m['knn5_leave_one_participant_out']:.4f}", flush=True)
        else:
            print(f"  !! real embedding length {len(emb_real)} != {T}; skipping re-score "
                  "(file set/crop does not match the original run)", flush=True)
        del emb_real

    for ps in args.perm_seeds:
        pdir = out / f"perm{ps}"
        if (pdir / "metrics.json").exists():
            print(f"[perm-control-indiv] SKIP perm{ps} (already complete)", flush=True)
            records.append(json.load(open(pdir / "metrics.json")))
            continue
        pdir.mkdir(parents=True, exist_ok=True)
        pmap = permute_participant_labels(pid_to_cls, ps)
        assert sorted(pmap.values()) == sorted(pid_to_cls.values()), "class counts changed!"
        cls_perm = np.empty_like(cls_real)
        for m in meta:
            cls_perm[m["start"]:m["end"]] = pmap[m["pid"]]
        assert np.bincount(cls_perm).tolist() == np.bincount(cls_real).tolist(), "sample counts changed!"
        n_moved = sum(1 for p in pmap if pmap[p] != pid_to_cls[p])
        json.dump({"perm_seed": ps,
                   "real_participant_to_aq_class": {str(k): v for k, v in sorted(pid_to_cls.items())},
                   "permuted_participant_to_aq_class": {str(k): v for k, v in sorted(pmap.items())},
                   "n_participants_with_changed_label": n_moved},
                  open(out / "permutation_mappings" / f"perm{ps}_mapping.json", "w"), indent=2)

        print(f"\n=== PERMUTATION {ps} ({n_moved}/{n_people} participants relabelled) ===", flush=True)
        t0 = time.time()
        np.random.seed(args.model_seed); torch.manual_seed(args.model_seed)
        model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
        model.fit(X, cls_perm)
        model.save(str(pdir / "model.pt"))
        emb = model.transform(X).astype(np.float32)
        train_s = time.time() - t0

        loss_hist = list(getattr(model, "state_dict_", {}).get("loss", []))
        final_loss = float(loss_hist[-1]) if loss_hist else None
        try:
            gof = float(cmetrics.goodness_of_fit_score(model, X, cls_perm))
        except Exception as ex:
            print("gof fail", ex); gof = None

        m, sub = evaluate(emb, cls_perm, dyadv, pidv, args.model_seed)
        m.update({"run": f"perm{ps}", "perm_seed": ps, "final_loss": final_loss,
                  "goodness_of_fit_bits": gof, "train_seconds": round(train_s, 1),
                  "n_participants_relabelled": n_moved})
        json.dump(m, open(pdir / "metrics.json", "w"), indent=2)
        np.save(pdir / "eval_indices.npy", sub)
        if loss_hist:
            np.save(pdir / "loss_history.npy", np.array(loss_hist, dtype=np.float32))
        records.append(m)
        print(f"  perm{ps}: loss={final_loss:.4f} GoF={gof} "
              f"LODO(dyad)5NN={m['knn5_leave_one_dyad_out']:.4f} "
              f"LOPO(pid)5NN={m['knn5_leave_one_participant_out']:.4f} "
              f"({train_s/60:.1f} min)", flush=True)
        del emb, model

    cols = ["run", "perm_seed", "n_participants_relabelled", "final_loss",
            "goodness_of_fit_bits", "knn5_leave_one_dyad_out",
            "knn5_leave_one_participant_out", "majority_chance", "train_seconds"]
    with open(out / "permutation_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"\n[perm-control-indiv] wrote {out/'permutation_metrics.csv'}  ({len(records)} rows)", flush=True)
    print("PERMUTATION_CONTROL_DONE", flush=True)


if __name__ == "__main__":
    main()
