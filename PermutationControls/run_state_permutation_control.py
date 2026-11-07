#!/usr/bin/env python
"""
PermutationControls/run_state_permutation_control.py
==========================================
WITHIN-PARTICIPANT, BLOCKED speaking/listening/resting permutation control,
with CEBRA retrained per permutation.

Written for the October 15 2026 mentor audit. Before this script existed,
the pooled state model (GenderStateRuns/run_gender_participant_states.py)
had no permutation control of its own at all for the STATE target -- the
only permutation control in that script is for GENDER, and it is a
frozen-embedding, decoder-only test (one `model.fit()` call, trained on
state labels; the gender-permutation loop only re-evaluates that same
embedding under shuffled labels -- confirmed by reading the code directly).

Design (mirrors the already-validated dyad/individual AQ permutation
controls one level down, see PermutationControls/run_dyad_aq_permutation_control.py
and PermutationControls/run_individual_aq_permutation_control.py):

  1. Load the validated pooled dataset exactly as
     GenderStateRuns/run_gender_participant_states.py does: 68 participants
     with all 3 states (speak/listen/rest), flagged participants excluded,
     global minimum crop applied, per-channel z-scored.
  2. Verified precondition (checked explicitly, not assumed): every
     participant contributes EXACTLY one recording per state (204
     recordings = 68 participants x 3 states, zero duplicates).
  3. For permutation seed s: WITHIN each participant, independently permute
     that participant's own 3 state labels among their own 3 recordings
     (a uniform-random permutation of {speak, listen, rest} per
     participant, drawn from a per-(participant, seed) RandomState so the
     full mapping is exactly reproducible). This preserves:
       * which recordings belong to which participant and dyad,
       * each participant's 3-recording structure,
       * the GLOBAL state-class counts exactly (68 speak + 68 listen +
         68 rest labels before and after, by construction, since every
         participant always contributes exactly one of each label).
     It does NOT preserve the true recording-to-state correspondence.
  4. Retrain CEBRA from scratch on the permuted state label, with the
     SAME config and the SAME model-init seed (0) as the real run -- only
     the label changes.
  5. Evaluate with the same grouped 5-NN pipeline as the real run:
     leave-one-dyad-out (primary) and leave-one-participant-out
     (secondary), plus the 3 pairwise sub-problems (speak-vs-listen,
     speak-vs-rest, listen-vs-rest) restricted to those two classes.

Checkpointed: each perm{seed}/metrics.json is written only once; reruns
skip seeds that already have a metrics.json.
"""
from __future__ import annotations

import argparse, csv, json, re, sys, time, collections
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
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix

mne.set_log_level("ERROR")

CEBRA_KW = dict(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                temperature=1.12, conditional="time_delta", output_dimension=3,
                distance="cosine", device="cuda_if_available", verbose=True, time_offsets=10)
STATES = ["speak", "listen", "rest"]
FILE_RE = re.compile(r"^fullband250_dyad(\d+)_(\d+)_(speak|listen|rest)\.edf$", re.I)
FAULTY = {(38, 98), (41, 104)}
SUBSAMPLE = 60000


def eeg_picks(names):
    return [i for i, c in enumerate(names)
            if not any(t in c.upper() for t in ("VREF", "STATUS", "TRIGGER", "STI"))]


def grouped_knn(E, y, g, k=5):
    yt, yp = [], []
    for q in sorted(set(g.tolist())):
        te = g == q; tr = ~te
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(E[tr])
        m = KNeighborsClassifier(k).fit(sc.transform(E[tr]), y[tr])
        yp.append(m.predict(sc.transform(E[te]))); yt.append(y[te])
    if not yt:
        return None, None
    return np.concatenate(yt), np.concatenate(yp)


def block(E, y, g, labels):
    yt, yp = grouped_knn(E, y, g)
    if yt is None:
        return {"note": "no valid folds"}
    return {"accuracy": float((yt == yp).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "per_class_recall": {labels[i]: float(v) for i, v in enumerate(
                recall_score(yt, yp, average=None, labels=list(range(len(labels))),
                             zero_division=0))},
            "confusion_matrix": confusion_matrix(
                yt, yp, labels=list(range(len(labels)))).tolist(),
            "majority_chance": float(np.bincount(yt, minlength=len(labels)).max() / len(yt)),
            "n_groups": int(len(set(g.tolist())))}


def pairwise_block(E, y, g, ca, cb):
    """Restrict to two classes (ca, cb) and report binary grouped 5-NN."""
    m = (y == ca) | (y == cb)
    if m.sum() < 10:
        return {"note": "too few samples"}
    yb = (y[m] == cb).astype(np.int64)
    yt, yp = grouped_knn(E[m], yb, g[m])
    if yt is None:
        return {"note": "no valid folds"}
    return {"accuracy": float((yt == yp).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "majority_chance": float(max((yb == 0).mean(), (yb == 1).mean())),
            "n_groups": int(len(set(g[m].tolist())))}


def permute_state_labels_within_participant(ppl, seed):
    """For each participant independently, draw a uniform-random permutation
    of their own {speak, listen, rest} assignment. Returns {pid: {state: new_state}}."""
    mapping = {}
    for p in ppl:
        rng = np.random.RandomState(seed * 100003 + p)
        perm = rng.permutation(3)
        mapping[p] = {STATES[i]: STATES[perm[i]] for i in range(3)}
    return mapping


def main():
    ap = argparse.ArgumentParser(
        description="Within-participant, blocked speak/listen/rest permutation "
                    "control with full CEBRA retraining per permutation.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--gender-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--perm-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--real-embedding", default=None,
                    help="existing real embedding.npy (results/gender_analysis/"
                         "participant_states/embedding.npy) -- re-scored with "
                         "identical evaluation code for a fair comparison")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "permutation_mappings").mkdir(exist_ok=True)
    pg = {int(k): v for k, v in json.load(open(args.gender_json))["participant_gender"].items()}

    files = []
    for p in sorted(Path(args.input_dir).glob("*.edf")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        d, pid, st = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        if (d, pid) in FAULTY or pid not in pg:
            continue
        files.append({"path": p, "dyad": d, "pid": pid, "state": st, "gender": pg[pid]})

    have = collections.defaultdict(set)
    for f in files:
        have[f["pid"]].add(f["state"])
    complete = {p for p, s in have.items() if set(STATES) <= s}
    files = [f for f in files if f["pid"] in complete]

    # ---- verify the "exactly one recording per state per participant" precondition ----
    counts = collections.Counter((f["pid"], f["state"]) for f in files)
    dupes = {k: v for k, v in counts.items() if v != 1}
    if dupes:
        sys.exit(f"ERROR: precondition violated -- these (pid,state) pairs do not have "
                  f"exactly one recording: {dupes}. Refusing to silently permute; a "
                  f"different grouping-preserving design is required.")
    print(f"[state-perm-control] precondition OK: {len(files)} recordings, "
          f"{len(complete)} participants, exactly 1 recording per (pid,state).", flush=True)

    crop = min(mne.io.read_raw_edf(str(f["path"]), preload=False, verbose="ERROR").n_times
               for f in files)
    print(f"[state-perm-control] global minimum crop = {crop} samples ({crop/250:.1f}s)",
          flush=True)

    blocks, meta, cur = [], [], 0
    for f in files:
        raw = mne.io.read_raw_edf(str(f["path"]), preload=True, verbose="ERROR")
        X = raw.get_data()[eeg_picks(raw.ch_names)][:, :crop].T.astype(np.float32)
        raw.close()
        blocks.append(X)
        meta.append({"file": f["path"].name, "dyad": f["dyad"], "pid": f["pid"],
                     "state": f["state"], "start": cur, "end": cur + X.shape[0]})
        cur += X.shape[0]
    X = np.concatenate(blocks); del blocks
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1
    X -= mu; X /= sd; X = np.ascontiguousarray(X, np.float32)
    T = X.shape[0]

    st_i = {s: i for i, s in enumerate(STATES)}
    state_real = np.zeros(T, np.int64); pidv = np.zeros(T, np.int64); dyadv = np.zeros(T, np.int64)
    for m in meta:
        s, e = m["start"], m["end"]
        state_real[s:e] = st_i[m["state"]]; pidv[s:e] = m["pid"]; dyadv[s:e] = m["dyad"]
    ppl = sorted(complete)
    print(f"[state-perm-control] X={X.shape} participants={len(ppl)} dyads="
          f"{len(set(dyadv.tolist()))} classes={np.bincount(state_real).tolist()}", flush=True)

    with open(out / "dataset_provenance.json", "w") as f:
        json.dump({"n_recordings": len(meta), "n_participants": len(ppl),
                   "n_dyads": len(set(dyadv.tolist())), "crop_samples": int(crop),
                   "crop_seconds": round(crop / 250, 2), "n_samples": int(T),
                   "class_counts": np.bincount(state_real).tolist(),
                   "cebra_config": {**CEBRA_KW, "max_iterations": args.max_iterations,
                                    "model_seed": args.model_seed},
                   "files": [m["file"] for m in meta]}, f, indent=2)

    records = []

    if args.real_embedding and Path(args.real_embedding).exists():
        print("[state-perm-control] scoring REAL embedding with identical evaluation code",
              flush=True)
        emb_real = np.load(args.real_embedding).astype(np.float32)
        if len(emb_real) == T:
            rng = np.random.RandomState(args.model_seed)
            sub = rng.choice(T, min(SUBSAMPLE, T), replace=False); sub.sort()
            E, S, P, D = emb_real[sub], state_real[sub], pidv[sub], dyadv[sub]
            m = {"run": "real", "perm_seed": None}
            m["lodo"] = block(E, S, D, STATES)
            m["lopo"] = block(E, S, P, STATES)
            for a, b, nm in [(0, 1, "speak_vs_listen"), (0, 2, "speak_vs_rest"),
                              (1, 2, "listen_vs_rest")]:
                m[nm] = pairwise_block(E, S, D, a, b)
            json.dump(m, open(out / "real_rescored_metrics.json", "w"), indent=2)
            records.append(m)
            print(f"  REAL  LODO={m['lodo']['accuracy']:.4f}  LOPO={m['lopo']['accuracy']:.4f}",
                  flush=True)
        else:
            print(f"  !! real embedding length {len(emb_real)} != {T}; skipping re-score",
                  flush=True)
        del emb_real

    for ps in args.perm_seeds:
        pdir = out / f"perm{ps}"
        if (pdir / "metrics.json").exists():
            print(f"[state-perm-control] SKIP perm{ps} (already complete)", flush=True)
            records.append(json.load(open(pdir / "metrics.json")))
            continue
        pdir.mkdir(parents=True, exist_ok=True)

        pmap = permute_state_labels_within_participant(ppl, ps)
        state_perm = np.empty_like(state_real)
        for m in meta:
            new_state = pmap[m["pid"]][m["state"]]
            state_perm[m["start"]:m["end"]] = st_i[new_state]
        # sanity: global class counts unchanged (guaranteed by construction, verified anyway)
        assert np.bincount(state_perm).tolist() == np.bincount(state_real).tolist(), \
            "global state-class counts changed!"
        n_identity = sum(1 for p in ppl if all(pmap[p][s] == s for s in STATES))
        json.dump({"perm_seed": ps,
                   "mapping_by_participant": {str(p): pmap[p] for p in ppl},
                   "n_participants_with_identity_permutation": n_identity},
                  open(out / "permutation_mappings" / f"perm{ps}_mapping.json", "w"), indent=2)

        print(f"\n=== PERMUTATION {ps} ({len(ppl)-n_identity}/{len(ppl)} participants "
              f"relabelled) ===", flush=True)
        t0 = time.time()
        np.random.seed(args.model_seed); torch.manual_seed(args.model_seed)
        model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
        model.fit(X, state_perm)
        model.save(str(pdir / "model.pt"))
        emb = model.transform(X).astype(np.float32)
        train_s = time.time() - t0

        loss_hist = list(getattr(model, "state_dict_", {}).get("loss", []))
        final_loss = float(loss_hist[-1]) if loss_hist else None
        try:
            gof = float(cmetrics.goodness_of_fit_score(model, X, state_perm))
        except Exception as ex:
            print("gof fail", ex); gof = None

        rng = np.random.RandomState(args.model_seed)
        sub = rng.choice(T, min(SUBSAMPLE, T), replace=False); sub.sort()
        E, S, P, D = emb[sub], state_perm[sub], pidv[sub], dyadv[sub]
        m = {"run": f"perm{ps}", "perm_seed": ps, "final_loss": final_loss,
             "goodness_of_fit_bits": gof, "train_seconds": round(train_s, 1),
             "n_participants_relabelled": len(ppl) - n_identity}
        m["lodo"] = block(E, S, D, STATES)
        m["lopo"] = block(E, S, P, STATES)
        for a, b, nm in [(0, 1, "speak_vs_listen"), (0, 2, "speak_vs_rest"),
                          (1, 2, "listen_vs_rest")]:
            m[nm] = pairwise_block(E, S, D, a, b)
        json.dump(m, open(pdir / "metrics.json", "w"), indent=2)
        np.save(pdir / "eval_indices.npy", sub)
        if loss_hist:
            np.save(pdir / "loss_history.npy", np.array(loss_hist, dtype=np.float32))
        records.append(m)
        print(f"  perm{ps}: loss={final_loss:.4f} GoF={gof} "
              f"LODO={m['lodo']['accuracy']:.4f} LOPO={m['lopo']['accuracy']:.4f} "
              f"({train_s/60:.1f} min)", flush=True)
        del emb, model

    cols = ["run", "perm_seed", "n_participants_relabelled", "final_loss",
            "goodness_of_fit_bits", "train_seconds", "lodo_accuracy", "lopo_accuracy"]
    with open(out / "permutation_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            flat = dict(r)
            for key in ("lodo", "lopo"):
                if key in flat and isinstance(flat[key], dict):
                    flat[f"{key}_accuracy"] = flat[key].get("accuracy")
            w.writerow(flat)
    print(f"\n[state-perm-control] wrote {out/'permutation_metrics.csv'} "
          f"({len(records)} rows)", flush=True)
    print("STATE_PERMUTATION_CONTROL_DONE", flush=True)


if __name__ == "__main__":
    main()
