#!/usr/bin/env python
"""
PermutationControls/run_individual_gender_permutation_control.py
=======================================================
INDIVIDUAL-PARTICIPANT gender (female/male) permutation control, with CEBRA
retrained per permutation. NOT dyad gender composition (FF/MF/MM).

Written for the October 15 2026 mentor audit.

IMPORTANT DESIGN NOTE (read before reusing any "real" number from elsewhere):
The existing gender_decoding_dyad_holdout number in
results/gender_analysis/participant_states/metrics.json (accuracy=0.7258,
balanced_accuracy=0.4893, p=0.4129) was obtained by decoding gender off an
embedding CEBRA trained supervised by STATE (speak/listen/rest), not by
gender -- see that script's single `model.fit(X, y_state)` call. Reusing
that embedding as the "real" baseline for a gender-supervised full-retraining
test would compare a gender-SUPERVISED null against a gender-UNSUPERVISED
real, which is not a valid like-for-like comparison. Per the mentor's own
framing ("determine whether the actual... labels produce better CEBRA
embeddings than permuted labels, with complete retraining"), this script
therefore trains its OWN real baseline -- CEBRA supervised by the TRUE
gender label, full retrain, identical config -- as permutation 's=None',
before training 200 further models supervised by PERMUTED gender labels.
The AQ controls instead re-score an existing real embedding, because theirs
was already trained on the same target.

STATUS: not yet run end to end; validate on a short run before relying on it.

Design:
  1. Load the identical pooled dataset as
     GenderStateRuns/run_gender_participant_states.py (68 participants, all
     3 states, flagged participants excluded, global min crop, per-channel
     z-score) -- same data, same grouped evaluation folds as the state
     analysis, per the mentor's "keep all EEG data and grouped evaluation
     folds fixed" instruction.
  2. Train ONE real-label model: CEBRA supervised by each recording's
     participant's TRUE gender (all 3 of a participant's recordings carry
     the same gender label, since gender does not vary within a person).
  3. For permutation seed s: permute the PARTICIPANT -> gender assignment
     BETWEEN participants (preserves the 15 M / 53 F split exactly). Every
     one of that participant's recordings (speak, listen, AND rest) gets
     the same permuted label -- never shuffled independently by state.
  4. Retrain CEBRA from scratch on the permuted gender label, same config,
     same model-init seed.
  5. Evaluate with LOPO (participant-holdout) and LODO (dyad-holdout),
     matching the original analysis's two reported holdouts.

Checkpointed: each perm{seed}/metrics.json is written only once; reruns
skip seeds that already have a metrics.json. The real-label model is cached
to real/ and only trained once across the whole run.
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


def permute_gender_between_participants(ppl, gender_by_p, seed):
    """Permute WHICH participant carries which gender label. Preserves the
    M/F multiset exactly (15 M / 53 F)."""
    labels = np.array([gender_by_p[p] for p in ppl])
    rng = np.random.RandomState(7000 + seed)
    perm = rng.permutation(len(labels))
    return {p: int(labels[perm[i]]) for i, p in enumerate(ppl)}


def load_dataset(input_dir, gender_json):
    pg = {int(k): v for k, v in json.load(open(gender_json))["participant_gender"].items()}
    files = []
    for p in sorted(Path(input_dir).glob("*.edf")):
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

    crop = min(mne.io.read_raw_edf(str(f["path"]), preload=False, verbose="ERROR").n_times
               for f in files)

    blocks, meta, cur = [], [], 0
    for f in files:
        raw = mne.io.read_raw_edf(str(f["path"]), preload=True, verbose="ERROR")
        X = raw.get_data()[eeg_picks(raw.ch_names)][:, :crop].T.astype(np.float32)
        raw.close()
        blocks.append(X)
        meta.append({"file": f["path"].name, "dyad": f["dyad"], "pid": f["pid"],
                     "state": f["state"], "gender": f["gender"],
                     "start": cur, "end": cur + X.shape[0]})
        cur += X.shape[0]
    X = np.concatenate(blocks); del blocks
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1
    X -= mu; X /= sd; X = np.ascontiguousarray(X, np.float32)

    pidv = np.zeros(len(X), np.int64); dyadv = np.zeros(len(X), np.int64)
    gen_real = np.zeros(len(X), np.int64)
    for m in meta:
        s, e = m["start"], m["end"]
        pidv[s:e] = m["pid"]; dyadv[s:e] = m["dyad"]
        gen_real[s:e] = 0 if m["gender"] == "M" else 1
    ppl = sorted(complete)
    gender_by_p = {p: (0 if pg[p] == "M" else 1) for p in ppl}
    return X, meta, pidv, dyadv, gen_real, ppl, gender_by_p


def train_and_eval(X, y, pidv, dyadv, max_iterations, model_seed, out_model_path):
    t0 = time.time()
    np.random.seed(model_seed); torch.manual_seed(model_seed)
    model = CEBRA(max_iterations=max_iterations, **CEBRA_KW)
    model.fit(X, y)
    if out_model_path:
        model.save(str(out_model_path))
    emb = model.transform(X).astype(np.float32)
    train_s = time.time() - t0
    loss_hist = list(getattr(model, "state_dict_", {}).get("loss", []))
    final_loss = float(loss_hist[-1]) if loss_hist else None
    try:
        gof = float(cmetrics.goodness_of_fit_score(model, X, y))
    except Exception:
        gof = None

    rng = np.random.RandomState(model_seed)
    T = len(X)
    sub = rng.choice(T, min(SUBSAMPLE, T), replace=False); sub.sort()
    E, Y, P, D = emb[sub], y[sub], pidv[sub], dyadv[sub]
    res = {"final_loss": final_loss, "goodness_of_fit_bits": gof, "train_seconds": round(train_s, 1)}
    res["lopo"] = block(E, Y, P, ["M", "F"])
    res["lodo"] = block(E, Y, D, ["M", "F"])
    return res, emb, sub, loss_hist


def main():
    ap = argparse.ArgumentParser(
        description="Individual-participant gender permutation control with full "
                    "CEBRA retraining per permutation (gender-supervised).")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--gender-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--perm-seeds", type=int, nargs="+", default=[])
    ap.add_argument("--skip-real", action="store_true",
                    help="skip (re)training the real-label baseline, e.g. when it "
                         "was already produced by a previous invocation")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "permutation_mappings").mkdir(exist_ok=True)
    real_dir = out / "real"; real_dir.mkdir(exist_ok=True)

    X, meta, pidv, dyadv, gen_real, ppl, gender_by_p = load_dataset(
        args.input_dir, args.gender_json)
    T = X.shape[0]
    gc = collections.Counter(gender_by_p.values())
    print(f"[gender-perm-control] X={X.shape} participants={len(ppl)} "
          f"M={gc.get(0,0)} F={gc.get(1,0)} chance={np.bincount(gen_real).max()/T:.4f}",
          flush=True)

    with open(out / "dataset_provenance.json", "w") as f:
        json.dump({"n_recordings": len(meta), "n_participants": len(ppl),
                   "n_dyads": len(set(dyadv.tolist())), "n_samples": int(T),
                   "gender_split": {"M": gc.get(0, 0), "F": gc.get(1, 0)},
                   "real_participant_to_gender": {str(p): v for p, v in sorted(gender_by_p.items())},
                   "cebra_config": {**CEBRA_KW, "max_iterations": args.max_iterations,
                                    "model_seed": args.model_seed},
                   "files": [m["file"] for m in meta]}, f, indent=2)

    records = []

    real_metrics_path = real_dir / "metrics.json"
    if args.skip_real and real_metrics_path.exists():
        print("[gender-perm-control] reusing existing real-label result", flush=True)
        real_res = json.load(open(real_metrics_path))
    else:
        print("[gender-perm-control] training REAL gender-supervised baseline "
              "(full retrain, true labels)", flush=True)
        real_res, real_emb, real_sub, real_loss = train_and_eval(
            X, gen_real, pidv, dyadv, args.max_iterations, args.model_seed,
            real_dir / "model.pt")
        real_res["run"] = "real"; real_res["perm_seed"] = None
        json.dump(real_res, open(real_metrics_path, "w"), indent=2)
        np.save(real_dir / "embedding.npy", real_emb)
        np.save(real_dir / "eval_indices.npy", real_sub)
        if real_loss:
            np.save(real_dir / "loss_history.npy", np.array(real_loss, dtype=np.float32))
        print(f"  REAL  LOPO={real_res['lopo']['accuracy']:.4f}  "
              f"LODO={real_res['lodo']['accuracy']:.4f}", flush=True)
    records.append(real_res)

    for ps in args.perm_seeds:
        pdir = out / f"perm{ps}"
        if (pdir / "metrics.json").exists():
            print(f"[gender-perm-control] SKIP perm{ps} (already complete)", flush=True)
            records.append(json.load(open(pdir / "metrics.json")))
            continue
        pdir.mkdir(parents=True, exist_ok=True)

        pmap = permute_gender_between_participants(ppl, gender_by_p, ps)
        assert sorted(pmap.values()) == sorted(gender_by_p.values()), "gender counts changed!"
        gen_perm = np.empty_like(gen_real)
        for m in meta:
            gen_perm[m["start"]:m["end"]] = pmap[m["pid"]]
        assert np.bincount(gen_perm).tolist() == np.bincount(gen_real).tolist(), \
            "sample counts changed!"
        n_moved = sum(1 for p in ppl if pmap[p] != gender_by_p[p])
        json.dump({"perm_seed": ps,
                   "real_participant_to_gender": {str(p): v for p, v in sorted(gender_by_p.items())},
                   "permuted_participant_to_gender": {str(p): v for p, v in sorted(pmap.items())},
                   "n_participants_with_changed_label": n_moved},
                  open(out / "permutation_mappings" / f"perm{ps}_mapping.json", "w"), indent=2)

        print(f"\n=== PERMUTATION {ps} ({n_moved}/{len(ppl)} participants relabelled) ===",
              flush=True)
        m_res, emb, sub, loss_hist = train_and_eval(
            X, gen_perm, pidv, dyadv, args.max_iterations, args.model_seed, pdir / "model.pt")
        m_res["run"] = f"perm{ps}"; m_res["perm_seed"] = ps
        m_res["n_participants_relabelled"] = n_moved
        json.dump(m_res, open(pdir / "metrics.json", "w"), indent=2)
        np.save(pdir / "eval_indices.npy", sub)
        if loss_hist:
            np.save(pdir / "loss_history.npy", np.array(loss_hist, dtype=np.float32))
        records.append(m_res)
        print(f"  perm{ps}: loss={m_res['final_loss']:.4f} GoF={m_res['goodness_of_fit_bits']} "
              f"LOPO={m_res['lopo']['accuracy']:.4f} LODO={m_res['lodo']['accuracy']:.4f} "
              f"({m_res['train_seconds']/60:.1f} min)", flush=True)
        del emb

    cols = ["run", "perm_seed", "n_participants_relabelled", "final_loss",
            "goodness_of_fit_bits", "train_seconds", "lopo_accuracy", "lodo_accuracy"]
    with open(out / "permutation_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            flat = dict(r)
            for key in ("lopo", "lodo"):
                if key in flat and isinstance(flat[key], dict):
                    flat[f"{key}_accuracy"] = flat[key].get("accuracy")
            w.writerow(flat)
    print(f"\n[gender-perm-control] wrote {out/'permutation_metrics.csv'} "
          f"({len(records)} rows)", flush=True)
    print("GENDER_PERMUTATION_CONTROL_DONE", flush=True)


if __name__ == "__main__":
    main()
