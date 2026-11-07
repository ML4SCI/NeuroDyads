#!/usr/bin/env python
"""
scripts/run_gender_participant_states.py
========================================
Participant-level analysis with speaker / listener / rest kept as SEPARATE
states, reported separately for M and F participants.

Design notes that matter:
  * The unit is the participant-recording. Each participant contributes up to
    three recordings (speak, listen, rest); they are never concatenated.
  * State (speak/listen/rest) varies WITHIN a participant, so it is the one
    target here that is not confounded with participant identity. Gender is
    constant within a participant, so it carries the same identity confound that
    AQ did at the dyad level -- hence both a participant-wise and a dyad-wise
    holdout, plus a permutation control.
  * Dyad-wise holdout matters because leave-one-participant-out still leaves the
    co-recorded dyad partner in training (same session, same headset).
  * Within-person similarity is computed for all three state pairs against a
    null of the same pair taken from different people.
"""
from __future__ import annotations
import argparse, csv, json, re, time, collections
from pathlib import Path
import numpy as np
try:
    import torch, cebra
    from cebra import CEBRA
    from cebra.integrations.sklearn import metrics as cmetrics
except ImportError:
    import sys; sys.exit("need torch + cebra")
import mne
from scipy.stats import mannwhitneyu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, balanced_accuracy_score, f1_score,
                             recall_score, confusion_matrix)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

mne.set_log_level("ERROR")
CEBRA_KW = dict(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                temperature=1.12, conditional="time_delta", output_dimension=3,
                distance="cosine", device="cuda_if_available", verbose=True, time_offsets=10)
STATES = ["speak", "listen", "rest"]
FILE_RE = re.compile(r"^fullband250_dyad(\d+)_(\d+)_(speak|listen|rest)\.edf$", re.I)
FAULTY = {(38, 98), (41, 104)}
SUB = 60000


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--gender-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perms", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
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
    print(f"participants with all 3 states: {len(complete)}  recordings: {len(files)}",
          flush=True)
    gc = collections.Counter(pg[p] for p in complete)
    print(f"  gender split: M={gc.get('M',0)} F={gc.get('F',0)}", flush=True)
    print(f"  dyads represented: {len({f['dyad'] for f in files})}", flush=True)

    crop = min(mne.io.read_raw_edf(str(f["path"]), preload=False,
                                   verbose="ERROR").n_times for f in files)
    print(f"global minimum crop = {crop} samples ({crop/250:.1f} s)", flush=True)

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

    st_i = {s: i for i, s in enumerate(STATES)}
    y_state = np.zeros(len(X), np.int64); y_gen = np.zeros(len(X), np.int64)
    pid_v = np.zeros(len(X), np.int64); dy_v = np.zeros(len(X), np.int64)
    for m in meta:
        s, e = m["start"], m["end"]
        y_state[s:e] = st_i[m["state"]]
        y_gen[s:e] = 0 if m["gender"] == "M" else 1
        pid_v[s:e] = m["pid"]; dy_v[s:e] = m["dyad"]
    print(f"X={X.shape}  states={np.bincount(y_state).tolist()}  "
          f"gender(M,F)={np.bincount(y_gen).tolist()}", flush=True)

    with open(out / "sample_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dyad", "pid", "state", "gender",
                                          "start", "end"])
        w.writeheader(); w.writerows(meta)
    json.dump({"n_participants": len(complete), "n_recordings": len(files),
               "n_dyads": len({m["dyad"] for m in meta}), "crop_samples": int(crop),
               "crop_seconds": round(crop / 250, 2), "n_samples": int(len(X)),
               "gender_split": dict(gc),
               "cebra_config": dict(CEBRA_KW, max_iterations=args.max_iterations,
                                    seed=args.seed)},
              open(out / "dataset_provenance.json", "w"), indent=2)

    # ---- train on STATE (the within-participant, non-confounded target) ----
    t0 = time.time()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
    model.fit(X, y_state); model.save(str(out / "model.pt"))
    emb = model.transform(X).astype(np.float32)
    np.save(out / "embedding.npy", emb)
    lh = list(getattr(model, "state_dict_", {}).get("loss", []))
    fl = float(lh[-1]) if lh else None
    try:
        gof = float(cmetrics.goodness_of_fit_score(model, X, y_state))
    except Exception:
        gof = None
    print(f"trained {(time.time()-t0)/60:.1f} min  loss={fl:.4f} GoF={gof:.4f}", flush=True)

    rng = np.random.RandomState(args.seed)
    sub = rng.choice(len(X), min(SUB, len(X)), replace=False); sub.sort()
    np.save(out / "eval_indices.npy", sub)
    E = emb[sub]; S = y_state[sub]; G = y_gen[sub]; P = pid_v[sub]; D = dy_v[sub]

    res = {"final_loss": fl, "goodness_of_fit_bits": gof}
    res["state_decoding_participant_holdout"] = block(E, S, P, STATES)
    res["state_decoding_dyad_holdout"] = block(E, S, D, STATES)
    res["gender_decoding_participant_holdout"] = block(E, G, P, ["M", "F"])
    res["gender_decoding_dyad_holdout"] = block(E, G, D, ["M", "F"])
    for gl, gi in (("M", 0), ("F", 1)):
        m = G == gi
        if m.sum() > 1000 and len(set(P[m].tolist())) > 2:
            res[f"state_decoding_{gl}_only_participant_holdout"] = block(
                E[m], S[m], P[m], STATES)
    try:
        s2 = rng.choice(len(E), min(10000, len(E)), replace=False)
        res["silhouette_state"] = float(silhouette_score(E[s2], S[s2]))
        res["silhouette_gender"] = float(silhouette_score(E[s2], G[s2]))
    except Exception:
        pass

    # ---- gender permutation across participants, dyad-wise holdout ----
    ppl = sorted(complete)
    gl = np.array([0 if pg[p] == "M" else 1 for p in ppl])
    null = []
    for i in range(args.n_perms):
        pr = np.random.RandomState(8000 + i).permutation(len(ppl))
        gm = {q: int(gl[pr[j]]) for j, q in enumerate(ppl)}
        yq = np.array([gm[int(q)] for q in P])
        yt, yp = grouped_knn(E, yq, D)
        if yt is not None:
            null.append(float((yt == yp).mean()))
    null = np.array(null)
    realg = res["gender_decoding_dyad_holdout"]["accuracy"]
    res["gender_permutation"] = {
        "n_perms": int(len(null)), "real": realg,
        "null_mean": float(null.mean()), "null_std": float(null.std()),
        "empirical_p": float((1 + int((null >= realg).sum())) / (1 + len(null)))}

    # ---- within-person state similarity ----
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    cent = {}
    for m in meta:
        cent[(m["pid"], m["state"])] = unit(emb[m["start"]:m["end"]].mean(0))
    sim_res = {}
    for a, b in (("speak", "listen"), ("speak", "rest"), ("listen", "rest")):
        A = np.array([cent[(q, a)] for q in ppl]); B = np.array([cent[(q, b)] for q in ppl])
        M = A @ B.T
        own = np.diag(M); n = len(ppl)
        oth = M[~np.eye(n, dtype=bool)]
        ranks = np.array([1 + int((M[i] > own[i]).sum()) for i in range(n)])
        u = mannwhitneyu(own, oth, alternative="greater")
        ent = {"own_mean": float(own.mean()), "own_std": float(own.std()),
               "other_mean": float(oth.mean()), "other_std": float(oth.std()),
               "frac_other_above_0.9": float((oth > 0.9).mean()),
               "median_rank_of_own": float(np.median(ranks)),
               "top1_frac": float(np.mean(ranks == 1)),
               "top5_frac": float(np.mean(ranks <= 5)),
               "chance_top1": 1.0 / n, "chance_top5": 5.0 / n,
               "mannwhitney_p": float(u.pvalue), "n_people": n}
        for glab, gi in (("M", "M"), ("F", "F")):
            idx = [i for i, q in enumerate(ppl) if pg[q] == gi]
            if len(idx) >= 2:
                ent[f"own_mean_{glab}"] = float(own[idx].mean())
                ent[f"top5_frac_{glab}"] = float(np.mean(ranks[idx] <= 5))
                ent[f"n_{glab}"] = len(idx)
        sim_res[f"{a}_vs_{b}"] = ent
    res["within_person_state_similarity"] = sim_res

    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print("\n=== RESULTS ===")
    print(json.dumps({k: v for k, v in res.items()
                      if k != "within_person_state_similarity"}, indent=1)[:2500])
    for k, v in sim_res.items():
        print(f"  {k}: own={v['own_mean']:.4f} other={v['other_mean']:.4f} "
              f"medrank={v['median_rank_of_own']:.1f}/{v['n_people']} "
              f"top5={v['top5_frac']*100:.1f}% (chance {v['chance_top5']*100:.1f}%)")
    print("PARTICIPANT_STATES_DONE")


if __name__ == "__main__":
    main()
