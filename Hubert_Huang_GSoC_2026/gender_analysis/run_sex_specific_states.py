#!/usr/bin/env python
"""
scripts/run_sex_specific_states.py
==================================
Sex-filtered replica of run_gender_participant_states.py: trains a SEPARATE
CEBRA embedding on only the female (or only the male) participants who have
complete speak/listen/rest triples, at the validated uniform 250 Hz.

This is not a subset of the pooled M+F embedding -- the model never sees the
other sex's data during training, matching the checklist's female-only /
male-only framing.

Reported per sex:
  * dataset counts (participants, dyads, recordings, samples, crop)
  * CEBRA loss / GoF / silhouette
  * 3-class state decoding: leave-one-participant-out (LOPO) and
    leave-one-dyad-out (LODO), majority chance, balanced accuracy, macro F1,
    per-state recall, confusion matrix
  * pairwise binary state decoding (speak/listen, speak/rest, listen/rest),
    LOPO holdout, accuracy / balanced accuracy / majority baseline / delta
  * within-person state-pair similarity (cosine of embedding centroids),
    own vs across-person, median rank, top-1%, top-5%, Mann-Whitney p
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
    import sys
    sys.exit("need torch + cebra")
import mne
from scipy.stats import mannwhitneyu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, balanced_accuracy_score, f1_score,
                             recall_score, confusion_matrix)

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
        te = g == q
        tr = ~te
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(E[tr])
        m = KNeighborsClassifier(k).fit(sc.transform(E[tr]), y[tr])
        yp.append(m.predict(sc.transform(E[te])))
        yt.append(y[te])
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


def pairwise_state(E, y, g, a_idx, b_idx, a_name, b_name):
    m = (y == a_idx) | (y == b_idx)
    if m.sum() < 20 or len(set(g[m].tolist())) < 2:
        return {"pair": a_name + " vs " + b_name, "note": "insufficient data/groups"}
    yb = (y[m] == b_idx).astype(np.int64)
    yt, yp = grouped_knn(E[m], yb, g[m])
    if yt is None:
        return {"pair": a_name + " vs " + b_name, "note": "no valid folds"}
    acc = float((yt == yp).mean())
    base = float(max(np.mean(yt == 0), np.mean(yt == 1)))
    return {"pair": a_name + " vs " + b_name, "n_samples": int(m.sum()),
            "n_groups": int(len(set(g[m].tolist()))),
            "accuracy": round(acc, 4), "majority_baseline": round(base, 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
            "delta_above_baseline": round(acc - base, 4)}


def within_person_similarity(emb, meta, ppl):
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    cent = {}
    for m in meta:
        if m["pid"] in ppl:
            cent[(m["pid"], m["state"])] = unit(emb[m["start"]:m["end"]].mean(0))
    sim_res = {}
    for a, b in (("speak", "listen"), ("speak", "rest"), ("listen", "rest")):
        valid = [q for q in ppl if (q, a) in cent and (q, b) in cent]
        A = np.array([cent[(q, a)] for q in valid])
        B = np.array([cent[(q, b)] for q in valid])
        n = len(valid)
        if n < 3:
            sim_res[a + "_vs_" + b] = {"note": "insufficient participants", "n_people": n}
            continue
        M = A @ B.T
        own = np.diag(M)
        oth = M[~np.eye(n, dtype=bool)]
        ranks = np.array([1 + int((M[i] > own[i]).sum()) for i in range(n)])
        u = mannwhitneyu(own, oth, alternative="greater")
        sim_res[a + "_vs_" + b] = {
            "own_mean": float(own.mean()), "own_std": float(own.std()),
            "other_mean": float(oth.mean()), "other_std": float(oth.std()),
            "frac_other_above_0.9": float((oth > 0.9).mean()),
            "median_rank_of_own": float(np.median(ranks)),
            "top1_frac": float(np.mean(ranks == 1)),
            "top5_frac": float(np.mean(ranks <= 5)),
            "chance_top1": 1.0 / n, "chance_top5": min(5.0, n) / n,
            "mannwhitney_p": float(u.pvalue), "n_people": n}
    return sim_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--gender-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sex", required=True, choices=["M", "F"])
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pg = {int(k): v for k, v in json.load(open(args.gender_json))["participant_gender"].items()}

    files = []
    for p in sorted(Path(args.input_dir).glob("*.edf")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        d, pid, st = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        if (d, pid) in FAULTY or pid not in pg or pg[pid] != args.sex:
            continue
        files.append({"path": p, "dyad": d, "pid": pid, "state": st, "gender": pg[pid]})

    have = collections.defaultdict(set)
    for f in files:
        have[f["pid"]].add(f["state"])
    complete = {p for p, s in have.items() if set(STATES) <= s}
    excluded_incomplete = sorted(set(have) - complete)
    files = [f for f in files if f["pid"] in complete]
    dyads_repr = sorted({f["dyad"] for f in files})
    print("[" + args.sex + "] participants with all 3 states: " + str(len(complete)) +
          "  recordings: " + str(len(files)) + "  dyads: " + str(len(dyads_repr)), flush=True)
    print("[" + args.sex + "] excluded (incomplete states): " + str(excluded_incomplete),
          flush=True)

    crop = min(mne.io.read_raw_edf(str(f["path"]), preload=False,
                                   verbose="ERROR").n_times for f in files)
    print("[" + args.sex + "] global minimum crop = " + str(crop) + " samples (" +
          str(round(crop / 250, 1)) + " s)", flush=True)

    blocks, meta, cur = [], [], 0
    for f in files:
        raw = mne.io.read_raw_edf(str(f["path"]), preload=True, verbose="ERROR")
        X = raw.get_data()[eeg_picks(raw.ch_names)][:, :crop].T.astype(np.float32)
        raw.close()
        blocks.append(X)
        meta.append({"file": f["path"].name, "dyad": f["dyad"], "pid": f["pid"],
                     "state": f["state"], "start": cur, "end": cur + X.shape[0]})
        cur += X.shape[0]
    X = np.concatenate(blocks)
    del blocks
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1
    X -= mu
    X /= sd
    X = np.ascontiguousarray(X, np.float32)

    st_i = {s: i for i, s in enumerate(STATES)}
    y_state = np.zeros(len(X), np.int64)
    pid_v = np.zeros(len(X), np.int64)
    dy_v = np.zeros(len(X), np.int64)
    for m in meta:
        s, e = m["start"], m["end"]
        y_state[s:e] = st_i[m["state"]]
        pid_v[s:e] = m["pid"]
        dy_v[s:e] = m["dyad"]
    class_counts = np.bincount(y_state, minlength=3).tolist()
    print("[" + args.sex + "] X=" + str(X.shape) +
          "  state class counts (speak,listen,rest)=" + str(class_counts), flush=True)

    with open(out / "sample_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dyad", "pid", "state", "start", "end"])
        w.writeheader()
        w.writerows(meta)
    prov = {"sex": args.sex, "n_participants": len(complete), "n_recordings": len(files),
            "n_dyads": len(dyads_repr), "dyads": dyads_repr,
            "excluded_incomplete_participants": excluded_incomplete,
            "crop_samples": int(crop), "crop_seconds": round(crop / 250, 2),
            "n_samples": int(len(X)), "class_counts_speak_listen_rest": class_counts,
            "cebra_config": dict(CEBRA_KW, max_iterations=args.max_iterations, seed=args.seed)}
    json.dump(prov, open(out / "dataset_provenance.json", "w"), indent=2)

    t0 = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
    model.fit(X, y_state)
    model.save(str(out / "model.pt"))
    emb = model.transform(X).astype(np.float32)
    np.save(out / "embedding.npy", emb)
    lh = list(getattr(model, "state_dict_", {}).get("loss", []))
    fl = float(lh[-1]) if lh else None
    try:
        gof = float(cmetrics.goodness_of_fit_score(model, X, y_state))
    except Exception:
        gof = None
    print("[" + args.sex + "] trained " + str(round((time.time() - t0) / 60, 1)) +
          " min  loss=" + str(fl) + " GoF=" + str(gof), flush=True)

    rng = np.random.RandomState(args.seed)
    sub = rng.choice(len(X), min(SUB, len(X)), replace=False)
    sub.sort()
    np.save(out / "eval_indices.npy", sub)
    E = emb[sub]
    S = y_state[sub]
    P = pid_v[sub]
    D = dy_v[sub]

    res = {"sex": args.sex, "final_loss": fl, "goodness_of_fit_bits": gof}
    res["state_decoding_LOPO"] = block(E, S, P, STATES)
    res["state_decoding_LODO"] = block(E, S, D, STATES)
    try:
        s2 = rng.choice(len(E), min(10000, len(E)), replace=False)
        res["silhouette_state"] = float(silhouette_score(E[s2], S[s2]))
    except Exception:
        res["silhouette_state"] = None

    res["pairwise_LOPO"] = [
        pairwise_state(E, S, P, st_i["speak"], st_i["listen"], "speak", "listen"),
        pairwise_state(E, S, P, st_i["speak"], st_i["rest"], "speak", "rest"),
        pairwise_state(E, S, P, st_i["listen"], st_i["rest"], "listen", "rest"),
    ]

    res["within_person_state_similarity"] = within_person_similarity(emb, meta, complete)

    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print("\n=== [" + args.sex + "] RESULTS ===")
    print("  LOPO: " + str(res["state_decoding_LOPO"]))
    print("  LODO: " + str(res["state_decoding_LODO"]))
    print("  silhouette_state: " + str(res["silhouette_state"]))
    for p in res["pairwise_LOPO"]:
        print("  pairwise: " + str(p))
    for k, v in res["within_person_state_similarity"].items():
        print("  " + k + ": " + str(v))
    print("[" + args.sex + "]_STATES_DONE")


if __name__ == "__main__":
    main()
