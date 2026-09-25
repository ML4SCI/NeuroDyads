#!/usr/bin/env python
"""
scripts/followup_individual_aq_checks.py
========================================
Follow-up checks on the individual-AQ run. No retraining -- reads the embedding.

Why these are needed
--------------------
1. Leave-one-PARTICIPANT-out does not remove session leakage. Each participant's
   dyad partner was recorded in the same session, on the same headset, at the
   same time, and that partner stays in the training set. Anything session-specific
   (impedance, electrode placement, ambient noise, time of day) is therefore
   still available to the decoder. The stricter control is leave-one-DYAD-out,
   which holds out both members of a dyad together. If AQ decoding survives LOPO
   but collapses under LODO, the effect was session leakage.

2. The permutation null (0.117) sits well BELOW the majority-class rate (0.201).
   With labels permuted, k-NN degenerates towards prior-weighted guessing rather
   than majority voting, so "beats the permutation null" is a weaker claim than
   "beats predicting the most common class". Both baselines are reported.

3. within-person cosine 0.957 vs across-person 0.178 looked decisive, yet a
   participant's own partner ranked first only 5% of the time. That combination
   only makes sense if the across-person distribution is wide and multi-modal --
   i.e. the embedding groups people coarsely rather than individuating them.
   Quantified here.
"""
from __future__ import annotations

import csv, json
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

RUN = Path("results/aug4_pipeline/individual_aq_role")


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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perms", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=20000,
                    help="subsample for the permutation null (kNN cost dominates)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb = np.load(RUN / "embedding.npy")
    meta = list(csv.DictReader(open(RUN / "sample_metadata.csv")))
    sub = np.load(RUN / "eval_indices.npy")

    aqv = np.zeros(len(emb), np.float32); pid = np.zeros(len(emb), np.int64)
    dy = np.zeros(len(emb), np.int64); role = np.zeros(len(emb), np.int64)
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        aqv[s:e] = float(m["aq"]); pid[s:e] = int(m["pid"])
        dy[s:e] = int(m["dyad"]); role[s:e] = 0 if m["role"] == "speak" else 1
    E, A, P, D, R = emb[sub], aqv[sub], pid[sub], dy[sub], role[sub]

    lv = sorted(set(A.tolist()))
    cls = np.array([lv.index(v) for v in A], np.int64)
    maj = float(np.bincount(cls).max() / len(cls))
    binlab = (A > 4.0).astype(np.int64)
    maj_bin = float(np.bincount(binlab).max() / len(binlab))

    out = {"majority_chance_exact": maj, "majority_chance_binary": maj_bin,
           "uniform_chance_exact": 1.0 / len(lv)}

    print("=== 1. participant-level vs DYAD-level holdout ===")
    for nm, grp in [("leave-one-participant-out", P), ("leave-one-DYAD-out", D)]:
        yt, yp = grouped_knn(E, cls, grp)
        a = float((yt == yp).mean()); b = float(balanced_accuracy_score(yt, yp))
        ytb, ypb = grouped_knn(E, binlab, grp)
        ab = float((ytb == ypb).mean()); bb = float(balanced_accuracy_score(ytb, ypb))
        out[nm] = {"exact_acc": a, "exact_balanced": b,
                   "binary_acc": ab, "binary_balanced": bb,
                   "n_groups": int(len(set(grp.tolist())))}
        print(f"  {nm:26s} groups={len(set(grp.tolist())):3d} | "
              f"exact acc={a:.4f} bal={b:.4f} (majority {maj:.4f}) | "
              f"binary acc={ab:.4f} bal={bb:.4f} (majority {maj_bin:.4f})")

    print("\n=== 2. AQ permutation across participants, evaluated leave-one-DYAD-out ===")
    # Shuffle which participant carries which AQ score, keeping the multiset of
    # scores identical, then evaluate with the same dyad-level holdout as the real
    # run. This is the null for "is AQ decodable at all from this embedding".
    ppl_aq = {}
    for m in meta:
        ppl_aq[int(m["pid"])] = float(m["aq"])
    people_all = sorted(ppl_aq)
    scores = np.array([ppl_aq[q] for q in people_all])

    # kNN cost dominates; evaluate real and null on the SAME reduced subsample so
    # the comparison stays apples-to-apples.
    rs = np.random.RandomState(args.seed)
    keep = rs.choice(len(E), min(args.n_eval, len(E)), replace=False)
    Ek, clsk, Pk, Dk = E[keep], cls[keep], P[keep], D[keep]
    yt0, yp0 = grouped_knn(Ek, clsk, Dk)
    real_lodo = float((yt0 == yp0).mean())
    print(f"  (null computed on {len(keep)} samples; real on the same subsample "
          f"= {real_lodo:.4f})", flush=True)

    null = []
    for i in range(args.n_perms):
        perm = np.random.RandomState(9000 + i).permutation(len(people_all))
        amap = {q: float(scores[perm[j]]) for j, q in enumerate(people_all)}
        yq = np.array([lv.index(amap[int(q)]) for q in Pk])
        yt, yp = grouped_knn(Ek, yq, Dk)
        null.append(float((yt == yp).mean()))
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{args.n_perms} null mean {np.mean(null):.4f}", flush=True)
    null = np.array(null)
    p_lodo = (1 + int((null >= real_lodo).sum())) / (1 + len(null))
    out["dyad_permutation"] = {"n_perms": len(null), "null_mean": float(null.mean()),
                               "null_std": float(null.std()), "real": real_lodo,
                               "empirical_p": p_lodo}
    print(f"  real (LODO) {real_lodo:.4f} vs dyad-permuted null "
          f"{null.mean():.4f} +- {null.std():.4f}  -> p = {p_lodo:.4f}")
    print(f"  (majority-class baseline {maj:.4f})")

    print("\n=== 3. is the embedding individuating people, or grouping them? ===")
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    cent = {}
    for m in meta:
        cent[(int(m["pid"]), m["role"])] = unit(emb[int(m["start"]):int(m["end"])].mean(0))
    people = sorted({int(m["pid"]) for m in meta})
    L = np.array([cent[(q, "listen")] for q in people if (q, "listen") in cent])
    S = np.array([cent[(q, "speak")] for q in people if (q, "speak") in cent])
    sim = S @ L.T
    n = len(sim)
    own = np.diag(sim)
    oth = sim[~np.eye(n, dtype=bool)]
    ranks = [1 + int((sim[i] > own[i]).sum()) for i in range(n)]
    out["individuation"] = {
        "n_people": n, "own_mean": float(own.mean()),
        "other_mean": float(oth.mean()), "other_std": float(oth.std()),
        "frac_other_above_0.9": float((oth > 0.9).mean()),
        "median_rank_of_own": float(np.median(ranks)),
        "top1_frac": float(np.mean([r == 1 for r in ranks])),
        "top5_frac": float(np.mean([r <= 5 for r in ranks])),
        "chance_top1": 1.0 / n}
    print(f"  own speak-listen cos      : {own.mean():.4f}")
    print(f"  other-person cos          : {oth.mean():.4f} +- {oth.std():.4f}")
    print(f"  fraction of OTHER pairs with cos > 0.9 : {(oth>0.9).mean()*100:.1f}%")
    print(f"  median rank of own partner: {np.median(ranks):.0f} of {n}")
    print(f"  top-1 {np.mean([r==1 for r in ranks])*100:.1f}%  "
          f"top-5 {np.mean([r<=5 for r in ranks])*100:.1f}%  (chance top-1 {100/n:.1f}%)")

    json.dump(out, open(RUN / "followup_checks.json", "w"), indent=2)
    print("\nFOLLOWUP_DONE")


if __name__ == "__main__":
    main()
