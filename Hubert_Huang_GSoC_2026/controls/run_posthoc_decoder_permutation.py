#!/usr/bin/env python
"""
scripts/run_posthoc_decoder_permutation.py
==========================================
TASK 4, SECONDARY control -- cheap dyad-level decoder permutation.

This holds the REAL embedding fixed and only permutes the dyad->label mapping
before re-decoding. It costs nothing compared to retraining CEBRA, so we can run
thousands of permutations, but it CANNOT detect label leakage that happens
during encoder training. It is therefore a supporting control, never a
substitute for run_dyad_aq_permutation_control.py.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def lodo(E, y, g, k=5):
    dy = sorted(set(g.tolist())); c = t = 0
    for d in dy:
        te = g == d; tr = ~te
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(E[tr])
        m = KNeighborsClassifier(k).fit(sc.transform(E[tr]), y[tr])
        c += int((m.predict(sc.transform(E[te])) == y[te]).sum()); t += int(te.sum())
    return c / t if t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--sample-metadata", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument("--n-samples", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    man = {r["output_npy"]: (int(r["dyad_id"]), int(r["aq_magnitude"]))
           for r in csv.DictReader(open(args.manifest))}
    meta = list(csv.DictReader(open(args.sample_metadata)))
    emb = np.load(args.embedding).astype(np.float32)

    idx, ys, gs = [], [], []
    per = max(1, args.n_samples // len(meta))
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        d, lab = man.get(m["npy"], (int(m["dyad"]), int(m["mag"])))
        take = rng.choice(np.arange(s, e), size=min(per, e - s), replace=False)
        idx.append(take); ys.append(np.full(len(take), lab)); gs.append(np.full(len(take), d))
    idx = np.concatenate(idx); ys = np.concatenate(ys); gs = np.concatenate(gs)
    E = emb[idx]
    dyads = sorted(set(gs.tolist()))
    d2l = {}
    for d, l in zip(gs, ys):
        d2l[int(d)] = int(l)
    labels = np.array([d2l[d] for d in dyads])

    real = lodo(E, ys, gs)
    print(f"real LODO 5-NN = {real:.4f}  ({len(dyads)} dyads, {len(idx)} samples)", flush=True)

    null = []
    for i in range(args.n_perms):
        p = rng.permutation(len(labels))
        pmap = {d: int(labels[p[j]]) for j, d in enumerate(dyads)}
        yp = np.array([pmap[int(d)] for d in gs])
        null.append(lodo(E, yp, gs))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.n_perms}  null mean so far {np.mean(null):.4f}", flush=True)
    null = np.array([x for x in null if x is not None])

    p_emp = (1 + int((null >= real).sum())) / (1 + len(null))
    res = {"control": "post-hoc decoder permutation (SECONDARY)",
           "note": "embedding held fixed; only the dyad->label map is permuted. "
                   "Does not test label leakage during CEBRA training.",
           "n_perms": int(len(null)), "n_dyads": len(dyads), "n_samples": int(len(idx)),
           "real_lodo_knn5": float(real),
           "null_mean": float(null.mean()), "null_std": float(null.std()),
           "null_min": float(null.min()), "null_max": float(null.max()),
           "null_p05": float(np.percentile(null, 5)),
           "null_p95": float(np.percentile(null, 95)),
           "empirical_p": float(p_emp),
           "delta_from_null_mean": float(real - null.mean()),
           "z_vs_null": float((real - null.mean()) / null.std()) if null.std() > 0 else None,
           "majority_chance": float(np.bincount(ys).max() / len(ys))}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out_json, "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(null, bins=40, color="#9ecae1", edgecolor="white", label=f"null ({len(null)} perms)")
    ax.axvline(real, color="#D55E00", lw=2.5, label=f"real = {real:.3f}")
    ax.axvline(null.mean(), color="#333333", ls="--", lw=1.5,
               label=f"null mean = {null.mean():.3f}")
    ax.axvline(res["majority_chance"], color="#999999", ls=":", lw=1.5,
               label=f"majority chance = {res['majority_chance']:.3f}")
    ax.set_xlabel("leave-one-dyad-out 5-NN accuracy"); ax.set_ylabel("count")
    ax.set_title(f"Post-hoc dyad-label permutation (secondary control)\n"
                 f"empirical p = {p_emp:.4f}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_png, dpi=300, facecolor="white",
                                    bbox_inches="tight"); plt.close(fig)
    print(json.dumps(res, indent=2))
    print("POSTHOC_PERM_DONE")


if __name__ == "__main__":
    main()
