#!/usr/bin/env python
"""
scripts/aq_delta6_svm_and_panels.py
===================================
Mentor items 1 and 2, on the existing six-class |dAQ| CEBRA embedding.

ITEM 1 -- pairwise multi-class SVM as an alternative decoder to 5-NN.
  "Pairwise" = one-vs-one: sklearn.svm.SVC fits K(K-1)/2 = 15 binary SVMs and
  votes. We report it the same way the KNN result is reported -- grouped
  leave-one-dyad-out -- so the two are directly comparable, plus the per-pair
  binary accuracies, which are what a pairwise scheme is actually informative
  about (KNN gives no equivalent).
  Linear and RBF kernels are both run. Features are standardised inside each
  fold (fit on train only) so no test information leaks.

ITEM 2 -- Figure 7 split into six per-class longitude/latitude panels, so each
  |dAQ| distribution is legible on its own instead of overplotted. Each panel
  shows its class against a grey backdrop of all samples for context, with the
  class median marked.

Both read the saved embedding; no CEBRA retraining.
"""
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score, f1_score,
                             recall_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#56B4E9"]


def wrap180(x):
    return (x + 180) % 360 - 180


def largest_gap_cut(lon, nbins=720):
    h, edges = np.histogram(wrap180(lon), bins=nbins, range=(-180, 180))
    empty = h == 0
    if not empty.any():
        k = max(1, nbins // 36)
        run = np.convolve(np.r_[h, h], np.ones(k), "valid")[:nbins]
        return float(wrap180(edges[int(np.argmin(run))] + (k / 2) * (360 / nbins)))
    doubled = np.r_[empty, empty]
    bl = bs = cur = 0
    for i, v in enumerate(doubled):
        cur = cur + 1 if v else 0
        if cur > bl:
            bl, bs = cur, i - cur + 1
    if bs >= nbins:
        bs -= nbins
    return float(wrap180(-180 + (bs + bl / 2.0) * (360 / nbins)))


def lodo_predict(clf_factory, X, y, g):
    """Grouped leave-one-dyad-out; returns pooled (y_true, y_pred)."""
    yt, yp = [], []
    for d in sorted(set(g.tolist())):
        te = g == d
        tr = ~te
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = clf_factory().fit(sc.transform(X[tr]), y[tr])
        yp.append(m.predict(sc.transform(X[te])))
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


def score_block(yt, yp, k, inv):
    return {"accuracy": float((yt == yp).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "per_class_recall": {str(inv[i]): float(v) for i, v in enumerate(
                recall_score(yt, yp, average=None, labels=list(range(k)), zero_division=0))},
            "confusion_matrix": confusion_matrix(yt, yp, labels=list(range(k))).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/aug4_pipeline/aq_delta6")
    ap.add_argument("--out-dir", default="results/aug4_pipeline/aq_delta6_svm")
    ap.add_argument("--n-samples", type=int, default=24000,
                    help="dyad-balanced subsample used for the SVM (kernel cost is O(n^2))")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    prov = json.load(open(run / "dataset_provenance.json"))
    kept = prov["kept_daq_values"]
    inv = {i: v for i, v in enumerate(kept)}
    K = len(kept)

    emb = np.load(run / "embedding.npy").astype(np.float32)
    meta = list(csv.DictReader(open(run / "sample_metadata.csv")))

    # dyad-balanced subsample (SVM with RBF is O(n^2)-ish; 24k is tractable)
    by_dyad = {}
    for m in meta:
        by_dyad.setdefault(int(m["dyad"]), []).append(m)
    n_dyads = len(by_dyad)
    per_dyad = max(50, args.n_samples // n_dyads)
    idx, y, g = [], [], []
    for d in sorted(by_dyad):
        segs = by_dyad[d]
        per_seg = max(10, per_dyad // len(segs))
        for m in segs:
            s, e = int(m["start"]), int(m["end"])
            take = rng.choice(np.arange(s, e), size=min(per_seg, e - s), replace=False)
            idx.append(take)
            y.append(np.full(len(take), int(m["cls"]), np.int64))
            g.append(np.full(len(take), d, np.int64))
    idx = np.concatenate(idx); y = np.concatenate(y); g = np.concatenate(g)
    order = np.argsort(idx); idx, y, g = idx[order], y[order], g[order]
    E = emb[idx]
    chance = float(np.bincount(y).max() / len(y))
    print(f"[svm] {len(idx)} samples over {n_dyads} dyads, {K} classes, "
          f"majority chance {chance:.4f}", flush=True)
    np.save(out / "svm_eval_indices.npy", idx)

    # ---------------- item 1: pairwise (one-vs-one) SVM ----------------
    results = {}
    for name, factory in [
        ("svm_ovo_linear", lambda: SVC(kernel="linear", C=1.0,
                                       decision_function_shape="ovo", cache_size=500)),
        ("svm_ovo_rbf", lambda: SVC(kernel="rbf", C=1.0, gamma="scale",
                                    decision_function_shape="ovo", cache_size=500)),
        ("knn5_reference", lambda: KNeighborsClassifier(5)),
    ]:
        t0 = time.time()
        yt, yp = lodo_predict(factory, E, y, g)
        r = score_block(yt, yp, K, inv)
        r["seconds"] = round(time.time() - t0, 1)
        r["majority_chance"] = chance
        results[name] = r
        print(f"  {name:16s} acc={r['accuracy']:.4f} bal={r['balanced_accuracy']:.4f} "
              f"macroF1={r['macro_f1']:.4f}  ({r['seconds']}s)", flush=True)

    # per-pair binary SVM accuracy -- the thing a pairwise scheme actually tells you
    pair_rows = []
    for a in range(K):
        for b in range(a + 1, K):
            m = (y == a) | (y == b)
            if m.sum() < 20:
                continue
            ya = (y[m] == b).astype(int)
            yt, yp = lodo_predict(
                lambda: SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=500),
                E[m], ya, g[m])
            acc = float((yt == yp).mean())
            base = float(max(np.mean(yt == 0), np.mean(yt == 1)))
            pair_rows.append({"class_a": a, "class_b": b, "daq_a": inv[a], "daq_b": inv[b],
                              "n_samples": int(m.sum()), "svm_rbf_lodo_acc": round(acc, 4),
                              "majority_baseline": round(base, 4),
                              "above_baseline": round(acc - base, 4)})
            print(f"    |dAQ| {inv[a]} vs {inv[b]}: acc={acc:.3f} (baseline {base:.3f})",
                  flush=True)
    with open(out / "svm_pairwise_accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader(); w.writerows(pair_rows)

    json.dump({"n_samples": int(len(idx)), "n_dyads": n_dyads, "n_classes": K,
               "kept_daq_values": kept, "majority_chance": chance,
               "decoder_comparison": results,
               "note": "grouped leave-one-dyad-out throughout; features standardised "
                       "within each fold on training data only"},
              open(out / "svm_metrics.json", "w"), indent=2)

    # ---- decoder comparison figure ----
    names = ["knn5_reference", "svm_ovo_linear", "svm_ovo_rbf"]
    pretty = ["5-NN\n(reference)", "SVM one-vs-one\nlinear", "SVM one-vs-one\nRBF"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(len(names))
    wid = 0.38
    axs[0].bar(xs - wid / 2, [results[n]["accuracy"] for n in names], wid,
               label="accuracy", color="#0173B2")
    axs[0].bar(xs + wid / 2, [results[n]["balanced_accuracy"] for n in names], wid,
               label="balanced accuracy", color="#DE8F05")
    axs[0].axhline(chance, ls="--", c="#D55E00", lw=1.5, label=f"majority chance {chance:.3f}")
    axs[0].axhline(1 / K, ls=":", c="#888888", lw=1.5, label=f"uniform chance {1/K:.3f}")
    axs[0].set_xticks(xs); axs[0].set_xticklabels(pretty, fontsize=8)
    axs[0].set_ylabel("leave-one-dyad-out score")
    axs[0].set_title("Six-class |ΔAQ| decoding: SVM vs 5-NN", fontsize=10)
    axs[0].legend(fontsize=7)
    labs = [str(inv[i]) for i in range(K)]
    M = np.array([[r["svm_rbf_lodo_acc"] if r else np.nan for r in
                   [next((x for x in pair_rows if {x["class_a"], x["class_b"]} == {i, j}), None)]][0]
                  if i != j else np.nan for i in range(K) for j in range(K)]).reshape(K, K)
    im = axs[1].imshow(M, cmap="viridis", vmin=0.5, vmax=1.0)
    axs[1].set_xticks(range(K)); axs[1].set_xticklabels(labs)
    axs[1].set_yticks(range(K)); axs[1].set_yticklabels(labs)
    axs[1].set_xlabel("|ΔAQ|"); axs[1].set_ylabel("|ΔAQ|")
    axs[1].set_title("Pairwise RBF-SVM accuracy\n(leave-one-dyad-out)", fontsize=10)
    for i in range(K):
        for j in range(K):
            if np.isfinite(M[i, j]):
                axs[1].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] < 0.8 else "black")
    fig.colorbar(im, ax=axs[1], label="accuracy")
    fig.tight_layout()
    fig.savefig(out / "aq_delta6_svm_vs_knn.png", dpi=300, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)

    # SVM confusion matrix
    cm = np.array(results["svm_ovo_rbf"]["confusion_matrix"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(labs)
    ax.set_yticks(range(K)); ax.set_yticklabels(labs)
    ax.set_xlabel("predicted |ΔAQ|"); ax.set_ylabel("true |ΔAQ|")
    ax.set_title(f"Pairwise (one-vs-one) RBF-SVM, leave-one-dyad-out\n"
                 f"acc={results['svm_ovo_rbf']['accuracy']:.3f}  "
                 f"bal={results['svm_ovo_rbf']['balanced_accuracy']:.3f}  "
                 f"chance={chance:.3f}", fontsize=9)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="row-normalised")
    fig.tight_layout()
    fig.savefig(out / "aq_delta6_svm_confusion_matrix.png", dpi=300, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)

    # ---------------- item 2: six separate lon/lat panels ----------------
    lonlat = np.load(run / "lonlat_radius.npy")   # rows follow eval_indices of the run
    ev = np.load(run / "eval_indices.npy")
    start2cls = {(int(m["start"]), int(m["end"])): int(m["cls"]) for m in meta}
    cls_of = np.empty(len(ev), np.int64)
    for i, s in enumerate(ev):
        for (a, b), c in start2cls.items():
            if a <= s < b:
                cls_of[i] = c
                break
    lon_u, lat = lonlat[:, 0], lonlat[:, 1]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for i in range(K):
        ax = axs[i // 3, i % 3]
        ax.scatter(lon_u, lat, s=2, alpha=0.06, color="#BBBBBB", linewidths=0, rasterized=True)
        m = cls_of == i
        ax.scatter(lon_u[m], lat[m], s=3, alpha=0.45, color=CLASS_COLORS[i],
                   linewidths=0, rasterized=True)
        if m.sum():
            ax.plot(np.median(lon_u[m]), np.median(lat[m]), marker="X", ms=13,
                    mfc=CLASS_COLORS[i], mec="black", mew=1.4, ls="none")
        ax.set_title(f"|ΔAQ| = {inv[i]}   (n={int(m.sum()):,}, "
                     f"{100*m.mean():.1f}% of samples)", fontsize=10)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.grid(alpha=0.15)
        if i // 3 == 1:
            ax.set_xlabel("Longitude (unwrapped)")
        if i % 3 == 0:
            ax.set_ylabel("Latitude")
    fig.suptitle("Six-class |ΔAQ|: longitude/latitude by class, plotted separately "
                 "(grey = all samples; X = class median)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "aq_delta6_lonlat_panels.png", dpi=300, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)

    # per-class summary of where each class sits
    srows = []
    for i in range(K):
        m = cls_of == i
        if not m.sum():
            continue
        srows.append({"class": i, "abs_daq": inv[i], "n": int(m.sum()),
                      "lon_median": round(float(np.median(lon_u[m])), 2),
                      "lon_iqr": round(float(np.subtract(*np.percentile(lon_u[m], [75, 25]))), 2),
                      "lat_median": round(float(np.median(lat[m])), 2),
                      "lat_iqr": round(float(np.subtract(*np.percentile(lat[m], [75, 25]))), 2)})
    with open(out / "aq_delta6_perclass_lonlat.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
        w.writeheader(); w.writerows(srows)

    print("\nper-class lon/lat medians:")
    for r in srows:
        print(f"  |dAQ|={r['abs_daq']}  n={r['n']:6d}  lon {r['lon_median']:+8.2f} "
              f"(IQR {r['lon_iqr']:6.2f})   lat {r['lat_median']:+7.2f} (IQR {r['lat_iqr']:6.2f})")
    print("SVM_AND_PANELS_DONE")


if __name__ == "__main__":
    main()
