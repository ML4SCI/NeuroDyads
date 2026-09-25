#!/usr/bin/env python
"""
scripts/run_aq_delta6_pipeline.py
=================================
TASK 5 -- six-class absolute AQ-difference pipeline (|dAQ| = 0,1,2,3,4,5).

The dyad whose |dAQ| = 7 is dropped (mentor instruction), but only after the
actual distribution is inspected and reported -- we never assume there is
exactly one such dyad.

Same CEBRA configuration as the full-band speaker-first AQ-magnitude baseline so
the two are directly comparable. Adds: balanced accuracy, macro F1, per-class
recall, a six-class confusion matrix, pairwise + omnibus coordinate tests with
Holm correction, and a label-agnostic GMM K-sweep scored post hoc against
|dAQ|, AQ magnitude and dyad identity.
"""
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path

import numpy as np

try:
    import torch, cebra
    from cebra import CEBRA
    from cebra.integrations.sklearn import metrics as cmetrics
except ImportError:
    import sys; sys.exit("ERROR: need torch + cebra")
from scipy.stats import ks_2samp, kruskal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, adjusted_rand_score, confusion_matrix,
                             balanced_accuracy_score, f1_score, recall_score,
                             normalized_mutual_info_score)
from sklearn.mixture import GaussianMixture
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

CEBRA_KW = dict(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                temperature=1.12, conditional="time_delta", output_dimension=3,
                distance="cosine", device="cuda_if_available", verbose=True, time_offsets=10)
SUB = 60000
CLASS_COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#56B4E9"]


def to_tc(x):
    return x.T if x.shape[0] < x.shape[1] else x


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


def holm(pvals):
    idx = np.argsort(pvals); n = len(pvals); adj = np.empty(n); run = 0.0
    for r, i in enumerate(idx):
        v = (n - r) * pvals[i]
        run = max(run, v)
        adj[i] = min(1.0, run)
    return adj


def leave_one_dyad_out(X, y, g, k=5):
    dy = sorted(set(g.tolist())); yt, yp = [], []
    for d in dy:
        te = g == d; tr = ~te
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = KNeighborsClassifier(k).fit(sc.transform(X[tr]), y[tr])
        yp.append(m.predict(sc.transform(X[te]))); yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-daq", type=int, nargs="*", default=[7])
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--label", default="fullband_speakerfirst")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dd = Path(args.data_dir)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # ---------- inspect the real |dAQ| distribution FIRST ----------
    allrows = []
    for r in csv.DictReader(open(args.manifest)):
        if str(r.get("flagged", "")).lower() == "true":
            continue
        p = dd / r["output_npy"]
        if p.exists():
            allrows.append({"npy": r["output_npy"], "path": p, "dyad": int(r["dyad_id"]),
                            "mag": int(r["aq_magnitude"]), "daq": int(float(r["abs_daq"]))})
    allrows.sort(key=lambda e: (e["dyad"], e["npy"]))

    import collections
    dyad_daq = {e["dyad"]: e["daq"] for e in allrows}
    dist = collections.Counter(dyad_daq.values())
    file_dist = collections.Counter(e["daq"] for e in allrows)
    print("=== |dAQ| distribution BEFORE dropping ===")
    for v in sorted(dist):
        print(f"  |dAQ|={v}: {dist[v]} dyads, {file_dist[v]} files")

    rows = [e for e in allrows if e["daq"] not in set(args.drop_daq)]
    dropped = [e for e in allrows if e["daq"] in set(args.drop_daq)]
    kept_vals = sorted(set(e["daq"] for e in rows))
    print(f"dropping |dAQ| in {args.drop_daq}: {len(dropped)} files "
          f"({len(set(e['dyad'] for e in dropped))} dyads)")
    print(f"kept |dAQ| values: {kept_vals}")
    if kept_vals != list(range(6)):
        print(f"!! WARNING: kept values {kept_vals} are not exactly 0..5")

    label_map = {v: i for i, v in enumerate(kept_vals)}   # |dAQ| -> class index
    inv_map = {i: v for v, i in label_map.items()}

    # ---------- build dataset ----------
    blocks, ys, gs, mags, meta = [], [], [], [], []
    cur = 0
    for e in rows:
        Xf = to_tc(np.load(e["path"]).astype(np.float32)); n = Xf.shape[0]
        blocks.append(Xf)
        ys.append(np.full(n, label_map[e["daq"]], np.int64))
        gs.append(np.full(n, e["dyad"], np.int64))
        mags.append(np.full(n, e["mag"], np.int64))
        meta.append({"npy": e["npy"], "dyad": e["dyad"], "daq": e["daq"],
                     "cls": label_map[e["daq"]], "mag": e["mag"], "start": cur, "end": cur + n})
        cur += n
    X = np.concatenate(blocks); y = np.concatenate(ys); g = np.concatenate(gs)
    y_mag = np.concatenate(mags); del blocks
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1
    X -= mu; X /= sd; X = np.ascontiguousarray(X, np.float32)
    T = X.shape[0]
    counts = np.bincount(y, minlength=len(kept_vals))
    chance = float(counts.max() / T)
    print(f"[delta6] X={X.shape} files={len(rows)} dyads={len(set(g.tolist()))} "
          f"class_counts={counts.tolist()} chance={chance:.4f}", flush=True)

    counts_tbl = []
    for i, v in enumerate(kept_vals):
        dl = [d for d, dq in dyad_daq.items() if dq == v]
        counts_tbl.append({"class": i, "abs_daq": v, "n_dyads": len(dl),
                           "n_files": file_dist[v], "n_samples": int(counts[i]),
                           "pct_samples": round(100 * counts[i] / T, 2)})
    with open(out / "class_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(counts_tbl[0].keys())); w.writeheader(); w.writerows(counts_tbl)
    json.dump({"distribution_before_drop_dyads": {str(k): v for k, v in sorted(dist.items())},
               "distribution_before_drop_files": {str(k): v for k, v in sorted(file_dist.items())},
               "dropped_daq_values": args.drop_daq,
               "dropped_files": [e["npy"] for e in dropped],
               "dropped_dyads": sorted(set(e["dyad"] for e in dropped)),
               "kept_daq_values": kept_vals, "label_map_daq_to_class": {str(k): v for k, v in label_map.items()},
               "n_files": len(rows), "n_dyads": len(set(g.tolist())), "n_samples": int(T),
               "class_counts": counts.tolist(), "majority_chance": chance,
               "cebra_config": {**CEBRA_KW, "max_iterations": args.max_iterations, "seed": args.seed}},
              open(out / "dataset_provenance.json", "w"), indent=2)

    # ---------- train ----------
    t0 = time.time()
    model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
    model.fit(X, y)
    model.save(str(out / "model.pt"))
    emb = model.transform(X).astype(np.float32)
    np.save(out / "embedding.npy", emb)
    train_s = time.time() - t0
    with open(out / "sample_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["npy", "dyad", "daq", "cls", "mag", "start", "end"])
        w.writeheader(); w.writerows(meta)
    loss_hist = list(getattr(model, "state_dict_", {}).get("loss", []))
    final_loss = float(loss_hist[-1]) if loss_hist else None
    try:
        gof = float(cmetrics.goodness_of_fit_score(model, X, y))
    except Exception as ex:
        print("gof fail", ex); gof = None
    try:
        ax = cebra.plot_loss(model); ax.get_figure().savefig(out / "loss.png", dpi=300,
            facecolor="white", bbox_inches="tight"); plt.close("all")
    except Exception:
        pass

    # ---------- evaluate ----------
    rng = np.random.RandomState(args.seed)
    sub = rng.choice(T, min(SUB, T), replace=False); sub.sort()
    np.save(out / "eval_indices.npy", sub)
    E, Y, G, YM = emb[sub], y[sub], g[sub], y_mag[sub]

    yt, yp = leave_one_dyad_out(E, Y, G)
    acc = float((yt == yp).mean())
    bal = float(balanced_accuracy_score(yt, yp))
    mf1 = float(f1_score(yt, yp, average="macro"))
    rec = recall_score(yt, yp, average=None, labels=list(range(len(kept_vals))), zero_division=0)
    cm = confusion_matrix(yt, yp, labels=list(range(len(kept_vals))))
    idx = rng.permutation(len(E)); cut = int(0.7 * len(E))
    sc = StandardScaler().fit(E[idx[:cut]])
    rnd = float(KNeighborsClassifier(5).fit(sc.transform(E[idx[:cut]]), Y[idx[:cut]])
                .score(sc.transform(E[idx[cut:]]), Y[idx[cut:]]))
    try:
        s2 = rng.choice(len(E), min(10000, len(E)), replace=False)
        sil = float(silhouette_score(E[s2], Y[s2]))
    except Exception:
        sil = None

    # ---------- spherical coords ----------
    Ec = E - np.median(E, axis=0)
    x_, y2_, z_ = Ec[:, 0], Ec[:, 1], Ec[:, 2]
    r = np.sqrt(x_**2 + y2_**2 + z_**2)
    lon = np.degrees(np.arctan2(y2_, x_)); lat = np.degrees(np.arctan2(z_, np.sqrt(x_**2 + y2_**2)))
    cut_deg = largest_gap_cut(lon); lon_u = wrap180(lon - cut_deg)
    np.save(out / "lonlat_radius.npy", np.c_[lon_u, lat, r])

    coord = {"longitude": lon_u, "latitude": lat, "radius": r}
    ks_rows, omni = [], {}
    for cname, cv in coord.items():
        groups = [cv[Y == i] for i in range(len(kept_vals)) if (Y == i).sum() > 1]
        try:
            H, pk = kruskal(*groups)
            omni[cname] = {"kruskal_H": float(H), "p": float(pk)}
        except Exception as ex:
            omni[cname] = {"error": str(ex)}
        raw = []
        for a in range(len(kept_vals)):
            for b in range(a + 1, len(kept_vals)):
                ma, mb = Y == a, Y == b
                if ma.sum() < 2 or mb.sum() < 2:
                    continue
                st = ks_2samp(cv[ma], cv[mb])
                raw.append({"coordinate": cname, "class_a": a, "class_b": b,
                            "daq_a": inv_map[a], "daq_b": inv_map[b],
                            "ks_stat": float(st.statistic), "p_raw": float(st.pvalue)})
        if raw:
            adj = holm(np.array([x["p_raw"] for x in raw]))
            for x, pa in zip(raw, adj):
                x["p_holm"] = float(pa); x["sig_holm_0.05"] = bool(pa < 0.05)
            ks_rows += raw
    with open(out / "ks_pairwise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ks_rows[0].keys())); w.writeheader(); w.writerows(ks_rows)
    json.dump(omni, open(out / "omnibus_kruskal.json", "w"), indent=2)

    # ---------- label-agnostic GMM ----------
    feat = np.c_[lon_u, lat]
    gsweep = []
    saved = {}
    for k in range(1, args.kmax + 1):
        gm = GaussianMixture(k, covariance_type="full", random_state=args.seed,
                             n_init=10, max_iter=500, reg_covar=1e-6).fit(feat)
        cl = gm.predict(feat)
        gsweep.append({"K": k, "BIC": float(gm.bic(feat)), "AIC": float(gm.aic(feat)),
                       "converged": bool(gm.converged_),
                       "ARI_vs_daq6": float(adjusted_rand_score(Y, cl)),
                       "NMI_vs_daq6": float(normalized_mutual_info_score(Y, cl)),
                       "ARI_vs_magnitude": float(adjusted_rand_score(YM, cl)),
                       "ARI_vs_dyad": float(adjusted_rand_score(G, cl)),
                       "purity_vs_daq6": float(sum(np.bincount(Y[cl == c]).max()
                                                   for c in np.unique(cl)) / len(Y)),
                       "purity_vs_dyad": float(sum(np.bincount(G[cl == c]).max()
                                                   for c in np.unique(cl)) / len(G))})
        if k in (2, 6):
            saved[k] = cl
    bestK = min(gsweep, key=lambda x: x["BIC"])["K"]
    bestK_aic = min(gsweep, key=lambda x: x["AIC"])["K"]
    with open(out / "gmm_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gsweep[0].keys())); w.writeheader(); w.writerows(gsweep)
    g6 = next(x for x in gsweep if x["K"] == 6)

    metrics = {"analysis": "aq_delta6", "condition": args.label, "label_scheme": "abs_AQ_delta_6class",
               "seed": args.seed, "n_files": len(rows), "n_dyads": len(set(g.tolist())),
               "n_samples": int(T), "class_counts": counts.tolist(),
               "kept_daq_values": kept_vals, "dropped_daq": args.drop_daq,
               "final_loss": final_loss, "goodness_of_fit_bits": gof,
               "knn5_leave_one_dyad_out": acc, "majority_chance": chance,
               "balanced_accuracy": bal, "macro_f1": mf1,
               "per_class_recall": {str(inv_map[i]): float(v) for i, v in enumerate(rec)},
               "knn5_random_split": rnd, "silhouette": sil,
               "confusion_matrix": cm.tolist(), "lon_cut_deg": cut_deg,
               "omnibus_kruskal": omni,
               "gmm_bestK_by_bic": bestK, "gmm_bestK_by_aic": bestK_aic,
               "gmm_K6_ARI_vs_daq6": g6["ARI_vs_daq6"], "gmm_K6_purity_vs_daq6": g6["purity_vs_daq6"],
               "gmm_K6_ARI_vs_dyad": g6["ARI_vs_dyad"],
               "train_seconds": round(train_s, 1), "cebra_version": cebra.__version__}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)

    # ---------------- figures ----------------
    def leg(i):
        return f"|dAQ|={inv_map[i]}"

    fig = plt.figure(figsize=(7.2, 6))
    ax = fig.add_subplot(111, projection="3d")
    pl = rng.choice(len(E), min(20000, len(E)), replace=False)
    for i in range(len(kept_vals)):
        m = Y[pl] == i
        ax.scatter(E[pl][m, 0], E[pl][m, 1], E[pl][m, 2], s=2, alpha=0.35,
                   color=CLASS_COLORS[i % 6], label=leg(i), linewidths=0)
    ax.set_title(f"CEBRA 3D embedding by |dAQ| ({args.label})")
    ax.legend(markerscale=6, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "aq_delta6_embedding.png", dpi=300,
                                    facecolor="white", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i in range(len(kept_vals)):
        m = Y == i
        ax.scatter(lon_u[m], lat[m], s=4, alpha=0.32, color=CLASS_COLORS[i % 6],
                   label=leg(i), linewidths=0)
    ax.set_xlabel(f"Longitude (unwrapped at {cut_deg:.0f}°)"); ax.set_ylabel("Latitude")
    ax.set_title(f"Longitude/latitude by |dAQ| ({args.label})")
    ax.legend(markerscale=4, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out / "aq_delta6_lonlat.png", dpi=300,
                                    facecolor="white", bbox_inches="tight"); plt.close(fig)

    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    labs = [str(inv_map[i]) for i in range(len(kept_vals))]
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    ax.set_xlabel("predicted |dAQ|"); ax.set_ylabel("true |dAQ|")
    ax.set_title(f"Leave-one-dyad-out 5-NN confusion\nacc={acc:.3f} bal={bal:.3f} "
                 f"chance={chance:.3f}", fontsize=10)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="row-normalised")
    fig.tight_layout(); fig.savefig(out / "aq_delta6_confusion_matrix.png", dpi=300,
                                    facecolor="white", bbox_inches="tight"); plt.close(fig)

    ks_arr = np.full((3, len(kept_vals), len(kept_vals)), np.nan)
    cnames = ["longitude", "latitude", "radius"]
    for row in ks_rows:
        ci = cnames.index(row["coordinate"])
        ks_arr[ci, row["class_a"], row["class_b"]] = row["ks_stat"]
        ks_arr[ci, row["class_b"], row["class_a"]] = row["ks_stat"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ci, cn in enumerate(cnames):
        im = axs[ci].imshow(ks_arr[ci], cmap="magma", vmin=0, vmax=np.nanmax(ks_arr))
        axs[ci].set_xticks(range(len(labs))); axs[ci].set_xticklabels(labs)
        axs[ci].set_yticks(range(len(labs))); axs[ci].set_yticklabels(labs)
        axs[ci].set_title(f"{cn}\nKruskal p={omni[cn].get('p', float('nan')):.2e}", fontsize=9)
        axs[ci].set_xlabel("|dAQ|")
        fig.colorbar(im, ax=axs[ci], label="KS statistic")
    axs[0].set_ylabel("|dAQ|")
    fig.suptitle(f"Pairwise KS by coordinate ({args.label})", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "aq_delta6_ks_summary.png", dpi=300,
                                    facecolor="white", bbox_inches="tight"); plt.close(fig)

    ks_ = [x["K"] for x in gsweep]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks_, [x["BIC"] for x in gsweep], "o-", color="#0173B2", label="BIC")
    ax.plot(ks_, [x["AIC"] for x in gsweep], "s--", color="#DE8F05", label="AIC")
    ax.axvline(bestK, ls=":", c="k", lw=1)
    ax.set_xlabel("K"); ax.set_ylabel("criterion"); ax.set_xticks(ks_[::2])
    ax.set_title(f"Label-agnostic GMM model selection, best K={bestK} ({args.label})")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "aq_delta6_gmm_bic.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    cl6 = saved.get(6)
    if cl6 is not None:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        cmap = plt.get_cmap("tab10", 6)
        for c in range(6):
            m = cl6 == c
            ax.scatter(lon_u[m], lat[m], s=4, alpha=0.32, color=cmap(c),
                       label=f"comp {c}", linewidths=0)
        ax.set_xlabel(f"Longitude (unwrapped at {cut_deg:.0f}°)"); ax.set_ylabel("Latitude")
        ax.set_title(f"Label-agnostic GMM (K=6) components ({args.label})")
        ax.legend(markerscale=4, fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(out / "aq_delta6_gmm_components.png", dpi=300,
                                        facecolor="white", bbox_inches="tight"); plt.close(fig)
        for nm, yv, fn, xl in [("|dAQ|", Y, "aq_delta6_gmm_vs_aq.png", "|dAQ| class"),
                               ("dyad", G, "aq_delta6_gmm_vs_dyad.png", "dyad id")]:
            cats = np.array(sorted(set(yv.tolist())))
            M = np.zeros((6, len(cats)))
            for i in range(6):
                for j, c in enumerate(cats):
                    M[i, j] = np.sum((cl6 == i) & (yv == c))
            Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
            fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(cats) + 3), 3.6))
            im = ax.imshow(Mn, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels([str(inv_map[c]) if nm == "|dAQ|" else str(c) for c in cats],
                               fontsize=7, rotation=90 if nm == "dyad" else 0)
            ax.set_yticks(range(6)); ax.set_yticklabels([f"comp {i}" for i in range(6)])
            ax.set_xlabel(xl); ax.set_title(f"K=6 GMM component composition by {nm}")
            fig.colorbar(im, ax=ax, label="row-normalised share")
            fig.tight_layout(); fig.savefig(out / fn, dpi=300, facecolor="white",
                                            bbox_inches="tight"); plt.close(fig)

    print(f"\n[delta6] loss={final_loss:.4f} GoF={gof} LODO acc={acc:.4f} "
          f"bal={bal:.4f} macroF1={mf1:.4f} chance={chance:.4f} sil={sil}")
    print(f"[delta6] GMM bestK={bestK}  K6 ARI vs |dAQ|={g6['ARI_vs_daq6']:.3f} "
          f"vs dyad={g6['ARI_vs_dyad']:.3f}")
    print("AQ_DELTA6_DONE")


if __name__ == "__main__":
    main()
