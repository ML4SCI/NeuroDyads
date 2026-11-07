#!/usr/bin/env python
"""
scripts/run_band_cebra_analysis.py
==================================
Steps 2-4 for one filtered EEG band (speaker-first, AQ-magnitude label):
  2) train CEBRA (normalized, supervised by AQ magnitude Low/High) + metric table
  3) re-map the 3D embedding to latitude/longitude, KS on AQ magnitude
  4) fit an UNSUPERVISED (label-agnostic) GMM on the lat/long map (Dr. Ames'
     suggestion) -- how well does the band separate on its own? -- then compare
     the GMM clusters to the AQ-magnitude labels (ARI/purity) and report the
     fitted Gaussian parameters.

Inputs come from batch_speaker_first_stack.py (a band's .npy + <band>_manifest.csv).
Normalization is ON (raw stacked EEG will not train). Flagged dyads excluded by
default. Deterministic seed.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np

try:
    import torch, cebra
    from cebra import CEBRA
    from cebra.integrations.sklearn import metrics as cmetrics
except ImportError:
    import sys; sys.exit("ERROR: need torch + cebra")
from scipy.stats import ks_2samp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EPS = 1e-12
C0, C1 = "#0173B2", "#DE8F05"


def load_manifest(mpath, data_dir, exclude_flagged):
    rows = []
    for r in csv.DictReader(open(mpath)):
        if exclude_flagged and str(r.get("flagged", "")).lower() == "true":
            continue
        p = data_dir / r["output_npy"]
        if p.exists():
            rows.append({"npy": r["output_npy"], "path": p, "dyad": int(r["dyad_id"]),
                         "mag": int(r["aq_magnitude"])})
    rows.sort(key=lambda e: (e["dyad"], e["npy"]))
    return rows


def to_tc(x):
    return x.T if x.shape[0] < x.shape[1] else x


def wrap180(x):
    return (x + 180) % 360 - 180


def grouped_knn(X, y, g, seed, k=5):
    gs = sorted(set(g.tolist())); te = set(gs[::3])
    tr = np.where(~np.isin(g, list(te)))[0]; te_i = np.where(np.isin(g, list(te)))[0]
    if not len(tr) or not len(te_i):
        return None
    sc = StandardScaler().fit(X[tr])
    return float(KNeighborsClassifier(k).fit(sc.transform(X[tr]), y[tr]).score(sc.transform(X[te_i]), y[te_i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--band", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples-per-file", type=int, default=None)
    ap.add_argument("--exclude-flagged", action="store_true", default=True)
    ap.add_argument("--gmm-kmax", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    rows = load_manifest(args.manifest, data_dir, args.exclude_flagged)
    if not rows:
        import sys; sys.exit("no files after filtering")

    # ---- build concatenated dataset (time, channels) ----
    blocks, ys, gs, meta = [], [], [], []
    cur = 0
    for e in rows:
        X = to_tc(np.load(e["path"]).astype(np.float32))
        if args.max_samples_per_file and X.shape[0] > args.max_samples_per_file:
            X = X[:args.max_samples_per_file]
        n = X.shape[0]
        blocks.append(X); ys.append(np.full(n, e["mag"], np.int64)); gs.append(np.full(n, e["dyad"], np.int64))
        meta.append({"npy": e["npy"], "dyad": e["dyad"], "mag": e["mag"], "start": cur, "end": cur + n})
        cur += n
    X = np.concatenate(blocks); y = np.concatenate(ys); g = np.concatenate(gs)
    del blocks
    T, Cn = X.shape
    dyads = sorted(set(g.tolist()))
    # normalize (required)
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1
    X = ((X - mu) / sd).astype(np.float32)
    print(f"[{args.band}] X={X.shape} files={len(rows)} dyads={len(dyads)} "
          f"classes={np.bincount(y).tolist()} (normalized)")

    # ---- Step 2: CEBRA ----
    model = CEBRA(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                  temperature=1.12, max_iterations=args.max_iterations, conditional="time_delta",
                  output_dimension=3, distance="cosine", device="cuda_if_available",
                  verbose=True, time_offsets=10)
    model.fit(X, y)
    model.save(str(out / "model.pt"))
    emb = model.transform(X).astype(np.float32)
    np.save(out / "embedding.npy", emb)
    with open(out / "sample_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["npy", "dyad", "mag", "start", "end"]); w.writeheader(); w.writerows(meta)

    def final_loss(m):
        l = list(getattr(m, "state_dict_", {}).get("loss", [])); return float(l[-1]) if l else None
    gof = None
    try:
        gof = float(cmetrics.goodness_of_fit_score(model, X, y))
    except Exception as ex:
        print("gof fail", ex)

    # subsample for decoding/plots/GMM
    rng = np.random.RandomState(args.seed)
    sub = rng.choice(T, min(60000, T), replace=False)
    E, Y, G = emb[sub], y[sub], g[sub]
    knn = grouped_knn(E, Y, G, args.seed)
    try:
        s2 = rng.choice(len(sub), min(10000, len(sub)), replace=False)
        sil = float(silhouette_score(E[s2], Y[s2])) if len(set(Y[s2])) > 1 else None
    except Exception:
        sil = None

    # loss plot
    try:
        ax = cebra.plot_loss(model); ax.get_figure().savefig(out / "loss.png", dpi=140, bbox_inches="tight"); plt.close("all")
    except Exception:
        pass

    # ---- Step 3: latitude/longitude ----
    Ec = E - np.median(E, axis=0)
    x_, y_, z_ = Ec[:, 0], Ec[:, 1], Ec[:, 2]
    r = np.sqrt(x_**2 + y_**2 + z_**2)
    lon = np.degrees(np.arctan2(y_, x_)); lat = np.degrees(np.arctan2(z_, np.sqrt(x_**2 + y_**2)))
    cm = np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(), np.cos(np.radians(lon)).mean()))
    lon_r = wrap180(lon - cm)
    np.save(out / "lonlat_radius.npy", np.c_[lon_r, lat, r])
    # KS on AQ magnitude: compare lat and radius distributions Low vs High
    ks_lat = float(ks_2samp(lat[Y == 0], lat[Y == 1]).statistic) if len(set(Y)) > 1 else None
    ks_lon = float(ks_2samp(lon_r[Y == 0], lon_r[Y == 1]).statistic) if len(set(Y)) > 1 else None
    ks_rad = float(ks_2samp(r[Y == 0], r[Y == 1]).statistic) if len(set(Y)) > 1 else None

    # lat/long scatter colored by magnitude
    fig, axp = plt.subplots(figsize=(8, 5.5))
    for c, col, lb in [(0, C0, "Low |dAQ|"), (1, C1, "High |dAQ|")]:
        m = Y == c
        axp.scatter(lon_r[m], lat[m], s=5, alpha=0.35, color=col, label=lb, linewidths=0)
    axp.set_xlim(-180, 180); axp.set_ylim(-90, 90)
    axp.set_xlabel(f"Longitude, rot {cm:.0f}deg"); axp.set_ylabel("Latitude")
    axp.set_title(f"{args.band}: lat/long by AQ magnitude (KS_lat={ks_lat:.3f}, KS_rad={ks_rad:.3f})")
    axp.legend(markerscale=3); fig.savefig(out / "lonlat_by_magnitude.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- Step 4: UNSUPERVISED (label-agnostic) GMM on lat/long ----
    feat = np.c_[lon_r, lat]
    bics, models = [], {}
    for k in range(1, args.gmm_kmax + 1):
        gm = GaussianMixture(k, covariance_type="full", random_state=args.seed, n_init=2, max_iter=300).fit(feat)
        bics.append(float(gm.bic(feat))); models[k] = gm
    best_k = int(np.argmin(bics) + 1)
    gm2 = models[2]; cl2 = gm2.predict(feat)
    ari = float(adjusted_rand_score(Y, cl2));
    pur = float(sum(np.bincount(Y[cl2 == c]).max() for c in np.unique(cl2)) / len(Y))
    gmm_params = {"best_k_by_bic": best_k, "bic_by_k": {k: round(b, 1) for k, b in zip(range(1, args.gmm_kmax + 1), bics)},
                  "K2_ARI_vs_magnitude": round(ari, 4), "K2_purity": round(pur, 4),
                  "K2_weights": gm2.weights_.tolist(),
                  "K2_means_lonlat": gm2.means_.tolist(),
                  "K2_covariances_lonlat": [c.tolist() for c in gm2.covariances_]}
    json.dump(gmm_params, open(out / "gmm_unsupervised.json", "w"), indent=2)

    # GMM plot
    from matplotlib.patches import Ellipse
    fig, axg = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10", 2)
    for c in range(2):
        m = cl2 == c
        axg.scatter(lon_r[m], lat[m], s=5, alpha=0.3, color=cmap(c), label=f"GMM comp {c}", linewidths=0)
    for c in range(2):
        mean, cov = gm2.means_[c], gm2.covariances_[c]
        vals, vecs = np.linalg.eigh(cov); order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        for ns in (1, 2):
            axg.add_patch(Ellipse(mean, 2*ns*np.sqrt(vals[0]), 2*ns*np.sqrt(vals[1]), angle=ang, fc="none", ec="k", lw=1.4))
    axg.set_xlim(-180, 180); axg.set_ylim(-90, 90); axg.set_xlabel("Longitude(rot)"); axg.set_ylabel("Latitude")
    axg.set_title(f"{args.band}: UNSUPERVISED GMM (K=2), ARI vs magnitude={ari:.3f}, bestK(BIC)={best_k}")
    axg.legend(markerscale=3); fig.savefig(out / "gmm_unsupervised.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    metrics = {"band": args.band, "n_files": len(rows), "n_dyads": len(dyads), "n_samples": int(T),
               "n_channels": int(Cn), "class_counts": np.bincount(y).tolist(), "chance": float(np.bincount(y).max()/T),
               "max_iterations": args.max_iterations, "seed": args.seed, "normalized": True,
               "final_loss": final_loss(model), "goodness_of_fit_bits": gof,
               "knn5_grouped_magnitude": knn, "silhouette_magnitude": sil,
               "ks_latitude": ks_lat, "ks_longitude": ks_lon, "ks_radius": ks_rad,
               "gmm_unsup_bestK": best_k, "gmm_unsup_ARI_vs_magnitude": round(ari, 4), "gmm_unsup_purity": round(pur, 4),
               "device": str(model.device_), "cebra_version": cebra.__version__}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(f"[{args.band}] loss={metrics['final_loss']:.4f} GoF={gof} 5NN_mag={knn} sil={sil}")
    print(f"[{args.band}] KS lat={ks_lat} lon={ks_lon} rad={ks_rad}")
    print(f"[{args.band}] unsup GMM: bestK={best_k} ARI={ari:.3f} purity={pur:.3f}")


if __name__ == "__main__":
    main()
