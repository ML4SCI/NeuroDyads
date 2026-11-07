#!/usr/bin/env python
"""
scripts/analyze_cebra_longitude_latitude.py
===========================================
Spherical-coordinate ("longitude/latitude") diagnostic for 3D CEBRA embeddings.

Treat each 3D point as spherical: after centering,
    r   = sqrt(x^2+y^2+z^2)
    lon = atan2(y, x)                     in [-180, 180] deg  (WRAPS)
    lat = atan2(z, sqrt(x^2+y^2))         in [-90, 90]  deg  (no wrap)

IMPORTANT: longitude wraps at +-180 and CEBRA orientation is arbitrary, so
longitude/latitude are used for PLOTTING and geometry diagnostics only. For any
decoding/distance we use wrap-safe features:
    unit vectors [x/r, y/r, z/r]  or  [sin(lon),cos(lon),sin(lat),cos(lat)].
Cross-run comparison requires Procrustes alignment first.

Single-run mode (Tasks 2-4): --embedding + --metadata + --out-dir
  -> spherical coords, plots, leave-dyads-out 5-NN on several feature sets.
Compare mode (Task 5): also pass --compare-embedding (+ --compare-metadata)
  -> Procrustes align, angular distance, circular/latitude correlations, KS,
     Kuiper, energy distance, cross-decoding, label-agreement.

Labels & groups come from the run's sample_metadata.csv (label_used, dyad_id).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import ks_2samp, spearmanr, pearsonr
    from scipy.linalg import orthogonal_procrustes
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("ERROR: needs scipy + scikit-learn.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

EPS = 1e-12
LON_LIM, LAT_LIM = (-180, 180), (-90, 90)


# ---------------- data ----------------
def load_embedding(path):
    Y = np.asarray(np.load(path), dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"{path}: expected 2D, got {Y.shape}")
    if Y.shape[1] != 3 and Y.shape[0] == 3:
        Y = Y.T
    if Y.shape[1] != 3:
        raise ValueError(f"{path}: expected (n,3), got {Y.shape}")
    return Y


def labels_from_metadata(meta_csv, n, group_column="dyad_id"):
    """Per-sample (label, group) from sample_metadata.csv row ranges."""
    y = np.full(n, -1, dtype=np.int64)
    g = np.full(n, -1, dtype=np.int64)
    for r in csv.DictReader(open(meta_csv, encoding="utf-8")):
        s, e = int(r["start_row"]), int(r["end_row"])
        y[s:e] = int(float(r["label_used"]))
        g[s:e] = int(r[group_column])
    return y, g


def clean(Y, *arrs):
    finite = np.isfinite(Y).all(axis=1)
    if not finite.all():
        print(f"  dropping {int((~finite).sum())} non-finite rows")
    return (Y[finite],) + tuple(a[finite] for a in arrs)


def center(Y, method="median"):
    c = np.median(Y, axis=0) if method == "median" else Y.mean(axis=0)
    return Y - c, c


def to_spherical(Y):
    x, y, z = Y[:, 0], Y[:, 1], Y[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return r, lon, lat


def unit_vectors(Y):
    r = np.linalg.norm(Y, axis=1, keepdims=True)
    return Y / (r + EPS)


# ---------------- features ----------------
def feat_sets(Y, lon_deg, lat_deg, r):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return {
        "unit_vectors": unit_vectors(Y),
        "longitude_only": np.c_[np.sin(lon), np.cos(lon)],
        "latitude_only": lat_deg.reshape(-1, 1),
        "lonlat_wrapsafe": np.c_[np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat)],
        "radius_only": r.reshape(-1, 1),
    }


# ---------------- decoding ----------------
def grouped_knn(X, y, g, seed, k=5):
    """Leave-dyads-out: hold out every 3rd group. Standardize on train."""
    groups = sorted(set(g.tolist()))
    test_g = set(groups[::3])
    tr = np.where(~np.isin(g, list(test_g)))[0]
    te = np.where(np.isin(g, list(test_g)))[0]
    if len(tr) == 0 or len(te) == 0:
        return None
    sc = StandardScaler().fit(X[tr])
    clf = KNeighborsClassifier(n_neighbors=k).fit(sc.transform(X[tr]), y[tr])
    return float(clf.score(sc.transform(X[te]), y[te]))


def random_knn(X, y, seed, k=5, frac=0.8):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X)); cut = int(frac * len(perm))
    tr, te = perm[:cut], perm[cut:]
    sc = StandardScaler().fit(X[tr])
    clf = KNeighborsClassifier(n_neighbors=k).fit(sc.transform(X[tr]), y[tr])
    return float(clf.score(sc.transform(X[te]), y[te]))


def chance(y):
    _, c = np.unique(y, return_counts=True)
    return float(c.max() / c.sum())


# ---------------- circular / distribution stats ----------------
def circ_corr(a_rad, b_rad):
    a = a_rad - np.arctan2(np.sin(a_rad).mean(), np.cos(a_rad).mean())
    b = b_rad - np.arctan2(np.sin(b_rad).mean(), np.cos(b_rad).mean())
    num = np.sum(np.sin(a) * np.sin(b))
    den = np.sqrt(np.sum(np.sin(a)**2) * np.sum(np.sin(b)**2))
    return float(num / (den + EPS))


def kuiper_2samp(a, b):
    """Two-sample Kuiper V = D+ + D- on the combined sorted support."""
    a = np.sort(a); b = np.sort(b)
    allv = np.concatenate([a, b])
    allv.sort()
    cdfa = np.searchsorted(a, allv, side="right") / len(a)
    cdfb = np.searchsorted(b, allv, side="right") / len(b)
    d = cdfa - cdfb
    return float(d.max() - d.min())


def energy_distance(A, B, seed=0, n=3000):
    rng = np.random.RandomState(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    def md(x, y):
        return np.mean(np.sqrt(((x[:, None, :] - y[None, :, :])**2).sum(-1)))
    return float(2 * md(A, B) - md(A, A) - md(B, B))


# ---------------- plots ----------------
def make_plots(out, name, lon, lat, r, y, seed, max_plot=20000):
    if not HAVE_PLT:
        return
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(lon), min(max_plot, len(lon)), replace=False)
    lo, la, rr, yy = lon[idx], lat[idx], r[idx], y[idx]
    classes = sorted(set(yy.tolist()))
    cmap = plt.get_cmap("tab10", max(len(classes), 2))

    # 1 lon/lat scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, c in enumerate(classes):
        m = yy == c
        ax.scatter(lo[m], la[m], s=3, alpha=0.4, color=cmap(i), label=f"label {c}")
    ax.set_xlim(LON_LIM); ax.set_ylim(LAT_LIM)
    ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)")
    ax.set_title(f"{name}: lon/lat colored by label"); ax.legend(markerscale=3)
    fig.savefig(out / "lonlat_scatter.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # 2-4 distributions
    for var, data, lim, fn in [("longitude", lo, LON_LIM, "hist_longitude.png"),
                               ("latitude", la, LAT_LIM, "hist_latitude.png"),
                               ("radius", rr, None, "hist_radius.png")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, c in enumerate(classes):
            ax.hist(data[yy == c], bins=60, density=True, alpha=0.5,
                    color=cmap(i), label=f"label {c}")
        if lim:
            ax.set_xlim(lim)
        ax.set_xlabel(f"{var} ({'deg' if var!='radius' else 'r'})")
        ax.set_ylabel("density"); ax.set_title(f"{name}: {var} by class"); ax.legend()
        fig.savefig(out / fn, dpi=130, bbox_inches="tight"); plt.close(fig)

    # 5 hexbin
    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(lo, la, gridsize=45, cmap="viridis", mincnt=1)
    ax.set_xlim(LON_LIM); ax.set_ylim(LAT_LIM)
    ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)")
    ax.set_title(f"{name}: lon/lat density"); fig.colorbar(hb, ax=ax, label="count")
    fig.savefig(out / "lonlat_hexbin.png", dpi=130, bbox_inches="tight"); plt.close(fig)


# ---------------- single run ----------------
def analyze_single(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    Y = load_embedding(args.embedding)
    n0 = Y.shape[0]
    y, g = labels_from_metadata(args.metadata, n0, args.group_column)
    Y, y, g = clean(Y, y, g)

    Yc, cvec = center(Y, args.center_method)
    r, lon, lat = to_spherical(Yc)

    # subsample for ML + plots + saved coords
    rng = np.random.RandomState(args.seed)
    n_use = min(args.max_samples, Yc.shape[0])
    idx = np.sort(rng.choice(Yc.shape[0], n_use, replace=False))
    np.save(out / "sampled_indices.npy", idx)
    Ys, ys, gs, rs, los, las = Yc[idx], y[idx], g[idx], r[idx], lon[idx], lat[idx]

    coords = np.c_[rs, los, las]
    np.save(out / "spherical_coordinates.npy", coords)
    with open(out / "spherical_coordinates.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["radius", "longitude_deg", "latitude_deg", "label", "dyad"])
        for i in range(len(idx)):
            w.writerow([rs[i], los[i], las[i], int(ys[i]), int(gs[i])])

    make_plots(out, args.name, los, las, rs, ys, args.seed)

    feats = feat_sets(Ys, los, las, rs)
    metrics = {"name": args.name, "n_used": int(n_use), "n_dyads": len(set(gs.tolist())),
               "center_method": args.center_method, "center_vector": cvec.tolist(),
               "seed": args.seed, "chance": chance(ys),
               "class_counts": {int(k): int(v) for k, v in zip(*np.unique(ys, return_counts=True))}}
    for nm, X in feats.items():
        metrics[f"grouped5nn_{nm}"] = grouped_knn(X, ys, gs, args.seed)
    metrics["random5nn_unit_vectors"] = random_knn(feats["unit_vectors"], ys, args.seed)
    try:
        s_idx = rng.choice(len(idx), min(10000, len(idx)), replace=False)
        if len(set(ys[s_idx].tolist())) > 1:
            metrics["silhouette_unit_vectors"] = float(
                silhouette_score(feats["unit_vectors"][s_idx], ys[s_idx]))
    except Exception as exc:
        print(f"  silhouette skipped: {exc}")

    json.dump({"embedding": str(args.embedding), "metadata": str(args.metadata),
               **vars(args)}, open(out / "analysis_config.json", "w"), indent=2, default=str)
    json.dump(metrics, open(out / "longlat_metrics.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k.startswith(("grouped5nn", "random5nn", "silhouette", "chance"))},
                     indent=2))
    return metrics, dict(idx=idx, Yc=Yc, y=y, g=g, lon=lon, lat=lat, r=r)


# ---------------- compare (Task 5) ----------------
def analyze_compare(args, base):
    """Compare the run's embedding (A) vs --compare-embedding (B), sample-aligned."""
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    YA = load_embedding(args.embedding)
    YB = load_embedding(args.compare_embedding)
    assert YA.shape == YB.shape, f"shape mismatch {YA.shape} vs {YB.shape}"
    nA = YA.shape[0]
    yA, gA = labels_from_metadata(args.metadata, nA, args.group_column)
    metaB = args.compare_metadata or args.metadata
    yB, gB = labels_from_metadata(metaB, nA, args.group_column)
    assert np.array_equal(gA, gB), "group arrays differ -> embeddings not sample-aligned"

    rng = np.random.RandomState(args.seed)
    n_use = min(args.max_samples, nA)
    idx = np.sort(rng.choice(nA, n_use, replace=False))
    A, B = YA[idx], YB[idx]
    yA_s, yB_s, g_s = yA[idx], yB[idx], gA[idx]

    Ac, _ = center(A, args.center_method)
    Bc, _ = center(B, args.center_method)

    # spherical of each (unaligned)
    rA, lonA, latA = to_spherical(Ac)
    rB, lonB, latB = to_spherical(Bc)

    # Procrustes: align B (aq_direction) onto A (direction)
    R, scale = orthogonal_procrustes(Bc, Ac)
    B_aligned = Bc @ R
    ss_tot = (Ac**2).sum()
    residual = float(((Ac - B_aligned)**2).sum() / (ss_tot + EPS))  # normalized

    uA = unit_vectors(Ac)
    uB_al = unit_vectors(B_aligned)
    dots = np.clip((uA * uB_al).sum(1), -1, 1)
    ang = np.degrees(np.arccos(dots))  # per-sample angular distance (deg)

    rA2, lonA2, latA2 = to_spherical(Ac)
    rBa, lonBa, latBa = to_spherical(B_aligned)

    comp = {
        "name_a": args.name, "name_b": args.compare_name or "B",
        "embedding_a": str(args.embedding), "embedding_b": str(args.compare_embedding),
        "n_used": int(n_use), "seed": args.seed, "center_method": args.center_method,
        "procrustes_residual_normalized": residual,
        "angular_distance_deg": {
            "mean": float(ang.mean()), "median": float(np.median(ang)),
            "std": float(ang.std()),
            "q25": float(np.percentile(ang, 25)), "q75": float(np.percentile(ang, 75)),
            "q95": float(np.percentile(ang, 95))},
        # unaligned
        "unaligned_longitude_circular_corr": circ_corr(np.radians(lonA), np.radians(lonB)),
        "unaligned_latitude_pearson": float(pearsonr(latA, latB)[0]),
        # aligned
        "aligned_longitude_circular_corr": circ_corr(np.radians(lonA2), np.radians(lonBa)),
        "aligned_latitude_pearson": float(pearsonr(latA2, latBa)[0]),
        "aligned_latitude_spearman": float(spearmanr(latA2, latBa)[0]),
        "aligned_latitude_ks": float(ks_2samp(latA2, latBa).statistic),
        "aligned_longitude_kuiper": kuiper_2samp(lonA2, lonBa),
        "unitvec_energy_distance_aligned": energy_distance(uA, uB_al, seed=args.seed),
    }

    # label agreement (direction=yA, aq_direction=yB), per-sample
    agree = int((yA_s == yB_s).sum()); disagree = int((yA_s != yB_s).sum())
    comp["label_agreement"] = {
        "agree": agree, "disagree": disagree, "n": int(n_use),
        "pct_agree": round(100 * agree / n_use, 2),
        "confusion": _confusion(yA_s, yB_s)}

    # cross-decoding (unit vectors, grouped)
    comp["cross_decoding"] = {
        "A_predict_dirLabel": grouped_knn(uA, yA_s, g_s, args.seed),
        "A_predict_aqdirLabel": grouped_knn(uA, yB_s, g_s, args.seed),
        "B_predict_aqdirLabel": grouped_knn(unit_vectors(Bc), yB_s, g_s, args.seed),
        "B_predict_dirLabel": grouped_knn(unit_vectors(Bc), yA_s, g_s, args.seed),
    }

    json.dump(comp, open(out / "compare_summary.json", "w"), indent=2, default=str)
    # aligned lon/lat 2D density comparison plot
    if HAVE_PLT:
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        axs[0].hexbin(lonA2, latA2, gridsize=45, cmap="viridis", mincnt=1)
        axs[0].set_title(f"{comp['name_a']} (A)")
        axs[1].hexbin(lonBa, latBa, gridsize=45, cmap="viridis", mincnt=1)
        axs[1].set_title(f"{comp['name_b']} aligned->A")
        for a in axs:
            a.set_xlim(LON_LIM); a.set_ylim(LAT_LIM)
            a.set_xlabel("longitude"); a.set_ylabel("latitude")
        fig.suptitle("Aligned lon/lat density")
        fig.savefig(out / "aligned_lonlat_density.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(json.dumps(comp, indent=2, default=str))
    return comp


def _confusion(a, b):
    la, lb = sorted(set(a.tolist())), sorted(set(b.tolist()))
    M = {}
    for i in la:
        for j in lb:
            M[f"A{i}_B{j}"] = int(((a == i) & (b == j)).sum())
    return M


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Longitude/latitude spherical diagnostic for 3D CEBRA embeddings.")
    p.add_argument("--embedding", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--metadata", required=True)
    p.add_argument("--name", default="run")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--center-method", choices=["mean", "median"], default="median")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=60000)
    p.add_argument("--group-column", default="dyad_id")
    p.add_argument("--compare-embedding", default=None)
    p.add_argument("--compare-labels", default=None)
    p.add_argument("--compare-metadata", default=None)
    p.add_argument("--compare-name", default=None)
    p.add_argument("--procrustes-align", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metrics, base = analyze_single(args)
    if args.compare_embedding:
        analyze_compare(args, base)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
