#!/usr/bin/env python
"""
scripts/analyze_cebra_gmm_lonlat.py
===================================
Gaussian Mixture Model (GMM) analysis of the longitude/latitude ("globe")
representation of 3D CEBRA embeddings.

WHY GMM
-------
The 5-NN decoding we ran is *supervised*: we hand it the labels and ask "can a
classifier separate them?". A GMM is *unsupervised*: it models the point cloud as
a mixture of K Gaussian blobs WITHOUT ever seeing the labels, then we ask "do the
blobs it found line up with the real labels?". That is a stronger, less circular
kind of evidence -- if the natural density structure of the embedding recovers
speaker/listener or AQ groups on its own, the structure is really in the geometry
and not just something a supervised classifier could carve out.

WHAT IT REPORTS
---------------
  * best K by BIC (how many natural blobs the density actually has)
  * ARI (adjusted Rand index) between GMM clusters (K=2) and the true labels
      1.0 = clusters exactly match labels, 0.0 = no better than random
  * purity = weighted fraction of each cluster that is its majority label
  * per-component label composition

LONGITUDE WRAPAROUND
--------------------
Longitude is circular (+-180 is the same place). Some runs sit right on that seam
(e.g. AQ-magnitude has ~47% of points beyond |lon|>150). Fitting a plain Gaussian
there would tear one blob into two. So we ROTATE longitude onto its circular mean
first, which moves the seam into the sparsest region. Latitude does not wrap.

FEATURE SETS
------------
  lonlat         : [longitude_rotated, latitude]   <- the angular "globe" mapping
  lonlat_radius  : the above + radius (standardized)
Comparing the two isolates whether a label lives in the ANGLE or needs the RADIUS.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

C0, C1 = "#0173B2", "#DE8F05"


def load_run(run_dir: Path):
    rows = list(csv.DictReader(open(run_dir / "spherical_coordinates.csv")))
    lon = np.array([float(r["longitude_deg"]) for r in rows])
    lat = np.array([float(r["latitude_deg"]) for r in rows])
    rad = np.array([float(r["radius"]) for r in rows])
    lab = np.array([int(r["label"]) for r in rows])
    dyad = np.array([int(r["dyad"]) for r in rows])
    return lon, lat, rad, lab, dyad


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def rotate_to_circular_mean(lon_deg):
    """Rotate longitudes so their circular mean sits at 0 -> seam lands in sparsest area."""
    r = np.radians(lon_deg)
    cm = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    return wrap180(lon_deg - cm), float(cm)


def purity(clusters, labels):
    tot = 0
    for c in np.unique(clusters):
        m = clusters == c
        if m.sum():
            tot += np.bincount(labels[m]).max()
    return float(tot / len(labels))


def component_composition(clusters, labels):
    out = {}
    for c in sorted(np.unique(clusters).tolist()):
        m = clusters == c
        cnt = np.bincount(labels[m], minlength=int(labels.max()) + 1)
        out[f"component_{c}"] = {"n": int(m.sum()),
                                 "label_counts": cnt.tolist(),
                                 "majority_label": int(cnt.argmax()),
                                 "majority_frac": round(float(cnt.max() / cnt.sum()), 4)}
    return out


def fit_bic_sweep(X, kmax, seed):
    bics, models = [], {}
    for k in range(1, kmax + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=seed, n_init=2, max_iter=300).fit(X)
        bics.append(float(gm.bic(X))); models[k] = gm
    best_k = int(np.argmin(bics) + 1)
    return best_k, bics, models


def plot_gmm(out, name, lon_r, lat, lab, clusters, gm, cm_shift):
    """lat/long scatter: true labels vs GMM components, with 2-sigma ellipses."""
    rng = np.random.RandomState(0)
    idx = rng.choice(len(lon_r), min(15000, len(lon_r)), replace=False)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.8))
    # left: true labels
    for c, col, lb in [(0, C0, "label 0"), (1, C1, "label 1")]:
        m = lab[idx] == c
        axs[0].scatter(lon_r[idx][m], lat[idx][m], s=5, alpha=0.35, color=col,
                       label=lb, linewidths=0)
    axs[0].set_title("True labels")
    # right: GMM components
    cmap = plt.get_cmap("tab10", max(gm.n_components, 2))
    for c in range(gm.n_components):
        m = clusters[idx] == c
        axs[1].scatter(lon_r[idx][m], lat[idx][m], s=5, alpha=0.35,
                       color=cmap(c), label=f"component {c}", linewidths=0)
    for c in range(gm.n_components):
        mean, cov = gm.means_[c], gm.covariances_[c][:2, :2]
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        for nsig in (1, 2):
            w, h = 2 * nsig * np.sqrt(vals)
            axs[1].add_patch(Ellipse(mean[:2], w, h, angle=ang, fc="none",
                                     ec="black", lw=1.4, alpha=0.8))
    axs[1].set_title(f"GMM components (K={gm.n_components}) + 1σ/2σ ellipses")
    for a in axs:
        a.set_xlim(-180, 180); a.set_ylim(-90, 90)
        a.set_xlabel(f"Longitude, rotated by {cm_shift:.0f}° (degrees)")
        a.set_ylabel("Latitude (degrees)")
        a.legend(markerscale=3, fontsize=10); a.grid(alpha=0.2)
    fig.suptitle(f"{name}: GMM on longitude/latitude", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "gmm_lonlat.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze(run_dir: Path, out_dir: Path, name: str, kmax=8, seed=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    lon, lat, rad, lab, dyad = load_run(run_dir)
    lon_r, cm = rotate_to_circular_mean(lon)

    res = {"name": name, "n": int(len(lab)), "longitude_rotation_deg": cm,
           "n_true_classes": int(len(np.unique(lab))), "seed": seed}

    for fname, X in [("lonlat", np.c_[lon_r, lat]),
                     ("lonlat_radius", StandardScaler().fit_transform(np.c_[lon_r, lat, rad]))]:
        best_k, bics, models = fit_bic_sweep(X, kmax, seed)
        gm2 = models[2]
        cl2 = gm2.predict(X)
        res[fname] = {
            "best_k_by_bic": best_k,
            "bic_by_k": {k: round(b, 1) for k, b in zip(range(1, kmax + 1), bics)},
            "K2_ARI_vs_labels": round(float(adjusted_rand_score(lab, cl2)), 4),
            "K2_purity": round(purity(cl2, lab), 4),
            "K2_components": component_composition(cl2, lab),
        }
        if fname == "lonlat":
            plot_gmm(out_dir, name, lon_r, lat, lab, cl2, gm2, cm)

    json.dump(res, open(out_dir / "gmm_metrics.json", "w"), indent=2)
    print(f"{name}: lonlat ARI={res['lonlat']['K2_ARI_vs_labels']} "
          f"purity={res['lonlat']['K2_purity']} bestK={res['lonlat']['best_k_by_bic']} | "
          f"+radius ARI={res['lonlat_radius']['K2_ARI_vs_labels']} "
          f"purity={res['lonlat_radius']['K2_purity']}")
    return res


def main():
    ap = argparse.ArgumentParser(description="GMM on longitude/latitude CEBRA embeddings.")
    ap.add_argument("--ll-root", default="results/stacked_cebra_final/longitude_latitude_analysis")
    ap.add_argument("--runs", default="direction_real,direction_shuffled,aq_direction_real,"
                                      "aq_direction_shuffled,aq_magnitude_real,aq_magnitude_shuffled")
    ap.add_argument("--out-root", default="results/stacked_cebra_final/gmm_lonlat_analysis")
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root, out_root = Path(args.ll_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    all_res = []
    for name in [r.strip() for r in args.runs.split(",") if r.strip()]:
        rd = root / name
        if not (rd / "spherical_coordinates.csv").exists():
            print(f"  SKIP {name}: no spherical_coordinates.csv")
            continue
        all_res.append(analyze(rd, out_root / name, name, args.kmax, args.seed))

    # summary CSV
    cols = ["run", "n_true_classes", "lonlat_bestK", "lonlat_ARI", "lonlat_purity",
            "lonlat_radius_bestK", "lonlat_radius_ARI", "lonlat_radius_purity",
            "longitude_rotation_deg"]
    sdir = out_root / "summary"; sdir.mkdir(parents=True, exist_ok=True)
    with open(sdir / "gmm_lonlat_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in all_res:
            w.writerow([r["name"], r["n_true_classes"],
                        r["lonlat"]["best_k_by_bic"], r["lonlat"]["K2_ARI_vs_labels"],
                        r["lonlat"]["K2_purity"],
                        r["lonlat_radius"]["best_k_by_bic"],
                        r["lonlat_radius"]["K2_ARI_vs_labels"],
                        r["lonlat_radius"]["K2_purity"],
                        round(r["longitude_rotation_deg"], 1)])
    print(f"\nSummary -> {sdir/'gmm_lonlat_summary.csv'}")


if __name__ == "__main__":
    main()
