#!/usr/bin/env python
"""
scripts/build_overleaf_figures_250.py
=====================================
Assemble the 12 Overleaf figures that must be replaced after the uniform 250 Hz
resampling, under their existing filenames so the LaTeX needs no edits.

Sources — all from the 250 Hz re-run:
  rerun250/fullband_baseline/    baseline geometry + unsupervised GMM
  rerun250/dyad_permutation/     permutation nulls
  rerun250/aq_delta6/            six-class confusion
  rerun250/aq_delta6_svm/        six-class panels + SVM comparisons
  rerun250/gmm_identity_check/   GMM K-sweep + K=5 vs dyad
  band_comparison/               band-removal figures (already 250 Hz, re-exported
                                 so the submission has one provenance)

Two figures are drawn here rather than copied, because the generic pipeline does
not emit them: aq_delta6_svm_multiclass.png and aq_delta6_pairwise_svm.png.
"""
from __future__ import annotations

import argparse, csv, json, shutil, struct
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUG = RESULTS / "aug4_pipeline"
RR = AUG / "rerun250"
SVMD = RR / "aq_delta6_svm"
OUT = AUG / "overleaf_figures_250hz"

C0, C1 = "#0173B2", "#DE8F05"


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  drew   {name}")


def svm_multiclass():
    s = json.load(open(SVMD / "svm_metrics.json"))
    dc = s["decoder_comparison"]
    kept = s["kept_daq_values"]; K = len(kept)
    labs = [str(v) for v in kept]
    chance = s["majority_chance"]
    names = ["knn5_reference", "svm_ovo_linear", "svm_ovo_rbf"]
    pretty = ["5-NN\n(reference)", "SVM one-vs-one\nlinear", "SVM one-vs-one\nRBF"]

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    xs = np.arange(len(names)); w = 0.26
    axs[0].bar(xs - w, [dc[n]["accuracy"] for n in names], w, label="accuracy", color=C0)
    axs[0].bar(xs, [dc[n]["balanced_accuracy"] for n in names], w,
               label="balanced accuracy", color=C1)
    axs[0].bar(xs + w, [dc[n]["macro_f1"] for n in names], w, label="macro F1",
               color="#029E73")
    axs[0].axhline(chance, ls="--", c="#D55E00", lw=1.5,
                   label=f"majority chance {chance:.3f}")
    axs[0].axhline(1 / K, ls=":", c="#888888", lw=1.5, label=f"uniform chance {1/K:.3f}")
    axs[0].set_xticks(xs); axs[0].set_xticklabels(pretty, fontsize=9)
    axs[0].set_ylabel("leave-one-dyad-out score"); axs[0].set_ylim(0, 0.72)
    axs[0].set_title("Six-class |ΔAQ| decoding (250 Hz)", fontsize=11)
    axs[0].legend(fontsize=7.5, loc="upper left")
    for i, n in enumerate(names):
        axs[0].text(i - w, dc[n]["accuracy"] + 0.012, f"{dc[n]['accuracy']:.3f}",
                    ha="center", fontsize=7)

    rec = np.array([[dc[n]["per_class_recall"][str(v)] for v in kept] for n in names])
    im = axs[1].imshow(rec, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
    axs[1].set_xticks(range(K)); axs[1].set_xticklabels(labs)
    axs[1].set_yticks(range(len(names)))
    axs[1].set_yticklabels(["5-NN", "SVM linear", "SVM RBF"], fontsize=9)
    axs[1].set_xlabel("|ΔAQ| class")
    axs[1].set_title("Per-class recall", fontsize=11)
    for i in range(len(names)):
        for j in range(K):
            axs[1].text(j, i, f"{rec[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if rec[i, j] > 0.45 else "black")
    fig.colorbar(im, ax=axs[1], label="recall")
    fig.suptitle("Multi-class decoders on the six-class |ΔAQ| embedding — "
                 "uniform 250 Hz, grouped leave-one-dyad-out", fontsize=12)
    save(fig, "aq_delta6_svm_multiclass.png")


def svm_pairwise():
    s = json.load(open(SVMD / "svm_metrics.json"))
    kept = s["kept_daq_values"]; K = len(kept)
    labs = [str(v) for v in kept]
    pr = list(csv.DictReader(open(SVMD / "svm_pairwise_accuracy.csv")))
    M = np.full((K, K), np.nan); B = np.full((K, K), np.nan)
    for r in pr:
        a, b = int(r["class_a"]), int(r["class_b"])
        M[a, b] = M[b, a] = float(r["svm_rbf_lodo_acc"])
        B[a, b] = B[b, a] = float(r["above_baseline"])

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axs[0].imshow(M, cmap="viridis", vmin=0.4, vmax=0.9)
    axs[0].set_title("Pairwise RBF-SVM accuracy", fontsize=11)
    vmax = float(np.nanmax(np.abs(B))) if np.isfinite(B).any() else 0.3
    im1 = axs[1].imshow(B, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axs[1].set_title("Accuracy minus majority baseline", fontsize=11)
    for ax, Mx, fmt in [(axs[0], M, "{:.2f}"), (axs[1], B, "{:+.2f}")]:
        ax.set_xticks(range(K)); ax.set_xticklabels(labs)
        ax.set_yticks(range(K)); ax.set_yticklabels(labs)
        ax.set_xlabel("|ΔAQ|"); ax.set_ylabel("|ΔAQ|")
        for i in range(K):
            for j in range(K):
                if np.isfinite(Mx[i, j]):
                    ax.text(j, i, fmt.format(Mx[i, j]), ha="center", va="center",
                            fontsize=7.5, color="black")
    fig.colorbar(im0, ax=axs[0], label="accuracy")
    fig.colorbar(im1, ax=axs[1], label="Δ over baseline")
    fig.suptitle("Pairwise (one-vs-one) separability of |ΔAQ| classes — uniform 250 Hz",
                 fontsize=11)
    save(fig, "aq_delta6_pairwise_svm.png")


def _baseline_lonlat():
    """Reproduce the baseline evaluation subsample and its spherical coordinates.

    run_band_cebra_analysis.py draws `sub` as the FIRST call on
    RandomState(seed), so it is exactly reproducible from the seed and the
    sample count. It saves lonlat_radius.npy for that subsample but not the
    indices, hence the reconstruction here.
    """
    base = RR / "fullband_baseline"
    m = json.load(open(base / "metrics.json"))
    T = int(m["n_samples"]); seed = int(m.get("seed", 0))
    sub = np.random.RandomState(seed).choice(T, min(60000, T), replace=False)
    ll = np.load(base / "lonlat_radius.npy")
    meta = list(csv.DictReader(open(base / "sample_metadata.csv")))
    y = np.zeros(T, np.int64)
    for r in meta:
        y[int(r["start"]):int(r["end"])] = int(r["mag"])
    return ll[:, 0], ll[:, 1], y[sub], m


def baseline_lonlat_fig():
    lon, lat, Y, m = _baseline_lonlat()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for c, col, lb in [(0, C0, "Low |ΔAQ|"), (1, C1, "High |ΔAQ|")]:
        k = Y == c
        ax.scatter(lon[k], lat[k], s=4, alpha=0.32, color=col, label=lb,
                   linewidths=0, rasterized=True)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Full-band speaker-first embedding by AQ magnitude (uniform 250 Hz)\n"
                 f"KS lat={m['ks_latitude']:.3f}  lon={m['ks_longitude']:.3f}  "
                 f"rad={m['ks_radius']:.3f};  grouped 5-NN="
                 f"{m['knn5_grouped_magnitude']:.3f} (chance {m['chance']:.3f})",
                 fontsize=10)
    ax.legend(markerscale=5)
    save(fig, "fullband_speakerfirst_lonlat_by_magnitude.png")


def baseline_gmm_fig():
    from sklearn.mixture import GaussianMixture
    from matplotlib.patches import Ellipse
    lon, lat, Y, m = _baseline_lonlat()
    g = json.load(open(RR / "fullband_baseline" / "gmm_unsupervised.json"))
    feat = np.c_[lon, lat]
    gm = GaussianMixture(2, covariance_type="full", random_state=int(m.get("seed", 0)),
                         n_init=2, max_iter=300).fit(feat)
    cl = gm.predict(feat)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    cmap = plt.get_cmap("tab10", 2)
    for c in range(2):
        k = cl == c
        ax.scatter(lon[k], lat[k], s=4, alpha=0.3, color=cmap(c),
                   label=f"GMM component {c} ({100*k.mean():.0f}%)",
                   linewidths=0, rasterized=True)
    for c in range(2):
        mean, cov = gm.means_[c], gm.covariances_[c]
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        for nsd in (1, 2):
            ax.add_patch(Ellipse(mean, 2 * nsd * np.sqrt(vals[0]),
                                 2 * nsd * np.sqrt(vals[1]), angle=ang,
                                 fc="none", ec="black", lw=1.4, ls="--", alpha=0.8))
        ax.plot(*mean, marker="X", ms=13, mfc=cmap(c), mec="black", mew=1.4, ls="none")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Label-agnostic GMM (K=2) on the full-band embedding (uniform 250 Hz)\n"
                 f"ARI vs AQ magnitude = {m['gmm_unsup_ARI_vs_magnitude']:.4f},  "
                 f"purity = {m['gmm_unsup_purity']:.4f},  "
                 f"BIC-preferred K = {m['gmm_unsup_bestK']}", fontsize=10)
    ax.legend(markerscale=5, fontsize=9)
    save(fig, "fullband_speakerfirst_gmm_unsupervised.png")


def build_copy_list():
    """Built as a function (not a module-level list) so it always reflects the
    current AUG/RR/SVMD globals, which main() may have overridden via
    --results-root -- a plain list literal here would freeze in the
    import-time default and silently ignore that flag."""
    return [
        # 33-dyad full-band baseline @250 Hz
        (RR / "dyad_permutation" / "fullband_speakerfirst_permutation_5nn_null.png",
         "fullband_speakerfirst_permutation_5nn_null.png"),
        (RR / "dyad_permutation" / "fullband_speakerfirst_permutation_gof_null.png",
         "fullband_speakerfirst_permutation_gof_null.png"),
        # six-class @250 Hz
        (RR / "aq_delta6" / "aq_delta6_confusion_matrix.png",
         "aq_delta6_confusion_matrix.png"),
        (SVMD / "aq_delta6_lonlat_panels.png", "aq_delta6_lonlat_panels.png"),
        # GMM identity @250 Hz
        (RR / "gmm_identity_check" / "gmm_k_sweep.png", "fullband_gmm_k_sweep.png"),
        (RR / "gmm_identity_check" / "gmm_k5_vs_dyad.png", "fullband_gmm_k5_vs_dyad.png"),
        # band-removal (already 250 Hz; re-exported for single provenance)
        (AUG / "band_comparison" / "band_metric_comparison.png", "band_metric_comparison.png"),
        (AUG / "band_comparison" / "band_ks_comparison.png", "band_ks_comparison.png"),
    ]

WANT = ["fullband_speakerfirst_lonlat_by_magnitude.png",
        "fullband_speakerfirst_gmm_unsupervised.png",
        "fullband_speakerfirst_permutation_5nn_null.png",
        "fullband_speakerfirst_permutation_gof_null.png",
        "aq_delta6_confusion_matrix.png",
        "aq_delta6_lonlat_panels.png",
        "aq_delta6_svm_multiclass.png",
        "aq_delta6_pairwise_svm.png",
        "fullband_gmm_k_sweep.png",
        "fullband_gmm_k5_vs_dyad.png",
        "band_metric_comparison.png",
        "band_ks_comparison.png"]


def main():
    global RESULTS, AUG, RR, SVMD, OUT
    ap = argparse.ArgumentParser(
        description="Assemble the 12 Overleaf figures for the uniform "
                    "250 Hz re-run from an existing results/aug4_pipeline/ "
                    "tree (expects rerun250/, aq_delta6_svm/, and "
                    "band_comparison/ subfolders, as produced by "
                    "training/run_250hz_rerun.py and "
                    "evaluation/build_band_comparison.py in this submission).")
    ap.add_argument("--results-root", type=Path, default=RESULTS,
                    help=f"Path to the results/ directory (default: {RESULTS}).")
    args = ap.parse_args()
    RESULTS = args.results_root
    AUG = RESULTS / "aug4_pipeline"
    RR = AUG / "rerun250"
    SVMD = RR / "aq_delta6_svm"
    OUT = AUG / "overleaf_figures_250hz"

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing to {OUT}\n")
    for src, dst in build_copy_list():
        if src.exists():
            shutil.copy2(src, OUT / dst)
            print(f"  copied {dst}")
        else:
            print(f"  !! MISSING SOURCE {src}")
    baseline_lonlat_fig()
    baseline_gmm_fig()
    svm_multiclass()
    svm_pairwise()

    print("\nverification (all must be valid 300-dpi PNGs):")
    ok = 0
    for w in WANT:
        p = OUT / w
        if not p.exists():
            print(f"  MISSING {w}"); continue
        d = open(p, "rb").read(33)
        sig = d[:8] == b"\x89PNG\r\n\x1a\n"
        wd, ht = struct.unpack(">II", d[16:24])
        fh = open(p, "rb"); fh.seek(-12, 2); end = fh.read(); fh.close()
        good = sig and (b"IEND" in end)
        ok += good
        print(f"  {'OK     ' if good else 'BAD    '} {w:48s} {wd:5d}x{ht:<5d} "
              f"{p.stat().st_size/1024:7.0f} KB")
    print(f"\n{ok}/{len(WANT)} figures ready in {OUT}")


if __name__ == "__main__":
    main()
