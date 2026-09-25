#!/usr/bin/env python
"""
evaluation/build_session_figures.py
=====================================
Produce a curated set of ten summary figures from already-computed results
(the six-class |dAQ| run, the participant-level AQ analysis, and the
non-oscillatory analysis), under fixed filenames, in one folder. Regenerates
the ones that need composing and copies the ones that already exist verbatim.

All figures: PNG, 300 dpi, white background, no local paths in titles.

By default this expects the standard results/aug4_pipeline/ layout produced
by the training/, controls/, and preprocessing/ scripts in this submission;
point --results-root elsewhere if your results tree lives somewhere else.
"""
from __future__ import annotations

import argparse
import csv, json, shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUG = RESULTS / "aug4_pipeline"
SVM = AUG / "aq_delta6_svm"
IND = AUG / "individual_aq_role"
NO = AUG / "nonosc_michelle"
OUT = AUG / "figures_session_aug11"

C0, C1 = "#0173B2", "#DE8F05"
CLS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#56B4E9"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- 1. copy
def copy_existing():
    for src, dst in [(SVM / "aq_delta6_lonlat_panels.png", "aq_delta6_lonlat_panels.png"),
                     (IND / "individual_aq_permutation_null.png",
                      "individual_aq_permutation_null.png")]:
        if src.exists():
            shutil.copy2(src, OUT / dst); print(f"  copied {dst}")


# ---------------------------------------------------------------- 2. SVM
def svm_figures():
    s = json.load(open(SVM / "svm_metrics.json"))
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
    axs[0].set_title("Six-class |ΔAQ| decoding", fontsize=11)
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
    fig.suptitle("Multi-class decoders on the six-class |ΔAQ| embedding "
                 "(grouped leave-one-dyad-out)", fontsize=12)
    save(fig, "aq_delta6_svm_multiclass.png")

    # ---- pairwise ----
    pr = list(csv.DictReader(open(SVM / "svm_pairwise_accuracy.csv")))
    M = np.full((K, K), np.nan); B = np.full((K, K), np.nan)
    for r in pr:
        a, b = int(r["class_a"]), int(r["class_b"])
        M[a, b] = M[b, a] = float(r["svm_rbf_lodo_acc"])
        B[a, b] = B[b, a] = float(r["above_baseline"])

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axs[0].imshow(M, cmap="viridis", vmin=0.4, vmax=0.9)
    axs[0].set_title("Pairwise RBF-SVM accuracy", fontsize=11)
    im1 = axs[1].imshow(B, cmap="RdBu_r", vmin=-0.31, vmax=0.31)
    axs[1].set_title("Accuracy minus majority baseline", fontsize=11)
    for ax, Mx, im, fmt in [(axs[0], M, im0, "{:.2f}"), (axs[1], B, im1, "{:+.2f}")]:
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
    fig.suptitle("Pairwise (one-vs-one) separability of |ΔAQ| classes — "
                 "adjacent 0 vs 1 is the most separable pair, distant 4 vs 5 is at chance",
                 fontsize=11)
    save(fig, "aq_delta6_pairwise_svm.png")


# ---------------------------------------------------------------- 3. individual
def individual_figures():
    m = json.load(open(IND / "metrics.json"))
    fu = json.load(open(IND / "followup_checks.json"))
    emb = np.load(IND / "embedding.npy")
    ll = np.load(IND / "lonlat_radius.npy")
    sub = np.load(IND / "eval_indices.npy")
    meta = list(csv.DictReader(open(IND / "sample_metadata.csv")))

    aq = np.zeros(len(emb), np.float32); role = np.zeros(len(emb), np.int64)
    for r in meta:
        s, e = int(r["start"]), int(r["end"])
        aq[s:e] = float(r["aq"]); role[s:e] = 0 if r["role"] == "speak" else 1
    A, R = aq[sub], role[sub]
    lon, lat = ll[:, 0], ll[:, 1]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    sc = ax.scatter(lon, lat, s=3, alpha=0.35, c=A, cmap="viridis", linewidths=0,
                    rasterized=True)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_title(f"Individual participants by their own AQ-10 level\n"
                 f"leave-one-dyad-out decoding {fu['leave-one-DYAD-out']['exact_acc']:.3f} "
                 f"vs majority {fu['majority_chance_exact']:.3f}, "
                 f"permutation p={fu['dyad_permutation']['empirical_p']:.4f}", fontsize=10)
    fig.colorbar(sc, ax=ax, label="AQ-10 score")
    save(fig, "individual_aq_lonlat.png")

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for rv, col, lb in [(0, C0, "speaking"), (1, C1, "listening")]:
        msk = R == rv
        ax.scatter(lon[msk], lat[msk], s=3, alpha=0.3, color=col, label=lb,
                   linewidths=0, rasterized=True)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_title(f"Same embedding by role — role is NOT encoded\n"
                 f"leave-one-participant-out decoding "
                 f"{m['decode_role_lopo_knn5']:.3f} vs chance "
                 f"{m['decode_role_chance']:.3f}; KS lat {m['ks_by_role']['latitude']:.3f}, "
                 f"lon {m['ks_by_role']['longitude']:.3f}", fontsize=10)
    ax.legend(markerscale=6)
    save(fig, "individual_role_lonlat.png")

    # ---- rank of own listening state ----
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    cent = {}
    for r in meta:
        cent[(int(r["pid"]), r["role"])] = unit(emb[int(r["start"]):int(r["end"])].mean(0))
    people = sorted({int(r["pid"]) for r in meta
                     if (int(r["pid"]), "speak") in cent and (int(r["pid"]), "listen") in cent})
    S = np.array([cent[(q, "speak")] for q in people])
    L = np.array([cent[(q, "listen")] for q in people])
    sim = S @ L.T
    own = np.diag(sim)
    n = len(people)
    ranks = np.array([1 + int((sim[i] > own[i]).sum()) for i in range(n)])
    oth = sim[~np.eye(n, dtype=bool)]

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axs[0].hist(oth, bins=45, alpha=0.75, color="#BBBBBB", density=True,
                label=f"different people (n={len(oth)})")
    axs[0].hist(own, bins=15, alpha=0.85, color="#D55E00", density=True,
                label=f"same person, both roles (n={n})")
    axs[0].axvline(own.mean(), color="#D55E00", lw=2)
    axs[0].axvline(oth.mean(), color="#555555", lw=2, ls="--")
    axs[0].set_xlabel("cosine similarity between embedding centroids")
    axs[0].set_ylabel("density")
    axs[0].set_title(f"own {own.mean():.3f} vs others {oth.mean():.3f} ± {oth.std():.3f}\n"
                     f"but {100*(oth>0.9).mean():.0f}% of OTHER pairs also exceed 0.9",
                     fontsize=10)
    axs[0].legend(fontsize=8)

    bins = np.arange(0.5, n + 1.5, 1)
    axs[1].hist(ranks, bins=bins, color=C0, edgecolor="white")
    axs[1].axvline(np.median(ranks), color="#D55E00", lw=2,
                   label=f"median rank {np.median(ranks):.1f}")
    axs[1].axvline(5.5, color="#029E73", ls="--", lw=1.5, label="top-5 cutoff")
    axs[1].set_xlabel(f"rank of own listening state among {n} candidates")
    axs[1].set_ylabel("participants")
    axs[1].set_xlim(0, n + 1)
    axs[1].set_title(f"top-1 {100*np.mean(ranks==1):.1f}% (chance {100/n:.1f}%),  "
                     f"top-5 {100*np.mean(ranks<=5):.1f}% (chance {500/n:.1f}%)",
                     fontsize=10)
    axs[1].legend(fontsize=8)
    fig.suptitle("Within-individual role similarity: real but not individuating",
                 fontsize=12)
    save(fig, "individual_role_similarity_rank.png")


# ---------------------------------------------------------------- 4. non-osc
def nonosc_figures():
    import glob
    rows = []
    for f in glob.glob(str(NO / "qc" / "*.csv")):
        rows += list(csv.DictReader(open(f)))
    rows = [r for r in rows if r.get("alpha_prominence_after_db")]
    b = np.array([float(r["alpha_prominence_before_db"]) for r in rows])
    a = np.array([float(r["alpha_prominence_after_db"]) for r in rows])
    sb = np.array([float(r["slope_before"]) for r in rows])
    sa = np.array([float(r["slope_after"]) for r in rows])

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    axs[0].scatter(b, a, s=45, color=C0, alpha=0.8, edgecolor="white")
    lim = [min(b.min(), a.min()) - 0.5, max(b.max(), a.max()) + 0.5]
    axs[0].plot(lim, lim, ls=":", c="#888888", label="no change")
    axs[0].axhline(0, color="#029E73", ls="--", lw=1.5, label="0 dB = fully aperiodic")
    axs[0].set_xlabel("alpha prominence BEFORE (dB)")
    axs[0].set_ylabel("alpha prominence AFTER (dB)")
    axs[0].set_title(f"Per recording (n={len(rows)})\n"
                     f"mean {b.mean():+.2f} → {a.mean():+.2f} dB", fontsize=10)
    axs[0].legend(fontsize=8)

    axs[1].hist(b, bins=22, alpha=0.7, color="#BBBBBB", label="before")
    axs[1].hist(a, bins=22, alpha=0.75, color=C1, label="after")
    axs[1].axvline(0, color="#029E73", ls="--", lw=1.5)
    axs[1].set_xlabel("alpha prominence over 1/f fit (dB)")
    axs[1].set_ylabel("recordings")
    axs[1].set_title(f"closer to 0 dB in {int((abs(a)<abs(b)).sum())}/{len(rows)} files;\n"
                     f"within 1 dB of 0 in {int((abs(a)<1).sum())}/{len(rows)}", fontsize=10)
    axs[1].legend(fontsize=8)

    axs[2].scatter(sb, sa, s=45, color="#CC78BC", alpha=0.8, edgecolor="white")
    lim2 = [min(sb.min(), sa.min()) - 0.2, max(sb.max(), sa.max()) + 0.2]
    axs[2].plot(lim2, lim2, ls=":", c="#888888")
    axs[2].set_xlabel("1/f slope BEFORE"); axs[2].set_ylabel("1/f slope AFTER")
    axs[2].set_title(f"aperiodic slope preserved\n{sb.mean():+.3f} → {sa.mean():+.3f}",
                     fontsize=10)
    fig.suptitle("Non-oscillatory extraction QC — 84 recordings, 64 channels fitted each, "
                 "all changed", fontsize=12)
    save(fig, "nonosc_extraction_qc.png")

    # ---- nonosc vs fullband ----
    n = json.load(open(NO / "cebra" / "metrics.json"))
    f = json.load(open(RESULTS / "filtered_bands_ours" / "fullband" / "metrics.json"))
    keys = [("final_loss", "loss"), ("goodness_of_fit_bits", "GoF (bits)"),
            ("knn5_grouped_magnitude", "grouped 5-NN"),
            ("silhouette_magnitude", "silhouette"),
            ("ks_latitude", "KS lat"), ("ks_longitude", "KS lon"),
            ("ks_radius", "KS rad"), ("gmm_unsup_ARI_vs_magnitude", "GMM K2 ARI"),
            ("gmm_unsup_purity", "GMM K2 purity")]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    xs = np.arange(len(keys)); w = 0.36
    nv = [n[k] for k, _ in keys]; fv = [f[k] for k, _ in keys]
    ax.bar(xs - w / 2, nv, w, label="non-oscillatory (aperiodic only)", color=C0)
    ax.bar(xs + w / 2, fv, w, label="full band", color="#BBBBBB")
    ax.axhline(n["chance"], ls="--", c="#D55E00", lw=1.4,
               label=f"decoding chance {n['chance']:.3f}")
    for i, (a_, b_) in enumerate(zip(nv, fv)):
        ax.text(i - w / 2, a_ + 0.02, f"{a_:.3f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, b_ + 0.02, f"{b_:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(xs); ax.set_xticklabels([p for _, p in keys], rotation=25, ha="right",
                                          fontsize=9)
    ax.set_title("Removing all oscillatory content changes nothing\n"
                 "19 dyads, 38 speaker-first files, identical model configuration",
                 fontsize=11)
    ax.legend(fontsize=8.5)
    save(fig, "nonosc_vs_fullband_metrics.png")

    # ---- permutation null ----
    perms = []
    for d in sorted((NO / "permutation").glob("perm*")):
        if (d / "metrics.json").exists():
            perms.append(json.load(open(d / "metrics.json")))
    rr = json.load(open(NO / "permutation" / "real_rescored_metrics.json"))["metrics"]
    rvn = {r["metric"]: r for r in
           csv.DictReader(open(NO / "permutation" / "real_vs_null_summary.csv"))}

    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax, key, nm in [(axs[0], "knn5_leave_one_dyad_out", "Leave-one-dyad-out 5-NN"),
                        (axs[1], "goodness_of_fit_bits", "Goodness of fit (bits)"),
                        (axs[2], "silhouette", "Silhouette")]:
        vals = np.array([p[key] for p in perms if p.get(key) is not None])
        real = float(rvn[key]["real"]); p_emp = float(rvn[key]["empirical_p"])
        ax.scatter(vals, np.full(len(vals), 0.5), s=110, color="#9ecae1",
                   edgecolor="#3182bd", zorder=3, label=f"permutations (n={len(vals)})")
        ax.axvline(real, color="#D55E00", lw=2.6, zorder=4, label=f"real = {real:.4f}")
        ax.axvline(vals.mean(), color="#333333", ls="--", lw=1.5,
                   label=f"null mean = {vals.mean():.4f}")
        ax.set_yticks([]); ax.set_ylim(0, 1)
        ax.set_xlabel(nm)
        ax.set_title(f"{nm}\nempirical p = {p_emp:.3f}", fontsize=10)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.suptitle("Non-oscillatory: dyad-level AQ permutation control — "
                 "CEBRA retrained per permutation, real value sits inside the null",
                 fontsize=12)
    save(fig, "nonosc_permutation_null.png")


def main():
    global RESULTS, AUG, SVM, IND, NO, OUT
    ap = argparse.ArgumentParser(
        description="Build ten curated summary PNGs (six-class |dAQ| SVM "
                    "panels, participant-level AQ figures, non-oscillatory "
                    "QC/comparison/permutation figures) from an existing "
                    "results/aug4_pipeline/ tree produced by the training/, "
                    "controls/, and preprocessing/ scripts in this "
                    "submission, plus the sibling results/filtered_bands_ours/ "
                    "tree.")
    ap.add_argument("--results-root", type=Path, default=RESULTS,
                    help=f"Path to the results/ directory (default: {RESULTS}).")
    args = ap.parse_args()
    RESULTS = args.results_root
    AUG = RESULTS / "aug4_pipeline"
    SVM = AUG / "aq_delta6_svm"
    IND = AUG / "individual_aq_role"
    NO = AUG / "nonosc_michelle"
    OUT = AUG / "figures_session_aug11"
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"writing to {OUT}")
    copy_existing()
    svm_figures()
    individual_figures()
    nonosc_figures()

    want = ["aq_delta6_lonlat_panels.png", "aq_delta6_svm_multiclass.png",
            "aq_delta6_pairwise_svm.png", "individual_aq_lonlat.png",
            "individual_role_lonlat.png", "individual_aq_permutation_null.png",
            "individual_role_similarity_rank.png", "nonosc_extraction_qc.png",
            "nonosc_vs_fullband_metrics.png", "nonosc_permutation_null.png"]
    print("\nverification:")
    ok = 0
    for w in want:
        p = OUT / w
        if p.exists():
            print(f"  OK      {w:44s} {p.stat().st_size/1024:7.0f} KB"); ok += 1
        else:
            print(f"  MISSING {w}")
    print(f"\n{ok}/{len(want)} figures present in {OUT}")


if __name__ == "__main__":
    main()
