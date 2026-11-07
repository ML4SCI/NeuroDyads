#!/usr/bin/env python
"""
scripts/build_gender_figures.py
===============================
Publication-ready figures for the dyad-level gender-composition analysis.

Every figure carries the dyad counts in its caption text, because the MM class
rests on very few dyads and no panel should be read without that context.
"""
from __future__ import annotations
import argparse, csv, json, glob, struct
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

TYPES = ["MM", "FF", "MF"]
COL = {"MM": "#0173B2", "FF": "#DE8F05", "MF": "#029E73"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/gender_analysis/dyad_cebra")
    ap.add_argument("--out-dir", default="results/gender_analysis/figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    prov = json.load(open(run / "dataset_provenance.json"))
    real = json.load(open(run / "real" / "metrics.json"))
    per_type = prov["dyads_per_type"]
    nd = {t: len(per_type[t]) for t in TYPES}
    sub = np.load(run / "real" / "eval_indices.npy")
    emb = np.load(run / "real" / "embedding.npy")
    ll = np.load(run / "real" / "lonlat_radius.npy")
    meta = list(csv.DictReader(open(run / "sample_metadata.csv")))

    cls = np.zeros(len(emb), np.int64)
    for m in meta:
        cls[int(m["start"]):int(m["end"])] = int(m["cls"])
    Y = cls[sub]
    E = emb[sub]
    lon, lat = ll[:, 0], ll[:, 1]
    rng = np.random.RandomState(args.seed)
    cap = (f"MM n={nd['MM']} dyads, FF n={nd['FF']}, MF n={nd['MF']}  |  "
           f"chance={prov['majority_chance']:.3f}")

    def save(fig, name):
        fig.savefig(out / name, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig); print(f"  wrote {name}")

    # ---- 1. 3D embedding ----
    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    pl = rng.choice(len(E), min(24000, len(E)), replace=False)
    for i, t in enumerate(TYPES):
        m = Y[pl] == i
        ax.scatter(E[pl][m, 0], E[pl][m, 1], E[pl][m, 2], s=2.5, alpha=0.35,
                   color=COL[t], label=f"{t} ({nd[t]} dyads)", linewidths=0)
    ax.set_title(f"CEBRA 3D embedding by dyad gender composition\n{cap}", fontsize=10)
    ax.legend(markerscale=6, fontsize=9, loc="upper left")
    fig.tight_layout(); save(fig, "gender_embedding_3d.png")

    # ---- 2. longitude / latitude, all classes ----
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for i, t in enumerate(TYPES):
        m = Y == i
        ax.scatter(lon[m], lat[m], s=4, alpha=0.32, color=COL[t],
                   label=f"{t} ({nd[t]} dyads)", linewidths=0, rasterized=True)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Longitude/latitude by gender composition\n{cap}", fontsize=10)
    ax.legend(markerscale=5)
    fig.tight_layout(); save(fig, "gender_lonlat.png")

    # ---- 3. one panel per class ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    for i, t in enumerate(TYPES):
        a = axs[i]
        a.scatter(lon, lat, s=2, alpha=0.05, color="#BBBBBB", linewidths=0,
                  rasterized=True)
        m = Y == i
        a.scatter(lon[m], lat[m], s=4, alpha=0.4, color=COL[t], linewidths=0,
                  rasterized=True)
        if m.sum():
            a.plot(np.median(lon[m]), np.median(lat[m]), marker="X", ms=14,
                   mfc=COL[t], mec="black", mew=1.5, ls="none")
        a.set_xlim(-180, 180); a.set_ylim(-90, 90)
        a.set_title(f"{t}  —  {nd[t]} dyads, n={int(m.sum()):,} samples "
                    f"({100*m.mean():.1f}%)", fontsize=10)
        a.set_xlabel("Longitude")
        a.grid(alpha=0.15)
        if i == 0:
            a.set_ylabel("Latitude")
    fig.suptitle("Gender composition, each class plotted separately "
                 "(grey = all samples; X = class median)", fontsize=12)
    fig.tight_layout(); save(fig, "gender_lonlat_panels.png")

    # ---- 4. confusion matrix ----
    cm = np.array(real["confusion_matrix"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(TYPES)
    ax.set_yticks(range(3)); ax.set_yticklabels(TYPES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Leave-one-dyad-out 5-NN confusion\n"
                 f"acc={real['lodo_acc']:.3f}  bal={real['lodo_balanced']:.3f}  "
                 f"chance={real['majority_chance']:.3f}\n{cap}", fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="row-normalised")
    fig.tight_layout(); save(fig, "gender_confusion_matrix.png")

    # ---- 5. pairwise separability ----
    pw = real.get("pairwise", [])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    names = [r["pair"] for r in pw]
    xs = np.arange(len(names)); w = 0.36
    acc = [r.get("accuracy", np.nan) for r in pw]
    base = [r.get("majority_baseline", np.nan) for r in pw]
    axs[0].bar(xs - w / 2, acc, w, label="LODO accuracy", color="#0173B2")
    axs[0].bar(xs + w / 2, base, w, label="majority baseline", color="#BBBBBB")
    axs[0].set_xticks(xs); axs[0].set_xticklabels(names, fontsize=9)
    axs[0].set_ylabel("accuracy"); axs[0].set_ylim(0, 1.05)
    axs[0].set_title("Pairwise gender-type separability", fontsize=11)
    axs[0].legend(fontsize=8)
    for i, r in enumerate(pw):
        if "accuracy" in r:
            axs[0].text(i - w / 2, r["accuracy"] + 0.02, f"{r['accuracy']:.3f}",
                        ha="center", fontsize=7.5)
    delta = [r.get("delta_above_baseline", np.nan) for r in pw]
    cols = ["#029E73" if (d == d and d > 0) else "#D55E00" for d in delta]
    axs[1].bar(xs, delta, 0.5, color=cols)
    axs[1].axhline(0, color="black", lw=1)
    axs[1].set_xticks(xs); axs[1].set_xticklabels(names, fontsize=9)
    axs[1].set_ylabel("accuracy − majority baseline")
    axs[1].set_title("Δ above baseline (positive = separable)", fontsize=11)
    for i, d in enumerate(delta):
        if d == d:
            axs[1].text(i, d + (0.004 if d >= 0 else -0.012), f"{d:+.3f}",
                        ha="center", fontsize=8)
    sub_n = "  |  ".join(f"{r['pair']}: {r.get('n_dyads','?')} dyads" for r in pw)
    fig.suptitle(f"Pairwise comparisons, leave-one-dyad-out\n{sub_n}", fontsize=10)
    fig.tight_layout(); save(fig, "gender_pairwise_separability.png")

    # ---- 6. permutation nulls ----
    perms = []
    for d in sorted(glob.glob(str(run / "perm*" / "metrics.json"))):
        perms.append(json.load(open(d)))
    if perms:
        keys = [("lodo_acc", "Leave-one-dyad-out 5-NN"),
                ("goodness_of_fit_bits", "Goodness of fit (bits)"),
                ("lodo_balanced", "Balanced accuracy"),
                ("silhouette", "Silhouette")]
        fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
        for ax, (k, nm) in zip(axs, keys):
            v = np.array([p[k] for p in perms if p.get(k) is not None])
            rv = real[k]
            cnt = int((v >= rv).sum())
            p_emp = (1 + cnt) / (1 + len(v))
            ax.scatter(v, np.full(len(v), 0.5), s=110, color="#9ecae1",
                       edgecolor="#3182bd", zorder=3, label=f"permutations (n={len(v)})")
            ax.axvline(rv, color="#D55E00", lw=2.6, zorder=4, label=f"real = {rv:.4f}")
            ax.axvline(v.mean(), color="#333333", ls="--", lw=1.5,
                       label=f"null = {v.mean():.4f}")
            if k == "lodo_acc":
                ax.axvline(real["majority_chance"], color="#999999", ls=":", lw=1.5,
                           label=f"chance = {real['majority_chance']:.3f}")
            ax.set_yticks([]); ax.set_ylim(0, 1); ax.set_xlabel(nm)
            ax.set_title(f"{nm}\np = {p_emp:.3f}", fontsize=10)
            ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        fig.suptitle("Dyad-level gender permutation control — CEBRA retrained per "
                     f"permutation  |  {cap}", fontsize=11)
        fig.tight_layout(); save(fig, "gender_permutation_null.png")

    print("\nverification:")
    for p in sorted(out.glob("*.png")):
        d = open(p, "rb").read(33)
        sig = d[:8] == b"\x89PNG\r\n\x1a\n"
        w_, h_ = struct.unpack(">II", d[16:24])
        fh = open(p, "rb"); fh.seek(-12, 2); e = fh.read(); fh.close()
        print(f"  {'OK ' if sig and b'IEND' in e else 'BAD'} {p.name:42s} "
              f"{w_:5d}x{h_:<5d} {p.stat().st_size/1024:7.0f} KB")


if __name__ == "__main__":
    main()
