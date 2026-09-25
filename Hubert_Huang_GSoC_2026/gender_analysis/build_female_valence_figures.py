#!/usr/bin/env python
"""
gender_analysis/build_female_valence_figures.py
=================================================
Figures for the female-only positive-vs-negative valence CEBRA analysis.
Label source: InteractionQuality/IntRating_code (Satisfactory=positive,
Deficient=negative) from the demographic spreadsheet, attached to each
female participant's own speak-role recording (see run_female_valence_cebra.py
docstring for the full, disclosed mapping rationale).

Like build_sex_state_figures.py, this uses fixed paths relative to the
current working directory rather than --results-root, since it only ever
reads run_female_valence_cebra.py's fixed output layout. Run it from the
directory that contains `results/`, after that script has produced:
  results/gender_analysis/part2/female_valence/
Figures are written to results/gender_analysis/part2/figures/.
"""
from __future__ import annotations
import csv, json, glob, struct
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

RUN = Path("results/gender_analysis/part2/female_valence")
OUT = Path("results/gender_analysis/part2/figures")
LAB = ["negative", "positive"]
COL = {"negative": "#D55E00", "positive": "#0173B2"}


def wrap180(x):
    return (x + 180) % 360 - 180


def lonlat(emb):
    Ec = emb - np.median(emb, axis=0)
    x_, y_, z_ = Ec[:, 0], Ec[:, 1], Ec[:, 2]
    lon = np.degrees(np.arctan2(y_, x_))
    lat = np.degrees(np.arctan2(z_, np.sqrt(x_ ** 2 + y_ ** 2)))
    cm = np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(),
                               np.cos(np.radians(lon)).mean()))
    return wrap180(lon - cm), lat


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prov = json.load(open(RUN / "dataset_provenance.json"))
    real = json.load(open(RUN / "real" / "metrics.json"))
    emb = np.load(RUN / "real" / "embedding.npy")
    sub = np.load(RUN / "real" / "eval_indices.npy")
    meta = list(csv.DictReader(open(RUN / "sample_metadata.csv")))

    y = np.zeros(len(emb), np.int64)
    for m in meta:
        y[int(m["start"]):int(m["end"])] = int(m["label"])
    Y = y[sub]
    E = emb[sub]
    lon, lat = lonlat(E)

    cap = (f"n={prov['n_files']} recordings ({prov['n_positive_files']} positive, "
           f"{prov['n_negative_files']} negative), {prov['n_participants']} participants, "
           f"{prov['n_dyads']} dyads  |  chance={prov['majority_chance']:.3f}")

    # ---- 1. 3D embedding ----
    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.RandomState(0)
    pl = rng.choice(len(E), min(len(E), 40000), replace=False)
    for i, lab in enumerate(LAB):
        m = Y[pl] == i
        ax.scatter(E[pl][m, 0], E[pl][m, 1], E[pl][m, 2], s=3, alpha=0.4,
                   color=COL[lab], label=f"{lab} ({int((Y==i).sum()):,})", linewidths=0)
    ax.set_title(f"CEBRA 3D embedding by valence\n{cap}", fontsize=9)
    ax.legend(markerscale=6, fontsize=9, loc="upper left")
    fig.tight_layout(); save(fig, "female_valence_embedding_3d.png")

    # ---- 2. longitude/latitude, both classes ----
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for i, lab in enumerate(LAB):
        m = Y == i
        ax.scatter(lon[m], lat[m], s=4, alpha=0.32, color=COL[lab],
                   label=f"{lab} ({int(m.sum()):,})", linewidths=0, rasterized=True)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Longitude/latitude by valence\n{cap}", fontsize=9)
    ax.legend(markerscale=5)
    fig.tight_layout(); save(fig, "female_valence_lonlat.png")

    # ---- 3. one panel per class ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for i, lab in enumerate(LAB):
        a = axs[i]
        a.scatter(lon, lat, s=2, alpha=0.06, color="#BBBBBB", linewidths=0,
                  rasterized=True)
        m = Y == i
        a.scatter(lon[m], lat[m], s=4, alpha=0.4, color=COL[lab], linewidths=0,
                  rasterized=True)
        if m.sum():
            a.plot(np.median(lon[m]), np.median(lat[m]), marker="X", ms=14,
                   mfc=COL[lab], mec="black", mew=1.5, ls="none")
        a.set_xlim(-180, 180); a.set_ylim(-90, 90)
        a.set_title(f"{lab}  —  n={int(m.sum()):,} ({100*m.mean():.1f}%)", fontsize=10)
        a.set_xlabel("Longitude")
        a.grid(alpha=0.15)
        if i == 0:
            a.set_ylabel("Latitude")
    fig.suptitle(f"Valence, each class plotted separately\n{cap}", fontsize=11)
    fig.tight_layout(); save(fig, "female_valence_lonlat_panels.png")

    # ---- 4. confusion matrix (LOPO by participant) ----
    lopo = real["LOPO_by_participant"]
    cm = np.array(lopo["confusion_matrix"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(2)); ax.set_xticklabels(LAB)
    ax.set_yticks(range(2)); ax.set_yticklabels(LAB)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Leave-one-participant-out 5-NN\n"
                 f"acc={lopo['accuracy']:.3f}  bal={lopo['balanced_accuracy']:.3f}  "
                 f"chance={lopo['majority_chance']:.3f}\n{cap}", fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="row-normalised")
    fig.tight_layout(); save(fig, "female_valence_confusion_matrix.png")

    # ---- 5. permutation null ----
    summary = json.load(open(RUN / "real_vs_null_summary.json"))
    perms = []
    for d in sorted(glob.glob(str(RUN / "perm*" / "metrics.json"))):
        perms.append(json.load(open(d)))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics_map = [("LOPO_accuracy", "accuracy", "Leave-one-participant-out accuracy"),
                   ("goodness_of_fit_bits", "goodness_of_fit_bits", "Goodness of fit (bits)"),
                   ("silhouette", "silhouette", "Silhouette")]
    for ax, (skey, pkey, nm) in zip(axs, metrics_map):
        if pkey == "silhouette":
            vals = np.array([p["LOPO_by_participant"]["silhouette"] for p in perms])
        elif pkey == "accuracy":
            vals = np.array([p["LOPO_by_participant"]["accuracy"] for p in perms])
        else:
            vals = np.array([p[pkey] for p in perms])
        s = summary[skey]
        ax.scatter(vals, np.full(len(vals), 0.5), s=110, color="#9ecae1",
                   edgecolor="#3182bd", zorder=3, label=f"permutations (n={len(vals)})")
        ax.axvline(s["real"], color="#D55E00", lw=2.6, zorder=4, label=f"real = {s['real']:.4f}")
        ax.axvline(s["null_mean"], color="#333333", ls="--", lw=1.5,
                   label=f"null = {s['null_mean']:.4f}")
        if skey == "LOPO_accuracy":
            ax.axvline(prov["majority_chance"], color="#999999", ls=":", lw=1.5,
                       label=f"chance = {prov['majority_chance']:.3f}")
        ax.set_yticks([]); ax.set_ylim(0, 1); ax.set_xlabel(nm)
        ax.set_title(f"{nm}\np = {s['empirical_p']:.3f}", fontsize=10)
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(f"Female valence: dyad/participant-level permutation control "
                 f"(CEBRA fully retrained per permutation)\n{cap}", fontsize=10)
    fig.tight_layout(); save(fig, "female_valence_permutation_null.png")

    print("\nverification:")
    for p in sorted(OUT.glob("female_valence_*.png")):
        d = open(p, "rb").read(33)
        sig = d[:8] == b"\x89PNG\r\n\x1a\n"
        w_, h_ = struct.unpack(">II", d[16:24])
        fh = open(p, "rb"); fh.seek(-12, 2); e = fh.read(); fh.close()
        print(f"  {'OK ' if sig and b'IEND' in e else 'BAD'} {p.name:42s} "
              f"{w_:5d}x{h_:<5d} {p.stat().st_size/1024:7.0f} KB")


if __name__ == "__main__":
    main()
