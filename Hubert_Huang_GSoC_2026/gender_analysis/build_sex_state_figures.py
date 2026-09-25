#!/usr/bin/env python
"""
gender_analysis/build_sex_state_figures.py
===========================================
Publication-ready figures for the female-only and male-only speak/listen/rest
CEBRA analyses (run_sex_specific_states.py), plus a female-vs-male comparison
panel. Lon/lat convention matches run_gender_dyad_cebra.py exactly: median-
center the embedding, arctan2 to spherical coordinates, then re-center
longitude on the circular mean so the wrap seam sits away from the data.

Unlike the other evaluation/ figure builders, this script uses fixed paths
relative to the current working directory rather than --results-root, since
it only ever reads run_sex_specific_states.py's fixed output layout. Run it
from the directory that contains `results/`, after that script has produced:
  results/gender_analysis/part2/female_states/
  results/gender_analysis/part2/male_states/
Figures are written to results/gender_analysis/part2/figures/.
"""
from __future__ import annotations
import csv, json, struct
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

STATES = ["speak", "listen", "rest"]
COL = {"speak": "#0173B2", "listen": "#DE8F05", "rest": "#029E73"}
ROOT = Path("results/gender_analysis/part2")


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


def load_run(sex_dir):
    prov = json.load(open(sex_dir / "dataset_provenance.json"))
    real = json.load(open(sex_dir / "metrics.json"))
    sub = np.load(sex_dir / "eval_indices.npy")
    emb = np.load(sex_dir / "embedding.npy")
    meta = list(csv.DictReader(open(sex_dir / "sample_metadata.csv")))
    st_i = {s: i for i, s in enumerate(STATES)}
    cls = np.zeros(len(emb), np.int64)
    for m in meta:
        cls[int(m["start"]):int(m["end"])] = st_i[m["state"]]
    Y = cls[sub]
    E = emb[sub]
    lon, lat = lonlat(E)
    return prov, real, E, Y, lon, lat


def save(fig, out, name):
    fig.savefig(out / name, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def confusion_fig(real, prov, out, name, sex_label):
    cm = np.array(real["state_decoding_LOPO"]["confusion_matrix"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(STATES)
    ax.set_yticks(range(3)); ax.set_yticklabels(STATES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    lopo = real["state_decoding_LOPO"]
    ax.set_title(f"{sex_label}-only speak/listen/rest, leave-one-participant-out 5-NN\n"
                 f"acc={lopo['accuracy']:.3f}  bal={lopo['balanced_accuracy']:.3f}  "
                 f"chance={lopo['majority_chance']:.3f}\n"
                 f"n={prov['n_participants']} participants, {prov['n_dyads']} dyads",
                 fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="row-normalised")
    fig.tight_layout(); save(fig, out, name)


def lonlat_fig(lon, lat, Y, prov, out, name, sex_label):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    for i, s in enumerate(STATES):
        a = axs[i]
        a.scatter(lon, lat, s=2, alpha=0.05, color="#BBBBBB", linewidths=0,
                  rasterized=True)
        m = Y == i
        a.scatter(lon[m], lat[m], s=4, alpha=0.4, color=COL[s], linewidths=0,
                  rasterized=True)
        if m.sum():
            a.plot(np.median(lon[m]), np.median(lat[m]), marker="X", ms=14,
                   mfc=COL[s], mec="black", mew=1.5, ls="none")
        a.set_xlim(-180, 180); a.set_ylim(-90, 90)
        a.set_title(f"{s}  —  n={int(m.sum()):,} ({100*m.mean():.1f}%)", fontsize=10)
        a.set_xlabel("Longitude")
        a.grid(alpha=0.15)
        if i == 0:
            a.set_ylabel("Latitude")
    fig.suptitle(f"{sex_label}-only: longitude/latitude by state "
                 f"({prov['n_participants']} participants, {prov['n_dyads']} dyads)\n"
                 f"grey = all samples; X = state median", fontsize=12)
    fig.tight_layout(); save(fig, out, name)


def pairwise_fig(real, prov, out, name, sex_label):
    pw = real["pairwise_LOPO"]
    names = [r["pair"] for r in pw]
    xs = np.arange(len(names)); w = 0.36
    acc = [r.get("accuracy", np.nan) for r in pw]
    base = [r.get("majority_baseline", np.nan) for r in pw]
    delta = [r.get("delta_above_baseline", np.nan) for r in pw]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axs[0].bar(xs - w / 2, acc, w, label="LOPO accuracy", color="#0173B2")
    axs[0].bar(xs + w / 2, base, w, label="majority baseline", color="#BBBBBB")
    axs[0].set_xticks(xs); axs[0].set_xticklabels(names, fontsize=9)
    axs[0].set_ylabel("accuracy"); axs[0].set_ylim(0, 1.05)
    axs[0].set_title(f"{sex_label}-only pairwise state separability", fontsize=11)
    axs[0].legend(fontsize=8)
    for i, r in enumerate(pw):
        if "accuracy" in r:
            axs[0].text(i - w / 2, r["accuracy"] + 0.02, f"{r['accuracy']:.3f}",
                        ha="center", fontsize=7.5)
    cols = ["#029E73" if (d == d and d > 0) else "#D55E00" for d in delta]
    axs[1].bar(xs, delta, 0.5, color=cols)
    axs[1].axhline(0, color="black", lw=1)
    axs[1].set_xticks(xs); axs[1].set_xticklabels(names, fontsize=9)
    axs[1].set_ylabel("accuracy - majority baseline")
    axs[1].set_title("delta above baseline (positive = separable)", fontsize=11)
    for i, d in enumerate(delta):
        if d == d:
            axs[1].text(i, d + (0.004 if d >= 0 else -0.012), f"{d:+.3f}",
                        ha="center", fontsize=8)
    fig.suptitle(f"{sex_label}-only ({prov['n_participants']} participants, "
                 f"{prov['n_dyads']} dyads), leave-one-participant-out", fontsize=10)
    fig.tight_layout(); save(fig, out, name)


def comparison_fig(fprov, freal, mprov, mreal, out, name):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))

    labels = ["Female-only", "Male-only"]
    n_ppl = [fprov["n_participants"], mprov["n_participants"]]
    n_dy = [fprov["n_dyads"], mprov["n_dyads"]]
    lopo = [freal["state_decoding_LOPO"], mreal["state_decoding_LOPO"]]
    xs = np.arange(2)

    axs[0].bar(xs - 0.2, [l["accuracy"] for l in lopo], 0.4,
               label="LOPO accuracy", color="#0173B2")
    axs[0].bar(xs + 0.2, [l["majority_chance"] for l in lopo], 0.4,
               label="majority chance", color="#BBBBBB")
    axs[0].set_xticks(xs); axs[0].set_xticklabels(
        [f"{lb}\n(n={n})" for lb, n in zip(labels, n_ppl)])
    axs[0].set_ylim(0, 1.0); axs[0].set_ylabel("3-class state accuracy")
    axs[0].set_title("Overall state decoding (LOPO)", fontsize=10)
    for i, l in enumerate(lopo):
        axs[0].text(i - 0.2, l["accuracy"] + 0.02, f"{l['accuracy']:.3f}",
                    ha="center", fontsize=8)
    axs[0].legend(fontsize=8)

    xr = np.arange(3)
    axs[1].bar(xr - 0.2, [lopo[0]["per_class_recall"][s] for s in STATES], 0.4,
               label="Female-only", color="#DE8F05")
    axs[1].bar(xr + 0.2, [lopo[1]["per_class_recall"][s] for s in STATES], 0.4,
               label="Male-only", color="#0173B2")
    axs[1].set_xticks(xr); axs[1].set_xticklabels(STATES)
    axs[1].set_ylabel("recall"); axs[1].set_ylim(0, 1.0)
    axs[1].set_title("Per-state recall (LOPO)", fontsize=10)
    axs[1].legend(fontsize=8)

    sim_f = freal["within_person_state_similarity"]
    sim_m = mreal["within_person_state_similarity"]
    pairs = list(sim_f.keys())
    xr2 = np.arange(len(pairs))
    top5_f = [sim_f[p].get("top5_frac", np.nan) for p in pairs]
    top5_m = [sim_m[p].get("top5_frac", np.nan) for p in pairs]
    chance_f = [sim_f[p].get("chance_top5", np.nan) for p in pairs]
    axs[2].bar(xr2 - 0.2, top5_f, 0.4, label="Female-only", color="#DE8F05")
    axs[2].bar(xr2 + 0.2, top5_m, 0.4, label="Male-only", color="#0173B2")
    axs[2].plot(xr2, chance_f, "k_", ms=20, mew=2, label="chance (female N)")
    axs[2].set_xticks(xr2); axs[2].set_xticklabels(pairs, fontsize=8, rotation=15)
    axs[2].set_ylabel("top-5 rank fraction")
    axs[2].set_title("Within-person state similarity: top-5", fontsize=10)
    axs[2].legend(fontsize=7)

    fig.suptitle(f"Female-only (n={n_ppl[0]} ppts, {n_dy[0]} dyads) vs "
                 f"Male-only (n={n_ppl[1]} ppts, {n_dy[1]} dyads) — speak/listen/rest",
                 fontsize=12)
    fig.tight_layout(); save(fig, out, name)


def main():
    out = Path("results/gender_analysis/part2/figures")
    out.mkdir(parents=True, exist_ok=True)

    fprov, freal, fE, fY, flon, flat = load_run(ROOT / "female_states")
    mprov, mreal, mE, mY, mlon, mlat = load_run(ROOT / "male_states")

    confusion_fig(freal, fprov, out, "female_state_confusion_matrix.png", "Female")
    lonlat_fig(flon, flat, fY, fprov, out, "female_state_lonlat.png", "Female")
    pairwise_fig(freal, fprov, out, "female_state_pairwise.png", "Female")

    confusion_fig(mreal, mprov, out, "male_state_confusion_matrix.png", "Male")
    lonlat_fig(mlon, mlat, mY, mprov, out, "male_state_lonlat.png", "Male")
    pairwise_fig(mreal, mprov, out, "male_state_pairwise.png", "Male")

    comparison_fig(fprov, freal, mprov, mreal, out, "sex_state_comparison.png")

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
