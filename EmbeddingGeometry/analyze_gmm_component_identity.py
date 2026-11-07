#!/usr/bin/env python
"""
scripts/analyze_gmm_component_identity.py
=========================================
TASK 6 -- what is the BIC-preferred GMM actually clustering?

The full-band speaker-first run preferred K=5 over K=2. Two competing stories:
  (a) K=5 reflects finer AQ structure (exact |dAQ| levels), or
  (b) K=5 is just DYAD IDENTITY leaking into the embedding.
This script decides between them WITHOUT retraining CEBRA.

Method notes that matter:
  * Longitude is CIRCULAR. Fitting a plain 2D Gaussian on (lon, lat) lets the
    +/-180 seam manufacture spurious components. We therefore find the LARGEST
    EMPTY CIRCULAR GAP in longitude and unwrap at that cut before the primary
    fit, and we save the chosen cut angle.
  * Sensitivity analysis: refit on wrap-safe features
    (cos lon, sin lon, lat_scaled) so no seam exists at all.
  * GMMs are fit LABEL-AGNOSTICALLY. Labels are used only afterwards to score.
  * Subsampling is deterministic and dyad-balanced (equal samples per dyad).
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_score, completeness_score)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

C0, C1 = "#0173B2", "#DE8F05"


def wrap180(x):
    return (x + 180) % 360 - 180


def largest_gap_cut(lon_deg, nbins=720):
    """Return the longitude angle at the centre of the largest EMPTY circular gap.
    Unwrapping there guarantees the seam sits where there is no data."""
    h, edges = np.histogram(wrap180(lon_deg), bins=nbins, range=(-180, 180))
    empty = h == 0
    if not empty.any():
        # no strictly empty bin -> use the sparsest run instead
        k = max(1, nbins // 36)
        run = np.convolve(np.r_[h, h], np.ones(k), "valid")[:nbins]
        i = int(np.argmin(run))
        return float(wrap180(edges[i] + (k / 2) * (360 / nbins)))
    # find longest circular run of empty bins
    doubled = np.r_[empty, empty]
    best_len = best_start = 0
    cur = 0
    for i, v in enumerate(doubled):
        cur = cur + 1 if v else 0
        if cur > best_len:
            best_len, best_start = cur, i - cur + 1
    if best_start >= nbins:
        best_start -= nbins
    centre_bin = best_start + best_len / 2.0
    return float(wrap180(-180 + centre_bin * (360 / nbins)))


def unwrap_at(lon_deg, cut):
    """Re-express longitude on a linear axis whose discontinuity sits at `cut`."""
    return wrap180(lon_deg - cut)


def purity(labels_true, cl):
    tot = 0
    for c in np.unique(cl):
        m = cl == c
        vals, cnts = np.unique(labels_true[m], return_counts=True)
        tot += cnts.max()
    return float(tot / len(labels_true))


def dyads_per_component(dyad, cl, frac=0.10):
    """How many dyads are 'substantially represented' (>=frac of the component)."""
    out = {}
    for c in np.unique(cl):
        m = cl == c
        vals, cnts = np.unique(dyad[m], return_counts=True)
        share = cnts / cnts.sum()
        out[int(c)] = {"n_dyads_present": int(len(vals)),
                       "n_dyads_ge_10pct": int((share >= frac).sum()),
                       "top_dyad": int(vals[np.argmax(cnts)]),
                       "top_dyad_share": float(share.max())}
    return out


def score_against(cl, y, name):
    return {f"ARI_vs_{name}": float(adjusted_rand_score(y, cl)),
            f"NMI_vs_{name}": float(normalized_mutual_info_score(y, cl)),
            f"homogeneity_vs_{name}": float(homogeneity_score(y, cl)),
            f"completeness_vs_{name}": float(completeness_score(y, cl)),
            f"purity_vs_{name}": purity(y, cl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--sample-metadata", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--per-dyad", type=int, default=2000, help="samples per dyad (balanced)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="fullband_speakerfirst")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # ---- labels per file from the manifest ----
    man = {}
    for r in csv.DictReader(open(args.manifest)):
        man[r["output_npy"]] = {"dyad": int(r["dyad_id"]), "mag": int(r["aq_magnitude"]),
                                "daq": int(float(r["abs_daq"]))}
    meta = list(csv.DictReader(open(args.sample_metadata)))

    emb = np.load(args.embedding).astype(np.float32)
    print(f"embedding {emb.shape}", flush=True)

    # ---- deterministic, dyad-balanced subsample ----
    by_dyad = {}
    for m in meta:
        d = int(m["dyad"]); by_dyad.setdefault(d, []).append(m)
    idx_all, y_mag, y_daq, y_dyad = [], [], [], []
    for d in sorted(by_dyad):
        segs = by_dyad[d]
        per_seg = max(1, args.per_dyad // len(segs))
        for m in segs:
            s, e = int(m["start"]), int(m["end"])
            info = man.get(m["npy"], {"mag": int(m["mag"]), "daq": -1})
            take = rng.choice(np.arange(s, e), size=min(per_seg, e - s), replace=False)
            idx_all.append(take)
            y_mag.append(np.full(len(take), info["mag"], np.int64))
            y_daq.append(np.full(len(take), info["daq"], np.int64))
            y_dyad.append(np.full(len(take), d, np.int64))
    idx = np.concatenate(idx_all)
    order = np.argsort(idx)
    idx = idx[order]
    y_mag = np.concatenate(y_mag)[order]
    y_daq = np.concatenate(y_daq)[order]
    y_dyad = np.concatenate(y_dyad)[order]
    np.save(out / "sample_indices.npy", idx)
    E = emb[idx]
    n_dyads = len(set(y_dyad.tolist()))
    print(f"balanced subsample: {len(idx)} samples over {n_dyads} dyads "
          f"({args.per_dyad}/dyad)", flush=True)

    # ---- spherical coords ----
    Ec = E - np.median(E, axis=0)
    x_, y_, z_ = Ec[:, 0], Ec[:, 1], Ec[:, 2]
    r = np.sqrt(x_**2 + y_**2 + z_**2)
    lon = np.degrees(np.arctan2(y_, x_))
    lat = np.degrees(np.arctan2(z_, np.sqrt(x_**2 + y_**2)))

    cut = largest_gap_cut(lon)
    lon_u = unwrap_at(lon, cut)
    print(f"longitude unwrap cut at {cut:.2f} deg (largest empty circular gap)", flush=True)
    np.save(out / "lonlat_radius.npy", np.c_[lon_u, lat, r])

    feat_primary = np.c_[lon_u, lat]
    # wrap-safe sensitivity features: unit circle for lon + scaled lat
    feat_safe = np.c_[np.cos(np.radians(lon)), np.sin(np.radians(lon)), lat / 90.0]

    # held-out split (grouped by dyad so likelihood is not inflated by within-dyad copies)
    dy = np.array(sorted(set(y_dyad.tolist())))
    hold = set(dy[::4].tolist())
    te_m = np.isin(y_dyad, list(hold)); tr_m = ~te_m

    results = {}
    for tag, feat in [("primary_lonlat_unwrapped", feat_primary), ("wrapsafe_unitvec", feat_safe)]:
        rows = []
        print(f"\n--- GMM sweep [{tag}] ---", flush=True)
        for k in range(1, args.kmax + 1):
            gm = GaussianMixture(k, covariance_type="full", random_state=args.seed,
                                 n_init=args.n_init, max_iter=500, reg_covar=1e-6).fit(feat)
            cl = gm.predict(feat)
            gmh = GaussianMixture(k, covariance_type="full", random_state=args.seed,
                                  n_init=max(2, args.n_init // 3), max_iter=500,
                                  reg_covar=1e-6).fit(feat[tr_m])
            row = {"K": k, "BIC": float(gm.bic(feat)), "AIC": float(gm.aic(feat)),
                   "train_ll": float(gm.score(feat)),
                   "heldout_ll": float(gmh.score(feat[te_m])),
                   "converged": bool(gm.converged_), "n_iter": int(gm.n_iter_),
                   "min_weight": float(gm.weights_.min()),
                   "max_cov_cond": float(max(np.linalg.cond(c) for c in gm.covariances_))}
            row.update(score_against(cl, y_mag, "aq_magnitude"))
            row.update(score_against(cl, y_daq, "aq_delta"))
            row.update(score_against(cl, y_dyad, "dyad"))
            rows.append(row)
            print(f"  K={k:2d} BIC={row['BIC']:12.1f} heldLL={row['heldout_ll']:7.3f} "
                  f"ARI_mag={row['ARI_vs_aq_magnitude']:.3f} ARI_daq={row['ARI_vs_aq_delta']:.3f} "
                  f"ARI_dyad={row['ARI_vs_dyad']:.3f} pur_dyad={row['purity_vs_dyad']:.3f}",
                  flush=True)
            if tag == "primary_lonlat_unwrapped" and k in (2, 5):
                np.save(out / f"K{k}_assignments.npy", cl)
                json.dump({"K": k, "weights": gm.weights_.tolist(),
                           "means_lon_lat": gm.means_.tolist(),
                           "covariances": [c.tolist() for c in gm.covariances_],
                           "dyads_per_component": dyads_per_component(y_dyad, cl),
                           "converged": bool(gm.converged_)},
                          open(out / f"K{k}_components.json", "w"), indent=2)
        results[tag] = rows
        with open(out / f"gmm_sweep_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- K=2 stability across seeds / subsamples ----
    stab = []
    for s in range(5):
        sub = np.random.RandomState(100 + s).choice(len(feat_primary),
                                                    int(0.7 * len(feat_primary)), replace=False)
        gm = GaussianMixture(2, covariance_type="full", random_state=s, n_init=args.n_init,
                             max_iter=500).fit(feat_primary[sub])
        cl = gm.predict(feat_primary)
        stab.append({"seed": s, "ARI_vs_magnitude": float(adjusted_rand_score(y_mag, cl)),
                     "purity_vs_magnitude": purity(y_mag, cl),
                     "ARI_vs_dyad": float(adjusted_rand_score(y_dyad, cl)),
                     "weights": gm.weights_.tolist()})
    with open(out / "K2_stability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "ARI_vs_magnitude", "purity_vs_magnitude",
                                          "ARI_vs_dyad", "weights"])
        w.writeheader(); w.writerows(stab)

    prim = results["primary_lonlat_unwrapped"]
    safe = results["wrapsafe_unitvec"]
    bestK_bic = min(prim, key=lambda r: r["BIC"])["K"]
    bestK_aic = min(prim, key=lambda r: r["AIC"])["K"]
    bestK_held = max(prim, key=lambda r: r["heldout_ll"])["K"]
    bestK_bic_safe = min(safe, key=lambda r: r["BIC"])["K"]
    r5 = next(r for r in prim if r["K"] == 5)
    r2 = next(r for r in prim if r["K"] == 2)

    # ---------------- figures ----------------
    ks = [r["K"] for r in prim]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, [r["BIC"] for r in prim], "o-", label="BIC", color=C0)
    ax[0].plot(ks, [r["AIC"] for r in prim], "s--", label="AIC", color=C1)
    ax[0].axvline(bestK_bic, ls=":", c="k", lw=1)
    ax[0].set_xlabel("K (GMM components)"); ax[0].set_ylabel("criterion (lower = better)")
    ax[0].set_title(f"GMM model selection ({args.label})\nBIC-preferred K={bestK_bic}")
    ax[0].legend(); ax[0].set_xticks(ks[::2])
    ax[1].plot(ks, [r["heldout_ll"] for r in prim], "o-", color="#029E73")
    ax[1].axvline(bestK_held, ls=":", c="k", lw=1)
    ax[1].set_xlabel("K"); ax[1].set_ylabel("held-out log-likelihood (dyad-grouped)")
    ax[1].set_title(f"Held-out likelihood, best K={bestK_held}")
    fig.tight_layout(); fig.savefig(out / "gmm_k_sweep.png", dpi=300, facecolor="white",
                                    bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ks, [r["ARI_vs_aq_magnitude"] for r in prim], "o-", label="vs AQ magnitude (2 cls)", color=C0)
    ax.plot(ks, [r["ARI_vs_aq_delta"] for r in prim], "s-", label="vs exact |dAQ|", color=C1)
    ax.plot(ks, [r["ARI_vs_dyad"] for r in prim], "^-", label="vs dyad identity", color="#D55E00")
    ax.set_xlabel("K"); ax.set_ylabel("Adjusted Rand Index"); ax.set_xticks(ks[::2])
    ax.set_title(f"What do the GMM components track? ({args.label})")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "gmm_ari_nmi_vs_k.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    for k in (2, 5):
        cl = np.load(out / f"K{k}_assignments.npy")
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        cmap = plt.get_cmap("tab10", k)
        for c in range(k):
            m = cl == c
            ax.scatter(lon_u[m], lat[m], s=4, alpha=0.35, color=cmap(c),
                       label=f"comp {c} ({m.mean()*100:.0f}%)", linewidths=0)
        ax.set_xlabel(f"Longitude (unwrapped at {cut:.0f}°)"); ax.set_ylabel("Latitude")
        ax.set_title(f"{args.label}: label-agnostic GMM, K={k}")
        ax.legend(markerscale=4, fontsize=8, loc="best")
        fig.tight_layout(); fig.savefig(out / f"gmm_k{k}_components.png", dpi=300,
                                        facecolor="white", bbox_inches="tight"); plt.close(fig)

    # heatmaps for K=5
    cl5 = np.load(out / "K5_assignments.npy")
    for lab, yv, fname, xlabel in [("dyad", y_dyad, "gmm_k5_vs_dyad.png", "dyad id"),
                                   ("AQ delta", y_daq, "gmm_k5_vs_aqdelta.png", "|dAQ|")]:
        cats = np.array(sorted(set(yv.tolist())))
        M = np.zeros((5, len(cats)))
        for i in range(5):
            for j, c in enumerate(cats):
                M[i, j] = np.sum((cl5 == i) & (yv == c))
        Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(max(6.5, 0.34 * len(cats) + 3), 3.6))
        im = ax.imshow(Mn, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=7, rotation=90)
        ax.set_yticks(range(5)); ax.set_yticklabels([f"comp {i}" for i in range(5)])
        ax.set_xlabel(xlabel); ax.set_title(f"{args.label}: K=5 GMM component composition by {lab}")
        fig.colorbar(im, ax=ax, label="row-normalised share")
        fig.tight_layout(); fig.savefig(out / fname, dpi=300, facecolor="white",
                                        bbox_inches="tight"); plt.close(fig)

    # ---------------- answers ----------------
    d5 = json.load(open(out / "K5_components.json"))["dyads_per_component"]
    single_dom = sum(1 for v in d5.values() if v["top_dyad_share"] >= 0.5)
    knear = [r for r in prim if abs(r["K"] - n_dyads) <= 2]
    bic5 = r5["BIC"]; bic_min = min(r["BIC"] for r in prim)
    bic_span = max(r["BIC"] for r in prim) - bic_min
    gain_near = ((bic5 - min(r["BIC"] for r in knear)) / bic_span * 100) if knear and bic_span else None
    ari_stab = [s["ARI_vs_magnitude"] for s in stab]

    answers = {
        "Q1_K5_aligns_with": ("dyad identity" if r5["ARI_vs_dyad"] > max(r5["ARI_vs_aq_magnitude"],
                              r5["ARI_vs_aq_delta"]) else "AQ difference"),
        "Q1_evidence": {"K5_ARI_vs_dyad": r5["ARI_vs_dyad"],
                        "K5_ARI_vs_aq_magnitude": r5["ARI_vs_aq_magnitude"],
                        "K5_ARI_vs_aq_delta": r5["ARI_vs_aq_delta"],
                        "K5_purity_vs_dyad": r5["purity_vs_dyad"],
                        "K5_purity_vs_aq_magnitude": r5["purity_vs_aq_magnitude"]},
        "Q2_components_dominated_by_single_dyad": f"{single_dom}/5 components have a dyad "
                                                  f"holding >=50% of their mass",
        "Q2_detail": d5,
        "Q3_K_near_n_dyads_major_BIC_gain": {
            "n_dyads": n_dyads, "BIC_at_K5": bic5,
            "best_BIC_for_K_near_n_dyads": (min(r["BIC"] for r in knear) if knear else None),
            "pct_of_total_BIC_range_gained": (round(gain_near, 1) if gain_near is not None else None),
            "note": f"K sweep only reaches {args.kmax}; n_dyads={n_dyads}."
                    + ("" if knear else " K never reaches n_dyads, so this is extrapolation-limited.")},
        "Q4_K2_stable_across_seeds": {"ARI_vs_magnitude_mean": float(np.mean(ari_stab)),
                                      "ARI_vs_magnitude_std": float(np.std(ari_stab)),
                                      "range": [float(min(ari_stab)), float(max(ari_stab))],
                                      "stable": bool(np.std(ari_stab) < 0.05)},
        "Q5_wrapsafe_changes_result": {"bestK_BIC_primary": bestK_bic,
                                       "bestK_BIC_wrapsafe": bestK_bic_safe,
                                       "changed": bestK_bic != bestK_bic_safe},
        "longitude_cut_deg": cut,
        "bestK_by_BIC": bestK_bic, "bestK_by_AIC": bestK_aic,
        "bestK_by_heldout_ll": bestK_held,
        "n_dyads": n_dyads, "n_samples_used": int(len(idx)),
    }
    json.dump(answers, open(out / "identity_answers.json", "w"), indent=2)

    # ---------------- markdown ----------------
    L = [f"# GMM component identity check -- {args.label}", "",
         f"Label-agnostic GMM sweep K=1..{args.kmax}, {args.n_init} inits, full covariance, "
         f"on a deterministic dyad-balanced subsample "
         f"({args.per_dyad}/dyad, {len(idx)} samples, {n_dyads} dyads).", "",
         f"Longitude is circular; the primary fit unwraps at **{cut:.1f}°**, the centre of the "
         "largest empty circular gap, so the seam carries no data. A wrap-safe "
         "(cos lon, sin lon, lat) fit is reported as sensitivity analysis.", "",
         "## Model selection", "",
         f"- BIC-preferred **K = {bestK_bic}**", f"- AIC-preferred **K = {bestK_aic}**",
         f"- Held-out (dyad-grouped) log-likelihood preferred **K = {bestK_held}**",
         f"- Wrap-safe features BIC-preferred **K = {bestK_bic_safe}**", "",
         "## The five questions", "",
         f"**1. Does K=5 align more with dyad identity or AQ?** "
         f"→ **{answers['Q1_K5_aligns_with']}**. "
         f"At K=5: ARI vs dyad = {r5['ARI_vs_dyad']:.3f}, vs |dAQ| = {r5['ARI_vs_aq_delta']:.3f}, "
         f"vs AQ magnitude = {r5['ARI_vs_aq_magnitude']:.3f}; "
         f"purity vs dyad = {r5['purity_vs_dyad']:.3f}.", "",
         f"**2. Are components dominated by single dyads?** → {answers['Q2_components_dominated_by_single_dyad']}.", "",
         f"**3. Does K near the dyad count give a major BIC gain?** → "
         f"{'not testable within this sweep' if not knear else f'{gain_near:.1f}% of the total BIC range'} "
         f"(n_dyads={n_dyads}, sweep max K={args.kmax}).", "",
         f"**4. Is the K=2 Low/High split stable?** → ARI vs magnitude "
         f"{np.mean(ari_stab):.3f} ± {np.std(ari_stab):.3f} across 5 seeds/subsamples "
         f"({'stable' if answers['Q4_K2_stable_across_seeds']['stable'] else 'UNSTABLE'}).", "",
         f"**5. Does wrap-safe featurisation change it?** → BIC-preferred K "
         f"{bestK_bic} (primary) vs {bestK_bic_safe} (wrap-safe): "
         f"{'CHANGED' if bestK_bic != bestK_bic_safe else 'unchanged'}.", "",
         "## Sweep table (primary features)", "",
         "| K | BIC | AIC | held-out LL | conv | ARI mag | ARI \\|dAQ\\| | ARI dyad | purity dyad |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in prim:
        L.append(f"| {r['K']} | {r['BIC']:.0f} | {r['AIC']:.0f} | {r['heldout_ll']:.3f} | "
                 f"{'y' if r['converged'] else 'N'} | {r['ARI_vs_aq_magnitude']:.3f} | "
                 f"{r['ARI_vs_aq_delta']:.3f} | {r['ARI_vs_dyad']:.3f} | {r['purity_vs_dyad']:.3f} |")
    (out / "gmm_identity_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L[:40]))
    print("GMM_IDENTITY_DONE")


if __name__ == "__main__":
    main()
