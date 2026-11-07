#!/usr/bin/env python
"""
scripts/analyze_cebra_cross_entropy_3d.py
=========================================
Adaptation of the Roca et al. (2023, Cell Reports Methods) cross-entropy test to
compare two 3D CEBRA embeddings.

Roca's method (STAR Methods), faithfully adapted
------------------------------------------------
Roca et al. compare two low-dimensional embeddings by computing, for EACH point,
a local cross-entropy value, then comparing the two resulting per-point
distributions with a Kolmogorov-Smirnov (KS) test. The KS statistic is reported
as the LN (L-infinity) distance:  LN(F,G) = max_x |F(x) - G(x)|  (Durbin KS).

Original paper computes, per point i, a local pair-probability over neighbours
using a Cauchy kernel q_ij ∝ (1 + ||y_i - y_j||^2)^-1 (Eq. 4/13), normalised so
the per-point sum ≈ 1, then a per-point (cross-)entropy H_i = -Σ_j q*_ij log q*_ij
(Eq. 10/11). t-SNE fixes the *entropy* of the high-dim distribution P, so all
the discriminative signal lives in this per-point cross-entropy distribution.

CEBRA gives us the embedding directly (no separate high-dim P / perplexity step),
so we adapt by computing each point's local-neighbourhood Cauchy distribution
over its k nearest neighbours and the per-point entropy of that distribution.
Two embeddings with the same local geometry/density yield similar per-point
distributions (small LN/KS); different geometry yields larger LN/KS. This is the
"local-neighbourhood version of Roca's per-point cross-entropy" the task asks for.

CAUTION (from Roca's Limitations): with many points, KS p-values become
significant for tiny differences. Therefore we report BOTH the LN/KS distance
(a magnitude, less sensitive to n) AND the p-value, use matched sample sizes,
and recommend interpreting against control comparisons — never the p-value alone.

Usage
-----
  python scripts/analyze_cebra_cross_entropy_3d.py \
    --embedding-a results/stacked_cebra_final/dir_real_norm/embedding.npy \
    --embedding-b results/stacked_cebra_final/dir_real_norm_s1/embedding.npy \
    --name-a direction_seed0 --name-b direction_seed1 \
    --max-samples 30000 --k 50 --seed 0 \
    --out-dir results/stacked_cebra_final/cross_entropy_3d/dir_seed0_vs_seed1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import ks_2samp
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    sys.exit("ERROR: needs scipy + scikit-learn.  pip install scipy scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

EPS = 1e-12


def load_embedding(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D array, got shape {arr.shape}")
    # ensure (n_samples, 3): transpose if it came in as (3, n)
    if arr.shape[1] != 3 and arr.shape[0] == 3:
        arr = arr.T
    if arr.shape[1] != 3:
        raise ValueError(f"{path}: expected an (n,3) embedding, got {arr.shape}")
    finite = np.isfinite(arr).all(axis=1)
    if not finite.all():
        print(f"  {path.name}: dropping {int((~finite).sum())} non-finite rows")
    return arr[finite]


def per_point_cross_entropy(Y: np.ndarray, k: int) -> np.ndarray:
    """Roca-adapted per-point cross-entropy on a 3D embedding.

    For each point: Cauchy kernel over its k nearest neighbours,
    w_j = (1 + d_j^2)^-1, normalised to a local probability p_j (sum 1),
    then H_i = -Σ_j p_j log p_j  (numerically stable).
    """
    n = Y.shape[0]
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(Y)   # +1 includes self
    dist, _ = nn.kneighbors(Y)
    dist = dist[:, 1:]                                     # drop self (distance 0)
    w = 1.0 / (1.0 + dist ** 2)                            # Cauchy kernel (Eq. 4)
    p = w / (w.sum(axis=1, keepdims=True) + EPS)           # local probability
    H = -(p * np.log(p + EPS)).sum(axis=1)                 # per-point entropy
    return H


def downsample(Y: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    if n >= Y.shape[0]:
        return Y, np.arange(Y.shape[0])
    idx = rng.choice(Y.shape[0], size=n, replace=False)
    idx.sort()
    return Y[idx], idx


def interpret(ln: float, p: float) -> str:
    mag = ("small" if ln < 0.05 else "moderate" if ln < 0.15 else "large")
    sig = "significant" if p < 0.05 else "not significant"
    return (f"LN/KS distance is {mag} ({ln:.4f}); KS p={p:.3g} ({sig}). "
            f"Interpret LN magnitude alongside controls; p-value alone is not "
            f"biological meaning (huge n inflates significance).")


def main():
    ap = argparse.ArgumentParser(description="Roca cross-entropy/KS test on two 3D CEBRA embeddings.")
    ap.add_argument("--embedding-a", required=True)
    ap.add_argument("--embedding-b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--max-samples", type=int, default=30000)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=15,
                    help="matched-downsample repeats for robustness (0 to disable)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    Ya = load_embedding(Path(args.embedding_a))
    Yb = load_embedding(Path(args.embedding_b))
    print(f"Loaded A={args.name_a} {Ya.shape}  B={args.name_b} {Yb.shape}")

    n_match = min(args.max_samples, Ya.shape[0], Yb.shape[0])

    # ---- primary comparison (single matched downsample, fixed seed) ----
    rng = np.random.RandomState(args.seed)
    Ya_s, idx_a = downsample(Ya, n_match, rng)
    Yb_s, idx_b = downsample(Yb, n_match, rng)
    np.save(out / "sampled_idx_a.npy", idx_a)
    np.save(out / "sampled_idx_b.npy", idx_b)

    Ha = per_point_cross_entropy(Ya_s, args.k)
    Hb = per_point_cross_entropy(Yb_s, args.k)
    np.save(out / "cross_entropy_a.npy", Ha)
    np.save(out / "cross_entropy_b.npy", Hb)

    ks = ks_2samp(Ha, Hb)
    ln = float(ks.statistic)
    pval = float(ks.pvalue)

    # ---- bootstrap robustness (repeat matched downsampling) ----
    boot = []
    for b in range(args.bootstrap):
        rb = np.random.RandomState(args.seed + 1000 + b)
        a_s, _ = downsample(Ya, n_match, rb)
        b_s, _ = downsample(Yb, n_match, rb)
        ha = per_point_cross_entropy(a_s, args.k)
        hb = per_point_cross_entropy(b_s, args.k)
        kb = ks_2samp(ha, hb)
        boot.append((float(kb.statistic), float(kb.pvalue)))
    boot = np.array(boot) if boot else np.empty((0, 2))

    summary = {
        "name_a": args.name_a, "name_b": args.name_b,
        "embedding_a": str(args.embedding_a), "embedding_b": str(args.embedding_b),
        "n_a_total": int(Ya.shape[0]), "n_b_total": int(Yb.shape[0]),
        "n_used": int(n_match), "k": int(args.k), "seed": int(args.seed),
        "ks_statistic": ln, "ln_distance": ln, "p_value": pval,
        "mean_H_a": float(Ha.mean()), "mean_H_b": float(Hb.mean()),
        "interpretation": interpret(ln, pval),
    }
    if boot.size:
        summary.update({
            "bootstrap_n": int(boot.shape[0]),
            "ln_mean": float(boot[:, 0].mean()), "ln_sd": float(boot[:, 0].std()),
            "p_mean": float(boot[:, 1].mean()), "p_median": float(np.median(boot[:, 1])),
        })

    with open(out / "cross_entropy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / "cross_entropy_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(list(summary.keys())); w.writerow(list(summary.values()))
    if boot.size:
        with open(out / "cross_entropy_robustness.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["rep", "ln", "p_value"])
            for i, (l, p) in enumerate(boot):
                w.writerow([i, l, p])

    # ---- plots ----
    if HAVE_PLT:
        # CDF
        fig, ax = plt.subplots(figsize=(7, 5))
        for H, nm, c in [(Ha, args.name_a, "C0"), (Hb, args.name_b, "C1")]:
            xs = np.sort(H); ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, label=nm, color=c, lw=1.5)
        ax.set_xlabel("per-point cross-entropy"); ax.set_ylabel("CDF")
        ax.set_title(f"Cross-entropy CDF\n{args.name_a} vs {args.name_b}\n"
                     f"LN/KS={ln:.4f}, p={pval:.3g}, n={n_match}, k={args.k}")
        ax.legend()
        fig.savefig(out / "cross_entropy_cdf.png", dpi=140, bbox_inches="tight"); plt.close(fig)
        # histogram / density
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(Ha, bins=60, density=True, alpha=0.5, label=args.name_a, color="C0")
        ax.hist(Hb, bins=60, density=True, alpha=0.5, label=args.name_b, color="C1")
        ax.set_xlabel("per-point cross-entropy"); ax.set_ylabel("density")
        ax.set_title(f"Cross-entropy distributions (LN/KS={ln:.4f}, p={pval:.3g})")
        ax.legend()
        fig.savefig(out / "cross_entropy_hist.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out}")
    return summary


if __name__ == "__main__":
    main()
