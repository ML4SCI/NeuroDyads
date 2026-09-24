#!/usr/bin/env python
"""
scripts/summarize_permutation.py
================================
Turn the per-permutation metrics into the real-vs-null comparison, the empirical
p-values, the Overleaf table and the null-distribution figures.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# metric -> (pretty name, higher_is_better)
METRICS = [("knn5_leave_one_dyad_out", "Leave-one-dyad-out 5-NN", True),
           ("knn5_grouped_fixed", "Grouped 5-NN (fixed holdout)", True),
           ("goodness_of_fit_bits", "Goodness of fit (bits)", True),
           ("silhouette", "Silhouette", True),
           ("ks_latitude", "KS latitude", True),
           ("ks_longitude", "KS longitude", True),
           ("ks_radius", "KS radius", True),
           ("gmm_K2_ARI", "GMM K=2 ARI", True),
           ("gmm_K2_purity", "GMM K=2 purity", True),
           ("final_loss", "Final InfoNCE loss", False)]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm-dir", required=True)
    ap.add_argument("--real-metrics", required=True,
                    help="metrics.json of the REAL run (for loss/GoF)")
    ap.add_argument("--real-rescored", required=True,
                    help="real_rescored_metrics.json produced by the perm script")
    ap.add_argument("--posthoc", default=None)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    pdir = Path(args.perm_dir)
    perms = []
    for d in sorted(pdir.glob("perm*")):
        m = d / "metrics.json"
        if m.exists():
            perms.append(json.load(open(m)))
    if not perms:
        raise SystemExit("no completed permutations")

    real_run = json.load(open(args.real_metrics))
    rr = json.load(open(args.real_rescored))["metrics"]
    real = dict(rr)
    # loss/GoF come from the original real training run
    real["final_loss"] = real_run.get("final_loss")
    real["goodness_of_fit_bits"] = real_run.get("goodness_of_fit_bits")

    prov = json.load(open(pdir / "dataset_provenance.json"))

    rows = []
    for key, pretty, hib in METRICS:
        nulls = [f(p.get(key)) for p in perms]
        nulls = [x for x in nulls if x is not None]
        rv = f(real.get(key))
        if rv is None or not nulls:
            continue
        na = np.array(nulls)
        if hib:
            cnt = int((na >= rv).sum())
        else:
            cnt = int((na <= rv).sum())
        p_emp = (1 + cnt) / (1 + len(na))
        rows.append({"metric": key, "pretty": pretty, "real": rv,
                     "null_mean": float(na.mean()), "null_std": float(na.std()),
                     "null_min": float(na.min()), "null_max": float(na.max()),
                     "n_null": len(na), "n_null_beating_real": cnt,
                     "empirical_p": p_emp,
                     "delta": rv - float(na.mean()),
                     "z": float((rv - na.mean()) / na.std()) if na.std() > 0 else None,
                     "higher_is_better": hib})
    with open(pdir / "real_vs_null_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    ph = json.load(open(args.posthoc)) if (args.posthoc and Path(args.posthoc).exists()) else None

    # ---------------- figures ----------------
    def nullplot(key, pretty, fname):
        r = next((x for x in rows if x["metric"] == key), None)
        if not r:
            return
        na = np.array([f(p.get(key)) for p in perms if f(p.get(key)) is not None])
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.scatter(na, np.zeros_like(na) + 0.5, s=90, color="#9ecae1",
                   edgecolor="#3182bd", zorder=3, label=f"permutations (n={len(na)})")
        ax.axvline(r["real"], color="#D55E00", lw=2.5, zorder=4,
                   label=f"real = {r['real']:.3f}")
        ax.axvline(r["null_mean"], color="#333333", ls="--", lw=1.5,
                   label=f"null mean = {r['null_mean']:.3f} ± {r['null_std']:.3f}")
        ch = f(prov.get("majority_chance"))
        if key.startswith("knn") and ch:
            ax.axvline(ch, color="#999999", ls=":", lw=1.5, label=f"chance = {ch:.3f}")
        ax.set_yticks([]); ax.set_ylim(0, 1)
        ax.set_xlabel(pretty)
        ax.set_title(f"Dyad-level AQ permutation control — {pretty}\n"
                     f"empirical p = {r['empirical_p']:.3f}  "
                     f"(Δ from null = {r['delta']:+.3f})", fontsize=10)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        fig.tight_layout(); fig.savefig(pdir / fname, dpi=300, facecolor="white",
                                        bbox_inches="tight"); plt.close(fig)

    nullplot("knn5_leave_one_dyad_out", "Leave-one-dyad-out 5-NN accuracy",
             "fullband_speakerfirst_permutation_5nn_null.png")
    nullplot("goodness_of_fit_bits", "Goodness of fit (bits)",
             "fullband_speakerfirst_permutation_gof_null.png")

    # combined panel
    keys = [("knn5_leave_one_dyad_out", "LODO 5-NN"), ("knn5_grouped_fixed", "Grouped 5-NN"),
            ("goodness_of_fit_bits", "GoF (bits)"), ("silhouette", "Silhouette")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (k, nm) in zip(axs, keys):
        r = next((x for x in rows if x["metric"] == k), None)
        if not r:
            ax.axis("off"); continue
        na = np.array([f(p.get(k)) for p in perms if f(p.get(k)) is not None])
        ax.scatter(np.zeros_like(na), na, s=70, color="#9ecae1", edgecolor="#3182bd", zorder=3)
        ax.axhline(r["real"], color="#D55E00", lw=2.2, zorder=4)
        ax.axhline(r["null_mean"], color="#333333", ls="--", lw=1.2)
        ax.set_xticks([]); ax.set_title(f"{nm}\nreal={r['real']:.3f}  p={r['empirical_p']:.3f}",
                                        fontsize=9)
    fig.suptitle("Real (orange) vs dyad-permuted null (blue) — full-band speaker-first",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(pdir / "permutation_panel.png", dpi=300,
                                    facecolor="white", bbox_inches="tight"); plt.close(fig)

    # ---------------- markdown ----------------
    lodo = next(x for x in rows if x["metric"] == "knn5_leave_one_dyad_out")
    verdict = ("**The AQ-magnitude decoding does NOT survive the dyad-level permutation "
               "control.**" if lodo["empirical_p"] > 0.05 else
               "**The AQ-magnitude decoding survives the dyad-level permutation control.**")

    L = ["# Dyad-level AQ permutation control — full-band speaker-first", "",
         "## What was done", "",
         f"The AQ-magnitude label was permuted **across whole dyads** "
         f"({prov['n_dyads']} dyads, {prov['n_files']} speaker-first files, "
         f"{prov['n_samples']:,} samples). Both speaker-first files of a dyad always receive "
         f"the same permuted label, and the number of Low/High **dyads** is preserved exactly, "
         f"so per-class sample counts are identical to the real run "
         f"({prov['class_counts']}, majority chance {prov['majority_chance']:.4f}).", "",
         f"For each of **{len(perms)} permutations** CEBRA was **retrained from scratch** with "
         f"the identical configuration and the identical model-init seed (0) as the real run — "
         f"only the label changed. This is the strong form of the control: it tests whether the "
         f"encoder can build an AQ-shaped geometry out of *any* random grouping of dyads.", "",
         "```", json.dumps(prov["cebra_config"], indent=2), "```", "",
         "## Result", "", verdict, "",
         "| metric | real | null mean ± sd | null range | Δ real−null | z | empirical p |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        z = f"{r['z']:+.2f}" if r["z"] is not None else "–"
        L.append(f"| {r['pretty']} | **{r['real']:.4f}** | {r['null_mean']:.4f} ± "
                 f"{r['null_std']:.4f} | {r['null_min']:.4f}–{r['null_max']:.4f} | "
                 f"{r['delta']:+.4f} | {z} | {r['empirical_p']:.3f} |")
    L += ["", f"Empirical p = (1 + #{{null ≥ real}}) / (1 + n_permutations), "
              f"n_permutations = {len(perms)}. With {len(perms)} permutations the smallest "
              f"attainable p is {1/(1+len(perms)):.3f}.", ""]

    L += ["## Interpretation", ""]
    if lodo["empirical_p"] > 0.05:
        L += [f"Retraining CEBRA on randomly re-assigned dyad labels reproduces the real "
              f"decoding accuracy almost exactly "
              f"(real {lodo['real']:.3f} vs null {lodo['null_mean']:.3f} ± "
              f"{lodo['null_std']:.3f}, p = {lodo['empirical_p']:.3f}). The model does not need "
              f"the true AQ labels to reach this accuracy — an arbitrary partition of the same "
              f"dyads works just as well.", "",
              "The mechanism is dyad identity. Each dyad is entirely Low or entirely High, so a "
              "label that is constant within a dyad can be recovered by recognising the dyad, "
              "and CEBRA's supervised (`time_delta`) objective is free to build exactly that "
              "geometry for whatever grouping it is handed. The previously reported "
              "AQ-magnitude accuracy therefore **cannot be read as evidence of neural AQ "
              "structure**; it is consistent with per-dyad recording identity.", "",
              "This does not say AQ has no neural signature — it says this design cannot "
              "separate an AQ effect from a dyad effect, because AQ is a between-dyad constant. "
              "Distinguishing them needs either within-dyad AQ variation or many more dyads with "
              "the label decorrelated from recording identity."]
    else:
        L += [f"The real run beats every permutation "
              f"(real {lodo['real']:.3f} vs null {lodo['null_mean']:.3f} ± "
              f"{lodo['null_std']:.3f}, p = {lodo['empirical_p']:.3f}): the AQ-magnitude "
              f"geometry is not reproducible from an arbitrary regrouping of the same dyads."]

    if ph:
        L += ["", "## Secondary control — post-hoc decoder permutation", "",
              f"With the real embedding held **fixed** and only the dyad→label map permuted "
              f"({ph['n_perms']} permutations): real {ph['real_lodo_knn5']:.4f} vs null "
              f"{ph['null_mean']:.4f} ± {ph['null_std']:.4f}, empirical p = "
              f"{ph['empirical_p']:.4f}.", "",
              "This is reported for completeness only. It cannot detect label leakage that "
              "happens while the encoder is being trained, which is exactly what the primary "
              "control above measures — so the primary result governs."]

    L += ["", "## Per-permutation detail", "",
          "| run | dyads relabelled | loss | GoF | LODO 5-NN | grouped 5-NN | silhouette | "
          "GMM K2 ARI | best K |", "|---|---|---|---|---|---|---|---|---|"]
    L.append(f"| **real** | 0 | {real.get('final_loss')} | "
             f"{real.get('goodness_of_fit_bits')} | {f(real.get('knn5_leave_one_dyad_out')):.4f} | "
             f"{f(real.get('knn5_grouped_fixed')):.4f} | {f(real.get('silhouette')):.4f} | "
             f"{f(real.get('gmm_K2_ARI')):.4f} | {real.get('gmm_bestK_by_bic')} |")
    for p in perms:
        L.append(f"| {p['run']} | {p.get('n_dyads_relabelled','–')}/{prov['n_dyads']} | "
                 f"{f(p.get('final_loss')):.4f} | {f(p.get('goodness_of_fit_bits')):.4f} | "
                 f"{f(p.get('knn5_leave_one_dyad_out')):.4f} | "
                 f"{f(p.get('knn5_grouped_fixed')):.4f} | {f(p.get('silhouette')):.4f} | "
                 f"{f(p.get('gmm_K2_ARI')):.4f} | {p.get('gmm_bestK_by_bic')} |")

    Path(args.out_md).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:60]))
    print("PERM_SUMMARY_DONE")


if __name__ == "__main__":
    main()
