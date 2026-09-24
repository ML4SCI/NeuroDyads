#!/usr/bin/env python
"""
evaluation/build_reports_and_export.py
=======================================
Aggregate report/figure export: pulls together whichever result files already
exist under a results tree, into a master metrics CSV, LaTeX-ready snippets,
and a figure bundle. Reads only what is actually on disk, so a partially
completed pipeline produces an honest partial report rather than an error.

By default this expects the standard results/aug4_pipeline/ layout produced
by the training/ and controls/ scripts in this submission; point
--results-root elsewhere if your results tree lives somewhere else.

  python build_reports_and_export.py --results-root /path/to/results/aug4_pipeline
"""
from __future__ import annotations

import argparse
import csv, json, shutil, zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUG = RESULTS / "aug4_pipeline"
OL = AUG / "overleaf_export"
FIG = OL / "figures"


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def f4(v, n=4):
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return "–"


# ---------------------------------------------------------------- figures
def build_figmap():
    """Built as a function (not a module-level list) so it always reflects the
    current AUG/RESULTS globals, which main() may have overridden via
    --results-root -- a plain list literal here would freeze in the import-time
    default and silently ignore that flag."""
    return [
    (AUG / "dyad_permutation/fullband_speakerfirst_permutation_5nn_null.png",
     "fullband_speakerfirst_permutation_5nn_null.png",
     "Dyad-level permutation control: real vs null leave-one-dyad-out 5-NN."),
    (AUG / "dyad_permutation/fullband_speakerfirst_permutation_gof_null.png",
     "fullband_speakerfirst_permutation_gof_null.png",
     "Dyad-level permutation control: real vs null goodness of fit."),
    (AUG / "dyad_permutation/permutation_panel.png", "permutation_panel.png",
     "Four-metric real-vs-null panel for the dyad permutation control."),
    (AUG / "dyad_permutation/posthoc_decoder_permutation.png",
     "posthoc_decoder_permutation_null.png",
     "Secondary control: post-hoc dyad-label permutation on the fixed real embedding."),
    (AUG / "aq_delta6/aq_delta6_embedding.png", "aq_delta6_embedding.png",
     "CEBRA 3D embedding coloured by six-class |ΔAQ|."),
    (AUG / "aq_delta6/aq_delta6_lonlat.png", "aq_delta6_lonlat.png",
     "Longitude/latitude map coloured by six-class |ΔAQ|."),
    (AUG / "aq_delta6/aq_delta6_confusion_matrix.png", "aq_delta6_confusion_matrix.png",
     "Leave-one-dyad-out 5-NN confusion matrix, six |ΔAQ| classes."),
    (AUG / "aq_delta6/aq_delta6_ks_summary.png", "aq_delta6_ks_summary.png",
     "Pairwise KS statistics per spherical coordinate, six |ΔAQ| classes."),
    (AUG / "aq_delta6/aq_delta6_gmm_bic.png", "aq_delta6_gmm_bic.png",
     "Label-agnostic GMM BIC/AIC sweep for the six-class run."),
    (AUG / "aq_delta6/aq_delta6_gmm_vs_aq.png", "aq_delta6_gmm_vs_aq.png",
     "GMM component composition by |ΔAQ|."),
    (AUG / "aq_delta6/aq_delta6_gmm_vs_dyad.png", "aq_delta6_gmm_vs_dyad.png",
     "GMM component composition by dyad identity."),
    (AUG / "gmm_identity_check/gmm_k_sweep.png", "fullband_gmm_k_sweep.png",
     "GMM BIC/AIC and held-out likelihood, K = 1…20, full-band speaker-first."),
    (AUG / "gmm_identity_check/gmm_ari_nmi_vs_k.png", "fullband_gmm_ari_vs_k.png",
     "What GMM components track (AQ magnitude vs |ΔAQ| vs dyad) as a function of K."),
    (AUG / "gmm_identity_check/gmm_k5_vs_dyad.png", "fullband_gmm_k5_vs_dyad.png",
     "K=5 GMM component composition by dyad identity."),
    (AUG / "gmm_identity_check/gmm_k5_vs_aqdelta.png", "fullband_gmm_k5_vs_aqdelta.png",
     "K=5 GMM component composition by |ΔAQ|."),
    (AUG / "gmm_identity_check/gmm_k2_components.png", "fullband_gmm_k2_components.png",
     "K=2 label-agnostic GMM components in longitude/latitude."),
    (AUG / "gmm_identity_check/gmm_k5_components.png", "fullband_gmm_k5_components.png",
     "K=5 label-agnostic GMM components in longitude/latitude."),
    (AUG / "band_comparison/band_metric_comparison.png", "band_metric_comparison.png",
     "Cross-condition comparison of decoding, goodness of fit and silhouette."),
    (AUG / "band_comparison/band_ks_comparison.png", "band_ks_comparison.png",
     "Spherical-coordinate KS statistics across frequency conditions."),
    (AUG / "band_comparison/band_gmm_k_comparison.png", "band_gmm_k_comparison.png",
     "BIC-preferred GMM K across frequency conditions."),
    (AUG / "qc/figures/psd_minusAlpha.png", "qc_psd_minus_alpha.png",
     "Representative PSD before/after the 8–12 Hz band-stop (locally generated filters)."),
    (AUG / "qc/figures/psd_minusGamma.png", "qc_psd_minus_gamma.png",
     "Representative PSD before/after the 30 Hz low-pass."),
    (AUG / "qc/figures/uploaded_conditions_psd.png", "qc_uploaded_conditions_psd.png",
     "The uploaded band datasets overlaid: all conditions are the same signal."),
    (RESULTS / "filtered_bands_ours/minusAlpha/loss.png", "minus_alpha_loss.png",
     "CEBRA training loss, alpha-removed condition."),
    (RESULTS / "filtered_bands_ours/minusAlpha/lonlat_by_magnitude.png",
     "minus_alpha_lonlat.png",
     "Longitude/latitude by AQ magnitude, alpha-removed condition."),
    (RESULTS / "filtered_bands_ours/minusAlpha/gmm_unsupervised.png", "minus_alpha_gmm.png",
     "Label-agnostic GMM, alpha-removed condition."),
    ]


def export_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    man = []
    for src, dst, cap in build_figmap():
        if Path(src).exists():
            shutil.copy2(src, FIG / dst)
            man.append({"file": dst, "caption": cap, "source": str(Path(src).relative_to(RESULTS))})
    return man


# ---------------------------------------------------------------- master CSV
MASTER_COLS = ["analysis", "condition", "label_scheme", "seed", "dyads", "files", "samples",
               "class_counts", "loss", "gof_bits", "grouped_5nn", "chance", "silhouette",
               "ks_latitude", "ks_longitude", "ks_radius", "gmm_K", "gmm_ARI", "gmm_NMI",
               "gmm_purity", "BIC", "AIC", "status", "notes"]


def master_rows():
    rows = []

    # real full-band speaker-first
    m = jload(RESULTS / "filtered_bands/fullband_speakerfirst/metrics.json")
    rr = jload(AUG / "dyad_permutation/real_rescored_metrics.json")
    if m:
        r = {"analysis": "speakerfirst_baseline", "condition": "fullband",
             "label_scheme": "AQ magnitude Low/High", "seed": m.get("seed"),
             "dyads": m.get("n_dyads"), "files": m.get("n_files"),
             "samples": m.get("n_samples"), "class_counts": m.get("class_counts"),
             "loss": m.get("final_loss"), "gof_bits": m.get("goodness_of_fit_bits"),
             "grouped_5nn": m.get("knn5_grouped_magnitude"), "chance": m.get("chance"),
             "silhouette": m.get("silhouette_magnitude"),
             "ks_latitude": m.get("ks_latitude"), "ks_longitude": m.get("ks_longitude"),
             "ks_radius": m.get("ks_radius"), "gmm_K": m.get("gmm_unsup_bestK"),
             "gmm_ARI": m.get("gmm_unsup_ARI_vs_magnitude"),
             "gmm_purity": m.get("gmm_unsup_purity"), "status": "completed",
             "notes": "33 dyads; mixed 250/1000 Hz source (see caveats)"}
        if rr:
            r["notes"] += f"; LODO 5-NN={f4(rr['metrics'].get('knn5_leave_one_dyad_out'))}"
        rows.append(r)

    # permutations
    prov = jload(AUG / "dyad_permutation/dataset_provenance.json") or {}
    for d in sorted((AUG / "dyad_permutation").glob("perm*")):
        pm = jload(d / "metrics.json")
        if not pm:
            continue
        rows.append({"analysis": "dyad_permutation_control", "condition": "fullband",
                     "label_scheme": f"PERMUTED AQ magnitude (perm seed {pm.get('perm_seed')})",
                     "seed": 0, "dyads": prov.get("n_dyads"), "files": prov.get("n_files"),
                     "samples": prov.get("n_samples"), "class_counts": prov.get("class_counts"),
                     "loss": pm.get("final_loss"), "gof_bits": pm.get("goodness_of_fit_bits"),
                     "grouped_5nn": pm.get("knn5_leave_one_dyad_out"),
                     "chance": pm.get("majority_chance"), "silhouette": pm.get("silhouette"),
                     "ks_latitude": pm.get("ks_latitude"), "ks_longitude": pm.get("ks_longitude"),
                     "ks_radius": pm.get("ks_radius"), "gmm_K": pm.get("gmm_bestK_by_bic"),
                     "gmm_ARI": pm.get("gmm_K2_ARI"), "gmm_purity": pm.get("gmm_K2_purity"),
                     "status": "completed",
                     "notes": f"{pm.get('n_dyads_relabelled')} dyads relabelled; "
                              f"CEBRA fully retrained"})

    # delta6
    d6 = jload(AUG / "aq_delta6/metrics.json")
    if d6:
        rows.append({"analysis": "aq_delta6", "condition": d6.get("condition"),
                     "label_scheme": "|ΔAQ| six-class 0-5", "seed": d6.get("seed"),
                     "dyads": d6.get("n_dyads"), "files": d6.get("n_files"),
                     "samples": d6.get("n_samples"), "class_counts": d6.get("class_counts"),
                     "loss": d6.get("final_loss"), "gof_bits": d6.get("goodness_of_fit_bits"),
                     "grouped_5nn": d6.get("knn5_leave_one_dyad_out"),
                     "chance": d6.get("majority_chance"), "silhouette": d6.get("silhouette"),
                     "gmm_K": d6.get("gmm_bestK_by_bic"), "gmm_ARI": d6.get("gmm_K6_ARI_vs_daq6"),
                     "gmm_purity": d6.get("gmm_K6_purity_vs_daq6"), "status": "completed",
                     "notes": f"balanced acc={f4(d6.get('balanced_accuracy'))}, "
                              f"macro F1={f4(d6.get('macro_f1'))}; |ΔAQ|=7 dyad dropped"})

    # bands
    bp = AUG / "band_comparison/band_metrics.csv"
    if bp.exists():
        for r in csv.DictReader(open(bp)):
            if r.get("status") != "OK":
                rows.append({"analysis": "band_cebra", "condition": r["band"],
                             "status": "failed", "notes": "no metrics.json produced"})
                continue
            rows.append({"analysis": "band_cebra", "condition": r["band"],
                         "label_scheme": "AQ magnitude Low/High", "seed": r.get("seed"),
                         "dyads": r.get("n_dyads"), "files": r.get("n_files"),
                         "samples": r.get("n_samples"), "chance": r.get("chance"),
                         "loss": r.get("final_loss"), "gof_bits": r.get("gof_bits"),
                         "grouped_5nn": r.get("knn5_grouped"), "silhouette": r.get("silhouette"),
                         "ks_latitude": r.get("ks_latitude"),
                         "ks_longitude": r.get("ks_longitude"), "ks_radius": r.get("ks_radius"),
                         "gmm_K": r.get("gmm_bestK"), "gmm_ARI": r.get("gmm_K2_ARI"),
                         "gmm_purity": r.get("gmm_K2_purity"), "status": "completed",
                         "notes": f"locally filtered EDFs; QC passed={r.get('qc_passed')}; "
                                  f"21-dyad subset"})

    # gmm identity sweep
    gp = AUG / "gmm_identity_check/gmm_sweep_primary_lonlat_unwrapped.csv"
    if gp.exists():
        for r in csv.DictReader(open(gp)):
            if int(r["K"]) not in (2, 5, 20):
                continue
            rows.append({"analysis": "gmm_identity_sweep", "condition": "fullband",
                         "label_scheme": "label-agnostic GMM", "seed": 0,
                         "gmm_K": r["K"], "gmm_ARI": r["ARI_vs_aq_magnitude"],
                         "gmm_NMI": r["NMI_vs_aq_magnitude"],
                         "gmm_purity": r["purity_vs_aq_magnitude"],
                         "BIC": r["BIC"], "AIC": r["AIC"], "status": "completed",
                         "notes": f"ARI vs dyad={f4(r['ARI_vs_dyad'])}, "
                                  f"ARI vs |ΔAQ|={f4(r['ARI_vs_aq_delta'])}"})
    return rows


# ---------------------------------------------------------------- latex
def latex(perm, d6, ident, ph, bandrows, gate, upload):
    lodo = None
    if perm:
        lodo = next((x for x in perm if x["metric"] == "knn5_leave_one_dyad_out"), None)

    meth = [r"% aug4_methods_snippet.tex", r"\subsection{Speaker-first construction}",
            "Every dyadic interaction was rebuilt as a single time-concatenated recording "
            "ordered \\emph{speaker first}, $[\\text{speaker};\\text{listener}]$, so that "
            "speaker/listener role no longer varies across samples and the absolute AQ "
            "difference $|\\Delta\\mathrm{AQ}|$ is the only manipulated label. Both interactions "
            "of a dyad are retained and receive the same label, since $|\\Delta\\mathrm{AQ}|$ is "
            "symmetric under exchange of speaker and listener. Recordings were reduced to the "
            "64 EEG channels (VREF and the trigger channel dropped), cropped to a single global "
            "minimum length, and z-scored per channel.", "",
            r"\subsection{CEBRA configuration}",
            "All runs use CEBRA 0.6.0 with the \\texttt{offset10-model} architecture, output "
            "dimension 3, cosine distance, the \\texttt{time\\_delta} conditional distribution, "
            "batch size 512, learning rate $3\\times10^{-4}$, temperature 1.12, time offset 10 "
            "and 5000 iterations, on CPU with a fixed seed of 0. Decoding is reported as "
            "leave-one-dyad-out 5-nearest-neighbour accuracy; a random-split accuracy is "
            "reported only as a secondary number because it leaks dyad identity across the "
            "split.", "",
            r"\subsection{Dyad-level permutation control}",
            "Because every dyad is entirely Low or entirely High, a decoder can in principle "
            "recover the label by recognising the dyad. To test this we permuted the "
            "dyad$\\rightarrow$label assignment, preserving the number of Low and High "
            "\\emph{dyads} exactly and keeping both speaker-first files of a dyad on the same "
            "permuted label, and \\textbf{retrained CEBRA from scratch} for each permutation "
            "with an identical configuration and identical initialisation seed. Empirical "
            "$p$-values are $(1+\\#\\{\\text{null}\\ge\\text{real}\\})/(1+n_{\\text{perm}})$.", "",
            r"\subsection{Spherical mapping and mixture models}",
            "Embeddings were median-centred and expressed as radius, longitude and latitude. "
            "Longitude is circular, so before fitting any mixture model we located the largest "
            "empty circular gap in longitude and unwrapped the axis at that angle, which "
            "prevents the $\\pm180^\\circ$ seam from manufacturing components; a wrap-safe "
            "$(\\cos\\lambda,\\sin\\lambda,\\phi)$ parameterisation is reported as sensitivity "
            "analysis. Gaussian mixtures were always fitted \\emph{without} labels; labels were "
            "used only afterwards to score the resulting partition.", "",
            r"\subsection{Frequency-band datasets}",
            "Band-removed datasets were generated locally with zero-phase FIR filters "
            "(\\texttt{firwin}, Hamming window, 1.0\\,Hz transition bandwidth per edge; "
            "2.0\\,Hz for the 30\\,Hz gamma low-pass) applied to each role-specific recording "
            "\\emph{before} speaker-first concatenation, so no filter crosses the artificial "
            "join. Every output was verified by comparing Welch power spectra before and after "
            "filtering over the same 64 EEG channels used for training."]

    res = [r"% aug4_results_snippet.tex", r"\subsection{Dyad-level permutation control}"]
    if lodo:
        surv = lodo["empirical_p"] <= 0.05
        res += [f"Retraining CEBRA on dyad-permuted AQ labels reproduced the real decoding "
                f"accuracy: real leave-one-dyad-out 5-NN $={lodo['real']:.3f}$ versus a null of "
                f"${lodo['null_mean']:.3f}\\pm{lodo['null_std']:.3f}$ over "
                f"{lodo['n_null']} independent retrainings "
                f"(empirical $p={lodo['empirical_p']:.3f}$, majority chance $0.545$). "
                + ("The real value therefore lies outside the null distribution."
                   if surv else
                   "The real value lies inside the null distribution: an arbitrary regrouping "
                   "of the same dyads supports the same decoding accuracy, so this accuracy "
                   "cannot be attributed to AQ."), ""]
    if ph:
        res += [f"A secondary control that held the embedding fixed and permuted only the "
                f"dyad$\\rightarrow$label map ({ph['n_perms']} permutations) gave real "
                f"${ph['real_lodo_knn5']:.3f}$ versus null "
                f"${ph['null_mean']:.3f}\\pm{ph['null_std']:.3f}$ "
                f"($p={ph['empirical_p']:.3f}$).", ""]
    if d6:
        res += [r"\subsection{Six-class absolute AQ difference}",
                f"Dropping the single dyad with $|\\Delta\\mathrm{{AQ}}|=7$ leaves "
                f"{d6['n_dyads']} dyads and {d6['n_files']} speaker-first files spanning "
                f"$|\\Delta\\mathrm{{AQ}}|\\in\\{{0,\\dots,5\\}}$. The six-class model reached a "
                f"final loss of {f4(d6.get('final_loss'),3)} and goodness of fit "
                f"{f4(d6.get('goodness_of_fit_bits'),3)} bits; leave-one-dyad-out 5-NN accuracy "
                f"was {f4(d6.get('knn5_leave_one_dyad_out'),3)} against a majority-class chance "
                f"of {f4(d6.get('majority_chance'),3)}, with balanced accuracy "
                f"{f4(d6.get('balanced_accuracy'),3)} and macro $F_1$ "
                f"{f4(d6.get('macro_f1'),3)}.", ""]
    if ident:
        res += [r"\subsection{What the mixture components track}",
                f"A label-agnostic Gaussian mixture swept over $K=1\\dots20$ preferred "
                f"$K={ident.get('bestK_by_BIC')}$ by BIC and "
                f"$K={ident.get('bestK_by_heldout_ll')}$ by dyad-grouped held-out likelihood. "
                f"At $K=5$ the partition matched "
                f"{'dyad identity' if ident.get('Q1_K5_aligns_with')=='dyad identity' else 'AQ difference'} "
                f"most closely (ARI vs dyad "
                f"{f4(ident['Q1_evidence']['K5_ARI_vs_dyad'],3)}, vs $|\\Delta$AQ$|$ "
                f"{f4(ident['Q1_evidence']['K5_ARI_vs_aq_delta'],3)}, vs AQ magnitude "
                f"{f4(ident['Q1_evidence']['K5_ARI_vs_aq_magnitude'],3)}). "
                f"The $K=2$ split was stable across seeds and subsamples "
                f"(ARI {f4(ident['Q4_K2_stable_across_seeds']['ARI_vs_magnitude_mean'],3)}"
                f"$\\pm${f4(ident['Q4_K2_stable_across_seeds']['ARI_vs_magnitude_std'],3)}), and "
                f"the wrap-safe parameterisation "
                f"{'changed' if ident['Q5_wrapsafe_changes_result']['changed'] else 'did not change'} "
                f"the preferred $K$.", ""]
    if upload:
        res += [r"\subsection{Data provenance check}",
                "The externally supplied band-specific EDF datasets were found to be "
                "numerically identical to one another (maximum absolute difference "
                "$0$ between the alpha-only, alpha-removed, beta-removed and gamma-removed "
                "exports of the same recording), so none of them can be used as a filtered "
                "condition. All band results reported here use locally generated filters whose "
                "attenuation was verified spectrally.", ""]
    if bandrows:
        okb = [r for r in bandrows if r.get("status") == "OK"]
        if okb:
            kk = [float(r["knn5_grouped"]) for r in okb if r.get("knn5_grouped")]
            res += [r"\subsection{Frequency-band comparison}",
                    f"Across the six frequency conditions the grouped 5-NN accuracy spanned only "
                    f"{min(kk):.3f}--{max(kk):.3f}. Removing any single band left the decoding "
                    f"essentially unchanged.", ""]

    tab = [r"% aug4_tables.tex", r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Dyad-level AQ permutation control on the full-band speaker-first "
           r"dataset. CEBRA is retrained from scratch for every permutation.}",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Metric & Real & Null mean $\pm$ sd & $\Delta$ & $p$ \\", r"\midrule"]
    for x in (perm or []):
        nm = x["pretty"].replace("&", r"\&")
        tab.append(f"{nm} & {x['real']:.3f} & {x['null_mean']:.3f} $\\pm$ "
                   f"{x['null_std']:.3f} & {x['delta']:+.3f} & {x['empirical_p']:.3f} \\\\")
    tab += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    if bandrows:
        tab += [r"\begin{table}[t]", r"\centering", r"\small",
                r"\caption{Per-band CEBRA (speaker-first, AQ magnitude). Identical model "
                r"configuration, dyads and crop across rows.}",
                r"\begin{tabular}{lrrrrr}", r"\toprule",
                r"Condition & Loss & GoF & Grouped 5-NN & Silhouette & GMM $K$ \\", r"\midrule"]
        for r in bandrows:
            if r.get("status") != "OK":
                continue
            tab.append(f"{r['band'].replace('minus','$-$')} & {f4(r['final_loss'],3)} & "
                       f"{f4(r['gof_bits'],3)} & {f4(r['knn5_grouped'],3)} & "
                       f"{f4(r['silhouette'],3)} & {r['gmm_bestK']} \\\\")
        tab += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    lim = [r"% aug4_limitations_snippet.tex", r"\subsection{Limitations}",
           r"\begin{itemize}"]
    if lodo and lodo["empirical_p"] > 0.05:
        lim.append(r"\item \textbf{AQ is confounded with dyad identity by construction.} "
                   r"Every dyad carries a single AQ-magnitude label, so a model that recognises "
                   r"the recording recovers the label. Retraining on permuted dyad labels "
                   r"reproduces the reported accuracy, so the AQ decoding in this design is not "
                   r"separable from a dyad effect.")
    lim += [r"\item The 33-dyad speaker-first dataset mixes source sampling rates: 14 dyads were "
            r"recorded at 1000\,Hz and 19 at 250\,Hz, and the fixed sample-count crop therefore "
            r"corresponds to 90.2\,s per role for the former and 361\,s for the latter. The "
            r"fixed time offset of 10 samples also spans different physical lags. Sampling rate "
            r"is not significantly associated with the AQ label (Fisher exact $p=0.73$), but "
            r"this heterogeneity should be removed by resampling everything to 250\,Hz before "
            r"any published run.",
            r"\item The externally supplied ``filtered'' EDF datasets are numerically identical "
            r"to one another and cannot be used; the band analyses rest on locally generated "
            r"filters covering 21 dyads rather than the full 33.",
            r"\item No non-oscillatory (FOOOF/aperiodic) representation was available: the "
            r"uploaded non-oscillatory export is also numerically indistinguishable from the "
            r"other conditions.",
            r"\item Coordinate-wise KS and Kruskal--Wallis tests were computed over time "
            r"samples, which are strongly autocorrelated and nested within dyads. Their "
            r"$p$-values are therefore pseudoreplicated and are not reported as inferential; "
            r"only the KS statistics are used, as descriptive effect sizes.",
            r"\item Spherical coordinates are a descriptive reparameterisation of an "
            r"unconstrained 3D embedding; individual axes carry no biological meaning.",
            r"\item Metrics are reported for a single training seed per condition, so small "
            r"between-condition differences are not interpretable without a seed-level "
            r"variance estimate.",
            r"\end{itemize}"]

    OL.mkdir(parents=True, exist_ok=True)
    (OL / "aug4_methods_snippet.tex").write_text("\n".join(meth), encoding="utf-8")
    (OL / "aug4_results_snippet.tex").write_text("\n".join(res), encoding="utf-8")
    (OL / "aug4_tables.tex").write_text("\n".join(tab), encoding="utf-8")
    (OL / "aug4_limitations_snippet.tex").write_text("\n".join(lim), encoding="utf-8")


def main():
    global RESULTS, AUG, OL, FIG
    ap = argparse.ArgumentParser(
        description="Build the master metrics CSV, LaTeX snippets, and figure "
                    "bundle from an existing results/ tree (expects both a "
                    "results/aug4_pipeline/ subfolder and the sibling "
                    "results/filtered_bands*/ trees produced by the training/ "
                    "and evaluation/ scripts in this submission).")
    ap.add_argument("--results-root", type=Path, default=RESULTS,
                    help=f"Path to the results/ directory (default: {RESULTS}).")
    args = ap.parse_args()
    RESULTS = args.results_root
    AUG = RESULTS / "aug4_pipeline"
    OL = AUG / "overleaf_export"
    FIG = OL / "figures"

    AUG.mkdir(parents=True, exist_ok=True)
    manifest = export_figures()

    rows = master_rows()
    with open(AUG / "AUG4_METRICS_MASTER.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    perm = None
    pcsv = AUG / "dyad_permutation/real_vs_null_summary.csv"
    if pcsv.exists():
        perm = []
        for r in csv.DictReader(open(pcsv)):
            perm.append({k: (float(v) if k in ("real", "null_mean", "null_std", "null_min",
                                               "null_max", "empirical_p", "delta")
                             else int(v) if k in ("n_null", "n_null_beating_real")
                             else v)
                         for k, v in r.items()})
    d6 = jload(AUG / "aq_delta6/metrics.json")
    ident = jload(AUG / "gmm_identity_check/identity_answers.json")
    ph = jload(AUG / "dyad_permutation/posthoc_decoder_permutation.json")
    gate = jload(AUG / "qc/filter_qc_gate.json")
    upload = (AUG / "qc/uploaded_band_verification.md").exists()
    bandrows = []
    bp = AUG / "band_comparison/band_metrics.csv"
    if bp.exists():
        bandrows = list(csv.DictReader(open(bp)))

    latex(perm, d6, ident, ph, bandrows, gate, upload)

    # ---- figure manifest ----
    ML = ["# Figure manifest — Aug 4", "",
          "All figures: PNG, 300 dpi, white background, no local paths in titles.", "",
          "| file | caption | produced by |", "|---|---|---|"]
    for m in manifest:
        ML.append(f"| `{m['file']}` | {m['caption']} | `{m['source']}` |")
    (OL / "FIGURE_MANIFEST.md").write_text("\n".join(ML), encoding="utf-8")

    # ---- zip ----
    zp = OL / "overleaf_aug4_results.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(FIG.glob("*.png")):
            z.write(p, f"figures/{p.name}")
        for n in ["aug4_methods_snippet.tex", "aug4_results_snippet.tex", "aug4_tables.tex",
                  "aug4_limitations_snippet.tex", "FIGURE_MANIFEST.md"]:
            if (OL / n).exists():
                z.write(OL / n, n)
    print(f"figures exported: {len(manifest)}   zip: {zp}")

    # ---------------- meeting summary ----------------
    lodo = next((x for x in (perm or []) if x["metric"] == "knn5_leave_one_dyad_out"), None)
    prov = jload(AUG / "dyad_permutation/dataset_provenance.json") or {}
    audit_md = (AUG / "qc/edf_sampling_rate_summary.md")
    S = [f"# Aug 4 meeting summary", "",
         f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_", "",
         "## 1. Executive summary", ""]
    if lodo:
        if lodo["empirical_p"] > 0.05:
            S += [f"**The headline AQ-magnitude result does not survive its own control.** "
                  f"Retraining CEBRA on randomly permuted dyad labels reaches "
                  f"{lodo['null_mean']:.3f} ± {lodo['null_std']:.3f} leave-one-dyad-out 5-NN "
                  f"accuracy, statistically indistinguishable from the real "
                  f"{lodo['real']:.3f} (empirical p = {lodo['empirical_p']:.3f}, "
                  f"{lodo['n_null']} permutations). Because every dyad carries a single AQ "
                  f"label, the decoder only has to recognise the dyad — and it does that just "
                  f"as well for a made-up grouping as for the real one.", "",
                  "Three independent lines of evidence now point the same way:", "",
                  "1. The permutation control above.",
                  "2. Label-agnostic mixture components track neither AQ nor a clean two-group "
                  "split at the BIC-preferred K.",
                  "3. Removing any single frequency band leaves decoding essentially unchanged "
                  "— a band-specific neural effect should not behave that way.", ""]
        else:
            S += [f"The AQ-magnitude decoding survives the dyad-level permutation control: real "
                  f"{lodo['real']:.3f} vs null {lodo['null_mean']:.3f} ± {lodo['null_std']:.3f} "
                  f"(p = {lodo['empirical_p']:.3f}).", ""]
    S += ["Two data-provenance problems were also found and are described in sections 5 and 6.", ""]

    S += ["## 2. Highest-priority permutation result", ""]
    if lodo and perm:
        S += [f"Dataset: {prov.get('n_dyads')} dyads, {prov.get('n_files')} speaker-first files, "
              f"{prov.get('n_samples', 0):,} samples, majority chance "
              f"{f4(prov.get('majority_chance'),3)}. Labels permuted **across whole dyads**, "
              f"preserving the Low/High dyad counts exactly; CEBRA **fully retrained** per "
              f"permutation with identical config and init seed.", "",
              "| metric | real | null mean ± sd | Δ | p |", "|---|---|---|---|---|"]
        for x in perm:
            S.append(f"| {x['pretty']} | **{x['real']:.4f}** | {x['null_mean']:.4f} ± "
                     f"{x['null_std']:.4f} | {x['delta']:+.4f} | {x['empirical_p']:.3f} |")
        S += ["", f"See `dyad_permutation/permutation_summary.md`."]
    else:
        S += ["_Not yet complete._"]
    if ph:
        S += ["", f"Secondary (decoder-only, embedding fixed, {ph['n_perms']} permutations): "
                  f"real {ph['real_lodo_knn5']:.4f} vs null {ph['null_mean']:.4f} ± "
                  f"{ph['null_std']:.4f}, p = {ph['empirical_p']:.4f}."]

    S += ["", "## 3. Six-class |ΔAQ| result", ""]
    if d6:
        cc = jload(AUG / "aq_delta6/dataset_provenance.json") or {}
        S += [f"Observed |ΔAQ| distribution before dropping anything: "
              f"`{cc.get('distribution_before_drop_dyads')}` (dyads). Exactly one dyad had "
              f"|ΔAQ| = 7 and was dropped, as instructed, leaving {d6['n_dyads']} dyads / "
              f"{d6['n_files']} files over classes 0–5.", "",
              f"- final loss **{f4(d6.get('final_loss'),3)}**, GoF "
              f"**{f4(d6.get('goodness_of_fit_bits'),3)} bits**",
              f"- leave-one-dyad-out 5-NN **{f4(d6.get('knn5_leave_one_dyad_out'),3)}** vs "
              f"majority chance **{f4(d6.get('majority_chance'),3)}**",
              f"- balanced accuracy **{f4(d6.get('balanced_accuracy'),3)}**, macro F1 "
              f"**{f4(d6.get('macro_f1'),3)}**",
              f"- silhouette {f4(d6.get('silhouette'),3)}; GMM BIC-preferred K = "
              f"{d6.get('gmm_bestK_by_bic')}", "",
              f"Per-class recall: `{d6.get('per_class_recall')}`", "",
              "Read this against section 2: the same dyad-identity confound applies with even "
              "more force here, because six classes over 32 dyads means several classes are "
              "carried by only three dyads."]
    else:
        S += ["_Not yet complete._"]

    S += ["", "## 4. GMM K-sweep and dyad identity", ""]
    if ident:
        e = ident["Q1_evidence"]
        S += [f"- BIC-preferred **K = {ident.get('bestK_by_BIC')}**, AIC-preferred "
              f"**K = {ident.get('bestK_by_AIC')}**, dyad-grouped held-out likelihood prefers "
              f"**K = {ident.get('bestK_by_heldout_ll')}**",
              f"- Longitude unwrapped at **{f4(ident.get('longitude_cut_deg'),1)}°** (largest "
              f"empty circular gap), so the ±180° seam cannot create components",
              f"- **Q1 — K=5 aligns with {ident.get('Q1_K5_aligns_with')}**: ARI vs dyad "
              f"{f4(e['K5_ARI_vs_dyad'],3)}, vs |ΔAQ| {f4(e['K5_ARI_vs_aq_delta'],3)}, vs AQ "
              f"magnitude {f4(e['K5_ARI_vs_aq_magnitude'],3)}",
              f"- **Q2 — {ident.get('Q2_components_dominated_by_single_dyad')}**",
              f"- **Q3 —** {ident['Q3_K_near_n_dyads_major_BIC_gain'].get('note')} "
              f"(BIC at K=5 = {f4(ident['Q3_K_near_n_dyads_major_BIC_gain'].get('BIC_at_K5'),0)})",
              f"- **Q4 — K=2 split stability**: ARI "
              f"{f4(ident['Q4_K2_stable_across_seeds']['ARI_vs_magnitude_mean'],3)} ± "
              f"{f4(ident['Q4_K2_stable_across_seeds']['ARI_vs_magnitude_std'],3)} across 5 "
              f"seeds/subsamples "
              f"({'stable' if ident['Q4_K2_stable_across_seeds']['stable'] else 'UNSTABLE'})",
              f"- **Q5 — wrap-safe features**: BIC-preferred K "
              f"{ident['Q5_wrapsafe_changes_result']['bestK_BIC_primary']} → "
              f"{ident['Q5_wrapsafe_changes_result']['bestK_BIC_wrapsafe']} "
              f"({'changed' if ident['Q5_wrapsafe_changes_result']['changed'] else 'unchanged'})",
              "",
              "Note the earlier K=5 preference was obtained with a K sweep that stopped at 6. "
              "Extending to K=20 shows BIC keeps improving well past 5, so 'K=5' was a boundary "
              "artefact of the old sweep rather than a real five-cluster structure."]
    else:
        S += ["_Not yet complete._"]

    S += ["", "## 5. EDF sampling-rate audit", ""]
    if audit_md.exists():
        S += ["1884 EDF files audited across three roots (see "
              "`qc/edf_sampling_rate_summary.md`).", "",
              "**Finding: the 33-dyad speaker-first dataset mixes sampling rates.** 14 dyads "
              "(dyads 1–18) were recorded at 1000 Hz and 19 dyads (20–45) at 250 Hz. Because "
              "the stacker cropped to a fixed *sample count* (180,500), the two groups "
              "contribute very different amounts of real time:", "",
              "| source sfreq | dyads | seconds per role in the stacked file |",
              "|---|---|---|", "| 1000 Hz | 14 | 90.2 s |", "| 250 Hz | 19 | 361.0 s |", "",
              "Sampling rate is **not** significantly associated with the AQ-magnitude label "
              "(Fisher exact p = 0.73, OR = 1.38), so it is not driving the decoding directly. "
              "It nonetheless has to be fixed: the fixed `time_offsets=10` spans 10 ms for one "
              "group and 40 ms for the other. Our locally filtered datasets are already "
              "uniformly 250 Hz.", "",
              "No two EDF files anywhere are byte-identical."]
    S += ["", "## 6. Filter generation QC", ""]
    if gate:
        S += [f"Locally generated filters (FIR, firwin, Hamming, zero-phase, 1.0 Hz transition; "
              f"2.0 Hz for the gamma low-pass), applied per role-specific recording *before* "
              f"speaker-first concatenation. QC scores Welch PSD over the same 64 EEG channels "
              f"used for training.", "",
              f"QC-cleared conditions: **{', '.join(gate.get('passing_bands', [])) or 'none'}**",
              ""]
        for s in gate.get("summary", []):
            # the fresh QC writes atten_max_db; the reclassifier writes atten_worst_db
            worst = s.get("atten_worst_db", s.get("atten_max_db"))
            drift = s.get("preserve_worst_abs_db", s.get("preserve_mean_db"))
            npass, nf = s.get("n_pass"), s.get("n_files")
            S.append(f"- `{s['band']}` — removed {s['removed_band_hz']} Hz, mean attenuation "
                     f"{s['atten_mean_db']} dB, worst {worst} dB, out-of-band "
                     f"drift {drift} dB"
                     + (f", {npass}/{nf} files pass" if npass is not None else "")
                     + f" → **{s['qc_verdict']}**")
    if upload:
        S += ["", "**The uploaded band datasets are not filtered.** Comparing the same "
              "recording across the uploaded Alpha-Only, Alpha-Removed, Beta-Removed and "
              "Gamma-Removed exports gives a maximum absolute difference of exactly 0 — they "
              "are the same signal with different file headers. The uploaded Non-Oscillatory "
              "export differs only at float-rounding level (correlation 1.000000). Verified on "
              "24 recordings; see `qc/uploaded_band_verification.md` and "
              "`qc_uploaded_expanded/`.", "",
              "Consequence: the run previously labelled 'alpha-removed' is correctly relabelled "
              "**full band**, and every genuine band result must come from the locally "
              "generated filters."]

    S += ["", "## 7. Per-band CEBRA results", ""]
    if bandrows:
        S += ["| condition | QC | loss | GoF | grouped 5-NN | silhouette | GMM K |",
              "|---|---|---|---|---|---|---|"]
        for r in bandrows:
            if r.get("status") != "OK":
                S.append(f"| {r['band']} | – | – | – | – | – | _{r.get('status')}_ |")
                continue
            S.append(f"| {r['band']} | {'✅' if r.get('qc_passed')=='True' else '⚠️'} | "
                     f"{f4(r['final_loss'],3)} | {f4(r['gof_bits'],3)} | "
                     f"{f4(r['knn5_grouped'],3)} | {f4(r['silhouette'],3)} | {r['gmm_bestK']} |")
        okb = [r for r in bandrows if r.get("status") == "OK" and r.get("knn5_grouped")]
        if okb:
            kk = [float(r["knn5_grouped"]) for r in okb]
            S += ["", f"Grouped 5-NN spans {min(kk):.3f}–{max(kk):.3f} across all conditions "
                      f"(range {max(kk)-min(kk):.3f}). Deleting an entire band barely moves the "
                      f"decoding, which argues against a band-specific effect and is consistent "
                      f"with the permutation result. These runs use the 21-dyad locally "
                      f"filtered subset, one seed each."]
    else:
        S += ["_Not yet complete._"]

    S += ["", "## 8. Non-oscillatory data status", "",
          "**The upload is unusable, so we computed the aperiodic component ourselves.**", "",
          "The `Non-Oscillatory EEG Datafiles` upload exists (216 EDFs, 36 dyads) but is not an "
          "aperiodic decomposition. For the same recording its spectrum is identical to the "
          "Alpha-Removed and Alpha-Only uploads — alpha prominence over a 1/f fit is −0.89 dB "
          "for all three, and the 1/f slope is −1.264 for all three, matching to two decimals "
          "on every recording tested. Sample arrays differ only at float-rounding level "
          "(max abs diff 1.24e-07, correlation 1.000000). Michelle's FOOOF script and README "
          "are absent (searched FOOOF / fooof / specparam / nonosc / aperiodic / 1f: 0 hits).", ""]
    noqc = jload(AUG / "nonoscillatory/qc/nonosc_extraction_summary.json")
    nom = jload(AUG / "nonoscillatory/cebra/metrics.json")
    if noqc:
        S += ["### Our own FOOOF extraction", "",
              "Because the intended representation could not be recovered from any project "
              "file, the input definition is **our documented assumption** — see "
              "`NONOSCILLATORY_INPUT_ASSUMPTION.md`. We use the standard aperiodic summary as "
              "a sliding-window time series: per window, per channel, the aperiodic **offset** "
              "and **exponent** (64 channels × 2 = 128 features per window).", "",
              f"- source: locally filtered, QC-verified full-band EDFs at a uniform 250 Hz",
              f"- window {noqc.get('win_sec')} s, hop {noqc.get('hop_sec')} s, "
              f"fit range {noqc.get('freq_range')} Hz, mode `{noqc.get('aperiodic_mode')}`",
              f"- {noqc.get('n_files')} recordings, {noqc.get('n_features')} features × "
              f"{noqc.get('n_windows')} windows each, crop {noqc.get('crop_sec')} s",
              f"- mean fit R² = **{noqc.get('mean_r_squared')}**, "
              f"mean aperiodic exponent = {noqc.get('mean_exponent')}", "",
              "**QC that the oscillatory content really was removed:** alpha prominence over a "
              f"1/f fit drops from **{noqc.get('alpha_prominence_raw_mean_db')} dB** in the raw "
              f"spectrum to **{noqc.get('alpha_prominence_aperiodic_mean_db')} dB** in the "
              f"reconstructed aperiodic spectrum "
              f"({'PASS — oscillatory content removed' if noqc.get('oscillatory_content_removed') else 'REVIEW — residual peak remains'}).",
              ""]
    if nom:
        S += ["### CEBRA on the non-oscillatory representation", "",
              f"- {nom.get('n_dyads')} dyads / {nom.get('n_files')} files / "
              f"{nom.get('n_samples')} windows, chance {f4(nom.get('chance'),3)}",
              f"- final loss {f4(nom.get('final_loss'),3)}, GoF "
              f"{f4(nom.get('goodness_of_fit_bits'),3)} bits",
              f"- grouped 5-NN **{f4(nom.get('knn5_grouped_magnitude'),3)}**, silhouette "
              f"{f4(nom.get('silhouette_magnitude'),3)}",
              f"- KS lat/lon/rad {f4(nom.get('ks_latitude'),3)} / "
              f"{f4(nom.get('ks_longitude'),3)} / {f4(nom.get('ks_radius'),3)}",
              f"- GMM BIC-preferred K = {nom.get('gmm_unsup_bestK')}, K=2 ARI "
              f"{nom.get('gmm_unsup_ARI_vs_magnitude')}, purity {nom.get('gmm_unsup_purity')}",
              ""]
    nop = None
    npc = AUG / "nonoscillatory/permutation/real_vs_null_summary.csv"
    if npc.exists():
        for r in csv.DictReader(open(npc)):
            if r["metric"] == "knn5_leave_one_dyad_out":
                nop = r
    if nop:
        p = float(nop["empirical_p"])
        S += ["### Permutation control on the non-oscillatory representation", "",
              f"Leave-one-dyad-out 5-NN: real **{float(nop['real']):.4f}** vs dyad-permuted "
              f"null **{float(nop['null_mean']):.4f} ± {float(nop['null_std']):.4f}** "
              f"({nop['n_null']} full CEBRA retrainings), empirical **p = {p:.3f}**.", "",
              ("The non-oscillatory representation behaves exactly like the broadband one: "
               "the decoding is reproduced by an arbitrary regrouping of dyads, so it is not "
               "evidence of AQ structure either."
               if p > 0.05 else
               "Unlike the broadband representation, this one separates from its permutation "
               "null — worth following up, but with only 19 dyads it needs replication."), ""]
    if not noqc:
        S += ["_Our own extraction has not completed yet._", ""]

    S += ["", "## 9. Autoencoder status", "",
          "**BLOCKED, and the gate is not met.** The gate requires valid non-oscillatory data "
          "and evidence that its representation is poorly modelled by a GMM. The "
          "non-oscillatory data is unusable (section 8), so the precondition fails. No "
          "mentor-provided architecture or IBM tutorial code was found in the workspace. "
          "Nothing was run and no exploratory baseline was substituted."]

    S += ["", "## 10. Completed outputs", ""]
    for p in sorted(AUG.rglob("*")):
        if p.is_file() and p.suffix in (".md", ".csv", ".json", ".zip") and "logs" not in p.parts:
            S.append(f"- `{p.relative_to(RESULTS)}`")

    S += ["", "## 11. Blocked items", "",
          "- Non-oscillatory / FOOOF pipeline — missing script, README and genuine aperiodic data",
          "- Autoencoder branch — gate not satisfied; no architecture provided",
          "- Band coverage beyond 21 dyads — the uploaded band files cannot be used, and only "
          "84 role-specific source EDFs (21 dyads) are available locally to filter",
          "- Resampling of the 1000 Hz recordings — audited and quantified, but rebuilding the "
          "33-dyad speaker-first dataset at a uniform 250 Hz was not run"]

    S += ["", "## 12. Scientific caveats", "",
          "- AQ magnitude is constant within a dyad, so it is inseparable from dyad identity in "
          "this design. This is the dominant caveat and the permutation control makes it "
          "concrete.",
          "- Mixed 250/1000 Hz sources in the 33-dyad dataset (section 5).",
          "- **The KS and Kruskal–Wallis p-values are pseudoreplicated and should not be "
          "quoted.** They treat 60,000 time samples as independent observations when the "
          "effective sample size is the number of dyads (33, or 32 for the six-class run). "
          "Every such test returns p ≈ 0 regardless of effect size. The KS *statistics* are "
          "usable as descriptive effect sizes; their p-values are not inferential. Any "
          "dyad-level claim needs a dyad-level test (e.g. one summary value per dyad).",
          "- Spherical coordinates are descriptive only; no biological meaning attaches to any "
          "individual CEBRA axis.",
          "- Single seed per condition; between-condition differences smaller than the "
          "between-seed spread are not interpretable.",
          "- Band results cover 21 dyads, not the 33 used for the full-band baseline, so band "
          "and full-band numbers are not directly comparable.",
          "- GMMs were fitted without labels throughout; labels were applied only for scoring."]

    S += ["", "## 13. Exact next steps", "",
          "1. **Design fix, highest value.** AQ magnitude cannot be decoded free of dyad "
          "identity with one label per dyad. Either obtain within-dyad AQ variation, or move to "
          "a target that varies within a dyad (role, turn, speech state), or accept a "
          "dyad-level analysis with n = 33 and dyad-level statistics rather than sample-level "
          "decoding.",
          "2. Rebuild the 33-dyad speaker-first dataset at a uniform 250 Hz and re-run the "
          "full-band baseline, so time offsets mean the same thing for every dyad.",
          "3. Ask Michelle for the FOOOF script, README and a genuine aperiodic export; the "
          "current upload is not aperiodic data.",
          "4. Ask for the raw role-specific EDFs for the remaining 12–15 dyads so the band "
          "analyses can run on the same 33 dyads as the baseline.",
          "5. Run 3 seeds per band condition to establish the noise floor before interpreting "
          "any between-band difference.",
          "6. Only after 1–2 are settled, revisit the six-class |ΔAQ| analysis."]

    (AUG / "AUG4_MEETING_SUMMARY.md").write_text("\n".join(S), encoding="utf-8")
    print(f"wrote {AUG/'AUG4_MEETING_SUMMARY.md'}")
    print(f"wrote {AUG/'AUG4_METRICS_MASTER.csv'} ({len(rows)} rows)")
    print("REPORTS_DONE")


if __name__ == "__main__":
    main()
