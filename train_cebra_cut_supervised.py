# scripts/train_cebra_cut_supervised.py
"""
Supervised CEBRA training on cut_60/raw dyads with
periodic checkpointing, full metrics, and visualisations.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import cebra
from cebra import CEBRA, KNNDecoder, L1LinearRegressor
from cebra.integrations.sklearn import metrics as cmetrics

from utils.cv_eval import (
    build_pooled_dataset,
    export_evaluation_results,
    normalize_train_fractions,
    run_learning_curve,
    run_repeated_stratified_cv,
)


# ------------------------------------------------------------------ #
# 1. Paths
# ------------------------------------------------------------------ #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "cut_60" / "raw"
MODELS_DIR = ROOT / "models" / "cut_60" / "raw"
RESULTS = ROOT / "results"
FIGS_DIR = RESULTS / "figures"
EVAL_DIR = RESULTS / "evaluation" / "supervised"
for path in (MODELS_DIR, RESULTS, FIGS_DIR):
    path.mkdir(parents=True, exist_ok=True)

METRICS_CSV = RESULTS / "metrics_log.csv"
CONSIST_TXT = RESULTS / "consistency_results.txt"
CONSIST_CSV = RESULTS / "consistency_log.csv"


# ------------------------------------------------------------------ #
# 2. Config
# ------------------------------------------------------------------ #
PAIRINGS = ["spk9-lst10", "lst9-spk10"]
N_RUNS = 5
TRAIN_FRAC = 0.8
CKPT_EVERY = 500  # iterations
DEFAULT_CV_FOLDS = 5
DEFAULT_CV_REPEATS = 1
DEFAULT_LC_FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)

MODEL_KWARGS = dict(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.12,
    max_iterations=5000,
    conditional="time_delta",
    output_dimension=3,
    distance="cosine",
    device="cuda_if_available",
    verbose=True,
    time_offsets=10,
)


# ------------------------------------------------------------------ #
# 3. Helpers
# ------------------------------------------------------------------ #
def to_T_C(x):
    return x.T if x.shape[0] > x.shape[1] else x


def make_labels(T, pair):
    return np.full((T,), 0 if pair == "spk9-lst10" else 1, dtype=np.int8)


def save_fig(fig_or_ax, path):
    if isinstance(fig_or_ax, tuple):
        fig = fig_or_ax[0]
    elif hasattr(fig_or_ax, "savefig"):
        fig = fig_or_ax
    else:
        fig = fig_or_ax.get_figure()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train supervised CEBRA models and optionally evaluate pooled embeddings."
    )
    parser.add_argument("--run-cv", action="store_true", help="Run repeated stratified CV on pooled embeddings.")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS, help="Requested number of CV folds.")
    parser.add_argument("--cv-repeats", type=int, default=DEFAULT_CV_REPEATS, help="Number of repeated CV passes.")
    parser.add_argument(
        "--run-learning-curve",
        action="store_true",
        help="Run learning-curve evaluation on pooled embeddings.",
    )
    parser.add_argument(
        "--lc-fracs",
        default=",".join(str(value) for value in DEFAULT_LC_FRACS),
        help="Comma-separated training fractions for learning-curve evaluation.",
    )
    return parser.parse_args()


def parse_learning_curve_fracs(raw_fracs):
    return normalize_train_fractions(float(value.strip()) for value in raw_fracs.split(",") if value.strip())


def initialise_results_files():
    for result_file in (METRICS_CSV, CONSIST_TXT, CONSIST_CSV):
        if result_file.exists():
            result_file.unlink()

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["pairing", "run", "gof_bits", "knn_r2", "cont_r2"])
    with open(CONSIST_CSV, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["pairing", "mean_consistency", "variability"])


def run_optional_evaluation(args, run_embeddings):
    if not (args.run_cv or args.run_learning_curve):
        return

    if not run_embeddings:
        print("No complete run embeddings available for evaluation.")
        return

    cv_rows = []
    learning_curve_rows = []

    for run in sorted(run_embeddings):
        pair_embeddings = run_embeddings[run]
        missing_pairs = [pair for pair in PAIRINGS if pair not in pair_embeddings]
        if missing_pairs:
            print(f"Skipping evaluation for run {run}: missing {', '.join(missing_pairs)}")
            continue

        pooled_X, pooled_y, _ = build_pooled_dataset(pair_embeddings)

        if args.run_cv:
            cv_rows.extend(
                run_repeated_stratified_cv(
                    pooled_X,
                    pooled_y,
                    folds=args.cv_folds,
                    repeats=args.cv_repeats,
                    random_state=run,
                    run_id=run,
                )
            )

        if args.run_learning_curve:
            learning_curve_rows.extend(
                run_learning_curve(
                    pooled_X,
                    pooled_y,
                    folds=args.cv_folds,
                    repeats=args.cv_repeats,
                    train_fractions=args.learning_curve_fracs,
                    random_state=run,
                    run_id=run,
                )
            )

    export_evaluation_results(
        EVAL_DIR,
        cv_rows=cv_rows,
        learning_curve_rows=learning_curve_rows,
        config={
            "pairings": PAIRINGS,
            "cv_folds": int(args.cv_folds),
            "cv_repeats": int(args.cv_repeats),
            "learning_curve_fractions": args.learning_curve_fracs,
        },
    )

    if cv_rows or learning_curve_rows:
        print(f"Saved evaluation outputs to {EVAL_DIR}")


def main():
    args = parse_args()
    args.learning_curve_fracs = parse_learning_curve_fracs(args.lc_fracs)
    initialise_results_files()

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    run_embeddings = defaultdict(dict)

    for pair in PAIRINGS:
        npy_path = DATA_DIR / f"{pair}.npy"
        if not npy_path.exists():
            print(f"  Missing {npy_path}")
            continue

        print(f"\n----> {pair} ({N_RUNS} supervised runs)")
        X = to_T_C(np.load(npy_path, mmap_mode="r").astype(np.float32))
        T = X.shape[0]
        y = make_labels(T, pair)

        idx_split = int(TRAIN_FRAC * T)
        idx_train = np.arange(idx_split)
        idx_test = np.arange(idx_split, T)

        embeddings = []

        for run in range(N_RUNS):
            torch.manual_seed(run)
            run_dir = MODELS_DIR / pair
            run_dir.mkdir(parents=True, exist_ok=True)

            def make_cb(rd):
                def cb(num_steps, solver):
                    if num_steps > 0:
                        solver.save(logdir=str(rd), filename=f"{pair}_ckpt_{num_steps}.pt")

                return cb

            model = CEBRA(**MODEL_KWARGS)
            model.fit(
                X,
                y,
                callback=make_cb(run_dir),
                callback_frequency=CKPT_EVERY,
            )

            model.save(run_dir / f"{pair}_run{run}.pt")

            emb = model.transform(X)
            emb16 = emb.astype(np.float16)
            np.save(run_dir / f"{pair}_emb_run{run}.npy", emb16)
            embeddings.append(emb16)

            if args.run_cv or args.run_learning_curve:
                run_embeddings[run][pair] = np.asarray(emb, dtype=np.float32)

            gof_bits = cmetrics.goodness_of_fit_score(model, X, y).item()
            gof_hist = cmetrics.goodness_of_fit_history(model)
            np.savetxt(
                RESULTS / f"gof_history_{pair}_run{run}.csv",
                gof_hist,
                delimiter=",",
                header="bits_per_iter",
                comments="",
            )

            train_emb, test_emb = emb[idx_train], emb[idx_test]
            train_lab, test_lab = y[idx_train], y[idx_test]
            knn = KNNDecoder()
            knn.fit(train_emb, train_lab)
            knn_r2 = knn.score(test_emb, test_lab)

            cont_path = DATA_DIR / f"{pair}_continuous.npy"
            if cont_path.exists():
                cont_y = np.load(cont_path)
                cont_train, cont_test = cont_y[idx_train], cont_y[idx_test]
                linreg = L1LinearRegressor()
                linreg.fit(train_emb, cont_train)
                cont_r2 = linreg.score(test_emb, cont_test)

                coef_path = RESULTS / f"linreg_coefs_{pair}_run{run}.csv"
                np.savetxt(
                    coef_path,
                    linreg.coef_.reshape(1, -1),
                    delimiter=",",
                    header="coef_dim1,coef_dim2,coef_dim3",
                    comments="",
                )
            else:
                cont_r2 = ""

            with open(METRICS_CSV, "a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [
                        pair,
                        run,
                        f"{gof_bits:.4f}",
                        f"{knn_r2:.4f}",
                        f"{cont_r2:.4f}" if cont_r2 != "" else "",
                    ]
                )

            emb_fig = cebra.plot_embedding(emb, embedding_labels=y, markersize=3)
            loss_fig = cebra.plot_loss(model)
            ovw_fig = cebra.plot_overview(model, X)

            save_fig(emb_fig, FIGS_DIR / f"{pair}_embed_run{run}.png")
            save_fig(loss_fig, FIGS_DIR / f"{pair}_loss_run{run}.png")
            save_fig(ovw_fig, FIGS_DIR / f"{pair}_overview_run{run}.png")

        scores, _, _ = cmetrics.consistency_score(embeddings, between="runs")
        mean_c = scores.mean().item()
        variability = 1.0 - mean_c

        with open(CONSIST_TXT, "a", encoding="utf-8") as handle:
            handle.write(f"{pair:<25}  consistency={mean_c:.4f}  variability={variability:.4f}\n")
        with open(CONSIST_CSV, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([pair, f"{mean_c:.4f}", f"{variability:.4f}"])

        print(f"{pair:<25}  consistency={mean_c:.4f}  variability={variability:.4f}")

    run_optional_evaluation(args, run_embeddings)
    print("\nAll done - artefacts in 'models/' and 'results/'")


if __name__ == "__main__":
    main()
