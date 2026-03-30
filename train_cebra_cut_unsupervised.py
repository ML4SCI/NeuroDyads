# scripts/train_cebra_cut_unsupervised.py
"""
Unsupervised CEBRA-Time training on cut_60/raw with
speaker-vs-listener colouring in the 3-D embedding plot.
"""

import torch, csv, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

import cebra
from cebra import CEBRA, KNNDecoder
from cebra.integrations.sklearn import metrics as cmetrics

# ------------------------------------------------------------------ #
# 1. Paths
# ------------------------------------------------------------------ #
ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data" / "processed" / "cut_60" / "raw"
MODELS_DIR = ROOT / "models"  / "cut_60" / "raw"
RESULTS    = ROOT / "results"
FIGS_DIR   = RESULTS / "figures"
for p in (MODELS_DIR, RESULTS, FIGS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# 2. Config
# ------------------------------------------------------------------ #
PAIRINGS   = ["spk9-lst10", "lst9-spk10"]
N_RUNS     = 5
TRAIN_FRAC = 0.8
CKPT_EVERY = 500

MODEL_KWARGS = dict(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.12,
    max_iterations=5000,
    conditional="time",
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

def save_fig(fig_or_ax, path):
    if isinstance(fig_or_ax, tuple):
        fig = fig_or_ax[0]
    elif hasattr(fig_or_ax, "savefig"):
        fig = fig_or_ax
    else:
        fig = fig_or_ax.get_figure()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

summary_rows = []

# ------------------------------------------------------------------ #
# 4. Training loop
# ------------------------------------------------------------------ #
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

for pair in PAIRINGS:
    np_path = DATA_DIR / f"{pair}.npy"
    if not np_path.exists():
        print(f"❌  Missing {np_path}")
        continue

    X = to_T_C(np.load(np_path, mmap_mode="r").astype(np.float32))   # (T, C)
    T, C = X.shape
    time_labels = np.arange(T, dtype=np.int32)      # for optional gradient plots

    split = int(TRAIN_FRAC * T)
    idx_train = np.arange(split)
    idx_val   = np.arange(split, T)

    # channel-wise speaker/listener label (0=speaker,1=listener)
    half = C // 2
    if pair == "spk9-lst10":
        ch_labels = np.array([0]*half + [1]*half)
    else:                               # lst9-spk10  → speaker second
        ch_labels = np.array([1]*half + [0]*half)

    emb_runs = []

    for run in range(N_RUNS):
        torch.manual_seed(run)
        run_dir = MODELS_DIR / pair
        run_dir.mkdir(parents=True, exist_ok=True)

        def cb_factory(out_dir):
            def cb(step, solver):
                if step:
                    solver.save(logdir=str(out_dir),
                                filename=f"{pair}_ckpt_{step}.pt")
            return cb

        model = CEBRA(**MODEL_KWARGS)
        model.fit(X,
                  callback=cb_factory(run_dir),
                  callback_frequency=CKPT_EVERY)
        model.save(run_dir / f"{pair}_run{run}.pt")

        emb = model.transform(X)                        # (T,3)
        np.save(run_dir / f"{pair}_emb_run{run}.npy", emb.astype(np.float16))
        emb_runs.append(emb.astype(np.float16))

        # ---------------- Kaggle-style colour trick ---------------
        # repeat each time-point for every channel so we can colour
        emb_big   = np.repeat(emb, C, axis=0)           # (T*C,3)
        lab_big   = np.tile(ch_labels, T)               # (T*C,)
        emb_ax    = cebra.plot_embedding(emb_big,
                                         embedding_labels=lab_big,
                                         markersize=3)
        save_fig(emb_ax, FIGS_DIR / f"{pair}_embed_run{run}.png")
        # ----------------------------------------------------------

        loss_ax = cebra.plot_loss(model)
        save_fig(loss_ax, FIGS_DIR / f"{pair}_loss_run{run}.png")

        ovw_fig = cebra.plot_overview(model, X)
        save_fig(ovw_fig, FIGS_DIR / f"{pair}_overview_run{run}.png")

        # metrics
        gof = cmetrics.goodness_of_fit_score(model, X).item()
        knn = KNNDecoder()
        knn.fit(emb[idx_train], lab_big.reshape(T, C)[:split, 0])  # dummy train
        acc = knn.score(emb[idx_val], lab_big.reshape(T, C)[split:, 0])

        with open(RESULTS / f"{pair}_metrics_run{run}.json", "w") as fp:
            json.dump({"pair": pair, "run": run,
                       "gof_bits": gof, "val_acc": acc}, fp, indent=2)

        summary_rows.append({"pair": pair, "run": run,
                             "gof_bits": gof, "val_acc": acc})

    # consistency
    cs, _, _ = cmetrics.consistency_score(emb_runs, between="runs")
    with open(RESULTS / f"{pair}_consistency.json", "w") as fp:
        json.dump({"mean_consistency": float(cs.mean()),
                   "variability": float(1-cs.mean())},
                  fp, indent=2)

# final CSV
pd.DataFrame(summary_rows).to_csv(
    RESULTS / "decoder_accuracy_summary.csv", index=False
)
print("✓ Done – coloured 3-D plots are in results/figures/")