# scripts/train_cebra.py
"""
Run 10× CEBRA-Time per processed .npy file and log consistency.

Folder assumptions  (created in prepare step):
data/processed/<clean>/<scale>/<pairing>.npy

Outputs:
  - results/consistency_results.txt
  - models/<clean>/<scale>/<pairing>_run<i>.pt
  - models/<clean>/<scale>/<pairing>_emb_run<i>.npy
"""
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))

from pathlib import Path
import numpy as np
from cebra import CEBRA
from cebra.integrations.sklearn import metrics as cmetrics
import os

# ---------------------------------------------------------------------
# 1.  Paths
# ---------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parents[1]          # repo root
PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
OUT_TXT = RESULTS / "consistency_results.txt"

# wipe old results file if it exists
if OUT_TXT.exists():
    OUT_TXT.unlink()

# ---------------------------------------------------------------------
# 2.  Config
# ---------------------------------------------------------------------
N_RUNS = 10  # embeddings per .npy
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

# ---------------------------------------------------------------------
# 3.  Helper
# ---------------------------------------------------------------------
def to_time_samples_first(arr):
    """Ensure shape = (T, channels)."""
    return arr.T if arr.shape[0] > arr.shape[1] else arr

def create_model():
    return CEBRA(**MODEL_KWARGS)

# ---------------------------------------------------------------------
# 4.  Walk through the 12 files
# ---------------------------------------------------------------------
for clean_dir in sorted(PROC.iterdir()): # zeropad_30, cut_60
    if not clean_dir.is_dir():
        continue
    for scale_dir in sorted(clean_dir.iterdir()):  # raw, normalized
        for npy_path in sorted(scale_dir.glob("*.npy")):
            pair_name = npy_path.stem  # e.g. spk9-lst10
            tag = f"{clean_dir.name}/{scale_dir.name}/{pair_name}"

            # Skip this condition if already processed
            if tag == "cut_60/normalized/lst9-spk10":
                print(f"\n⏭  Skipping {tag}")
                continue

            print(f"\n--->  {tag}")

            X = np.load(npy_path, mmap_mode="r")
            X = to_time_samples_first(X.astype(np.float32))

            # --------------- run N times -----------------
            embeds = []
            for run in range(N_RUNS):
                torch.manual_seed(run)   #for reproducible randomness
                model = create_model()
                model.fit(X)
                emb = model.transform(X)          # (T, D)
                embeds.append(emb.astype(np.float16))

                # save model & embedding
                run_dir = MODELS / clean_dir.name / scale_dir.name
                run_dir.mkdir(parents=True, exist_ok=True)
                model.save(run_dir / f"{pair_name}_run{run}.pt")
                np.save(run_dir / f"{pair_name}_emb_run{run}.npy", emb)

            # --------------- consistency -----------------
            scores, _, _ = cmetrics.consistency_score(
                embeddings=embeds,
                between="runs"
            )
            mean_consistency = scores.mean().item()
            variability      = 1.0 - mean_consistency

            # --------------- log -------------------------
            line = f"{tag:<40}  consistency={mean_consistency:.4f}  variability={variability:.4f}\n"
            with open(OUT_TXT, "a") as f:
                f.write(line)
            print(line.strip())

print(f"\nResults saved to {OUT_TXT}")