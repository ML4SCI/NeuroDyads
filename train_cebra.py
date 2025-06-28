import torch
import numpy as np
from pathlib import Path
import logging
from cebra import CEBRA
from cebra.integrations.sklearn import metrics as cmetrics

# ---------------------------------------------------------------------
# 1. Logging Configuration
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 2. Paths and Constants
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
OUT_TXT = RESULTS / "consistency_results.txt"

MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Remove old results file if it exists
if OUT_TXT.exists():
    OUT_TXT.unlink()

# ---------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------
N_RUNS = 10
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
# 4. Helper Functions
# ---------------------------------------------------------------------
def to_time_samples_first(arr: np.ndarray) -> np.ndarray:
    """Ensure shape = (T, channels)."""
    return arr.T if arr.shape[0] > arr.shape[1] else arr

def create_model() -> CEBRA:
    """Create a new CEBRA model instance."""
    return CEBRA(**MODEL_KWARGS)

def run_cebra_on_file(npy_path: Path, clean_name: str, scale_name: str) -> None:
    pair_name = npy_path.stem
    tag = f"{clean_name}/{scale_name}/{pair_name}"

    if tag == "cut_60/normalized/lst9-spk10":
        logger.info(f"Skipping {tag}")
        return

    logger.info(f"Processing {tag}")

    try:
        X = np.load(npy_path, mmap_mode="r")
        X = to_time_samples_first(X.astype(np.float32))
    except Exception as e:
        logger.error(f"Failed to load or process {npy_path}: {e}")
        return

    embeds = []
    for run in range(N_RUNS):
        try:
            torch.manual_seed(run)
            model = create_model()
            model.fit(X)
            emb = model.transform(X).astype(np.float16)
            embeds.append(emb)

            run_dir = MODELS / clean_name / scale_name
            run_dir.mkdir(parents=True, exist_ok=True)
            model.save(run_dir / f"{pair_name}_run{run}.pt")
            np.save(run_dir / f"{pair_name}_emb_run{run}.npy", emb)
        except Exception as e:
            logger.error(f"Run {run} failed for {tag}: {e}")

    try:
        scores, _, _ = cmetrics.consistency_score(embeddings=embeds, between="runs")
        mean_consistency = scores.mean().item()
        variability = 1.0 - mean_consistency
        line = f"{tag:<40}  consistency={mean_consistency:.4f}  variability={variability:.4f}\n"
        with open(OUT_TXT, "a") as f:
            f.write(line)
        logger.info(line.strip())
    except Exception as e:
        logger.error(f"Failed to compute consistency for {tag}: {e}")

# ---------------------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------------------
def main():
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU device: %s", torch.cuda.get_device_name(0))

    for clean_dir in sorted(PROC.iterdir()):
        if not clean_dir.is_dir():
            continue
        for scale_dir in sorted(clean_dir.iterdir()):
            for npy_path in sorted(scale_dir.glob("*.npy")):
                run_cebra_on_file(npy_path, clean_dir.name, scale_dir.name)

    logger.info(f"Results saved to {OUT_TXT}")

if __name__ == "__main__":
    main()
