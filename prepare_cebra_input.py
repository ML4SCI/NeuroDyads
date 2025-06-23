# scripts/prepare_cebra_input.py
"""
Create NumPy inputs for CEBRA sanity-check grid
----------------------------------------------
Outputs: 12 .npy files in data/processed/<clean>/<scale>/
         shape = (n_channels_total, n_times)

Grid:
    cleaning  : zeropad_30  |  cut_60
    pairing   : spk9-lst10, lst9-spk10, stacked
    scaling   : raw         |  normalized
"""

from pathlib import Path
import numpy as np
import mne
import os
import json

# ---------------------------------------------------------------------
# 1.  Paths
# ---------------------------------------------------------------------
ROOT   = Path(__file__).resolve().parents[1]          # repo root
RAW    = ROOT / "data" / "raw"
PROC   = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 2.  Helper functions
# ---------------------------------------------------------------------
def load_eeg(edf_path):
    """Return EEG data as np.ndarray (channels, time)."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data(picks="eeg")
    return data

def align_lengths(a, b):
    """Trim the longer array so a and b share the same #samples."""
    T = min(a.shape[1], b.shape[1])
    return a[:, :T], b[:, :T]

def minmax_per_channel(x):
    """Scale each channel to [0, 1] independently."""
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    rng  = np.where((xmax - xmin) == 0, 1, xmax - xmin)
    return (x - xmin) / rng

def save_npy(array, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, array)
    print(f"✓ Saved {out_path.name:45}  {array.shape}")

# ---------------------------------------------------------------------
# 3.  Grid definition
# ---------------------------------------------------------------------
GRID = [
    # cleaning, pairing_name,  EDF paths (relative to RAW/<clean>/...)
    (
        "zeropad_30",
        "spk9-lst10",
        ("individual/nt9_speak_zeropad_30_components_preprocessed.edf",
         "individual/nt10_listen_zeropad_30_components_preprocessed.edf"),
    ),
    (
        "zeropad_30",
        "lst9-spk10",
        ("individual/nt9_listen_zeropad_30_components_preprocessed.edf",
         "individual/nt10_speak_zeropad_30_components_preprocessed.edf"),
    ),
    (
        "zeropad_30",
        "stacked",
        ("stacked/nt9_zeropad_speak_listen_stacked.edf",
         "stacked/nt10_zeropad_listen_speak_stacked.edf"),
    ),

    (
        "cut_60",
        "spk9-lst10",
        ("individual/nt9_speak_cut_60_components_preprocessed.edf",
         "individual/nt10_listen_cut_60_components_preprocessed.edf"),
    ),
    (
        "cut_60",
        "lst9-spk10",
        ("individual/nt9_listen_cut_60_components_preprocessed.edf",
         "individual/nt10_speak_cut_60_components_preprocessed.edf"),
    ),
    (
        "cut_60",
        "stacked",
        ("stacked/nt9_cut_speak_listen_stacked.edf",
         "stacked/nt10_cut_listen_speak_stacked.edf"),
    ),
]

# ---------------------------------------------------------------------
# 4.  Main loop
# ---------------------------------------------------------------------
for clean, pairing, (edf_a_rel, edf_b_rel) in GRID:
    edf_a = RAW / clean / edf_a_rel
    edf_b = RAW / clean / edf_b_rel

    # ---------- load ----------
    A = load_eeg(edf_a)   # (ch, t)
    B = load_eeg(edf_b)

    # ---------- align ----------
    A, B = align_lengths(A, B)            # ensure equal length T
    combined = np.vstack([A, B])          # (ch_A+ch_B, T)

    # ---------- save raw ----------
    out_raw = PROC / clean / "raw" / f"{pairing}.npy"
    save_npy(combined, out_raw)

    # ---------- save normalized ----------
    combined_norm = minmax_per_channel(combined)
    out_norm = PROC / clean / "normalized" / f"{pairing}.npy"
    save_npy(combined_norm, out_norm)

print("\nAll NumPy files generated")
