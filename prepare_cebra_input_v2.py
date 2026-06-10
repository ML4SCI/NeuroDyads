# scripts/prepare_cebra_input.py
"""
Create NumPy inputs for CEBRA from stacked EDF files.
-----------------------------------------------------
Edit the PAIRS and OUTPUT_DIR variables below before running.

Output: one .npy file per pair, shape = (n_channels_total, n_times)
        saved to OUTPUT_DIR/<pairing_name>.npy
"""

from pathlib import Path
import numpy as np
import mne

# ---------------------------------------------------------------------
# 1.  Configure these before running
# ---------------------------------------------------------------------

# List of (pairing_name, edf_path_1, edf_path_2) tuples.
# Add one entry per dyad/pairing you want to process.
PAIRS = [

    (
        "dyad39_100sp-101ln",   # P1sp-P2ln
        r"C:\Users\miche\Downloads\EEG-MalaiaLab-Research\GSoC 2026\Cut EEG Datafiles\100-101_dyad39_cut_example\cut_dyad39_100_speak.edf",   # P1sp
        r"C:\Users\miche\Downloads\EEG-MalaiaLab-Research\GSoC 2026\Cut EEG Datafiles\100-101_dyad39_cut_example\cut_dyad39_101_listen.edf",   # P2ln
    ),
    (
        "dyad39_100ln-101sp",   # P1ln-P2sp
        r"C:\Users\miche\Downloads\EEG-MalaiaLab-Research\GSoC 2026\Cut EEG Datafiles\100-101_dyad39_cut_example\cut_dyad39_100_listen.edf",   # P1ln
        r"C:\Users\miche\Downloads\EEG-MalaiaLab-Research\GSoC 2026\Cut EEG Datafiles\100-101_dyad39_cut_example\cut_dyad39_101_speak.edf",   # P2sp
    ),
    # Add more dyads here, e.g.:
    # (
    #     "dyad20_60sp-61ln",   # P1sp-P2ln
    #     r"C:\path\to\cut_dyad20_60_speak.edf",    # P1sp
    #     r"C:\path\to\cut_dyad20_61_listen.edf",   # P2ln
    # ),
    # (
    #     "dyad20_60ln-61sp",   # P1ln-P2sp
    #     r"C:\path\to\cut_dyad20_60_listen.edf",   # P1ln
    #     r"C:\path\to\cut_dyad20_61_speak.edf",    # P2sp
    # ),
]

# Folder where .npy files will be saved (must already exist)
OUTPUT_DIR = Path(r"C:\Users\miche\Downloads\EEG-MalaiaLab-Research\GSoC 2026\CEBRA Input Files")

# ---------------------------------------------------------------------
# 2.  Helper functions
# ---------------------------------------------------------------------

def load_eeg(edf_path):
    """Return EEG data as np.ndarray (channels, time)."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    return raw.get_data(picks="eeg")

def align_lengths(a, b):
    """Trim the longer array so a and b share the same number of samples."""
    T = min(a.shape[1], b.shape[1])
    return a[:, :T], b[:, :T]

def save_npy(array, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, array)
    print(f"✓ Saved {out_path.name:45}  shape={array.shape}")

# ---------------------------------------------------------------------
# 3.  Main loop
# ---------------------------------------------------------------------

for pairing_name, edf_a_path, edf_b_path in PAIRS:
    print(f"\nProcessing: {pairing_name}")

    A = load_eeg(edf_a_path)   # (channels, time)
    B = load_eeg(edf_b_path)

    A, B = align_lengths(A, B)
    combined = np.vstack([A, B])   # (channels_A + channels_B, time)

    out_path = OUTPUT_DIR / f"{pairing_name}.npy"
    save_npy(combined, out_path)

print("\nAll NumPy files generated.")
