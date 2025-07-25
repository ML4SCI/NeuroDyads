<img width="1052" height="436" alt="Screenshot 2025-07-14 205450" src="https://github.com/user-attachments/assets/db6e1cab-3b23-4900-817b-ff3c918c670a" />

<img width="668" height="363" alt="Screenshot 2025-07-14 205321" src="https://github.com/user-attachments/assets/38f7802c-88da-4d84-9df7-011e73fe6f2c" />

<img width="324" height="316" alt="Screenshot 2025-07-14 205423" src="https://github.com/user-attachments/assets/1d143939-8829-4cf2-94dc-98b84e0b7d4b" />

# NeuroDyads

A reproducible toolkit for extracting and interpreting low-dimensional neural embeddings from EEG hyperscanning of dyadic social interactions using the CEBRA framework.

---

## Background

We record 64-channel EEG simultaneously from two people (“speaker” ↔ “listener”) during a natural turn-taking conversation. CEBRA (Contrastive Embedding for Behavioral and Neural Analysis) learns a shared latent space that captures the joint dynamics of the pair—potentially revealing neural synchrony, connectivity patterns, and hidden signatures of social communication.

---

## Scientific Goals

- **Inter-brain Synchrony**  
  Quantify how two brains co-fluctuate during speaker vs. listener turns.

- **Clinical vs Neurotypical Comparison**  
  Use embeddings to distinguish dyads involving autistic participants from neurotypical pairs.

- **Behavioral Correlates**  
  Incorporate continuous measures (AQ-10, PRCA, RSAS, self-report ratings) to decode individual traits from the joint neural manifold.

- **Biomarker Discovery**  
  Identify embedding features (e.g. latent dimensions, synchrony metrics) that reliably index social-communication differences in autism spectrum disorder.

---

## Data & Conditions

- **Preprocessing**  
  We trim (“cut_60”) each EEG stream to the middle 60 s (no zero-padding), stack the two 64-channel arrays into a 128-ch time series, and save raw voltages as NumPy arrays.

- **Pairings**  
  - `spk9-lst10.npy` (participant 9 speaks → 10 listens)  
  - `lst9-spk10.npy` (participant 9 listens ← 10 speaks)

- **Scaling**  
  Raw only (no normalization)—this proved most effective in initial tests.

---

## Pipeline Overview

1. **Input Generation** (`prepare_cebra_input.py`)  
   - Load EDF, align lengths, stack channels, output raw `.npy`.

2. **Model Training** (`train_cebra_cut_supervised.py` & `train_cebra_cut_unsupervised.py`)  
   - **Supervised** (uses speaker/listener labels) vs **Unsupervised** (time-only contrast)  
   - 5 independent GPU runs per pairing  
   - Periodic checkpointing every 500 iterations  
   - Save full-dataset embeddings (`.npy`) and final model checkpoints (`.pt`)

3. **Metrics & Visualization**  
   - **Variability:** consistency across runs → variability = 1 − mean consistency  
   - **Goodness-of-Fit:** Info-NCE bits history  
   - **Decoding:** KNN on latent coords → R² for speaker/listener labels (and later continuous traits)  
   - **Plots:** embedding scatter, loss curves, combined overview

4. **Reproducibility**  
   - All seeds, versions, and paths are logged  
   - Results and figures organized under `models/…` and `results/…`

---

## What We’re Comparing Now

- **Supervised vs Unsupervised CEBRA-Time** on the same cut/raw inputs  
- **Variability** and **decoding performance** across runs  
- **Checkpointing** to inspect intermediate model states  

This experiment grid helps us pinpoint which training setup yields the most stable, behaviorally meaningful embeddings.

---

## Next Steps

- **Behavioral labels:** add AQ-10, PRCA, RSAS scores as continuous decoders  
- **ICA & frequency analysis:** test if ICA-cleaned or band-passed data alters embedding quality  
- **Generalization:** package the pipeline so new dyads/datasets plug in seamlessly  

---

## Getting Started

1. Clone this repo  
2. `pip install -r requirements.txt` (CEBRA 0.6.0a2, PyTorch ≥ 2.0)  
3. `python scripts/prepare_cebra_input.py` to generate inputs  
4. Run the supervised or unsupervised training script  
5. Inspect results under `results/` and `figures/`  

---

## Citation
Barde, A., Saffaryazdi, N., Withana, P., Patel, N., Sasikumar, P., & Billinghurst, M. (2019). Inter-brain connectivity: Comparisons between real and virtual environments using hyperscanning. In 2019 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct) (pp. 338–339). IEEE. https://doi.org/10.1109/ISMAR-Adjunct.2019.00-17

Jazayeri, M., & Afraz, A. (2017). Navigating the neural space in search of the neural code. Neuron, 93(5), 1003–1014. https://doi.org/10.1016/j.neuron.2017.02.019

Algumaei, A., Hettiarachchi, I. T., Farghaly, M., & Bhatti, A. (2023). The neuroscience of team dynamics: Exploring neurophysiological measures for assessing team performance. IEEE Access, 11, 129173–129194. https://doi.org/10.1109/ACCESS.2023.3332907 

---

Feel free to open an issue or PR if you have questions or suggestions!  
