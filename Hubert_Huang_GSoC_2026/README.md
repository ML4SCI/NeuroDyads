# NeuroDyads — CEBRA Analysis Pipeline (GSoC 2026)

## 1. Title

**CEBRA-Based Decoding of Autism Quotient, Dyad Identity, and Conversational
State from Two-Person EEG ("NeuroDyads")**

## 2. Contributor

**Hubert Huang** — Google Summer of Code 2026, ML4Sci

## 3. GSoC / ML4Sci NeuroDyads

This folder is one contributor's submission to the
[ML4Sci NeuroDyads](https://github.com/ML4SCI/NeuroDyads) project, developed
during GSoC 2026. It builds on the project's existing EEG preprocessing and
CEBRA-training scaffolding (see the top-level `README.md`,
`PreprocessingPipeline/`, and `train_cebra*.py`) and adds a complete,
independently reproducible analysis pipeline: uniform-sampling-rate
preprocessing, CEBRA embedding training, decoding evaluations, full-retraining
permutation controls, embedding-geometry diagnostics, and a gender/state
extension.

## 4. Problem statement

Two people (a "dyad") each wear EEG while alternating speaking, listening, and
resting roles in conversation. The scientific questions this submission
addresses:

- Does a CEBRA embedding of dyadic EEG encode a *participant-level* trait —
  Autism Quotient (AQ) — in a way that is robust to confounds like dyad
  identity?
- Does it encode *dyad-level* AQ-difference information (|ΔAQ| between the two
  members of a dyad), and if so, does that signal survive a control that
  retrains the entire encoder under permuted labels (not just a downstream
  decoder)?
- Do these results depend on oscillatory (band-limited) content, or do they
  persist in a purely aperiodic (non-oscillatory) representation of the same
  signal?
- Do conversational states (speak/listen/rest) and participant gender modulate
  any of the above?

## 5. Dataset description

The dataset is two-person (dyadic) EEG recorded during conversation, cut into
per-role (speak/listen/rest) segments per participant, across up to 33 dyads
(a subset of dyads/files is excluded per-analysis after QC/flagging — see each
run's own manifest CSV for the exact count used). Each dyad has an associated
Autism Quotient (AQ) score per participant and, for the gender extension,
an explicit per-participant gender label.

**No raw or participant-identifiable data is included in this submission.**
Specifically excluded:
- Raw or filtered EDF recordings.
- The demographic spreadsheet (participant IDs, AQ scores, gender labels,
  interaction-quality ratings). Scripts that need it take a
  `--demographics-xlsx` / `--valence-xlsx` argument — point it at your own
  copy to reproduce those analyses.
- Any manifest CSV or JSON generated from the above (these live under
  `results/`, which is git-ignored).

Section 11 below describes the expected *shape* of these inputs precisely
enough to reproduce the pipeline against your own copy of the data.

## 6. Final preprocessing pipeline

Executed roughly in this order (each stage is a standalone, resumable script
under `preprocessing/` and `datasets/`):

1. **`preprocessing/audit_edf_sampling_rates.py`** — read-only integrity/rate
   audit of every source EDF (sampling rate, channel count, duration,
   SHA-256), never modifies originals.
2. **`preprocessing/organize_edf_by_dyad.py`** *(optional)* — reorganize a
   flat folder of cut EDFs into `dyad/participant/role` subfolders; several
   downstream scripts can also scan a flat folder directly.
3. **`preprocessing/downsample_and_filter_edf.py`** — downsample to 250 Hz
   where needed and produce band-removed variants with a local, verified FIR
   filter (firwin, Hamming window, zero-phase, 1 Hz transition edges).
4. **`preprocessing/resample_33dyad_to_250.py`** — resample the remaining
   1000 Hz recordings to a uniform 250 Hz. This step exists because the
   original 33-dyad set mixed 1000 Hz and 250 Hz sources: a fixed
   sample-count crop under mixed rates gave wildly different real-time
   windows per dyad (90.2 s vs. 361 s) and a fixed `time_offsets=10` spanned
   10 ms vs. 40 ms. Uniform 250 Hz makes a sample-count crop a duration crop.
5. **`preprocessing/qc_filtered_edfs.py`** / **`reclassify_filter_qc.py`** —
   PSD-based proof that each "filtered" output is actually attenuated in its
   intended band (catches silently-unfiltered files).
6. **`preprocessing/compare_new_nonosc_upload.py`** /
   **`verify_uploaded_band_files.py`** — provenance/QC checks confirming
   whether externally supplied "filtered" or "non-oscillatory" EDF sets are
   genuinely different from the raw signal (both were found NOT to be, which
   motivated step 7).
7. **`preprocessing/extract_aperiodic_fooof.py`** +
   **`preprocessing/run_aperiodic_extractor_adapter.py`** — a self-built,
   FOOOF/specparam-based aperiodic (non-oscillatory) extraction, used because
   the externally supplied "non-oscillatory" data turned out to be
   unmodified raw signal. The adapter script wraps an extraction module
   written by another NeuroDyads collaborator (see its docstring); that
   module itself is not bundled here.
8. **`datasets/batch_speaker_first_stack.py`** /
   **`batch_stack_edfs_for_cebra.py`** — stack per-role recordings into the
   fixed-sample-count, speaker-first arrays CEBRA trains on, with a manifest
   CSV recording dyad/participant/role/AQ-magnitude per array.
9. **`datasets/relabel_manifest_magnitude.py`** /
   **`build_nonosc_speakerfirst.py`** — derive AQ-magnitude labels
   (Low/High, or six-class |ΔAQ|) onto a manifest, or build the
   non-oscillatory-only counterpart dataset.

All stages are idempotent (skip work whose output already exists) and take
explicit `--input-dir`/`--output-dir`/`--data-root`-style arguments — run any
script with `--help` for its exact interface.

## 7. CEBRA configuration

Every CEBRA model in this submission (`training/`, `controls/`,
`gender_analysis/*_cebra.py`) uses the same fixed configuration, so results
are comparable across analyses:

```python
CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.12,
    conditional="time_delta",
    output_dimension=3,
    distance="cosine",
    time_offsets=10,
    max_iterations=5000,   # per training/run_250hz_rerun.py's default stages
    device="cuda_if_available",
)
```

Model seed is fixed at `0` for baseline runs; permutation controls (Section 9)
additionally fix a **label**-permutation seed (`0..4`, five retrainings) that
is separate from the model seed.

## 8. Major analysis modules

- **AQ magnitude (Low/High) decoding** — `training/run_band_cebra_analysis.py`
- **Six-class |ΔAQ| decoding** — `training/run_aq_delta6_pipeline.py`
- **Leave-one-dyad-out evaluation** — built into the training scripts' metric
  computation (grouped 5-NN and GMM metrics never train/test on the same
  dyad)
- **SVM comparisons** (one-vs-one linear/RBF vs. 5-NN reference, pairwise
  class separability) — `evaluation/aq_delta6_svm_and_panels.py`
- **Longitude/latitude embedding geometry** —
  `geometry/analyze_cebra_longitude_latitude.py`,
  `geometry/analyze_cebra_gmm_lonlat.py`
- **GMM K-sweep / dyad-identity check** —
  `geometry/analyze_gmm_component_identity.py` (does a data-driven GMM
  component count correspond to dyad identity, independent of any label?)
- **Cross-entropy / 3D manifold diagnostics** —
  `geometry/analyze_cebra_cross_entropy_3d.py`
- **Full-retraining permutation controls** (dyad-level and participant-level)
  — `controls/run_dyad_aq_permutation_control.py`,
  `controls/run_posthoc_decoder_permutation.py`,
  `controls/summarize_permutation.py`,
  `controls/followup_individual_aq_checks.py`
- **Frequency-band removal comparison** —
  `evaluation/build_band_comparison.py`
- **Aperiodic / non-oscillatory analysis** — `preprocessing/extract_aperiodic_fooof.py`
  + the `nonosc_*` stages in `training/run_250hz_rerun.py` / figures in
  `evaluation/build_session_figures.py`
- **Participant-level AQ** — `training/run_individual_aq_role_mapping.py`
- **Gender and speak/listen/rest state analyses** — everything under
  `gender_analysis/` (`audit_gender_metadata.py`,
  `run_gender_dyad_cebra.py`, `run_gender_participant_states.py`,
  `run_sex_specific_states.py`, `run_female_valence_cebra.py`, and their
  figure builders)

## 9. Key scientific conclusion

**Dyad-level AQ-difference decoding did not survive a full-encoder-retraining
permutation control, while participant-level AQ decoding did survive its
corresponding control.**

Concretely, in the finalized uniform-250 Hz re-run: the dyad-level
leave-one-dyad-out 5-NN accuracy for |ΔAQ| sat *inside* the null distribution
from five independent CEBRA retrainings under permuted dyad labels
(real ≈ 0.759 vs. permutation-null 0.756 ± 0.009, empirical p ≈ 0.50) — i.e.
statistically indistinguishable from chance-level structure once the encoder
itself is retrained under the null, even though naive (non-retrained) decoding
looked strong. By contrast, participant-level AQ decoding cleared its own
full-retraining permutation null clearly (real ≈ 0.247 vs. null 0.116 ± 0.020,
empirical p ≈ 0.0099). The practical lesson: **a downstream decoder trained on
a fixed embedding can look like strong evidence for a dyad-level effect that
disappears once you check whether the embedding itself, not just the decoder,
could have produced that accuracy by chance.** This same permutation-control
pattern (dyad-level result null, participant-level result real) also held
after removing all oscillatory content (Section 8, aperiodic analysis),
arguing the conclusion is not an artifact of any one frequency band.

See `figures/fullband_speakerfirst_permutation_5nn_null.png` (dyad-level,
null) and `figures/individual_aq_permutation_null.png` (participant-level,
survives) for the corresponding plots, and `figures/README.md` for the full
figure index.

## 10. Reproduction commands

Every script takes explicit path arguments and documents them under
`--help`; the commands below show the intended order for the core pipeline.
Adjust paths to your own data layout (Section 11).

```bash
pip install -r requirements.txt

# 1. Preprocessing (see Section 6 for the full stage list / rationale)
python preprocessing/audit_edf_sampling_rates.py --input-dir "/path/to/EDF Files cut" --out-csv audit.csv
python preprocessing/resample_33dyad_to_250.py --input-dir "/path/to/EDF Files cut" --output-dir "/path/to/Resampled250"
python preprocessing/qc_filtered_edfs.py --help   # see flags for your filtered-band layout

# 2. Build the CEBRA input dataset + manifest
python datasets/batch_speaker_first_stack.py --input-dir "/path/to/Resampled250" --out-dir "/path/to/Filtered CEBRA Inputs/fullband250"

# 3. Train + evaluate on the uniform 250 Hz dataset (baseline, permutation
#    control, six-class, SVM comparisons, GMM identity check)
python training/run_250hz_rerun.py --data-root "/path/to" --dry-run   # preview the plan
python training/run_250hz_rerun.py --data-root "/path/to"             # run it

# 4. Participant-level AQ
python training/run_individual_aq_role_mapping.py --help

# 5. Aperiodic / non-oscillatory extraction (optional; requires `fooof`)
python preprocessing/extract_aperiodic_fooof.py --help

# 6. Gender / state extension (requires your own demographics spreadsheet)
python gender_analysis/audit_gender_metadata.py \
    --demographics-xlsx /path/to/demographics.xlsx \
    --speakerfirst-manifest "/path/to/Filtered CEBRA Inputs/fullband250/fullband250_manifest.csv" \
    --out-dir results/gender_analysis

# 7. Reports, LaTeX snippets, and the final figure bundle
python evaluation/build_reports_and_export.py --results-root results
python evaluation/build_session_figures.py --results-root results
python evaluation/build_overleaf_figures_250.py --results-root results
```

Every other script (`controls/`, `geometry/`, `evaluation/aq_delta6_svm_and_panels.py`,
etc.) is invoked similarly against the outputs of the stage before it — run
`python <script>.py --help` for its exact required inputs.

## 11. Expected input directory structure

```
<data-root>/
  EDF Files cut/                          # per-dyad, per-participant, per-role EDFs
    dyad01_1_speak.edf
    dyad01_1_listen.edf
    dyad01_1_rest.edf
    dyad01_2_speak.edf
    ...
  Resampled250/                           # output of resample_33dyad_to_250.py
  Filtered CEBRA Inputs/
    fullband250/
      fullband250_manifest.csv            # dyad_id, speaker_id, listener_id, flagged, ...
      dyad01_speakerfirst.npy
      ...
<demographics>.xlsx                       # sheet 'Full-Info': dyad_id, Speaker ID,
                                           # Speaker Gender, Listener ID, Listener Gender,
                                           # AQ scores, (optional) Gender Congruence
```

The exact filename conventions (regexes for dyad/participant/role) are
documented in each script's own `--help`/docstring, since a couple of stages
tolerate more than one naming convention found across the project's source
data drops.

## 12. Expected outputs

Running the pipeline against a populated `<data-root>` produces, under
`results/` (git-ignored — regenerate rather than expecting it to be present
in this repository):

- `results/aug4_pipeline/rerun250/{fullband_baseline,dyad_permutation,aq_delta6,aq_delta6_svm,gmm_identity_check}/` —
  metrics JSON, embeddings (`.npy`), trained model weights, and PNGs for each
  stage of `training/run_250hz_rerun.py`.
- `results/aug4_pipeline/individual_aq_role/` — participant-level AQ metrics,
  embedding, and permutation-null figure.
- `results/aug4_pipeline/overleaf_figures_250hz/` — the 12-figure bundle
  assembled by `evaluation/build_overleaf_figures_250.py`.
- `results/aug4_pipeline/figures_session_aug11/` — the 10-figure curated set
  from `evaluation/build_session_figures.py`.
- `results/aug4_pipeline/AUG4_METRICS_MASTER.csv`, `AUG4_MEETING_SUMMARY.md`,
  and `overleaf_export/` (LaTeX snippets + zipped figures) from
  `evaluation/build_reports_and_export.py`.
- `results/gender_analysis/` — gender-metadata audit JSON, per-sex/state
  CEBRA runs, and their figures.

## 13. Software / package versions

Developed and run with:
- Python 3.12
- `cebra` (see `requirements.txt`; pin your installed version with
  `python -c "import cebra; print(cebra.__version__)"` — recorded
  automatically into each run's `metrics.json` as `cebra_version`)
- `torch` (CUDA build recommended; CEBRA falls back to CPU automatically via
  `device="cuda_if_available"`)
- `mne >= 1.0`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- Optional: `fooof` (aperiodic extraction), `python-pptx` + `Pillow`
  (lightning-talk deck), `openpyxl` (reading `.xlsx` demographics)

See `requirements.txt` for the full pinned-free dependency list.

## 14. Blog post

BLOG_URL: _TODO — add the final GSoC 2026 blog post URL here._

A curated set of publication-ready figures for that post (not part of this
code submission) was prepared separately under `results/gsoc_blog_figures/`.

## 15. Citation / acknowledgments

- Google Summer of Code 2026, hosted by **ML4Sci**, project **NeuroDyads**.
- Built on the existing NeuroDyads preprocessing scaffolding in this
  repository (`PreprocessingPipeline/`, `prepare_cebra_input*.py`,
  `train_cebra*.py`).
- Embeddings trained with [CEBRA](https://cebra.ai)
  (Schneider, Lee & Mathis, *Nature* 2023).
- The aperiodic/non-oscillatory extraction adapter in
  `preprocessing/run_aperiodic_extractor_adapter.py` wraps an extraction
  module written by another NeuroDyads collaborator (Michelle); that module
  is not bundled in this submission — see the adapter's docstring for the
  interface it expects.
- Filter specification for locally-produced band-removed EDFs
  (`preprocessing/downsample_and_filter_edf.py`) follows a specification
  provided by that same collaborator.

If you use this code, please cite the ML4Sci NeuroDyads project and CEBRA
as above.
