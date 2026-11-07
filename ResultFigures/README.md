# Figures index

Representative, already-generated result figures from the finalized
uniform-250 Hz pipeline, included here so the analysis can be understood
without re-running anything. All were produced by scripts in `Evaluation/`,
`CEBRATraining/`, and `GenderStateRuns/` against the full results tree (not
included in this repository — see `CEBRA_250Hz_Pipeline.md`, section 12).

| file | analysis | what it shows |
|---|---|---|
| `aq_delta6_embedding.png` | Six-class \|ΔAQ\| CEBRA embedding | The 3D neural manifold produced by the core pipeline, colored by |ΔAQ| class. |
| `dyad_aq_full_retraining_null_20perms.png` | Dyad-level AQ magnitude, full retraining | Real leave-one-dyad-out accuracy sits inside the null of 20 retrainings (p = 0.571). |
| `individual_aq_full_retraining_null_200perms.png` | Participant-level exact AQ, full retraining | Real accuracy sits inside the null of 200 retrainings (p = 0.692 leave-one-dyad-out, 0.468 leave-one-participant-out). |
| `individual_aq_FROZEN_decoder_only_null.png` | Participant-level exact AQ, frozen encoder | **Superseded.** Labels were shuffled on one trained embedding, so this does not test the encoder. Kept only to show why the earlier p-value misled. |
| `sex_state_comparison.png` | Speak/listen/rest state decoding, female-only vs. male-only | State separation (driven by rest-vs-task) replicates across both sexes. |
| `fullband_speakerfirst_lonlat_by_magnitude.png` | Full-band embedding geometry | Longitude/latitude projection of the embedding, colored by Low/High AQ magnitude. |
| `aq_delta6_confusion_matrix.png` | Six-class \|ΔAQ\| decoding | Confusion matrix for the six-class classifier. |
| `aq_delta6_svm_multiclass.png` | Six-class \|ΔAQ\| decoding | 5-NN vs. linear-SVM vs. RBF-SVM comparison, with per-class recall. |
| `fullband_gmm_k_sweep.png` | Dyad-identity check | BIC-based GMM component-count sweep on the baseline embedding. |
| `fullband_gmm_k5_vs_dyad.png` | Dyad-identity check | Whether GMM components at K=5 correspond to dyad identity (they do not; this does not rule out the dyad-identity confound shown by the permutation controls). |
| `band_metric_comparison.png` | Frequency-band removal | Effect of removing each frequency band on decoding/geometry metrics. |
| `nonosc_vs_fullband_metrics.png` | Aperiodic / non-oscillatory analysis | Removing all oscillatory content changes decoding/geometry metrics negligibly. |
| `nonosc_permutation_null.png` | Aperiodic / non-oscillatory analysis | Dyad-level AQ permutation control repeated on the non-oscillatory-only signal. |
