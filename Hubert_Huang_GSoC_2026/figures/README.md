# Figures index

Twelve representative, already-generated result figures from the finalized
uniform-250 Hz pipeline, included here so the analysis can be understood
without re-running anything. All were produced by scripts in `evaluation/`,
`training/`, and `gender_analysis/` against the full results tree (not
included in this repository — see the top-level `README.md`'s "Reproducing
the results" section).

| file | analysis | what it shows |
|---|---|---|
| `aq_delta6_embedding.png` | Six-class \|ΔAQ\| CEBRA embedding | The 3D neural manifold produced by the core pipeline, colored by |ΔAQ| class. |
| `fullband_speakerfirst_permutation_5nn_null.png` | Dyad-level AQ magnitude, full-retraining permutation control | **Headline negative result** — real leave-one-dyad-out accuracy vs. a 5-retraining permutation null; the real value sits inside the null. |
| `individual_aq_permutation_null.png` | Participant-level AQ, full-retraining permutation control | **Headline positive result** — participant-level AQ decoding clears its own permutation null. |
| `sex_state_comparison.png` | Speak/listen/rest state decoding, female-only vs. male-only | State separation (driven by rest-vs-task) replicates across both sexes. |
| `fullband_speakerfirst_lonlat_by_magnitude.png` | Full-band embedding geometry | Longitude/latitude projection of the embedding, colored by Low/High AQ magnitude. |
| `aq_delta6_confusion_matrix.png` | Six-class \|ΔAQ\| decoding | Confusion matrix for the six-class classifier. |
| `aq_delta6_svm_multiclass.png` | Six-class \|ΔAQ\| decoding | 5-NN vs. linear-SVM vs. RBF-SVM comparison, with per-class recall. |
| `fullband_gmm_k_sweep.png` | Dyad-identity check | BIC-based GMM component-count sweep on the baseline embedding. |
| `fullband_gmm_k5_vs_dyad.png` | Dyad-identity check | Whether GMM components at K=5 correspond to dyad identity (they do not, ruling out a trivial dyad-identity confound). |
| `band_metric_comparison.png` | Frequency-band removal | Effect of removing each frequency band on decoding/geometry metrics. |
| `nonosc_vs_fullband_metrics.png` | Aperiodic / non-oscillatory analysis | Removing all oscillatory content changes decoding/geometry metrics negligibly. |
| `nonosc_permutation_null.png` | Aperiodic / non-oscillatory analysis | Dyad-level AQ permutation control repeated on the non-oscillatory-only signal. |
