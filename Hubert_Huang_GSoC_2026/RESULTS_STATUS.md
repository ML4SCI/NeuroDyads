# Results status and corrections

Last updated: 6 November 2026. This file supersedes any conflicting statement in
earlier reports or in the first version of this folder (PR #4).

## Permutation controls

"Full retraining" means CEBRA is trained from scratch for every permuted labelling.
All controls use the same configuration and model-initialization seed (0) as the
real run. Empirical p = (1 + null results >= real) / (1 + completed permutations).

| analysis | status | retrainings | real | null mean +/- SD | null range | null >= real | p |
|---|---|---|---|---|---|---|---|
| Dyad AQ magnitude, leave-one-dyad-out (LODO) | complete at 20; 200 requested | 20 | 0.7589 | 0.7649 +/- 0.0166 | 0.7427-0.8061 | 11 | 0.571 |
| Participant exact AQ, LODO | complete | 200 | 0.2437 | 0.2589 +/- 0.0335 | 0.1222-0.3717 | 138 | 0.692 |
| Participant exact AQ, leave-one-participant-out (LOPO) | complete | 200 | 0.2300 | 0.2306 +/- 0.0386 | 0.1529-0.3644 | 93 | 0.468 |
| Speaking/listening/resting, LODO | **incomplete** | 10 of 200 | 0.5689 | 0.3502 +/- 0.0169 | 0.3310-0.3885 | 0 | not final (lowest possible at n=10: 0.091) |
| Non-oscillatory dyad AQ, LODO | 19 dyads, low power | 5 | 0.9206 | 0.9142 +/- 0.0116 | not re-derived | 1 | 0.333 |
| Individual gender | **not run** | 0 of 200 | - | - | - | - | - |

**Interpretation.** Neither AQ decoding result is established under this control.
That is not evidence that AQ has no neural signature. The labels are constant within
a dyad or participant, so the design cannot separate AQ from recording identity.

**Speaking/listening/resting so far (10 retrainings, not a final result).**

| LODO comparison | real | null mean | null maximum |
|---|---|---|---|
| Speaking vs resting | 0.8806 | 0.5149 | 0.5680 |
| Listening vs resting | 0.8143 | 0.5173 | 0.5507 |
| Speaking vs listening | 0.5354 | 0.5229 | 0.5941 |

Resting currently separates well beyond the null, but speaking vs listening does not.
These are interim numbers. No relationship between state decoding and AQ has been
tested. Surviving this control would not exclude EMG, movement or session confounds.

**Gender.** The earlier p = 0.4129 is not a retraining result. It shuffled gender
labels on one embedding that was trained on state, never on gender.

Per-seed values are in `results_summary/`, with no participant-level labels.

## Limitations

- **Transductive evaluation.** The encoder trains on all groups, including the group
  later held out by the decoder. Grouped decoding alone cannot rule out leakage, which
  is why retraining is the governing control.
- **Fixed initialization.** Every run uses seed 0, so the nulls reflect label changes,
  not initialization variance. Each real result comes from one trained model.
- **Mixed compute.** The real dyad baseline recorded `device=cpu`. Device was not
  recorded for the October runs. The PyTorch build changed from CPU-only to a GPU build
  on 9 October, and timings point to GPU for dyad seeds 5-19, all 200 participant
  seeds and the state seeds.
- **Blocked state permutation.** Labels are permuted within each participant, and about
  1 in 6 participants keep their true labels by chance in any iteration. This makes the
  null somewhat conservative.
- **Participant AQ metrics.** The 200-run control recorded accuracy only. Balanced
  accuracy was requested but not saved. It can be recomputed from the saved models.
- **No retraining control yet.** Six-class |dAQ|, band removal, SVM comparisons and
  female-only/male-only state models have no full-retraining null.

## Cohorts

Different analyses legitimately use different cohorts.

| analysis | cohort | samples | crop per recording |
|---|---|---|---|
| Enrolled in demographic sheet and raw export | 36 dyads | - | - |
| Dyad AQ magnitude (speaker-first) | 33 dyads, 66 files | 9,801,000 | 297 s |
| Six-class \|dAQ\| 0-5 | 32 dyads, 64 files | 9,504,000 | 297 s |
| Band removal, non-oscillatory | 19 dyads, 38 files | 5,776,000 | 304 s |
| Participant exact AQ | 40 participants, 80 recordings, 21 dyads | 6,080,000 | 304 s |
| Speaking/listening/resting, gender | 68 participants, 204 recordings, 35 dyads | 14,535,000 | 285 s |

- **36 to 33:** removed dyads 18, 38 and 41. Dyads 38 and 41 have one artifact-flagged
  participant each (98 and 104); their partners remain usable. Dyad 18's IDs (`59a`,
  `59b`) do not match the resampling script's integer-only filename pattern. That is
  the likely cause of its loss, though no log states it.
- **33 to 32:** one dyad has |dAQ| = 7, outside the 0-5 class range. This is a label
  definition, not a quality exclusion, and only affects the six-class analysis.
- **33 to 19 (separate branch):** the 19 dyads are the natively 250 Hz recordings (IDs
  20 and above) minus flagged dyads 38 and 41. These analyses were never rerun on all 33.
- **"20 dyads":** no such cohort was found in the project. Its source is unresolved.

## Possible cohort expansion (not applied)

- **The 28 participants from dyads below 20** already have 250 Hz speak, listen and rest
  files in the resampled folder. Adding them would give the participant AQ analysis
  68 participants at a 297 s crop. It would require new training and a new control.
  The recorded resampling QC is technical only, so artifact review is still needed.
- **Dyad 18** needs filename handling and artifact review. Its shortest speak or listen
  recording is 294.5 s, below the 297 s crop, so including it there would shorten the
  crop to 294.5 s for every recording and require retraining. Its recordings exceed the
  285 s crop used by the state analysis.

## Corrections to earlier versions and reports

1. **Participant-level AQ "survived" its control: wrong.** That p = 0.0099 came from a
   frozen-encoder test. It fails under full retraining.
2. **The non-oscillatory pattern "also held": unsupported.** No participant-level
   non-oscillatory control exists.
3. **`training/run_250hz_rerun.py` could not run as published.** It called a `scripts/`
   folder absent from this submission. Only its dry run had been tested. Paths are fixed;
   a full run has not been repeated.
4. **README reproduction commands used wrong flags.** They now match each parser.
5. **"Current crop is 304 s, not 297 s": misleading.** Both are real: 297 s for the
   33-dyad dataset and 304 s for the 40-participant dataset.
6. **"The 28 participants need new extraction": wrong.** Their role-separated 250 Hz
   files exist, as described above.
7. **"Never QC-reviewed" was overstated.** Technical resampling QC exists. No artifact
   review was found.
8. **The six-class sample count was wrong.** It is 9,504,000.
9. **A figure named `200_perm_dyad_aq.png` holds 20 permutations.** The published copy
   is `figures/dyad_aq_full_retraining_null_20perms.png`.
10. **State counts were overstated by one.** Seed 0 was a 50-iteration benchmark and was
    excluded, leaving 10 valid runs. Seed 0 must be rerun at 5000 iterations.
11. **Concurrent GPU workers were not "tested and found unsafe".** Only one process's
    memory use was observed, and that reading included other applications.
12. **"CPU-only" descriptions of the October runs are likely wrong.** See Mixed compute.

## Remaining work

1. Finish the state control: seed 0 and seeds 11-199.
2. Run the individual-gender control, including its real gender-supervised baseline.
3. Decide whether to extend the dyad AQ control from 20 to 200 retrainings.
4. Decide whether to expand the participant AQ cohort to 68 and do artifact review.
5. Confirm the source of the "20 dyads" figure with the mentors.
