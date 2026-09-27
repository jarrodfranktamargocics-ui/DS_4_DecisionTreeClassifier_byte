# Accuracy and recall target study

Run with `python -u explore_targets.py`. This study did not access the reserved test outcomes and did not replace the current model.

Twelve Decision Tree configurations were evaluated: the six best configurations from the earlier experiment with revised cutoff selection, and six larger trees (depths 14–18, 128–256 leaf limits, minimum leaf sizes 20–50, with and without class weighting). Call duration remained excluded.

Each configuration used the same five training-validation folds as the previous experiment. In each fold, three inner folds selected the cutoff, first looking for accuracy of at least 90%, recall of at least 58%, and precision/F1 above the original baseline values. If that combination was unavailable, the cutoff maximized F1 among eligible recall-focused choices. The 58% inner recall floor is a selection constraint, not a guarantee on separate validation records.

No inner fold found a cutoff satisfying the full target. No configuration met the target on mean outer validation metrics either.

| Candidate | Validation accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Best F1: expanded_45_recall | 86.37% | 43.71% | 50.08% | 46.36% |
| Highest recall: larger_7 | 81.49% | 33.60% | 59.25% | 42.84% |
| Current model, prior validation | 87.46% | 46.62% | 48.24% | 47.36% |

The current model remains selected because none of these candidates improves its validation F1 or achieves the requested combination. These are training-validation results, not new test results. Reusing validation folds for experiments introduces selection optimism; the study does not establish a ceiling on achievable performance.

See `design.json`, `fold_results.csv`, and `summary.csv` for the configuration and per-fold results.
