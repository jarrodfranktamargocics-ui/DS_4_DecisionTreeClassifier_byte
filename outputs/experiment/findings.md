# Experiment findings

All values in the first table are means over the same five training-validation folds. Cutoff tuning occurs in separate inner folds. Candidate means were used for selection, so the winning score is not an unbiased final estimate. Historical configurations had already been explored on this training pool.

| Variant | Validation precision | Validation recall | Validation F1 |
|---|---:|---:|---:|
| Original tree, fixed cutoff | 30.01% | 53.20% | 38.37% |
| Original tree, tuned cutoff | 30.63% | 50.91% | 38.24% |
| Larger Gini tree + indicator, fixed cutoff | 32.90% | 54.19% | 40.92% |
| Larger Gini tree + indicator, tuned cutoff | 35.34% | 50.72% | 41.65% |
| Previous entropy tree, fixed cutoff | 32.86% | 53.91% | 40.81% |
| Previous entropy tree, tuned cutoff | 35.24% | 50.74% | 41.59% |
| Previous tree without indicator, tuned cutoff | 35.24% | 50.74% | 41.59% |
| Selected expanded/pruned tree | 46.62% | 48.24% | 47.36% |

The larger Gini configuration was stronger than the original small tree. Tuning helped the larger trees but did not help the original small tree in this comparison. Entropy did not clearly outperform Gini for the previous configuration. Removing the derived contact indicator from the previous tree produced identical validation scores. The final search changed several settings together, so we cannot attribute its gain uniquely to pruning, weight changes, or capacity.

Selected parameters: `{'min_samples_leaf': 20, 'max_leaf_nodes': 128, 'max_depth': 12, 'criterion': 'entropy', 'class_weight': {'0': 1, '1': 4}, 'ccp_alpha': 0.0005}`. Pruning left 57 leaves (cap 128). Score cutoff 0.579186.

## Same-split test comparison

| Metric | Previous | Current | Change (pp) |
|---|---:|---:|---:|
| accuracy | 83.40% | 87.61% | +4.21 |
| precision | 35.42% | 47.21% | +11.79 |
| recall | 50.85% | 49.62% | -1.23 |
| f1 | 41.75% | 48.39% | +6.63 |

394 fewer false positives, 13 additional false negatives. This test split was already inspected for earlier versions; it is a regression comparison, not a new untouched benchmark. External/prospective evaluation remains necessary.
