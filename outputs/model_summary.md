# Model summary

UCI Bank Marketing; 45,211 records. Same 80/20 stratified split, seed 42. Call duration excluded.

## Training-only selection

55 candidates: seven comparisons of prior settings, and 48 seeded random configurations spanning depths 6/8/10/12, leaf caps 32/64/128, minimum leaf sizes 20/50/100, pruning alpha 0/.0001/.0005/.001, Gini/entropy, and class weights none/balanced/2x/4x.

Five validation folds (seed 2026). Inside each validation-training partition, three inner folds produce out-of-fold scores for cutoff tuning. Choose the F1 cutoff using only those inner predictions, refit the candidate on that partition, then measure F1 on the separate validation fold. Fixed-cutoff comparison candidates use 0.5. Candidate means are selection scores, not unbiased final estimates: they are used to pick the winner.

Winner: expanded_28. Parameters: `{'min_samples_leaf': 20, 'max_leaf_nodes': 128, 'max_depth': 12, 'criterion': 'entropy', 'class_weight': {0: 1, 1: 4}, 'ccp_alpha': 0.0005}`.
Validation F1: 0.4736; previous configuration retuned under the same procedure: 0.4159.

After selection, tune the final cutoff on three out-of-fold partitions of all training data (seed 314), then fit the final pipeline on all training data. Cutoff: 0.579186; scores >= cutoff predict subscription. The model and cutoff are frozen before test evaluation.

## Same-split test evaluation

This test set was inspected for earlier versions. It is a regression comparison, not a fresh untouched benchmark. No test outcomes were used to choose this experiment's winner. External/prospective data is needed for an independent final assessment.

- accuracy: 0.8761
- precision: 0.4721
- recall: 0.4962
- f1: 0.4839
- average_precision: 0.4101
- roc_auc: 0.7829
- baseline_accuracy: 0.8830

Actual depth 12, 57 leaves. Top five features: [('poutcome', 0.3255151065624472), ('month', 0.23132849774722494), ('contact', 0.14458537684606215), ('day', 0.09151635105445106), ('housing', 0.06957879306103312)].

## Preprocessing and explanations

One-hot encoding is fitted inside each training fold; numerical fields pass through. Unknown categories and pdays=-1 are retained. When used, previously_contacted = int(pdays != -1) is derived automatically. Schedule and contact count must be known before the predicted call. Scores are uncalibrated, especially with class weights.

SHAP TreeExplainer uses training path counts. Baseline plus contributions equals the score. Browser inference sums exact per-leaf Shapley games and is checked against Python across test profiles and all test-reached leaves. The final cutoff determines class labels separately. Correlated features can share attribution; SHAP is not causal. Fairness and future-period performance are untested.

Saved pipeline and cutoff: outputs/trained_pipeline.joblib. Browser export: web/model.json. Load only trusted joblib files.

CSV SHA256: `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`
