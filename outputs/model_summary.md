# Model summary

UCI Bank Marketing; 45,211 records. Target: term-deposit subscription.

80/20 stratified split, seed 42. Preprocessing fitted inside each of five CV folds. Selection metric: F1. Test set untouched until selection.

Selected parameters: `{'model__class_weight': 'balanced', 'model__max_depth': 5, 'model__min_samples_leaf': 150}`. CV F1: 0.3779.

- accuracy: 0.7944
- precision: 0.2974
- recall: 0.5558
- f1: 0.3875
- baseline_accuracy: 0.8830

Top five features: [('poutcome', 0.4445514705543877), ('contact', 0.2535508884459269), ('month', 0.14021398506989802), ('housing', 0.11700718246384371), ('campaign', 0.028673260458385724)].

Duration excluded because it is unavailable before the call. Categorical fields use one-hot encoding, numeric fields pass through; trees do not require scaling. Unknown categories are retained. The call schedule and current campaign contact count must be known at prediction time.

SHAP explains class-1 model scores using training path counts. Class weighting, if selected, makes these scores uncalibrated and unsuitable as literal purchase probabilities. Explanations describe the model, not causal effects. Historical Portuguese campaigns may not generalize to other populations or current campaigns. Random splitting does not measure future-period performance. Age is included for this educational experiment; fairness has not been evaluated.

CSV SHA256: `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`
