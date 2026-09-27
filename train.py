"""Reproducible training and browser artifact export. Run with Python 3.12+."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import zipfile
import os
import warnings
from urllib.request import urlopen
os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd()/'.runtime/matplotlib'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings('ignore', message='IProgress not found.*')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, roc_auc_score
from joblib import dump, load

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'

def load_data():
    path = ROOT / 'data/bank-full.csv'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with urlopen(URL, timeout=90) as response:
            outer = zipfile.ZipFile(BytesIO(response.read()))
        if 'bank-full.csv' in outer.namelist():
            raw = outer.read('bank-full.csv')
        else:
            inner = zipfile.ZipFile(BytesIO(outer.read('bank.zip')))
            raw = inner.read('bank-full.csv')
        path.write_bytes(raw)
    return pd.read_csv(path, sep=';')

def train(reuse_selection=False):
    df = load_data()
    assert df.shape == (45211, 17) and not df.isna().any().any()
    # Unknown is an explicit category; pdays=-1 means no previous contact.
    X, y = df.drop(columns=['y', 'duration']), df.y.eq('yes').astype(int)
    X = X.assign(previously_contacted=(X.pdays != -1).astype(int))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    from experiment import run_experiment
    selection_path = ROOT/'outputs/experiment/selected_pipeline.joblib'
    if not reuse_selection or not selection_path.exists():
        run_experiment()
    selected = load(selection_path)
    selection = selected['selection']
    assert selection['csv_sha256'] == hashlib.sha256((ROOT/'data/bank-full.csv').read_bytes()).hexdigest(), 'Dataset changed: rerun experiment'
    assert selection['training_indices_sha256'] == hashlib.sha256(np.asarray(X_train.index,dtype=np.int64).tobytes()).hexdigest(), 'Split changed: rerun experiment'
    pipeline, threshold = selected['pipeline'], selected['decision_threshold']
    if not selection['winner']['indicator']:
        X_train = X_train.drop(columns='previously_contacted', errors='ignore')
        X_test = X_test.drop(columns='previously_contacted', errors='ignore')
        X = X.drop(columns='previously_contacted', errors='ignore')
    categorical = X_train.select_dtypes(include=['object', 'str']).columns.tolist()
    numeric = [c for c in X_train if c not in categorical]
    prep, model = pipeline.named_steps['preprocess'], pipeline.named_steps['model']
    threshold_table = pd.read_csv(ROOT/'outputs/experiment/final_threshold_search.csv') if selection['winner']['tuned'] else pd.DataFrame()
    transformed = prep.transform(X_test)
    scores = pipeline.predict_proba(X_test)[:, 1]
    pred = (scores >= threshold).astype(int)
    metrics = {k: float(fn(y_test, pred)) for k, fn in [('accuracy',accuracy_score),('precision',precision_score),('recall',recall_score),('f1',f1_score)]}
    metrics.update(train_rows=len(X_train), test_rows=len(X_test), positive_rate=float(y.mean()), baseline_accuracy=float((y_test==0).mean()), confusion_matrix=confusion_matrix(y_test,pred).tolist(), best_params=selection['winner']['params'], validation_f1=selection['mean_validation_f1'], incumbent_validation_f1=selection['incumbent_validation_f1'], selected_candidate=selection['winner']['name'], decision_threshold=threshold, average_precision=float(average_precision_score(y_test,scores)), roc_auc=float(roc_auc_score(y_test,scores)), leaves=int(model.get_n_leaves()), depth=int(model.get_depth()))
    descriptors = [{'source':c,'category':None} for c in numeric]
    for c, categories in zip(categorical, prep.named_transformers_['cat'].categories_):
        descriptors.extend({'source':c,'category':str(v)} for v in categories)
    tree = model.tree_
    nodes = [{'left':int(tree.children_left[i]),'right':int(tree.children_right[i]),'feature':int(tree.feature[i]),'threshold':float(tree.threshold[i]),'weight':float(tree.weighted_n_node_samples[i]),'probability':float(tree.value[i,0,1]/tree.value[i,0].sum())} for i in range(tree.node_count)]
    importance = {c:0. for c in X}
    for descriptor, value in zip(descriptors, model.feature_importances_):
        importance[descriptor['source']] += float(value)
    ranking = sorted(importance.items(), key=lambda x:x[1], reverse=True)
    schema = []
    for c in X:
        if c == 'previously_contacted': continue  # Always derived from pdays, never independently editable.
        field = {'name':c,'default':str(X_train[c].mode().iloc[0]) if c in categorical else int(X_train[c].median())}
        if c in categorical: field['options'] = sorted(X_train[c].unique().tolist())
        else: field.update(min=int(X_train[c].min()),max=int(X_train[c].max()))
        schema.append(field)
    # Path-dependent SHAP explains the raw classifier output (class-1 leaf score).
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent', model_output='raw')
    # Cover every leaf represented in the test set, plus 100 held-out profiles.
    validation = X_test.iloc[:100].copy()
    leaf_ids = model.apply(transformed)
    representatives = [int(np.flatnonzero(leaf_ids==leaf)[0]) for leaf in np.unique(leaf_ids)]
    validation = pd.concat([validation, X_test.iloc[representatives]], ignore_index=True)
    validation_z = prep.transform(validation)
    values = explainer.shap_values(validation_z)[:,:,1]
    base = float(explainer.expected_value[1])
    np.testing.assert_allclose(base + values.sum(axis=1), model.predict_proba(validation_z)[:,1], atol=1e-8)
    fixtures = [{'input':validation.drop(columns='previously_contacted', errors='ignore').iloc[i].to_dict(),'probability':float(model.predict_proba(validation_z[i:i+1])[0,1]),'prediction':int(model.predict_proba(validation_z[i:i+1])[0,1]>=threshold),'shap':values[i].tolist()} for i in range(len(validation))]
    examples = []
    for label in [0,1]:
        ids = np.flatnonzero(pred==label)
        if len(ids): examples.append({'label':'Predicted subscriber' if label else 'Predicted non-subscriber','input':X_test.drop(columns='previously_contacted', errors='ignore').iloc[int(ids[0])].to_dict()})
    artifact = {'nodes':nodes,'features':descriptors,'schema':schema,'metrics':metrics,'importance':ranking[:5],'base':base,'examples':examples,'weighted':model.class_weight is not None,'decision_threshold':threshold}
    for directory in ['web','outputs','tests']: (ROOT/directory).mkdir(exist_ok=True)
    for path,obj in [('web/model.json',artifact),('outputs/metrics.json',metrics),('tests/fixtures.json',fixtures)]:
        (ROOT/path).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    pd.read_csv(ROOT/'outputs/experiment/candidate_results.csv').to_csv(ROOT/'outputs/cv_results.csv',index=False)
    threshold_table.to_csv(ROOT/'outputs/threshold_search.csv',index=False)
    dump({'pipeline':pipeline,'decision_threshold':threshold,'derived_features':{'previously_contacted':'int(pdays != -1)'}}, ROOT/'outputs/trained_pipeline.joblib')
    ConfusionMatrixDisplay(confusion_matrix(y_test,pred),display_labels=['No subscription','Subscription']).plot(cmap='Blues',colorbar=False)
    plt.title('Decision tree · held-out test set'); plt.tight_layout(); plt.savefig(ROOT/'outputs/confusion_matrix.png',dpi=170); plt.close()
    fig,ax=plt.subplots(figsize=(8,4)); ax.barh([x[0] for x in ranking[:5]][::-1],[x[1] for x in ranking[:5]][::-1],color='#277568'); ax.set_xlabel('Aggregated impurity importance'); ax.set_title('Top five features'); fig.tight_layout(); fig.savefig(ROOT/'outputs/feature_importance.png',dpi=170); plt.close(fig)
    summary = f"""# Model summary

UCI Bank Marketing; 45,211 records. Same 80/20 stratified split, seed 42. Call duration excluded.

## Training-only selection

55 candidates: seven comparisons of prior settings, and 48 seeded random configurations spanning depths 6/8/10/12, leaf caps 32/64/128, minimum leaf sizes 20/50/100, pruning alpha 0/.0001/.0005/.001, Gini/entropy, and class weights none/balanced/2x/4x.

Five validation folds (seed 2026). Inside each validation-training partition, three inner folds produce out-of-fold scores for cutoff tuning. Choose the F1 cutoff using only those inner predictions, refit the candidate on that partition, then measure F1 on the separate validation fold. Fixed-cutoff comparison candidates use 0.5. Candidate means are selection scores, not unbiased final estimates: they are used to pick the winner.

Winner: {selection['winner']['name']}. Parameters: `{selection['winner']['params']}`.
Validation F1: {selection['mean_validation_f1']:.4f}; previous configuration retuned under the same procedure: {selection['incumbent_validation_f1']:.4f}.

After selection, tune the final cutoff on three out-of-fold partitions of all training data (seed 314), then fit the final pipeline on all training data. Cutoff: {threshold:.6f}; scores >= cutoff predict subscription. The model and cutoff are frozen before test evaluation.

## Same-split test evaluation

This test set was inspected for earlier versions. It is a regression comparison, not a fresh untouched benchmark. No test outcomes were used to choose this experiment's winner. External/prospective data is needed for an independent final assessment.

""" + '\n'.join(f'- {k}: {metrics[k]:.4f}' for k in ['accuracy','precision','recall','f1','average_precision','roc_auc','baseline_accuracy']) + f"""

Actual depth {model.get_depth()}, {model.get_n_leaves()} leaves. Top five features: {ranking[:5]}.

## Preprocessing and explanations

One-hot encoding is fitted inside each training fold; numerical fields pass through. Unknown categories and pdays=-1 are retained. When used, previously_contacted = int(pdays != -1) is derived automatically. Schedule and contact count must be known before the predicted call. Scores are uncalibrated, especially with class weights.

SHAP TreeExplainer uses training path counts. Baseline plus contributions equals the score. Browser inference sums exact per-leaf Shapley games and is checked against Python across test profiles and all test-reached leaves. The final cutoff determines class labels separately. Correlated features can share attribution; SHAP is not causal. Fairness and future-period performance are untested.

Saved pipeline and cutoff: outputs/trained_pipeline.joblib. Browser export: web/model.json. Load only trusted joblib files.

CSV SHA256: `{selection['csv_sha256']}`
"""
    (ROOT/'outputs/model_summary.md').write_text(summary,encoding='utf-8')
    baseline_path = ROOT/'outputs/baseline_metrics.json'
    if baseline_path.exists():
        old = json.loads(baseline_path.read_text(encoding='utf-8'))
        names = ['accuracy','precision','recall','f1']
        previous = json.loads((ROOT/'outputs/previous_metrics.json').read_text(encoding='utf-8'))
        comparison = pd.DataFrame({'metric':names, 'original':[old[k] for k in names], 'previous':[previous[k] for k in names], 'current':[metrics[k] for k in names]})
        comparison['change_vs_previous_pp'] = 100*(comparison.current-comparison.previous)
        comparison.to_csv(ROOT/'outputs/model_comparison.csv',index=False)
        fig,ax=plt.subplots(figsize=(8,4)); positions=np.arange(len(names))
        for offset,key,color in [(-.25,'original','#bcc6cb'),(0,'previous','#719998'),(.25,'current','#1b5c4b')]:
            ax.bar(positions+offset, comparison[key]*100, width=.24, label=key.capitalize(),color=color)
        ax.set_xticks(positions,[n.capitalize() for n in names]); ax.set_ylim(0,100); ax.set_ylabel('Percent'); ax.set_title('Same test split · three model versions'); ax.legend()
        fig.tight_layout(); fig.savefig(ROOT/'outputs/model_comparison.png',dpi=170); plt.close(fig)
    print(json.dumps(metrics,indent=2))
    return pipeline, X_test, y_test, explainer, metrics

if __name__ == '__main__': train()
