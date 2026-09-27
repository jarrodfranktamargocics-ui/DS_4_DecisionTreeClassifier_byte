"""Create and execute the self-contained submission notebook."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
(root/'.runtime').mkdir(exist_ok=True)
os.environ.setdefault('IPYTHONDIR', str(root/'.runtime/ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(root/'.runtime/jupyter'))
source=(root/'train.py').read_text(encoding='utf-8')
source=source.replace("ROOT = Path(__file__).resolve().parent", "ROOT = Path.cwd()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent")
source=source.replace("if __name__ == '__main__': train()", '')
experiment_source=(root/'experiment.py').read_text(encoding='utf-8').replace("if __name__=='__main__': run_experiment()", '')
nb=nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell('# Mind Reader: Bank Marketing Decision Tree\n\nB.Y.T.E Data Science — Task 4\n\nPredict term-deposit subscription and explain each prediction with SHAP.\n\nDataset: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), Moro, Rita & Cortez, CC BY 4.0. Use `bank-full.csv`, 45,211 rows, 16 original inputs and target `y`. Balances are euros. Download is automatic; raw data stays out of Git.'),
    nbformat.v4.new_markdown_cell('## Experimental design\n\nExclude duration; preserve unknown categories and pdays=-1. Same stratified 80/20 split, seed 42. One-hot encoding is fitted inside each training fold. The optional contact indicator is derived from pdays.\n\nexperiment.py compares seven prior-setting variants and 48 seeded random configurations: depths 6/8/10/12; maximum leaves 32/64/128; minimum leaf sizes 20/50/100; pruning alpha 0/.0001/.0005/.001; Gini/entropy; no weights, balanced, 2x or 4x positive weights.\n\nFor each of five validation folds (seed 2026), use three inner folds on its training portion to select an F1 cutoff, refit on that training portion, and evaluate on the separate validation fold. Choose the highest mean validation F1; ties prefer fewer leaves. These means are model-selection scores, not unbiased performance estimates. Historical configurations were already explored on the training data, so these comparisons are diagnostic, not a pristine benchmark.\n\nFreeze the winner. Tune its final cutoff on three out-of-fold partitions of all training data (seed 314), then fit on all training data. The existing test split is excluded from this selection but has been inspected before. Report its scores as a same-split regression comparison. Schedule and contact count must be known before prediction.\n\nThe full experiment and export code are embedded below. This notebook reuses the saved training-only selection to avoid repeating the expensive search, validates the dataset/split fingerprints, reloads the fitted model, and recomputes evaluation and exports. Set reuse_selection=False (or call train()) to rerun the entire search. No test metric is used to choose the winner.\n'),
    nbformat.v4.new_code_cell(experiment_source),
    nbformat.v4.new_code_cell(source),
    nbformat.v4.new_code_cell('pipeline, X_test, y_test, explainer, metrics = train(reuse_selection=True)'),
    nbformat.v4.new_markdown_cell('## Held-out evaluation and global feature ranking\n\nImpurity importance is summed across one-hot columns to the original feature. It describes this fitted model and is not a causal ranking.'),
    nbformat.v4.new_code_cell("from IPython.display import display, Image, Markdown\ndisplay(pd.read_csv(ROOT/'outputs/experiment/candidate_results.csv').head(10))\ndisplay(pd.read_csv(ROOT/'outputs/model_comparison.csv'))\ndisplay(Image(filename=str(ROOT/'outputs/model_comparison.png')))\ndisplay(Image(filename=str(ROOT/'outputs/confusion_matrix.png')))\ndisplay(Image(filename=str(ROOT/'outputs/feature_importance.png')))\ndisplay(Markdown((ROOT/'outputs/model_summary.md').read_text(encoding='utf-8')))"),
    nbformat.v4.new_markdown_cell('## Explain an individual prediction\n\nTreeExplainer uses tree-path-dependent training counts to explain class-1 scores. The baseline plus SHAP contributions equals the score. For a weighted tree, scores are not calibrated real-world probabilities. SHAP signals describe model associations; they do not prove why a customer acts.'),
    nbformat.v4.new_code_cell("row = X_test.iloc[[0]]\nz = pipeline.named_steps['preprocess'].transform(row)\nphi = explainer.shap_values(z)[0,:,1]\nscore = pipeline.predict_proba(row)[0,1]\nassert abs(explainer.expected_value[1] + phi.sum() - score) < 1e-8\nartifact = json.loads((ROOT/'web/model.json').read_text())\ncontributions = {}\nfor f, value in zip(artifact['features'], phi):\n    contributions[f['source']] = contributions.get(f['source'], 0) + float(value)\nyes = score > .5\nranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)\nreasons = [(k,v) for k,v in ranked if (v>0 if yes else v<0)][:2]\nprint(f'The model predicts this customer will {\"\" if yes else \"not \"}subscribe because ' + ' and '.join(f'{k}={row.iloc[0][k]} pushes the subscription score {\"up\" if v>0 else \"down\"}' for k,v in reasons) + '.')\nprint(f'Model score: {score:.3f}; baseline: {explainer.expected_value[1]:.3f}')\ndisplay(pd.DataFrame(ranked, columns=['Feature', 'SHAP contribution']))"),
    nbformat.v4.new_markdown_cell('## Reload the trained Python model\n\nThe joblib artifact contains the fitted preprocessing/model pipeline and decision cutoff. Only load trusted joblib files. Raw inputs need previously_contacted derived from pdays; predict.py does this automatically.'),
    nbformat.v4.new_code_cell("import joblib\nsaved = joblib.load(ROOT/'outputs/trained_pipeline.joblib')\nnp.testing.assert_allclose(saved['pipeline'].predict_proba(X_test.iloc[:10]), pipeline.predict_proba(X_test.iloc[:10]))\nassert saved['decision_threshold'] == metrics['decision_threshold']\nprint('Reloaded pipeline matches the fitted model.')"),
    nbformat.v4.new_markdown_cell('## Limitations and deployment\n\nScores are uncalibrated; fairness and future-period performance are untested. Precision and recall may trade off. The browser uses exact per-leaf Shapley games, summing coalition products by size with polynomial coefficients to support deeper trees efficiently. Tests verify predictions, class labels, SHAP vectors and additivity across 100 test profiles plus each test-reached leaf. The cutoff determines labels separately from SHAP. Vercel remains paused.'),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=root/'notebooks/bank_marketing.ipynb';path.parent.mkdir(exist_ok=True)
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.source = cell.source.replace('yes = score > .5', "yes = score >= metrics['decision_threshold']")
        cell.source = cell.source.replace("print(f'Model score: {score:.3f}; baseline:", "print(f'Model score: {score:.3f}; cutoff: {metrics[\"decision_threshold\"]:.3f}; baseline:")
nbformat.write(nb,path)
NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata':{'path':str(root)}}).execute()
nbformat.write(nb,path)
print(f'Executed notebook saved: {path}')
