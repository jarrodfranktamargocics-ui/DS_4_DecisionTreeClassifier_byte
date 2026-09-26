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
nb=nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell('# Mind Reader: Bank Marketing Decision Tree\n\nB.Y.T.E Data Science — Task 4\n\nPredict term-deposit subscription and explain each prediction with SHAP.\n\nDataset: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), Moro, Rita & Cortez, CC BY 4.0. Use `bank-full.csv`, 45,211 rows, 16 original inputs and target `y`. Balances are euros. Download is automatic; raw data stays out of Git.'),
    nbformat.v4.new_markdown_cell('## Experimental design\n\nDrop call duration: it is unknown before the call. Retain `unknown` as a category and `pdays=-1` as the no-prior-contact sentinel. Assert schema and missingness. Stratify the 80/20 train/test split with seed 42. Fit one-hot encoding only inside each training fold; numerical inputs need no scaling. Use 5-fold training CV to select depth, minimum leaf size, and class weighting by F1; cap the tree at 8 leaves. Evaluate once on the held-out test split. Scheduled contact day/month and contact number must be available at prediction time.\n\nBecause only about 12% subscribe, accuracy alone is misleading. Compare to the majority-class baseline and report positive-class precision, recall, F1, and confusion matrix.'),
    nbformat.v4.new_code_cell(source),
    nbformat.v4.new_code_cell('pipeline, X_test, y_test, explainer, metrics = train()'),
    nbformat.v4.new_markdown_cell('## Held-out evaluation and global feature ranking\n\nImpurity importance is summed across one-hot columns to the original feature. It describes this fitted model and is not a causal ranking.'),
    nbformat.v4.new_code_cell("from IPython.display import display, Image, Markdown\ndisplay(Image(filename=str(ROOT/'outputs/confusion_matrix.png')))\ndisplay(Image(filename=str(ROOT/'outputs/feature_importance.png')))\ndisplay(Markdown((ROOT/'outputs/model_summary.md').read_text(encoding='utf-8')))"),
    nbformat.v4.new_markdown_cell('## Explain an individual prediction\n\nTreeExplainer uses tree-path-dependent training counts to explain class-1 scores. The baseline plus SHAP contributions equals the score. For a weighted tree, scores are not calibrated real-world probabilities. SHAP signals describe model associations; they do not prove why a customer acts.'),
    nbformat.v4.new_code_cell("row = X_test.iloc[[0]]\nz = pipeline.named_steps['preprocess'].transform(row)\nphi = explainer.shap_values(z)[0,:,1]\nscore = pipeline.predict_proba(row)[0,1]\nassert abs(explainer.expected_value[1] + phi.sum() - score) < 1e-8\nartifact = json.loads((ROOT/'web/model.json').read_text())\ncontributions = {}\nfor f, value in zip(artifact['features'], phi):\n    contributions[f['source']] = contributions.get(f['source'], 0) + float(value)\nyes = score > .5\nranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)\nreasons = [(k,v) for k,v in ranked if (v>0 if yes else v<0)][:2]\nprint(f'The model predicts this customer will {\"\" if yes else \"not \"}subscribe because ' + ' and '.join(f'{k}={row.iloc[0][k]} pushes the subscription score {\"up\" if v>0 else \"down\"}' for k,v in reasons) + '.')\nprint(f'Model score: {score:.3f}; baseline: {explainer.expected_value[1]:.3f}')\ndisplay(pd.DataFrame(ranked, columns=['Feature', 'SHAP contribution']))"),
    nbformat.v4.new_markdown_cell('## Limitations and deployment\n\nHistorical data and a random split do not establish future campaign performance. The small tree trades flexibility for interpretability. Class-weighted scores require calibration before interpreting them as probabilities. Fairness has not been audited. This is a learning demonstration, not a customer targeting recommendation.\n\nThe browser receives the actual fitted tree as JSON and computes exact Shapley values over the at-most-seven split features using the same path-count value function. `node tests/model.test.mjs` checks 100 predictions and complete SHAP vectors against Python. This lets Vercel serve a static demo without a Python API.'),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=root/'notebooks/bank_marketing.ipynb';path.parent.mkdir(exist_ok=True)
nbformat.write(nb,path)
NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata':{'path':str(root)}}).execute()
nbformat.write(nb,path)
print(f'Executed notebook saved: {path}')
