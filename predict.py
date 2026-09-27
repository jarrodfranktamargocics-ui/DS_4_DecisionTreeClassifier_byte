"""Run the saved Python pipeline: python predict.py --example, or --input profile.json."""
from pathlib import Path
import argparse
import json
import warnings
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.runtime/matplotlib'))
warnings.filterwarnings('ignore', message='IProgress not found.*')
import joblib
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent

def predict(profile):
    artifact = json.loads((ROOT/'web/model.json').read_text(encoding='utf-8'))
    for field in artifact['schema']:
        value = profile.get(field['name'])
        valid = value in field['options'] if 'options' in field else isinstance(value, int) and not isinstance(value, bool) and field['min'] <= value <= field['max']
        if not valid: raise ValueError(f"Invalid {field['name']}")
    row = {field['name']:profile[field['name']] for field in artifact['schema']}
    row['previously_contacted'] = int(row['pdays'] != -1)
    # Only load this locally generated, trusted artifact; joblib uses pickle.
    saved = joblib.load(ROOT/'outputs/trained_pipeline.joblib')
    pipeline, threshold = saved['pipeline'], saved['decision_threshold']
    frame = pd.DataFrame([row])
    score = float(pipeline.predict_proba(frame)[0,1])
    z = pipeline.named_steps['preprocess'].transform(frame)
    explainer = shap.TreeExplainer(pipeline.named_steps['model'], feature_perturbation='tree_path_dependent', model_output='raw')
    phi = explainer.shap_values(z)[0,:,1]
    contributions = {}
    for descriptor, value in zip(artifact['features'], phi):
        key = descriptor['source']
        contributions[key] = contributions.get(key,0.) + float(value)
    return {'prediction':'yes' if score >= threshold else 'no', 'score':score, 'threshold':threshold, 'baseline':float(explainer.expected_value[1]), 'shap':contributions}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--input', type=Path)
    choice.add_argument('--example', action='store_true')
    args = parser.parse_args()
    profile = json.loads(args.input.read_text(encoding='utf-8')) if args.input else json.loads((ROOT/'web/model.json').read_text(encoding='utf-8'))['examples'][1]['input']
    print(json.dumps(predict(profile), indent=2))
