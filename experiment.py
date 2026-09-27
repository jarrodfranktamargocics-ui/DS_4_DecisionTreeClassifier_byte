"""Training-only tree experiments with separately validated threshold selection.

Run: python -u experiment.py
The existing 20% test partition is excluded from this module's model selection.
"""
import json
import hashlib
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from train import load_data, ROOT

def make_pipeline(X, params):
    categorical = X.select_dtypes(include=['object', 'str']).columns.tolist()
    numeric = [c for c in X if c not in categorical]
    prep = ColumnTransformer([('num','passthrough',numeric), ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
    return Pipeline([('preprocess',prep),('model',DecisionTreeClassifier(random_state=42,**params))])

def choose_cutoff(y, scores):
    precision, recall, thresholds = precision_recall_curve(y,scores)
    total = precision[:-1]+recall[:-1]
    f1 = np.divide(2*precision[:-1]*recall[:-1],total,out=np.zeros_like(total),where=total>0)
    best = np.flatnonzero(np.isclose(f1,f1.max(),atol=1e-12,rtol=0))[-1]
    return float(thresholds[best]), pd.DataFrame({'threshold':thresholds,'precision':precision[:-1],'recall':recall[:-1],'f1':f1})

def tuned_fit(X,y,params,seed):
    pipeline = make_pipeline(X,params)
    inner = StratifiedKFold(3,shuffle=True,random_state=seed)
    oof = np.zeros(len(y))
    for fit_ids,calibration_ids in inner.split(X,y):
        fitted=clone(pipeline).fit(X.iloc[fit_ids],y.iloc[fit_ids])
        oof[calibration_ids]=fitted.predict_proba(X.iloc[calibration_ids])[:,1]
    threshold,curve=choose_cutoff(y,oof)
    pipeline.fit(X,y)
    return pipeline,threshold,curve

def candidate_score(candidate,X,y,splits):
    features=X if candidate['indicator'] else X.drop(columns='previously_contacted')
    rows=[]
    for fold,(fit_ids,validation_ids) in enumerate(splits):
        fit_X,fit_y=features.iloc[fit_ids],y.iloc[fit_ids]
        if candidate['tuned']:
            pipeline,threshold,_=tuned_fit(fit_X,fit_y,candidate['params'],100+fold)
        else:
            pipeline=make_pipeline(fit_X,candidate['params']).fit(fit_X,fit_y)
            threshold=.5
        scores=pipeline.predict_proba(features.iloc[validation_ids])[:,1]
        pred=(scores>=threshold).astype(int)
        actual=y.iloc[validation_ids]
        rows.append({'candidate':candidate['name'],'fold':fold+1,'threshold':threshold,
                     'f1':float(f1_score(actual,pred)), 'precision':float(precision_score(actual,pred,zero_division=0)),
                     'recall':float(recall_score(actual,pred)), 'accuracy':float(accuracy_score(actual,pred)),
                     'average_precision':float(average_precision_score(actual,scores)),
                     'leaves':pipeline.named_steps['model'].get_n_leaves()})
    return candidate,rows

def run_experiment():
    df=load_data()
    X=df.drop(columns=['y','duration']).assign(previously_contacted=(df.pdays!=-1).astype(int))
    y=df.y.eq('yes').astype(int)
    X_train,_,y_train,_=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    original={'class_weight':'balanced','criterion':'gini','max_depth':5,'max_leaf_nodes':8,'min_samples_leaf':150}
    capacity={**original,'max_depth':6,'max_leaf_nodes':32,'min_samples_leaf':50}
    previous={**capacity,'criterion':'entropy'}
    candidates=[]
    for name,params in [('v1',original),('capacity_only',capacity),('v2',previous)]:
        for tuned in [False,True]:
            candidates.append({'name':f'{name}_{"tuned" if tuned else "fixed"}','params':params,'tuned':tuned,'indicator':name!='v1'})
    candidates.append({'name':'v2_no_indicator_tuned','params':previous,'tuned':True,'indicator':False})
    grid={'max_depth':[6,8,10,12], 'max_leaf_nodes':[32,64,128], 'min_samples_leaf':[20,50,100],
          'ccp_alpha':[0.,.0001,.0005,.001], 'criterion':['gini','entropy'],
          'class_weight':[None,'balanced',{0:1,1:2},{0:1,1:4}]}
    for i,params in enumerate(ParameterSampler(grid,n_iter=48,random_state=73)):
        candidates.append({'name':f'expanded_{i+1:02d}','params':params,'tuned':True,'indicator':True})
    splits=list(StratifiedKFold(5,shuffle=True,random_state=2026).split(X_train,y_train))
    out=ROOT/'outputs/experiment';out.mkdir(parents=True,exist_ok=True)
    (out/'design.json').write_text(json.dumps({'candidates':candidates,'outer_folds':5,'inner_folds':3,'selection':'mean outer validation F1; ties prefer fewer leaves','test_used':False,'seed':2026},indent=2),encoding='utf-8')
    all_rows=[]; summary=[]
    # Thread workers avoid Windows process shutdown/memmap issues; limit to two.
    jobs=Parallel(n_jobs=2,prefer='threads',return_as='generator')(delayed(candidate_score)(c,X_train,y_train,splits) for c in candidates)
    for count,(candidate,rows) in enumerate(jobs,1):
        all_rows.extend(rows)
        values=pd.DataFrame(rows)
        record={'candidate':candidate['name'],'mean_f1':float(values.f1.mean()),'std_f1':float(values.f1.std(ddof=1)),
                **{f'mean_{key}':float(values[key].mean()) for key in ['precision','recall','accuracy','average_precision','leaves']},
                'params':json.dumps(candidate['params'],sort_keys=True),'tuned':candidate['tuned'],'indicator':candidate['indicator']}
        summary.append(record)
        pd.DataFrame(all_rows).to_csv(out/'fold_results.csv',index=False)
        pd.DataFrame(summary).sort_values(['mean_f1','mean_leaves'],ascending=[False,True]).to_csv(out/'candidate_results.csv',index=False)
        print(f"{count}/{len(candidates)} {candidate['name']}: validation F1={record['mean_f1']:.4f}",flush=True)
    table=pd.DataFrame(summary).sort_values(['mean_f1','mean_leaves'],ascending=[False,True])
    winner_name=table.iloc[0].candidate
    winner=next(c for c in candidates if c['name']==winner_name)
    # Lock the winner using training validation before accessing test outcomes.
    selected_X=X_train if winner['indicator'] else X_train.drop(columns='previously_contacted')
    if winner['tuned']:
        pipeline,threshold,curve=tuned_fit(selected_X,y_train,winner['params'],314)
    else:
        pipeline=make_pipeline(selected_X,winner['params']).fit(selected_X,y_train);threshold=.5;curve=pd.DataFrame()
    curve.to_csv(out/'final_threshold_search.csv',index=False)
    selection={'winner':winner,'threshold':threshold,'mean_validation_f1':float(table.iloc[0].mean_f1),
               'incumbent_validation_f1':float(table.loc[table.candidate=='v2_tuned','mean_f1'].iloc[0]),
               'csv_sha256':hashlib.sha256((ROOT/'data/bank-full.csv').read_bytes()).hexdigest(),
               'training_indices_sha256':hashlib.sha256(np.asarray(X_train.index,dtype=np.int64).tobytes()).hexdigest(),
               'selection_bias_note':'Outer folds are used to choose among 55 candidates; the winning mean is a selection score, not an unbiased final estimate. Existing test split has been inspected before.'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    dump({'pipeline':pipeline,'decision_threshold':threshold,'selection':selection},out/'selected_pipeline.joblib')
    print(json.dumps(selection,indent=2),flush=True)
    return table,selection

if __name__=='__main__': run_experiment()
