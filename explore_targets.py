"""Training-only feasibility study for 90% accuracy and stronger subscriber recall.

Does not change the deployed model or inspect the reserved test partition.
Run: python -u explore_targets.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from joblib import Parallel, delayed
from experiment import make_pipeline
from train import load_data, ROOT

def cutoff(y,scores):
    precision,recall,threshold=precision_recall_curve(y,scores)
    precision,recall=precision[:-1],recall[:-1]
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    positives=int(y.sum()); negatives=len(y)-positives
    tp=recall*positives
    fp=np.divide(tp,precision,out=np.zeros_like(tp),where=precision>0)-tp
    accuracy=(tp+negatives-fp)/len(y)
    # A 58% inner recall floor leaves a small margin over baseline test recall.
    eligible=(recall>=.58)&(precision>.29742033383915023)&(f1>.3874794069192751)
    target=eligible&(accuracy>=.90)
    pool=np.flatnonzero(target if target.any() else eligible)
    if not len(pool): pool=np.arange(len(threshold))
    chosen=pool[np.argmax(f1[pool])]
    return float(threshold[chosen]),bool(target.any())

def evaluate(name,params,X,y,splits):
    rows=[]
    for fold,(fit,val) in enumerate(splits):
        a,b=X.iloc[fit],y.iloc[fit]; scores=np.zeros(len(b))
        for tr,cal in StratifiedKFold(3,shuffle=True,random_state=100+fold).split(a,b):
            pipe=make_pipeline(a,params).fit(a.iloc[tr],b.iloc[tr])
            scores[cal]=pipe.predict_proba(a.iloc[cal])[:,1]
        threshold,feasible=cutoff(b,scores)
        pipe=make_pipeline(a,params).fit(a,b)
        pred=pipe.predict_proba(X.iloc[val])[:,1]>=threshold
        actual=y.iloc[val]
        rows.append(dict(candidate=name,fold=fold+1,threshold=threshold,inner_target_feasible=feasible,
            accuracy=accuracy_score(actual,pred),precision=precision_score(actual,pred,zero_division=0),
            recall=recall_score(actual,pred),f1=f1_score(actual,pred)))
    return rows

def run():
    df=load_data(); X=df.drop(columns=['y','duration']).assign(previously_contacted=(df.pdays!=-1).astype(int)); y=df.y.eq('yes').astype(int)
    X,_,y,_=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    prior=pd.read_csv(ROOT/'outputs/experiment/candidate_results.csv').head(6)
    candidates=[]
    for r in prior.itertuples():
        p=json.loads(r.params)
        if isinstance(p['class_weight'],dict):p['class_weight']={int(k):v for k,v in p['class_weight'].items()}
        candidates.append((r.candidate+'_recall',p))
    for depth,leaves,minimum in [(14,128,30),(16,256,50),(18,256,20)]:
        for weight in [None,{0:1,1:3}]:
            candidates.append((f'larger_{len(candidates)}',dict(max_depth=depth,max_leaf_nodes=leaves,min_samples_leaf=minimum,criterion='entropy',class_weight=weight,ccp_alpha=.0001)))
    out=ROOT/'outputs/target_study';out.mkdir(exist_ok=True)
    (out/'design.json').write_text(json.dumps(dict(candidates=candidates,outer_folds=5,inner_folds=3,inner_recall_floor=.58,target_accuracy=.90,test_used=False),indent=2))
    splits=list(StratifiedKFold(5,shuffle=True,random_state=2026).split(X,y))
    rows=[]
    for result in Parallel(n_jobs=2,prefer='threads',return_as='generator')(delayed(evaluate)(name,p,X,y,splits) for name,p in candidates):
        rows.extend(result); frame=pd.DataFrame(rows);frame.to_csv(out/'fold_results.csv',index=False)
        summary=frame.groupby('candidate').mean(numeric_only=True).drop(columns='fold').sort_values('f1',ascending=False)
        summary.to_csv(out/'summary.csv')
        print(result[0]['candidate'],summary.loc[result[0]['candidate']].round(4).to_dict(),flush=True)
    print(summary.to_string(),flush=True)

if __name__=='__main__':run()
