const $=id=>document.getElementById(id), pct=x=>`${(x*100).toFixed(2)}%`;
const names={accuracy:'Accuracy',precision:'Precision',recall:'Recall',f1:'F1 score',poutcome:'Previous campaign outcome',month:'Contact month',contact:'Contact channel',housing:'Housing loan',campaign:'Campaign contact count',day:'Contact day'};
const stories={poutcome:'The outcome of an earlier campaign carries information about a customer’s prior response.',month:'The tree uses differences between historical campaign months. These patterns may change in future campaigns.',contact:'The method of contact helps separate groups in the historical data.',day:'The day of the month helps distinguish historical campaign patterns; it is not proof of an ideal calling day.',housing:'Housing-loan status is associated with different outcomes in parts of the tree.'};
function bars(id,items){for(const [name,value] of items){const row=document.createElement('div');row.className='bar-row';row.innerHTML=`<div class="bar-label"><span>${names[name]||name}</span><strong>${pct(value)}</strong></div><div class="bar-track"><div class="bar-fill" style="width:${value*100}%"></div></div>`;$(id).append(row);}}
function matrix(id,m){$(id).innerHTML='<caption class="sr-only">Actual versus predicted subscription counts</caption><thead><tr><th scope="col">Actual outcome</th><th scope="col">Predicted no</th><th scope="col">Predicted yes</th></tr></thead><tbody>'+m.map((r,i)=>`<tr><th scope="row">${i?'Yes':'No'}</th>${r.map((n,j)=>`<td class="${i===j?'correct':''}">${n.toLocaleString()}</td>`).join('')}</tr>`).join('')+'</tbody>';}
if($('report'))try{
 const [base,model]=await Promise.all(['baseline.json','model.json'].map(async url=>{const r=await fetch(url);if(!r.ok)throw Error(url);return r.json();}));
 const b=base.metrics,m=model.metrics;
 if($('comparison')){
  for(const key of ['accuracy','precision','recall','f1']){
   const delta=(m[key]-b[key])*100;
   $('baseline-metrics').insertAdjacentHTML('beforeend',`<div class="metric"><span>${names[key]}</span><strong>${pct(b[key])}</strong></div>`);
   $('tuned-metrics').insertAdjacentHTML('beforeend',`<div class="metric"><span>${names[key]}</span><strong>${pct(m[key])}</strong><small class="delta ${delta<0?'down':''}">${delta>=0?'+':''}${delta.toFixed(2)} pts</small></div>`);
   $('comparison').insertAdjacentHTML('beforeend',`<div class="comparison-row"><h3>${names[key]}</h3><div class="compare-bar baseline-bar" style="--value:${100*b[key]}%"><span>Baseline</span><b>${pct(b[key])}</b></div><div class="compare-bar tuned-bar" style="--value:${100*m[key]}%"><span>Tuned</span><b>${pct(m[key])}</b></div></div>`);
  }
  matrix('baseline-matrix',b.confusion_matrix);matrix('tuned-matrix',m.confusion_matrix);
  $('tradeoff').textContent=`Compared with the baseline, the tuned tree makes ${b.confusion_matrix[0][1]-m.confusion_matrix[0][1]} fewer false-positive predictions, but misses ${m.confusion_matrix[1][0]-b.confusion_matrix[1][0]} more actual subscribers. Higher precision comes with lower recall.`;
  $('accuracy-context').textContent=`An always-“no” classifier achieves ${pct(m.baseline_accuracy)} accuracy on this test set but detects zero subscribers. That is why accuracy alone is misleading here. Subscription is the positive class for precision, recall and F1.`;
  const p=m.best_params;
  const config={'Algorithm':'Decision Tree','Split criterion':p.criterion,'Actual depth / leaves':`${m.depth} / ${m.leaves}`,'Maximum leaf nodes':p.max_leaf_nodes,'Minimum samples per leaf':p.min_samples_leaf,'Pruning alpha':p.ccp_alpha,'Class weights (no : yes)':`${p.class_weight['0']} : ${p.class_weight['1']}`,'Decision cutoff':pct(model.decision_threshold),'Mean validation F1':pct(m.validation_f1),'Train / test records':`${m.train_rows.toLocaleString()} / ${m.test_rows.toLocaleString()}`};
  for(const [k,v] of Object.entries(config))$('config').insertAdjacentHTML('beforeend',`<div><dt>${k}</dt><dd>${v}</dd></div>`);
 }
 if($('feature-stories')){
  bars('baseline-importance',base.importance);bars('tuned-importance',model.importance);
  model.importance.forEach(([name,value],i)=>$('feature-stories').insertAdjacentHTML('beforeend',`<article class="panel feature-story"><span class="rank">0${i+1}</span><div><h3>${names[name]||name}</h3><p>${stories[name]||'A signal used by the selected tree.'}</p></div><strong>${pct(value)}</strong></article>`));
  $('class-description').textContent=`Only ${pct(m.positive_rate)} of the 45,211 historical customers subscribed. Most examples belong to the “no” class.`;
  $('class-chart').setAttribute('aria-label',`${pct(m.positive_rate)} subscribed; ${pct(1-m.positive_rate)} did not subscribe`);
  $('class-chart').innerHTML=`<div style="width:${m.positive_rate*100}%"></div><span>${pct(m.positive_rate)} yes / ${pct(1-m.positive_rate)} no</span>`;
 }
 $('report-status').hidden=true;$('report').hidden=false;
}catch(e){$('report-status').textContent='Evaluation data could not load. Please refresh or serve this folder over HTTP.';console.error(e);}
