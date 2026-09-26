export function explain(model, input) {
  for (const field of model.schema) {
    const v = input[field.name];
    if (field.options ? !field.options.includes(v) : !Number.isInteger(v) || v < field.min || v > field.max) throw new Error(`Invalid ${field.name}`);
  }
  const x = model.features.map(f => f.category === null ? Math.fround(input[f.source]) : Number(input[f.source] === f.category));
  const active = [...new Set(model.nodes.filter(n => n.left >= 0).map(n => n.feature))];
  const m = active.length, position = new Map(active.map((f,i) => [f,i]));
  function expectation(i, mask) {
    const n = model.nodes[i];
    if (n.left < 0) return n.probability;
    if (mask & (1 << position.get(n.feature))) return expectation(x[n.feature] <= n.threshold ? n.left : n.right, mask);
    return (model.nodes[n.left].weight * expectation(n.left, mask) + model.nodes[n.right].weight * expectation(n.right, mask)) / n.weight;
  }
  const values = Array.from({length:1<<m}, (_,mask) => expectation(0,mask));
  const factorial = n => n < 2 ? 1 : n * factorial(n-1);
  const shap = model.features.map(() => 0);
  active.forEach((feature,i) => {
    for(let mask=0; mask<(1<<m); mask++) if (!(mask & (1<<i))) {
      const count = mask.toString(2).replaceAll('0','').length;
      shap[feature] += factorial(count)*factorial(m-count-1)/factorial(m)*(values[mask | (1<<i)]-values[mask]);
    }
  });
  const contributions = {};
  shap.forEach((v,i) => { const key=model.features[i].source; contributions[key]=(contributions[key]||0)+v; });
  const path=[];
  let i=0;
  while(model.nodes[i].left>=0) {
    const n=model.nodes[i], f=model.features[n.feature], left=x[n.feature]<=n.threshold;
    path.push(f.category===null ? `${f.source} ${left?'≤':'>'} ${n.threshold}` : `${f.source} ${left?'is not':'is'} ${f.category}`);
    i=left?n.left:n.right;
  }
  return {probability:model.nodes[i].probability,base:values[0],shap,contributions,path};
}
