export function explain(model, input) {
  for (const field of model.schema) {
    const v = input[field.name];
    if (field.options ? !field.options.includes(v) : !Number.isInteger(v) || v < field.min || v > field.max) throw new Error(`Invalid ${field.name}`);
  }
  // Derive this field internally so the form cannot contradict pdays.
  const prepared = {...input, previously_contacted:Number(input.pdays !== -1)};
  const x = model.features.map(f => f.category === null ? Math.fround(prepared[f.source]) : Number(prepared[f.source] === f.category));
  const factorial = n => n < 2 ? 1 : n * factorial(n-1);
  const shap = model.features.map(() => 0);
  let base=0;
  // The path-dependent value function is a sum of leaf games. By Shapley
  // linearity and dummy-player invariance, compute each leaf separately.
  // Polynomial coefficients sum coalition products by size without enumerating
  // 2^depth subsets, keeping explanations responsive for deeper trees.
  function visit(index, path) {
    const node=model.nodes[index];
    if(node.left>=0) {
      for(const child of [node.left,node.right]) {
        const next=new Map(path), prior=path.get(node.feature)||{zero:1,one:1};
        const follows=(x[node.feature]<=node.threshold ? node.left : node.right)===child;
        next.set(node.feature,{zero:prior.zero*model.nodes[child].weight/node.weight,one:prior.one*Number(follows)});
        visit(child,next);
      }
      return;
    }
    const entries=[...path], m=entries.length;
    base+=node.probability*entries.reduce((v,[,p])=>v*p.zero,1);
    entries.forEach(([feature,p],i)=>{
      const others=entries.filter((_,j)=>j!==i);
      let coefficients=[1];
      for(const [,q] of others) {
        const next=Array(coefficients.length+1).fill(0);
        coefficients.forEach((value,k)=>{next[k]+=value*q.zero; next[k+1]+=value*q.one;});
        coefficients=next;
      }
      const weighted=coefficients.reduce((sum,value,k)=>sum+factorial(k)*factorial(m-k-1)/factorial(m)*value,0);
      shap[feature]+=node.probability*(p.one-p.zero)*weighted;
    });
  }
  visit(0,new Map());
  const contributions = {};
  shap.forEach((v,i) => { const key=model.features[i].source; contributions[key]=(contributions[key]||0)+v; });
  const path=[];
  let i=0;
  while(model.nodes[i].left>=0) {
    const n=model.nodes[i], f=model.features[n.feature], left=x[n.feature]<=n.threshold;
    path.push(f.category===null ? `${f.source} ${left?'≤':'>'} ${n.threshold}` : `${f.source} ${left?'is not':'is'} ${f.category}`);
    i=left?n.left:n.right;
  }
  const probability=model.nodes[i].probability;
  const threshold=model.decision_threshold ?? .5;
  return {probability,prediction:Number(probability>=threshold),threshold,base,shap,contributions,path};
}
