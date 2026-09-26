import assert from 'node:assert/strict';
import fs from 'node:fs';
import {explain} from '../web/engine.js';
const model=JSON.parse(fs.readFileSync(new URL('../web/model.json',import.meta.url)));
const fixtures=JSON.parse(fs.readFileSync(new URL('./fixtures.json',import.meta.url)));
for(const fixture of fixtures) {
  const result=explain(model,fixture.input);
  assert.ok(Math.abs(result.probability-fixture.probability)<1e-9);
  result.shap.forEach((v,i)=>assert.ok(Math.abs(v-fixture.shap[i])<1e-8,`SHAP feature ${i}: ${v} vs ${fixture.shap[i]}`));
  assert.ok(Math.abs(result.base+result.shap.reduce((a,b)=>a+b,0)-result.probability)<1e-8);
}
assert.throws(()=>explain(model,{}));
assert.throws(()=>explain(model,{...fixtures[0].input,age:NaN}));
console.log(`Passed: ${fixtures.length} predictions and SHAP vectors match Python; additivity and invalid inputs checked.`);
