const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('server_4090/templates/index.html','utf8');
const source=html.slice(html.indexOf('function evalVideoGroupSummary('),html.indexOf('function renderEvalVideoScopes('));
test('family has one selection entry and variant details preserve setting differences',()=>{
 const select={value:'all',innerHTML:''};
 const family={id:'family',label:'PPO-Lag',count:1796,run_count:5,variants:[
  {id:'fixed',label:'固定场景训练',count:756},{id:'random',label:'随机场景训练 / PPO v3',count:1040,independent_eval_count:144}]};
 const context=vm.createContext({Set,Map,esc:value=>String(value),selectedEvalVideoGroups:new Set(['family']),
  activeEvalVideoGroups:['family'],evalVideoGroupData:[family],document:{getElementById:()=>select}});
 vm.runInContext(source,context);
 const markup=context.evalVideoScopeGroup(family);
 assert.equal((markup.match(/type="checkbox"/g)||[]).length,1);
 assert(markup.includes('场景版本明细（2）'));assert(markup.includes('固定场景训练'));assert(markup.includes('随机场景训练'));
 context.fillEvalVideoVariantFilter();assert(select.innerHTML.includes('value="fixed"'));assert(select.innerHTML.includes('value="random"'));
 select.value='fixed';context.fillEvalVideoVariantFilter();assert.equal(select.value,'fixed');
 context.activeEvalVideoGroups=[];context.selectedEvalVideoGroups.clear();context.fillEvalVideoVariantFilter();assert.equal(select.value,'all');
});
