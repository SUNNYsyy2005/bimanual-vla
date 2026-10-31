const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('server_4090/templates/index.html','utf8');
const summary=html.slice(html.indexOf('function evalVideoGroupSummary('),html.indexOf('function renderEvalVideoScopes('));
test('group summary distinguishes training, periodic and independent evaluation',()=>{
 const context=vm.createContext({});vm.runInContext(summary,context);
 const text=context.evalVideoGroupSummary({run_count:3,process_run_count:2,independent_eval_run_count:1,count:1040,
  phase_counts:{train:704,eval:336,unknown:0},independent_eval_count:144,remote_count:896});
 assert(text.includes('2 次训练 / 过程评测运行'));assert(text.includes('1 次独立评测'));
 assert(text.includes('训练期评测 192'));assert(text.includes('独立评测 144'));assert(text.includes('1040 个视频'));
});
test('refresh migrates selected legacy scopes without clearing the selected experiment',async()=>{
 const source=html.slice(html.indexOf('async function loadEvalVideoScopes('),html.indexOf('async function loadEvalVideos('));
 const elements=new Map();
 function el(id){if(!elements.has(id))elements.set(id,{value:'all',innerHTML:'',textContent:'',options:[{value:'all'}]});return elements.get(id)}
 const c=vm.createContext({Date,Set,Map,tokenInput:{value:'fixture'},document:{getElementById:el},
  evalVideoGroupData:[],evalVideoGroupsFetchedAt:0,evalVideoGroupsPromise:null,
  selectedEvalVideoGroups:new Set(['old-final','training']),activeEvalVideoGroups:['old-final','training'],
  evalVideoSourceLabel:x=>x,esc:x=>x,renderEvalVideoScopes(){},clearEvalVideoPage(){throw Error('selection cleared')},
  api:async()=>({ready:true,total:10,updated_at:1,scope_aliases:{'old-final':'training'},
   groups:[{id:'training',video_source:'libero-safety',task:'task',experiment:'training'}]})});
 vm.runInContext(source,c);await c.loadEvalVideoScopes(true);
 assert.deepEqual([...c.selectedEvalVideoGroups],['training']);assert.deepEqual([...c.activeEvalVideoGroups],['training']);
});
