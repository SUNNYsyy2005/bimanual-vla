const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('server_4090/templates/index.html', 'utf8');
const source = html.slice(html.indexOf('let evalVideoGroupData = []'), html.indexOf('function evalPlayerId(videoId)'));
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value:'all',innerHTML:'',textContent:'',checked:false,options:[],classList:{toggle(){}},querySelectorAll(){return []}});
  return elements.get(id);
}
element('evalVideoQueryFilter').value='';
const calls=[];let resolveOld;
const context=vm.createContext({console,URLSearchParams,AbortController,Date,Map,Set,
  document:{getElementById:element},tokenInput:{value:'fixture'},
  listPageStates:{evalVideos:{page:1,pageSize:25}},selectedEvalVideoIds:new Set(),expandedEvalVideoIds:new Set(),
  updateEvalVideoBulkControls(){},resetListPage(){},renderListPagination(){return 'pages'},
  detachEvalVideoPlayers(){return []},restoreEvalVideoPlayers(){},
  renderEvalVideoGroups(v){return v.map(x=>x.id).join(',')},show(m){throw new Error(m)},
  api:async(url)=>{calls.push(url);return {videos:[],total:0,facets:{},ready:true}},
});
vm.runInContext(source,context);
vm.runInContext('renderEvalVideoScopes=()=>{};loadEvalVideoScopes=async()=>{await api("/api/eval-videos/groups")}',context);
(async()=>{
  await vm.runInContext('loadEvalVideos()',context);
  assert.deepEqual(calls,['/api/eval-videos/groups']);calls.length=0;
  vm.runInContext('chooseEvalVideoScopes("a,b",true)',context);
  assert.equal(calls.length,0,'checkboxes only prepare a selection');
  vm.runInContext('activeEvalVideoGroups=[...selectedEvalVideoGroups]',context);
  await vm.runInContext('loadEvalVideos()',context);
  const url=new URL(calls[0],'http://test');
  assert.deepEqual(url.searchParams.getAll('scope'),['a','b']);
  assert.equal(url.searchParams.get('limit'),'25');assert.equal(url.searchParams.get('browse'),'1');
  assert.equal(url.searchParams.has('include_remote'),false);
  context.api=()=>new Promise(resolve=>{resolveOld=resolve});
  const old=vm.runInContext('loadEvalVideos()',context);
  vm.runInContext('clearEvalVideoScopes()',context);
  resolveOld({videos:[{id:'stale'}],total:1,offset:0,facets:{}});await old;
  assert.equal(element('evalVideoList').innerHTML,'','stale response must not repopulate cleared selection');
  assert.ok(!html.includes('if (evalVideoModule && !evalVideoModule.hidden) loadEvalVideos('));
  console.log('PASS: groups-only entry, multi-selection, bounded requests, stale response suppression, no automatic list polling');
})().catch(e=>{console.error(e);process.exitCode=1});
