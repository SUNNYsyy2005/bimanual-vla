const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const template=fs.readFileSync(path.join(__dirname,'../server_4090/templates/index.html'),'utf8');
const code=template.slice(template.indexOf('function formatBytes('),template.indexOf('\nfunction filteredEvaluationTasks('));
function setup() {
  const elements=new Map();
  const context={latestTransferTasks:null,transferTasksRequest:null,tokenInput:{value:'fake'},
    document:{getElementById:id=>{if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:''});return elements.get(id);}},
    api:async()=>({tasks:[{id:'new-transfer',type:'transfer',state:'running',metadata:{}}]}),
    esc:String,shortTaskId:String,shortPath:String,stateClass:String,selectedTaskIds:new Set(),
    ACTIVE_STATES:new Set(['running']),TERMINAL_STATES:new Set(['completed','failed']),
    paginateList:(_,items)=>items,renderListPagination:()=>'',updateTaskBulkControls(){},show(){}};
  vm.createContext(context);vm.runInContext(code,context);
  return {context,elements};
}
test('transfer refresh renders independently of whole-dashboard status',async()=>{
  const {context,elements}=setup();
  await context.loadTransferTasks();
  assert.match(elements.get('transferJobs').innerHTML,/new-transfer/);
  context.fillTransferTasks([]);
  assert.match(elements.get('transferJobs').innerHTML,/new-transfer/);
});
test('failed refresh retains the last known transfer list',async()=>{
  const {context,elements}=setup();await context.loadTransferTasks();
  const previous=elements.get('transferJobs').innerHTML;
  context.api=async()=>{throw new Error('timeout');};await context.loadTransferTasks(true);
  assert.equal(elements.get('transferJobs').innerHTML,previous);
});
test('dataset deletion has its own status rather than a fake byte progress bar',()=>{
  const {context,elements}=setup();
  context.fillTransferTasks([{id:'deleted-copy',type:'transfer',state:'completed',metadata:{transfer_kind:'dataset_delete'}}]);
  assert.match(elements.get('transferJobs').innerHTML,/删除已完成/);
  assert.ok(!elements.get('transferJobs').innerHTML.includes('ETA'));
});
