const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const html = fs.readFileSync(path.join(__dirname,'../server_4090/templates/index.html'),'utf8');
function merge(items, data) {
  const context={VISIBLE_DATASET_ORIGINS:new Set(['simulation']),datasetLocationLabel:x=>x};
  vm.createContext(context);
  vm.runInContext(html.slice(html.indexOf('function mergeDatasetLocationInfo('), html.indexOf('\nfunction renderDatasetLocations(')), context);
  return context.mergeDatasetLocationInfo(items,data);
}
test('local NVMe and archive inventory entries are never labelled remote-only',()=>{
  for(const kind of ['local','local_archive']) {
    const rows=merge([],{datasets:[{id:'ds',locations:[{target:'local_4090',kind,host:null,origin:'simulation'}]}]});
    assert.equal(rows[0].remote_only,false);
  }
});
test('actual local dataset metadata restores a complete missing main-list row',()=>{
  const rows=merge([],{local_datasets:[{id:'ds',schema:'joint',training_supported:true,dataset_origin:'simulation'}],datasets:[{id:'ds',locations:[{target:'local_4090',kind:'local',host:null,origin:'simulation'}]}]});
  assert.equal(rows[0].remote_only,false);
  assert.equal(rows[0].schema,'joint');
  assert.equal(rows[0].training_supported,true);
});
test('a new location snapshot removes stale synthetic inventory rows',()=>{
  const previous=merge([],{datasets:[{id:'ds',locations:[{target:'h100',kind:'ssh',host:'login-server',origin:'simulation'}]}]});
  assert.equal(previous[0].remote_only,true);
  assert.equal(merge(previous,{datasets:[]}).length,0);
});
