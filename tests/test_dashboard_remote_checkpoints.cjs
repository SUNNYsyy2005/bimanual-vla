const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const template=fs.readFileSync('server_4090/templates/index.html','utf8');
const source=template.slice(template.indexOf('let remoteCheckpointRequest'),template.indexOf('function fillCheckpoints(items)'));
test('remote list renders source errors and sends only the selected step to sync',async()=>{
 const elements=new Map();const requests=[];
 const ctx={Date,Promise,Map,Array,JSON,encodeURIComponent,latestStatus:{checkpoints:[]},
 esc:x=>String(x).replaceAll('<','&lt;'),modelVariantLabel:x=>x,show:()=>{},
 document:{getElementById:id=>{if(!elements.has(id))elements.set(id,{});return elements.get(id);}},
 api:async(path,options)=>{requests.push([path,options]);return options?{id:'transfer-test'}:{locations:[
 {source:'h100',label:'H100',state:'ready',checkpoints:[{config_name:'pi05',experiment:'fix',step:20000,dataset_ids:['fix'],model_variant:'pi05'}]},
 {source:'h200',label:'H200',state:'error',error:'missing root',checkpoints:[]} ]};}};
 vm.createContext(ctx);vm.runInContext(source,ctx);
 await ctx.loadRemoteCheckpoints(true);
 assert.match(elements.get('remoteCheckpoints').innerHTML,/同步此 step 到 4090/);
 assert.match(elements.get('remoteCheckpoints').innerHTML,/missing root/);
 const button={};await ctx.syncRemoteCheckpoint(0,button);
 const body=JSON.parse(requests.at(-1)[1].body);
 assert.deepEqual(body,{source:'h100',target:'local_4090',config_name:'pi05',exp_name:'fix',step:20000,skip_existing:true});
 assert.equal(button.disabled,true);
});
