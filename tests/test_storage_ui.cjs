const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('server_4090/templates/index.html','utf8');
const code=html.slice(html.indexOf('const storageRows=new Map();'),html.indexOf('</script>',html.indexOf('const storageRows=new Map();')));
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{value:id.startsWith('storageNode')?'all':id.startsWith('storageKind')?'rollout':'',innerHTML:'',textContent:'',isConnected:true});return elements.get(id);}
let calls=[],deferred=null;
const context={Map,URLSearchParams,Date,JSON,console,esc:String,confirm:()=>false,show:()=>{},document:{getElementById:element,querySelectorAll:()=>[]},api:async url=>{
 calls.push(url);if(deferred)return await deferred;
 if(url.includes('/files?'))return {files:[],total:0};
 return {items:[{id:'fixture',node:'h100',path:'/data/run/eval/c/f/e',kind:'rollout',size_bytes:10,file_count:1,file_types:{trajectory:1},file_states:{present_unverified:1},mutable:true,status:'present'}],total:1,locations:[{id:'h100',label:'H100'},{id:'local',label:'4090'}],scans:[]};
}};
vm.createContext(context);vm.runInContext(code,context);
(async()=>{
 assert.equal(calls.length,0,'page definitions must not fetch inventories');
 await context.loadStorageItems('eval-videos','rollout',0);
 assert.equal(calls.length,1);assert(calls[0].includes('limit=25'));assert(calls[0].includes('area=rollout'));
 assert(!calls.some(url=>url.includes('/files')),'directory query must not fetch per-file details');
 await context.loadStorageFiles('fixture','eval-videos',0);
 assert(calls[1].includes('limit=100'));
 const before=calls.length;await context.operateStorageItem('fixture','delete','eval-videos','rollout');assert.equal(calls.length,before,'no mutation without user selection/confirmation');
 console.log('storage UI lazy-query tests passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
