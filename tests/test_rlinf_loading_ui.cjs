const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('server_4090/templates/index.html','utf8');
const code=html.slice(html.indexOf('async function loadRlinfSelectedMetrics()'),html.indexOf('function renderRlinfMetricData()'));
const status={textContent:''};let finishSlow;let draws=0;
const c={Map,Set,Promise,AbortController,setTimeout,clearTimeout,encodeURIComponent,
 rlinfMetricRequest:0,rlinfSelected:new Set(['fast','slow']),rlinfMetricData:new Map(),rlinfRuns:[{id:'fast',name:'Fast'},{id:'slow',name:'Slow'}],
 document:{getElementById:()=>status},renderRlinfRuns:()=>{},renderRlinfMetricData:()=>draws++,
 api:async url=>url.includes('/fast/')?{run:{id:'fast'},training_points:[{round:1,loss:.5}]}:await new Promise(resolve=>finishSlow=resolve)};
vm.createContext(c);vm.runInContext(html.slice(html.indexOf("const rlinfLiveRequests=new Map();"),html.indexOf("async function loadRlinfRunLog(")),c);vm.runInContext(code,c);
(async()=>{
 const pending=c.loadRlinfSelectedMetrics();await new Promise(setImmediate);
 assert(c.rlinfMetricData.has('fast'),'a slow comparison must not hide a ready curve');
 assert(!c.rlinfMetricData.has('slow'));assert(status.textContent.includes('1'));assert(draws>1);
 finishSlow({run:{id:'slow'},training_points:[]});await pending;assert(c.rlinfMetricData.has('slow'));
 c.api=async()=>{throw new Error('source unavailable')};await c.loadRlinfSelectedMetrics();
 assert(status.textContent.includes('source unavailable'),'failures must be visible');
 assert(c.rlinfMetricData.has('fast'),'refresh failures retain the previous valid chart');
 console.log('PASS: progressive curves, visible failures, retained previous data');
})().catch(e=>{console.error(e);process.exitCode=1});
