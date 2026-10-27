const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('server_4090/templates/index.html','utf8');
const code=html.slice(html.indexOf('const rlinfLiveRequests=new Map();'),html.indexOf('async function loadRlinfRunLog('));
let calls=0,finish;const c={Map,AbortController,setTimeout,clearTimeout,api:()=>{calls++;return new Promise(r=>finish=r)}};vm.createContext(c);vm.runInContext(code,c);
(async()=>{const first=c.readRlinfLive('/metrics');const second=c.readRlinfLive('/metrics');assert.equal(calls,1);assert.strictEqual(first,second);finish({points:[1]});assert.deepEqual(await second,{points:[1]});const next=c.readRlinfLive('/metrics');assert.equal(calls,2);finish({points:[2]});await next;console.log('PASS: overlapping refreshes share one request; subsequent refresh gets fresh data');})().catch(e=>{console.error(e);process.exitCode=1});
