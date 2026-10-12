const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const template = fs.readFileSync(path.join(__dirname, '../server_4090/templates/index.html'), 'utf8');
function extract(start, end) {
  return template.slice(template.indexOf(start), template.indexOf(end, template.indexOf(start)));
}
function chartContext(width = 400, ratio = 2) {
  const moves = [], labels = [], clips = [];
  const ctx = new Proxy({
    measureText: text => ({width: text.length * 7}),
    moveTo: (x,y) => moves.push([x,y]), lineTo: (x,y) => moves.push([x,y]),
    fillText: (text,x,y) => labels.push([text,x,y]), rect: (...args) => clips.push(args),
  }, {get: (target,key) => target[key] || (() => {})});
  const elements = new Map([
    ['rlinfChart', {width:300,height:150,getBoundingClientRect:()=>({width}),getContext:()=>ctx}],
    ['rlinfMetric', {value:'train.loss'}], ['rlinfXAxis', {value:'actor_updates'}],
    ['rlinfMetricSummary', {}], ['rlinfLegend', {}],
  ]);
  const context = {document:{body:{},getElementById:id=>elements.get(id)}, window:{devicePixelRatio:ratio},
    getComputedStyle:()=>({getPropertyValue:()=> '#8799aa'}), esc:String, METRIC_COLORS:['red','blue'], rlinfMetricData:new Map()};
  vm.createContext(context);
  vm.runInContext(extract('function drawRlinfChart()', "\ndocument.getElementById('rlinfTarget')"), context);
  return {context,elements,moves,labels,clips};
}

test('high DPI redraw keeps canvas width stable and all curves within shared axes', () => {
  const {context,elements,moves,labels,clips} = chartContext();
  context.rlinfMetricData.set('a',{run:{name:'a'},training_points:[{actor_updates:20,loss:2},{actor_updates:0,loss:1}]});
  context.rlinfMetricData.set('b',{run:{name:'b'},training_points:[{actor_updates:2000,loss:3},{actor_updates:100,loss:4}]});
  context.drawRlinfChart(); context.drawRlinfChart();
  assert.equal(elements.get('rlinfChart').width,800);
  assert.equal(elements.get('rlinfChart').height,680);
  assert.ok(moves.every(([x,y]) => x >= 0 && x <= 400 && y >= 0 && y <= 340));
  assert.ok(labels.some(([text]) => text === '2,000'));
  assert.equal(clips.length,2);
  assert.match(elements.get('rlinfLegend').innerHTML,/Actor updates 2,000/);
});

test('a selected round axis does not silently mix actor updates from other runs', () => {
  const {context,elements} = chartContext();
  elements.get('rlinfXAxis').value='round';
  context.rlinfMetricData.set('a',{run:{name:'rounds'},training_points:[{actor_updates:200,round:2,loss:1}]});
  context.rlinfMetricData.set('b',{run:{name:'updates-only'},training_points:[{actor_updates:2000,loss:3}]});
  context.drawRlinfChart();
  assert.match(elements.get('rlinfMetricSummary').textContent,/1 条曲线/);
  assert.ok(!elements.get('rlinfLegend').innerHTML.includes('updates-only'));
});

test('pagination clamps after deletion and keeps selections across pages', () => {
  const context = {LIST_PAGE_SIZE_OPTIONS:[10,25,50,100],listPageStates:{rlinfRuns:{page:3,pageSize:10}},esc:String};
  vm.createContext(context);
  vm.runInContext(extract('function listPageState(', '\nfunction rerenderPaginatedList('), context);
  const rows=Array.from({length:25},(_,id)=>({id}));
  assert.equal(context.paginateList('rlinfRuns',rows).length,5);
  assert.equal(context.paginateList('rlinfRuns',rows.slice(0,10)).length,10);
  assert.equal(context.listPageStates.rlinfRuns.page,1);
  assert.match(context.renderListPagination('rlinfRuns',10),/1-10 \/ 10/);
});

test('metric discovery includes keys introduced after the first training point', async () => {
  const elements = new Map([['rlinfMetric',{value:'',innerHTML:''}]]);
  const context = {rlinfMetricRequest:0,rlinfSelected:new Set(['run']),rlinfRuns:[{id:'run'}],
    api:async()=>({run:{id:'run'},training_points:[{actor_updates:1,loss:1},{actor_updates:2,gradient_norm:3}],evaluation_points:[]}),
    document:{getElementById:id=>elements.get(id)},renderRlinfRuns(){},drawRlinfChart(){},esc:String};
  vm.createContext(context);
  vm.runInContext(extract('async function loadRlinfSelectedMetrics()', '\nfunction drawRlinfChart()'), context);
  await context.loadRlinfSelectedMetrics();
  assert.match(elements.get('rlinfMetric').innerHTML,/train.gradient_norm/);
});
