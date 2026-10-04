const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const template = fs.readFileSync(
  path.join(__dirname, '../server_4090/templates/index.html'), 'utf8',
);
function extract(start, end) {
  const begin = template.indexOf(start);
  assert.ok(begin >= 0);
  const finish = template.indexOf(end, begin);
  assert.ok(finish > begin);
  return template.slice(begin, finish);
}

function dashboard(remoteFailure = false) {
  const requests = [];
  const renders = [];
  const gpus = [{ index: 0, processes: [], memory_total_mib: 24564, memory_used_mib: 16 }];
  const context = {
    console: { warn() {} }, Date, tokenInput: { value: 'test' },
    DASHBOARD_PROFILE: 'simulation', DASHBOARD_BUILD: 'test-build',
    latestStatus: null, latestDatasetLocations: null,
    datasetLocationsRequest: null, datasetLocationsFetchedAt: 0,
    clusterResourcesRequest: null, clusterResourcesQueriedAt: 0,
    episodeDatasetData: {},
    document: { getElementById: () => ({ value: '', hidden: true, textContent: '' }) },
    api: async endpoint => {
      requests.push(endpoint);
      if (endpoint === '/api/status') return { gpus, datasets: [], config: {} };
      if (remoteFailure) throw new Error('SSH connection timed out');
      return new Promise(() => {});
    },
    mergeDatasetLocationInfo: datasets => datasets,
    fillGpus: items => renders.push(['gpus', items]),
    updateSummaries: data => renders.push(['summary', data.gpus]),
    show: message => { throw new Error(message); },
  };
  for (const name of [
    'fillDatasets', 'fillTrainingExperiments', 'fillBaseModels',
    'fillExecutionTargets', 'fillCheckpoints', 'fillTasks', 'fillTrainingTasks',
    'fillTransferTasks', 'refreshSelectedLogs', 'fillObservation',
    'loadDatasetEpisodes', 'loadEvalVideos', 'loadTrainingMetrics',
  ]) context[name] = () => {};
  vm.createContext(context);
  vm.runInContext([
    extract('async function loadDatasetLocations(', '\nasync function syncDatasetTo('),
    extract('async function loadClusterResources(', '\nfunction fillGpus('),
    extract('async function refreshAll()', '\nfunction setTrainingOutputView('),
  ].join('\n'), context);
  return { context, requests, renders, gpus };
}

test('local GPU refresh completes while remote inventory and Slurm never respond', async () => {
  const { context, requests, renders, gpus } = dashboard();
  let timer;
  try {
    await Promise.race([
      context.refreshAll(),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('local refresh blocked on remote query')), 200);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
  assert.equal(renders[0][0], 'gpus');
  assert.equal(renders[0][1], gpus);
  await context.refreshAll();
  assert.equal(requests.filter(url => url.startsWith('/api/dataset-locations')).length, 1);
  assert.equal(requests.filter(url => url === '/api/cluster-resources').length, 1);
  assert.equal(renders.filter(([kind]) => kind === 'gpus').length, 2);
});

test('remote failures remain isolated and automatic refresh respects the cooldown', async () => {
  const { context, requests, renders } = dashboard(true);
  await context.refreshAll();
  await context.refreshAll();
  assert.equal(renders.filter(([kind]) => kind === 'gpus').length, 2);
  assert.equal(requests.filter(url => url.startsWith('/api/dataset-locations')).length, 1);
  assert.equal(requests.filter(url => url === '/api/cluster-resources').length, 1);
  await context.loadClusterResources();
  assert.equal(requests.filter(url => url === '/api/cluster-resources').length, 2);
});
