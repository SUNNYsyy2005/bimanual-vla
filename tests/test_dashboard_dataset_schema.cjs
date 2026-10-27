const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../server_4090/templates/index.html'), 'utf8');
const context = {esc: s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'), cameraDisplayName: s => s};
vm.createContext(context);
vm.runInContext(html.slice(html.indexOf('function datasetSchemaLabel('), html.indexOf('function fillDatasets(')), context);
const eef = {schema:'delivery', arm_mode:'bimanual', arm_side:'both', state_dim:20, raw_action_dim:20, model_action_dim:14,
  contract_version:3, dataset_layout:'canonical', training_supported:true, raw_action_convention:'absolute_eef_target', model_action_convention:'chunk_origin',
  raw_gripper_semantics:'absolute_opening_fraction_0_closed_1_open', wire_gripper_semantics:'absolute_opening_fraction_0_closed_1_open'};
test('EEF dataset dimensions remain 20D while the actual 14D model conversion is confined to details', () => {
  const schema = context.datasetSchemaCell(eef), action = context.datasetActionCell(eef);
  assert.match(schema, /双臂 EEF/); assert.match(schema, /状态 \/ 动作 20D/);
  assert.doesNotMatch(schema, /14D|canonical|v3|训练兼容|observation.state/);
  assert.doesNotMatch(action.split('<details>')[0], /14D|6D|chunk_origin/);
  assert.match(action, /模型动作：14D/); assert.match(action, /每臂旋转由 6D 转为 3D/);
});
test('joint actions and gripper direction use short readable labels', () => {
  const action = context.datasetActionCell({...eef, schema:'joint', state_dim:14, raw_action_dim:14, raw_action_convention:'absolute_joint_target'});
  assert.match(action, /关节绝对位置/); assert.match(action, /0 关 \/ 1 开/);
  assert.doesNotMatch(action, /旋转由/);
  assert.match(context.datasetGripperLabel('absolute_closed_fraction_0_open_1_closed'), /0 开 \/ 1 关/);
});
test('legacy dimensions and contract errors stay visible and escaped', () => {
  const schema = context.datasetSchemaCell({...eef, raw_action_dim:14, contract_version:2, dataset_layout:'legacy', training_supported:false, contract_error:'<invalid>'});
  assert.match(schema, /状态 20D · 动作 14D/); assert.match(schema, /旧版 v2/); assert.match(schema, /仅管理/);
  assert.match(schema, /&lt;invalid&gt;/); assert.doesNotMatch(schema, /<invalid>/);
});
