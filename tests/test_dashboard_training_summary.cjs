const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
const template = fs.readFileSync('server_4090/templates/index.html','utf8');
const source = template.slice(template.indexOf('function trainingJobParameters('),template.indexOf('function fillTasks(items)'));
const ctx = {esc: x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;'), modelVariantLabel: x=>x, trainingMetricInline:()=>'<br>step 300'};
vm.createContext(ctx);vm.runInContext(source,ctx);
test('training summary shows actual tuning and cluster cards; moves technical fields into details',()=>{
 const html=ctx.trainingJobParameters({id:'task',metadata:{dataset_id:'fix',model_variant:'pi05',runtime:'slurm',execution_target:'h100',gpu_ids:[0],batch_size:8,steps:20000,finetuning:{action_expert_tuning:'full',llm_lora:false,rtc_training:true,rtc_training_max_delay:10,rtc_training_delay_weights:[1,2],image_resize_mode:'stretch'}}});
 const summary=html.split('<details')[0];
 assert.match(summary,/Expert 全量 · LLM 冻结/);assert.match(summary,/h100 · 1 卡/);
 assert.match(summary,/RTC ≤10 步 · 加权采样/);assert.match(summary,/224×224 · 无补边/);
 assert.doesNotMatch(summary,/contract|夹爪|FSDP|等待\/未知|GPU 0/);
 assert.match(html,/<summary>更多参数<\/summary>/);assert.match(html,/RTC 权重/);
});
test('missing historical settings are not presented as frozen defaults and values are escaped',()=>{
 const html=ctx.trainingJobParameters({id:'old',metadata:{dataset_id:'<script>',gpu_ids:[2]}});
 assert.match(html,/微调配置：未记录/);assert.match(html,/GPU 2/);assert.doesNotMatch(html,/LLM 冻结|<script>/);
});
