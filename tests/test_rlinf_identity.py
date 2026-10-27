from server_4090.rlinf_identity import annotate_run,annotate_model,contract_from_config

CONTRACT=dict(robot_type='franka-panda',arm_mode='single',action_space='eef_pose')
CONFIG={'rlinf_dashboard':{'verified_action_environments':{'/verified':dict(safety_cost_protocol='native-v1',action_contract=CONTRACT)},'verified_run_contracts':[dict(node='h100',paths=['/runs/one'],action_contract=CONTRACT,evidence='audited config')]}}

def test_only_verified_environment_and_protocol():
    assert contract_from_config(CONFIG,dict(benchmark='/verified',safety_cost_protocol='native-v1'))==CONTRACT
    assert contract_from_config(CONFIG,dict(benchmark='/other',safety_cost_protocol='native-v1'))=={}
    assert contract_from_config(CONFIG,dict(benchmark='/verified'))=={}
    assert contract_from_config(CONFIG,dict(benchmark='/verified',safety_cost_protocol='native-v1',robots=['Other']))=={}

def test_identity_uses_exact_node_and_path():
    assert annotate_run(CONFIG,dict(path='/runs/one',target='h100'))['action_contract']==CONTRACT
    for path,node in [('/runs/one-other','h100'),('/runs/one','h200-ali-01')]:
        assert not annotate_run(CONFIG,dict(path=path,target=node)).get('action_contract')

def test_unknown_fill_and_conflict_preservation():
    r=annotate_run(CONFIG,dict(path='/runs/one',target='h100',action_contract={'robot_type':'unknown'}))
    assert r['action_contract']==CONTRACT
    r=annotate_run(CONFIG,dict(path='/runs/one',target='h100',action_contract={'robot_type':'piper'}))
    assert r['action_contract']['robot_type']=='piper' and r['identity_conflicts']

def test_only_checkpoint_descendants_inherit():
    r=annotate_model(CONFIG,dict(path='/runs/one/checkpoints/step_2',node='h100'))
    assert r['action_contract']==CONTRACT and r['path']=='/runs/one/checkpoints/step_2'
    assert not annotate_model(CONFIG,dict(path='/runs/one/rollouts/x',node='h100')).get('action_contract')
