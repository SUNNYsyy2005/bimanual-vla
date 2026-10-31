from server_4090.rlinf_run_dedup import deduplicate_runs


def test_canonical_run_absorbs_native_output_but_keeps_nested_independent_run():
    canonical = dict(id='canonical', target='h100', path='/runs/a', _canonical=True)
    native = dict(id='native', target='h100-ksy-01', path='/runs/a/artifacts/native/ppo')
    nested = dict(id='nested', target='h100', path='/runs/a/artifacts/native/independent', _canonical=True)
    rows, aliases = deduplicate_runs([native, nested, canonical])
    assert {r['id'] for r in rows} == {'canonical', 'nested'}
    assert aliases == {'native':'canonical'}


def test_identical_paths_on_different_nodes_are_independent():
    rows, aliases = deduplicate_runs([
        dict(id='a', target='h100', path='/runs/a'),
        dict(id='b', target='h200-ali-02', path='/runs/a')])
    assert len(rows) == 2 and aliases == {}
