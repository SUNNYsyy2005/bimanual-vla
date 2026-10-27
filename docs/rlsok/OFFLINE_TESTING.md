# RLSOK project integration: offline test

Run from the repository root:

```sh
./scripts/test_rlsok_offline.sh
```

Set `BIMANUAL_VLA_PYTHON` to another Conda Python path when needed. The script
uses `/home/user/miniconda3/envs/dual_arm/bin/python` by default.

The test suite creates a temporary executable from
`tests/fixtures/fake_rlsok_resolver.py` and calls it through the same stdin /
stdout JSON protocol that the production device bridge uses. It covers both
arm modes, the three RLSOK review decisions, stale evidence, nonce mismatch,
collection and inference pre-open refusal, reviewed locator handoff, and an
endpoint change during a session. Existing mock tests additionally cover GUI
return paths, camera stream loss and CAN feedback loss. Hardware factories and
the policy child are mocked before either could open a device.

The fake resolver is **only a project interface fixture**. It does not install
or invoke RLSOK, create an approved baseline, prove physical identity, open a
camera/CAN handle, or authorize movement. Real RLSOK integration still needs
the private project-side resolver and operator-approved configuration described
in `DEVICE_RESOLVER_GUI_GLUE.md`.
