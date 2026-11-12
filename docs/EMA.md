# OpenPI EMA precision and frozen parameters

The Piper trainer uses `trainable_fp32_frozen_raw.v1`:

- Frozen parameters always come directly from the current Raw parameter tree.
- Only trainable parameters are averaged, using FP32 accumulators.
- Initialization and checkpoint restore preserve this policy and the complete
  NNX parameter-tree structure used by OpenPI checkpoint saving and inference.
- With EMA disabled, the upstream training step is unchanged. The adapter does
  not change the optimizer, gradient, loss, or inference architecture.

The adapter is installed on the freshly loaded trainer, after any external
full-state restore hook. It proxies checkpoint restoration for that trainer
without patching the shared checkpoint module.

## Existing checkpoints

Older checkpoints may contain corrupted frozen EMA parameters: repeatedly
applying EMA to BF16 constants can cause rounding drift even though their Raw
values never changed. One observed frozen norm element changed from
`0.87109375` to `0.83203125`. Merely loading that EMA in FP32 does not undo the
drift.

When repairing an existing checkpoint, use frozen parameters from the **same
training step's Raw state**, retaining the original trainable EMA parameters.
An initialization model is a valid substitute only after every frozen tensor
has been verified to match that Raw state exactly. Keep the original checkpoint
and provenance in a checksum-verified archive; publish repaired weights only
after save/reload and complete tensor checks. Weights-only exports do not
provide optimizer-state recovery. Historical rounding in trainable EMA values
cannot be reconstructed without the original update sequence.

For the audited 20k model, replacing only the frozen EMA tensors changed the
offline t=1 noise coefficient from approximately `0.968` to `0.999`. This is an
offline diagnostic, not evidence of real-robot task success.

## Regression checks

In the project's OpenPI Conda environment:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p test_training_ema.py -v
```

The tests cover the observed BF16 drift, 20,000 compiled updates, a known FP32
EMA sequence, restore migration, Orbax save/reload continuation, disabled EMA,
trainer-local restore hooks, and parity with a real upstream optimizer step.
