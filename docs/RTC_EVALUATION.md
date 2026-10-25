# Prefix-conditioned held-out evaluation

`server_4090/eval_heldout_loss.py` retains the existing `eval_loss_*` metrics as
the no-prefix (`d=0`) baseline. For a trained-RTC checkpoint it additionally
evaluates `d=0/4/5/6/7/8/10`, restricted to the checkpoint's supported maximum,
and includes that maximum when it is not already listed. Non-RTC models retain
only the original no-prefix condition. No training or inference setting changes.

Each batch uses the same held-out observations, action labels, noise and sampled
flow times for all delays. Image preprocessing stays in evaluation mode. Clean
demonstration prefix rows are excluded from the error, and `H/(H-d)` weighting
makes the reported mean an average over only the generated suffix. Increasing
`d` changes both conditioning and which target rows are scored; lower loss at a
larger `d` is not by itself evidence of better online behavior.

Outputs include:

- `eval_rtc_demo_d6_loss_model`, `eval_rtc_demo_d6_loss_objective`, and component
  metrics including `eval_rtc_demo_d6_loss_gripper` (and equivalent keys per d).
- `rtc_evaluation.by_delay`: per-dimension losses and generated suffix length.
- `rtc_evaluation.planner_prefix_evaluated=false`: no planner-prefix experiment
  has been performed by the standard held-out dataset loader.

The Dashboard's existing numeric `eval_*` ingestion accepts the new flat keys.
Historical `eval_loss_model` and best-checkpoint selection remain the d=0 metric.
Use `--rtc-eval-delays 6 7 8` to restrict the conditions; d=0 is always included.
The default adds six forward passes for a maximum-10 RTC model, increasing
evaluation duration. Existing results are not retroactively recomputed.

## Planner-prefix evaluation interface

`rtc_evaluation.prefix_loss_per_dim` accepts a separate `planned_prefix` tensor
in B x P x D form. It must be paired with the exact observation, encoded relative
to that new observation's chunk origin with the checkpoint's normalization,
and cover at least `delay` rows. Unused action dimensions are cleared. The
demonstration suffix remains the label; changing the prefix does not replace it.
Bind the function to the model with `types.MethodType`, then use OpenPI's
`module_jit(..., static_argnames=("delay",))`.

The standard loader does not fabricate planner prefixes from real-world failure
recordings: those recordings have no reference future demonstration. A paired
planner-prefix dataset/ablation is still needed to measure that condition.
Flow loss is not sampled action MAE, boundary smoothness, or grasp success.

## Validation

`tests/test_rtc_evaluation.py` tests delay selection, suffix-only averaging,
disabled augmentation, planner-prefix separation/padding, JIT execution and
end-to-end metric aggregation/JSON output. Run with the OpenPI Conda environment
and `JAX_PLATFORMS=cpu`; no checkpoint or GPU is required.
