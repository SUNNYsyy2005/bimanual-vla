# RTC questions: why the trained-RTC model still jitters on the arm

Date: 2026-10-22. Scope: the **trained** (clean-prefix) RTC path, `RTCConfig.mode="trained"`.
Status: diagnosis plus proposed fixes, updated with deployed-code and CPU numerical verification.
Existing robot recordings were analyzed; the proposed fixes have not been verified in closed loop.
No training, policy service, or robot motion was started to produce this document.

**Evidence scope:** the current remote 4090:8099 checkpoint is
`fix_trimstart_rtc35_h200_20261021/20000` (rtc35). The saved prefix/noise ablations below use the
older `fix_nopad_rtc_empirical_d10_20261019/20000` (d10). Their results must not be treated as
paired experiments on the current checkpoint. The independent verification is recorded in
[RTC normalization audit](analysis/rtc_normalization_20261022/README.md).

## 1. Background: how trained RTC works here

- Training (`bimanual_vla/training/openpi_model.py`, `compute_loss_per_dim`): sample a prefix length `d`
  per example from `rtc_training_delay_weights`. The first `d` ground-truth actions get flow time 0 (clean);
  the rest get the sampled flow time. The loss covers only the remaining rows, scaled by `H/(H-d)`.
  Per-action time conditioning is enabled by `install_per_action_adarms()`.
- Inference (`PiperPi0.sample_actions`): every denoising step overwrites the first `inference_delay` rows
  with the supplied prefix at time 0. There is no guidance, and `execution_horizon`, the prefix schedule and
  `max_guidance_weight` do not affect this sampler's blending/guidance. Upstream configuration validation
  still uses the horizon.
- Client (`bimanual_vla/deployment/client.py`, `rtc_request_metadata`): with Ruckig on, the prefix is read
  from the client's own 100 Hz plan (`servo.py: rtc_prefix`) at `capture + (i+1)/action_hz`, sent as
  `planned_prefix`, and re-encoded on the server with DeltaActions/Normalize (`rtc_policy.py: infer`).
- `inference_delay_steps` is a fixed user setting (typically 6-8), not a predicted latency.

## 2. Observed problem

From `analysis/test_review_20261022_late/README.md` and `boundary_followup.md`:

- Across the six runs, small backward steps occur at **47-63% of eligible chunk switches**
  (>=0.005 rad on some joint within 250 ms), 27-40% at >=0.01 rad, and 3-17% at >=0.02 rad.
  Eligibility excludes startup/gaps and requires an incoming moving joint; the window ends at the next
  switch if earlier. Counts can include legitimate task reversals, not only unwanted oscillations.
- Example (10:47:28, generation 31, right J2): last prefix point 0.1768, then predicted
  0.1753 / 0.1578 / 0.2131 rad. The step backward is already in the server's raw output.
- In the 10:47 run, uploaded prefixes match the returned first rows to within 1.8e-7. This verifies
  physical prefix transport and re-encoding for those samples, but does not establish velocity or
  acceleration continuity into the generated suffix. The example backward step is already in that suffix.
- Six steps is not hard-coded: 10:43 used 7; 09:51 and 09:53 used 8. Boundary analysis must use each
  request's actual `d`, with the first generated point at `d+1` (one-based).

## 3. Findings

### Q1. Is the model output itself much rougher than the training data? Yes, about 10x.

Mean |second difference| per joint ("roughness"), 12 arm joints, rad:

| Source | 5 denoising steps | 10 denoising steps | 50 denoising steps |
|---|---:|---:|---:|
| Demonstration actions (25 episodes) | 0.0011 | - | - |
| Model, held-out obs, no prefix | 0.0139 | 0.0118 | 0.0119 |
| Model, held-out obs, true demo prefix | 0.0125 | 0.0103 | 0.0103 |
| Model, held-out obs, its own prefix | 0.0139 | 0.0115 | 0.0114 |
| Model, reconstructed real-robot obs, no prefix | 0.0161 | 0.0132 | 0.0128 |

More denoising steps help only a little (5 -> 10) and then level off. The roughness is about the same
right after the prefix (rows 3-10) as later in the chunk (rows 10+).

Method: section 5.1. Predictions come from the d10 20k checkpoint (`fix_nopad_rtc_empirical_d10_20261019/20000`),
saved by `analysis/rtc_padding_fix_20261021` on 4x4090.

**Current-model verification:** the trimmed dataset `fix_nopad_trimstart_20261021` has 25 episodes,
mean absolute second difference **0.001139 rad**, and mean absolute first difference **0.004103 rad**
(episode means averaged over the 12 arm joints). Accepted current-model predictions give:

| Run | Generated suffix mean absolute second difference (rad) | Rows 10+ (zero-based) | Suffix after [1,2,1]/4 filter |
|---|---:|---:|---:|
|10:41:31|0.01260|0.01269|0.00307|
|10:43:06|0.01139|0.01142|0.00334|
|10:44:16|0.01219|0.01202|0.00330|
|10:45:34|0.01360|0.01344|0.00385|
|10:46:53|0.01340|0.01310|0.00341|
|10:47:28|0.01373|0.01348|0.00363|

The generated suffix starts after each request's actual `d`. Thus the roughly 10–12x difference also
exists in current recordings and is not confined to the prefix boundary. These are geometric action
differences, not measured actuator acceleration. Scenes and motion speeds are not matched, so this
is not a controlled accuracy comparison. The filter column is a numerical diagnostic, not a replay
of the full planner or a newly deployed setting.

### Q2. Where does the roughness come from? Strong evidence for initial-noise residuals in the old models.

- `analysis/no_rtc_comparison_20261021/REPORT.md` (4x4090) found the per-joint second differences of
  the output are about 0.98 correlated with those of the initial flow noise, with a gain of about
  0.006 rad per noise unit. This is not a claim that 98% of physical robot shaking has been explained.
- Unit white noise has mean |second difference| of about sqrt(6) * 0.8 = 1.95, and 1.95 * 0.006 = 0.0117 rad.
  That accounts for nearly all of the measured 0.012 rad.
- The model trained **without** RTC shows a similar correlation (0.976 vs 0.978). RTC training is
  therefore not necessary for this phenomenon, but could still affect its amplitude or boundary behavior.

Independent CPU recomputation from the saved predictions and original seeds reproduced:

| Old checkpoint | Denoising steps | Mean per-joint correlation | Gain (rad/unit) | Mean absolute second difference | After subtracting fitted noise component |
|---|---:|---:|---:|---:|---:|
|d10 RTC 20k|5|0.97799|0.00679|0.01386|0.00262|
|d10 RTC 20k|50|0.99131|0.00588|0.01193|0.00141|
|No RTC 20k|5|0.97576|0.00669|0.01362|0.00278|
|No RTC 20k|50|0.98929|0.00539|0.01096|0.00141|

Each row uses 24 matched held-out observation/seed pairs. Subtracting the fitted component is a
diagnostic, not an online correction method. The current rtc35 checkpoint still needs this same
paired noise experiment; its roughness alone does not establish the same correlation or cause.

### Q3. Does the normalization scale amplify small errors? Yes; whether it causes the residual is unverified.

- Actions are joint deltas from the chunk start (`DeltaActions`), normalized with one quantile range
  (q01/q99) across all 50 steps of the chunk.
- With the current checkpoint (`fix_trimstart_rtc35_h200_20261021/20000`), one normalized unit equals
  **0.13-0.93 rad**, depending on the joint, because far-future deltas are large.
- The average real step is about **0.004 rad**. Relative to one normalized unit, this is about
  **0.43–3.1%** across the quoted joint scales, not below 1% for every joint.
- For right J2 specifically, one normalized unit is **0.9055 rad**, so normalized error 0.01 becomes
  about **0.00905 rad**. This makes fine physical errors small in the normalized loss, but does not
  prove that the scale is why the model leaves noise behind.

The transform is continuous and invertible; it does not quantize output actions or reverse the sign
of their differences. A matched retraining comparison is required to establish the effect of changing
the normalization scale. Implementation-axis and precision checks are in Q7.

Method: section 5.1 (reads `norm_stats.json` and the dataset).

### Q4. Why does it look like a rhythmic shake, especially at chunk switches?

- In the 10:47 run, after each replan the arm mostly executes rows 7-13 (median range; zero-based 6-12),
  so a new noisy segment starts about every 300 ms.
- `temporal_consistency=True` reuses the same initial-noise key every request in a session. If the model
  retains a similar residual pattern, consuming the same early suffix rows repeatedly could produce
  periodic retreats. The initial noise repeats; identical output is not guaranteed as observations change.
- Retreat amplitude and mean absolute second difference are different statistics. Similar orders of
  magnitude do not rule out a separate boundary/conditioning contribution.
- Not yet proven: whether the shift between training prefixes (true demo positions) and deployment prefixes
  (planned commands) adds to it. See Q6.

### Q5. Evidence that narrows, but does not settle, other explanations

- The prefix is passed through exactly (section 2).
- Training labels are the robot's **measured** next state: `action_offset=1`, and the probe measures
  `|action[t] - state[t+1]| = 0` on all 25 episodes. They are not commanded targets.
- Recomputing the existing [1,2,1]/4 filter on current outputs reduces suffix second differences by
  about **71–76%**, to **0.0031–0.0038 rad** (Q1). It substantially reduces high-frequency variation,
  but does not guarantee removal of direction reversals or physical shaking. For ideal white noise,
  this filter reduces position standard deviation to about 0.612x and second-difference standard
  deviation to 0.25x; those effects must not be confused.
- Speed saturation was 0 in the 10:44 / 10:45 / 10:47 runs, and moving command/feedback error P95 was
  about 0.014–0.018 rad. These runs do not support increasing the speed cap as the primary remedy;
  this does not exclude other planner dynamics or establish end-effector accuracy.

### Q6. Open questions

- How much does training on demo prefixes vs. deploying with planner prefixes contribute? This needs
  the ablation in section 4.1, including a planner-shaped prefix condition.
- Does fixed noise per session (`temporal_consistency`) make it better or worse on the robot?
- Delays 1-4 were never trained in the d10 model (weights `[39,0,0,0,0,574,873,275,101,32,22]`).
  **The current deployed rtc35 checkpoint uses `[35,2,3,4,6,20,12,3,10,3,2]`, covering every delay 0–10.**
- The ordinary validation loss forces `d=0` (`train=False`), so that metric does not evaluate nonzero-prefix
  continuation. Add explicit evaluation by prefix length and source rather than interpreting low d=0 loss
  as evidence that planner-prefix continuation is accurate.

**Evaluation implementation update (2026-10-22):** the standard held-out evaluator now adds
separate demonstration-prefix losses for supported d=0/4/5/6/7/8/10, while retaining
the historical d=0 metric. The evaluation-only API also accepts a separately supplied
planner prefix; the normal dataset loader does not supply one and reports this explicitly.
CPU regression tests pass, but current-checkpoint prefix evaluation results are still pending.
See [evaluation behavior and limitations](docs/RTC_EVALUATION.md).

### Q7. Is RTC normalization applied at the wrong granularity? No such axis error was found in the deployed code.

The audit used the remote serving code and current checkpoint assets directly:

- State and action statistics are separate 14-element vectors; each joint has its own scale, shared
  across chunk time. Neither prefix length nor other action rows change a given row's normalization.
- Training and inference subtract the current chunk-origin state for joint dimensions before
  normalization. Grippers remain absolute opening fractions. Padding to 32 dimensions happens afterward.
- The deployed clean-prefix padding repair already zeros the unused 18 dimensions. This was a historical
  bug, not an unfixed issue newly identified in this audit.
- RMSNorm/adaRMSNorm computes the mean square over hidden channels **within each token**, not across
  time or across the prefix. Per-action conditioning supplies separate flow times without changing this axis.

CPU checks with production transforms and a recorded sample found:

| Check | Maximum difference |
|---|---:|
|Encode 6/7/8/10/50 rows vs. encode full chunk then slice|0|
|RTC re-encoding vs. training action transform|about 3e-8|
|Physical action encode/decode round trip|about 1.02e-7|
|Stock vs. expanded RMSNorm, matching conditioning, float32/BF16 input and parameter combinations|0|
|Change other tokens while inspecting one token's RMSNorm output|0|

These checks are much tighter than the observed retreats, but are not a full-model precision benchmark
or a reconstruction of every historical training-code version.

There is a separate discretization: π0.5 converts normalized **measured state** into 256-bin text input.
With current statistics, one bin is about 0.008835 rad (0.506°) for right J2, 0.008184 rad for left J2,
and 0.0134–0.0135 rad for J1. Actions themselves remain continuous. Retreats occur both within and across
state bins; these observations do not establish state quantization as the cause. Images and continuous
RTC prefixes also supply information.

## 4. Proposed fixes

### 4.1 Offline check first (4x4090 GPU, no robot)

Use the **current rtc35 checkpoint** and adapt `analysis/prefix_ablation_20261021/evaluate.py`
(same saved observations and seeds, including actual planner-shaped prefixes) to
compare roughness plus held-out MAE for:

1. normal Gaussian initial noise (baseline);
2. zero initial noise;
3. initial noise smoothed over time (for example, a Gaussian filter along the 50-step axis, rescaled to unit variance);
4. baseline output, then smoothing only the generated rows after the prefix (Savitzky-Golay or a cubic spline),
   keeping the prefix fixed and matching position/velocity at the prefix end.

Pass criteria: roughness close to the demo level (about 0.001-0.003 rad) **and** held-out near/full MAE
no worse than baseline, including gripper error. Also check boundary velocity, stopping/correction delay,
and preservation of already committed prefixes. Run only when GPU0 is not serving a robot test.

Zero or temporally smoothed noise changes the independent-Gaussian distribution used in training;
rescaling variance does not restore independence. The old script's self-prefix comes from a previous
model prediction, not the current planner. Prefix conditions must be compared on the same observation
subset (the old oracle/self groups have different sample counts). Two held-out episodes remain only two
independent episodes, regardless of the number of frames or seeds.

### 4.2 Candidate intervention without retraining (server side)

Apply the winner of section 4.1 inside the trained sampler or in `RTCAwarePolicy.infer`, after the model
output and before unnormalization. Keep the prefix rows unchanged, and record the change in `result["rtc"]`
so recordings show which smoothing was active. Check boundary constraints in physical units. Offline
smoothness alone is not a deployment criterion. Then run a closed-loop robot test and count backward steps
at switches with `switch_alignment.py`.

### 4.3 Candidate retraining experiments, not yet validated fixes

- Evaluate normalizing the deltas **separately for each of the 50 steps** (per-step mean/std or quantiles).
  Measure the resulting early-step scales rather than assuming they will be a few mrad. This requires
  recomputing statistics and retraining; replacing the deployed stats alone would violate the learned contract.
- The current Normalize transform slices only the feature dimension. A short 6/7/8-row RTC prefix cannot
  directly broadcast against `[50,14]` statistics: encoding must select statistics for the new chunk's
  time positions. Re-audit retained normalized gripper values in the server-previous-chunk path as well.
- Predicting step-to-step deltas and accumulating them is another action contract, with possible cumulative
  drift. It needs coordinated training, RTC encoding, and output decoding changes, not an inference switch.
- Keep the current RTC delay weights. Retrain with the same data split, then compare roughness, held-out MAE,
  and backward steps at switches against the current checkpoint.
- Add evaluation by `d` (0/4/5/6/7/8/10) and by true/planner prefix source, so continuation is measured.

## 5. Methods (how the numbers were produced)

### 5.1 Jitter probe (read-only)

Script: `analysis/rtc_jitter_probe_20261022/jitter_probe.py`. It only reads files and doesn't load a model
or use the GPU.

- Reads `norm_stats.json` from the checkpoint and prints `(q99 - q01) / 2` per joint
  (rad per normalized unit).
- For every dataset episode, computes mean |d2| and |d1| of `action`, and checks `action[t]` vs `state[t+1]`.
- For every saved prediction `<obs>_n<steps>_s<seed>_<mode>.npz`, computes mean |d2| over the whole chunk,
  rows 3-10, and rows 10+, grouped by split / prefix mode / denoising steps.

Run on 4x4090 (reached via everwise-friend), from any directory:

```bash
/home/sunny/miniconda3/envs/openpi/bin/python analysis/rtc_jitter_probe_20261022/jitter_probe.py
```

The script lives in this repo on everwise-friend; to run it on 4x4090 without copying, run this from the repo
root on everwise-friend:

```bash
ssh 4x4090 '/home/sunny/miniconda3/envs/openpi/bin/python -' < analysis/rtc_jitter_probe_20261022/jitter_probe.py
```

Optional env vars (defaults in the script): `TSQ_STORAGE`, `PROBE_CHECKPOINT`, `PROBE_DATASET`, `PROBE_PREDICTIONS`.

### 5.2 Other sources

- Boundary backward-step statistics and the J2 example: `analysis/test_review_20261022_late/`
  (`details.py`, `switch_alignment.py`).
- Noise correlation and RTC vs. no-RTC training: `analysis/no_rtc_comparison_20261021/REPORT.md` on 4x4090.
- Prefix padding fix and paired self-prefix evaluation: `analysis/rtc_padding_fix_20261021/REPORT.md` on 4x4090.
- Latency and prefix-length distribution: `analysis/rtc_training_audit_20261020/README.md`,
  `analysis/offline_paired_20261020/LATENCY_PREFIX_RECOMMENDATION.md`.

### 5.3 Independent verification added on 2026-10-22

Full report: [RTC normalization and document audit](analysis/rtc_normalization_20261022/README.md).
All added verification used CPU and saved data; no new model inference or robot execution was performed.

- [CPU normalization/RMSNorm checks](analysis/rtc_normalization_20261022/cpu_audit.json), reproduced by
  `audit_cpu.py` in an OpenPI Conda environment with `JAX_PLATFORMS=cpu`, `--stats-dir` and `--sample`.
- [Current-model roughness](analysis/rtc_normalization_20261022/current_roughness.json), reproduced from
  deployment recordings by `current_roughness.py`.
- [Trimmed demonstration statistics](analysis/rtc_normalization_20261022/trimmed_demo_stats.json).
- [Old-checkpoint noise recomputation](analysis/rtc_normalization_20261022/old_noise_recheck.json),
  reproduced by `recheck_noise_cpu.py` using saved predictions and original seeds on CPU.
- [Original jitter-probe rerun](analysis/rtc_normalization_20261022/document_probe.txt). Its defaults
  combine current rtc35 normalization statistics with old d10 predictions; preserve that provenance.
