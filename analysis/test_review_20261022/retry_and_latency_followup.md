# Retry reservation fix and non-network latency

## Changes

- ServoLoop remembers a deferred installation while a valid old plan remains.
  Requesting a new RTC prefix raises ChunkPlanDeferred until installation
  succeeds, instead of extending the old commitment by another 7–8 steps.
  Existing promises are retained. Hold/fault invalidation, candidate rejection,
  successful installation and candidate expiry release the gate. Expiry permits
  fresh inference rather than an indefinite retry deadlock.
- The client's existing snapshot-error path reschedules the request; it now also
  restores pending configuration delivery, because that snapshot was never sent.
- Periodic inference wakes at its own future deadline between 20 Hz control
  ticks. An in-flight request or an overdue retry keeps the normal control
  wakeup, avoiding a busy loop. Extra wakes preserve the regular control clock.
- Model recording retains model_phase_timing. Client transport diagnostics add
  feedback_to_snapshot_start_ms, camera_select_ms, rtc_prefix_build_ms,
  execution_metadata_ms, snapshot_freeze_ms, request_worker_queue_ms,
  request_payload_build_ms and capture_to_rpc_ms. These expose the prior
  aggregate observation-preparation interval without changing observation times.

## Measurements

In the recorded 09:51 run, capture-to-launch median was 52 ms, full chunk
compilation 28 ms, local splice computation 1.2 ms, and server policy 303 ms.
The 52 ms includes client work after state capture, not just camera capture.
The new substage diagnostics require a new client run; no retrospective
substage timings are invented for old records.

Eight independent requests to the existing local service reused the recorded
state and nearest decoded video frames from the 09:53 run. The first request
bootstrapped the separate RTC session; the seven remaining requests used the
saved planner prefix. No CAN, camera devices, or robot commands were involved.
Decoded recording frames are not claimed to reproduce the original input bytes.

| Warm request phase | Median |
| --- | ---: |
| RTC request parsing | 0.006 ms |
| RTC lock wait | 0.003 ms |
| RTC preparation | 0.945 ms |
| OpenPI input transform | 0.131 ms |
| Observation construction | 2.423 ms |
| Model sampler | 331.152 ms |
| OpenPI post-sampler | 0.315 ms |
| RTC total | 335.435 ms |

Input construction overlaps/encloses some reported phases; do not add nested
durations. This probe's load/scene differs from deployment and is not an
optimization before/after benchmark. The sampler is the dominant non-network
cost. The 150 ms target remains unmet. Image encoder, prefix prefill, and each
denoising iteration are not individually timed by this probe; GPU optimization
requires a separate sampler profile and output-equivalence checks.

The request wakeup fix removes control-clock quantization (up to approximately
50 ms when no other work blocks it). It does not make a 300 ms sampler sustain
4 independent requests per second, remove configured RTC commitments, or repair
model predictions after a failed grasp.

## Validation

Regression reproduces a deferred installation, denies two attempted old-prefix
reservations without changing existing promises, and installs on the next retry
within one writer period rather than another 350 ms. Hold, rejection and expiry
release are covered. Scheduler tests check independent deadlines, in-flight and
overdue behavior. Recorder test verifies nested server phase persistence.
No post-change real-robot trial has been run by the agent.
