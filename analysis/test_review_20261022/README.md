# 2026-10-22 correction and grasp audit

Scope: completed local policy port 8099 recordings at 09:50:05, 09:51:09,
and 09:53:36 (the last completed during this audit). Read-only analysis; no
motion commands, controller settings, or production code changed.

## Timing

All values below are medians unless stated otherwise. First publication uses
actual per-arm JointCtrl timestamps and the generation of the published plan
sample, excluding braking, holds, and samples outside the recording interval.

| Run | Configured Hz / RTC delay | Capture to return | Return to first publication | Capture to first publication | First source index | Wait >=300 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 09:50:05 | 4 / 7 | 360 ms | 76 ms | 436 ms | 8 | 14/137 |
| 09:51:09 | 4 / 8 | 364 ms | 76 ms | 440 ms | 8 | 12/147 |
| 09:53:36 | 2.5 / 8 | 310 ms | 109 ms | 420 ms | 8 | 0/96 |

Source indices are zero based. Medians are computed separately, so summing
column medians need not produce the median of the total.

The earlier installation-order fix improved the usual path, but did not cover
installation deferral. All 26 long waits in the two 4 Hz runs have a planning
deferral near result arrival and an outstanding commitment matching the next
request's capture plus its configured delay. execute_next catches
ChunkPlanDeferred and returns; the loop then still allows the next request to
reserve the old trajectory before the returned result can be installed on retry.
This remains an execution-side issue requiring a separate fix.

The 2.5 Hz run has no long-wait example. It still has a roughly 400 ms committed
prefix: 84/96 installed plan boundaries match that request's capture + 8 steps.
Median server policy processing is 295 ms, RPC 298 ms, capture-to-launch 10 ms,
full plan compilation 26 ms, and local splice computation 1.3 ms. Thus the
remaining delay is not principally network transfer. The camera timestamp age
at launch has median 32 ms/P95 61 ms; these software timestamps do not establish
sensor exposure-to-delivery latency.

Actual capture intervals at 2.5 Hz have median 447 ms (~2.24 Hz). A deviation
arising just after a sampled observation must wait for another observation in
addition to the ~420 ms capture-to-command delay; a representative response
window is therefore roughly 0.42–0.87 s, not a measured hard upper bound.

## Grasp evidence, latest run

Video brackets the right bottle tipping between +19 and +20 seconds. The right
wrist camera sees the cap near +14 s, and largely background from +16 s onward.
The overhead camera continues to see the bottle. This is a visibility change,
not proof of a camera freeze or proof that the policy recognized the failure.

The latest returned predictions evaluated at the publication clock continue to
move right joint 2: 1.991 rad at +20 s, 2.123 at +24 s, and 2.251 at +28 s.
Published targets are respectively 1.908, 2.123, and 2.239 rad. Right gripper
prediction remains near fully open (0.986, 1.000, 0.991), with approximately
70 mm commands. This is not a sustained stop/recovery request that the actuator
ignores. Some individual reversals exist, but the broader predicted trajectory
continues after the bottle tips.

During +18–28 s, the largest-joint command/feedback discrepancy per sample has
P95 0.0145 rad, while latest-model/command same-clock discrepancy has P95
0.0491 rad. Across moving execution, the prior-issued-command/feedback P95 is
0.0447 rad, max 0.0901 rad. These joint-space errors do not establish end-effector
millimetre accuracy. No samples reach 99% of the configured 0.6 rad/s velocity
cap in the audit's moving-sample metric. Raising the velocity cap is not supported
as the primary fix by this run.

See `right_grasp_095336.png` and `grasp_095336.jpg` in the local analysis directory.
`timing.py` reproduces the metrics from existing recordings; `summary.json`
contains compact numeric results. Same-clock model targets are interpolated
from absolute wire actions with source time capture + (index+1)/20; no cumulative
sum of deltas is used. These targets include server RTC conditioning, so they
cannot isolate intrinsic model ability from the closed-loop prefix feedback.

## Priorities supported by these records

1. Close the deferred-installation prefix reservation path, without removing
   physical safeguards or revoking already-issued prefix commitments silently.
2. Reduce policy computation and observation-to-request overhead, then recalibrate
   the RTC delay. Reducing delay alone trades away coverage; increasing it adds
   a longer committed trajectory interval.
3. Evaluate grasp-local prediction/recovery on recorded observations and matched
   training examples. The sustained continuation after tipping cannot be explained
   by a single additional 100–400 ms wait. Visibility, model conditioning, and
   recovery coverage require investigation; the logs alone do not identify which
   one is decisive.
