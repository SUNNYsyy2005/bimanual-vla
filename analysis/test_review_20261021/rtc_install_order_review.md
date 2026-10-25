# RTC plan installation before the next prefix reservation

The control loop reserved the next request's RTC prefix before installing the
just-returned plan. When inference exceeded the launch period, the next request
was already due. Its reservation extended the old plan by another delay window
(350 ms for seven steps), even though the new result was available.

Install the accepted plan before constructing the next inference snapshot.
Keep the original observation timestamp and all execution checks. The next
request now samples the trajectory actually scheduled for publication.

## Recorded prevalence (2026-10-21, policy port 8099)

Wait means result arrival to scheduled splice, not inference duration. Denominator
is installed plans. These recordings precede the installation-order fix.

| Run | Configured Hz | Plans | Wait >= 300 ms | Next request exactly accounts for commitment | Median wait ms | Transitions / active second |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16:39:23 | 2 | 35 | 0 | 0 | 115.49 | 1.882 |
| 16:56:28 | 2 | 70 | 1 | 1 | 67.64 | 1.944 |
| 16:59:14 | 2 | 26 | 1 | 1 | 62.95 | 1.827 |
| 17:02:00 | 4 | 46 | 33 | 33 | 356.50 | 2.172 |
| 17:03:12 | 4 | 54 | 43 | 42 | 406.02 | 2.164 |
| 17:19:31 | 4 | 67 | 50 | 50 | 357.30 | 2.285 |
| 17:21:06 | 4 | 44 | 35 | 34 | 355.44 | 2.326 |

Latest two runs: 85/111 (76.6%) waited at least 300 ms; 84/111 match the
next-request reservation. Observation-to-splice medians were about 710 and 715 ms.

Publication transitions count changes in servo context generation during actual
command publication, excluding braking, holds and out-of-recording samples.
Active-execution cadence did not fall between 16:39 and 16:56. However, cadence
over the first-to-last publication interval fell from 1.888 to 1.647 Hz due to
interruptions. These runs are not a controlled comparison; less jitter cannot
be attributed solely to better planning. A pipeline can retain similar cadence
while delaying every result by another 350 ms.

## Validation

The regression test reproduces >=350 ms delay when reserving first and <=20 ms
when installing first. It verifies that the next prefix samples generation 2's
scheduled trajectory and that the reservation remains valid.

151 local tests passed across chunk planner, servo, observation bridge, RTC
client, recording and inference wakeup suites. No real-robot recording after
this installation-order fix has yet been validated. Compare the same 4 Hz / d7
settings next; reducing request frequency would confound that comparison.
