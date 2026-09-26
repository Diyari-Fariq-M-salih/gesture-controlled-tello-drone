# Preregistration — in-flight classifier comparison

Committed before the first launch of the re-measurement. Fixing the primary
metric and the sample size in advance is what stops a favourable subgroup from
becoming the headline after the fact.

**Date:** 2026-09-20
**Frozen SVM:** `models/production/model.joblib`,
sha256 `e34704a1317101572a7604ae6ec84484c35d323637faf1a40ba11fbe88a934d4`
Verified by hash at every load; `--allow-unpinned-model` is required to override
and must not be used for these flights.

## Configuration change being measured

The controller previously selected the gesture classifier by testing whether a
model path was supplied, defaulting to the geometric rule. The earlier flight
campaign ran the rule on that default. Selection is now explicit, and this study
measures both arms in flight. Guards in place: `--classifier` is mandatory with no
default, the SVM arm validates model, hash and label set before arming, the
resolved arm and model hash go in the manifest, a startup self-test confirms the
executed branch, and a regression test fails on any manifest that disagrees with
the latency column its run wrote.

## Question

Does the SVM, flown, differ from the rule, flown, in the accuracy with which a
cued gesture reaches the aircraft as the intended command?

## Primary metric — fixed

**Per-hold accuracy on the depth channel (FORWARD, BACK), each classifier on its
own vocabulary.** This is the channel the change is predicted to alter, it is the
best-powered comparison available within the flight budget, and the prediction
was recorded before flying.

A hold is one cue: the operator is told a class, given 5 s to form it and
recorded for 6 s. A hold counts correct when the majority of its detected frames
carry the cued class. The hold, not the frame, is the independent unit: frames
within a hold are the same pose milliseconds apart, and treating them as
independent repeats in miniature the leakage error this paper identifies in
within-session splitting.

Vocabularies differ deliberately. The SVM arm is cued with the static poses it
was trained on, the rule arm with the pointing gestures it decodes.
Cross-vocabulary scoring measures the mismatch, not the classifier, and is not a
metric here.

## Secondary metrics — reported, not claimed

- Overall per-hold accuracy across all seven classes. **Underpowered by design:**
  82 holds per arm are needed to resolve the ground-observed 0.809 against 0.611
  at 80% power, and the flight budget affords 42. Reported with Wilson intervals
  and explicitly not claimed as a significant difference.
- Per-frame accuracy, as the optimistic figure.
- Per-class F1 and the 7x8 confusion matrix including a no-detection column.
- Hand-detection rate per class.
- Median and p95 stage latency conditioned on frames where the stage ran.
- Uncommanded-depth-contamination rate for the SVM arm.
- Per-run stream fps and reconnection count.

## Sample size — fixed

Two-proportion test, two-sided, alpha 0.05.

| comparison | priors | n/arm at 80% |
|---|---|---|
| depth channel (primary) | 0.45 vs 0.05 | 18 |
| all classes (secondary) | 0.809 vs 0.611 | 82 |

**Committed: 42 holds per arm**, six rounds of seven classes. That is 12 depth
holds per arm against the 18 required, so the depth comparison is reported at
84 depth holds pooled across both arms where the pooled test applies, and
per-arm where it does not.

Budget: 11 s per hold, 7.7 min of cueing per arm, 15.4 min across both, inside
the ~28 min airborne afforded by three batteries with manoeuvre and repositioning
overhead.

Amended 2026-09-20, before the first launch, when the flight budget was costed.
The original draft committed 84 holds per arm for the all-class comparison; that
does not fit three batteries, and flying an underpowered version of it while
calling it primary would have been the error this document exists to prevent.

## Stopping rule

Fly all 42 holds per arm, or stop and report the count reached. **No interim
analysis.**

## Blinding — not achievable, and why

The two arms require physically different gestures: the rule decodes the
direction of a pointing index finger, the SVM recognises static hand
configurations. The operator cannot perform a cue without being told which
vocabulary applies, and the vocabulary identifies the arm. Operator blinding is
therefore impossible for this comparison, in the way it is impossible for any
trial in which the participant must execute the condition.

`--arm-randomise` exists in the controller and is correct for comparisons whose
arms share a vocabulary. **It is not used for these flights**, and the runs are
flown with an explicit `--classifier`.

What is randomised instead:

- which arm is flown first, by one recorded coin flip;
- the arm order across batteries, alternating rather than blocked, so link
  quality and battery state cannot align with arm;
- class order within every round, by the existing seeded schedule.

Not controlled: operator expectation. The operator knows which classifier is
flying and may perform gestures differently as a result. This is a limitation of
the comparison and is reported as one; it is not corrected for.

## Confound control

Stream rate varies 4–24 fps by venue and run, so arms alternate rather than
block by day. The analysis reports per-arm fps distributions and checks they
overlap; if they do not, the comparison is reported as confounded by link quality
and the primary metric is not claimed.

## What will not change

- The frozen SVM: no retraining, no hyperparameter change, no threshold tuning
  after seeing flight results.
- The identity gate, arbitration logic and thresholds.
- Any existing run directory or log. The original rule-only runs are evidence and
  stay untouched; new runs go in new directories.
- Any claim about the original 149 trials. They ran the rule and the paper will
  continue to say so.

## Exclusions

Declared operator error only — a hold in which the operator demonstrably
performed a gesture other than the one cued — excluded, never relabelled, with
the count and reason stated. Relabelling from the classifier's own output would
be circular.

---

## Amendment 2026-09-25 — hand-face association, second day (n7)

Written on 2026-09-25 before the day's first launch. It adds a comparison; nothing
above changes.

**Status of the first day.** One run per arm was flown on 2026-09-24
(`20260924-170628_drone_n7-hover-cued-svm-association`, then
`20260924-184619_drone_n7-hover-cued-svm-no-association`) and analysed before this
amendment was written. Those two runs are exploratory. Only the second day is
tested against what is fixed here.

**Question.** With the frozen SVM flown under the hover lock, does enabling
hand-face association (`--associate-hands --target-side left`, as implemented on
2026-09-24: pose every 20th hand detection, up to 4 hands) change the accuracy with
which a cued gesture reaches the aircraft?

**Protocol, fixed.** `--classifier svm --cue-rounds 6 --no-actuate --record`, cue
seed 7, same room, standoff about 1 m, one fully charged battery per run. Order on
the second day is association off, then association on (the reverse of the first
day), so across both days each arm is flown once first and once second.

**Primary metric.** Per-frame accuracy of the emitted gesture against the cue over
hold frames, per arm, on the second day, counting only holds that were airborne
throughout and have at least 80% of the run's median frames per hold. Reported
with the first-day difference beside it; the direction is the finding. No
significance test is claimed at one run per arm per day.

**Secondary, reported not claimed.** Per-hold accuracy (majority of hand-detected
frames); per-class recall; share of hold frames with the face not verified; frame
rate and hand-detection time during holds; association rejection reasons.

**Stopping and exclusions.** A run is not repeated because of its outcome. A run
the operator declares aborted for a control error, before its results are looked
at, is named `_aborted`, kept, excluded, and may be flown again on a fresh battery.
Holds after a failsafe landing fall out by the airborne rule. Declared operator
error within a hold is excluded, never relabelled, as above.
