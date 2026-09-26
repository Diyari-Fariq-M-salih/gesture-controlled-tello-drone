# Response to external review

Every item verified against the logs, the generated tables and the paper source
before any edit. Items that did not hold are recorded as REFUTED and no edit was
made. Numbers below are recomputed, not quoted from the draft.

Audit after this pass: **74/74 log-derived claims reproduce** (was 64). The
offline EER remains EXTERNAL, attributed to I. Chaabeni and derived by no script
here. PDF builds clean, 0 undefined references, 11 pages.

---

## Part 1 — factual checks

### 1.1 "Four of five static classes exceed 0.92 on both classifiers" — **VERIFIED, and worse than stated**

| class | SVM | rule |
|---|---|---|
| CENTER | 0.988 | 0.881 |
| LEFT | 0.980 | 0.999 |
| RIGHT | 0.985 | 0.968 |
| UP | 0.999 | 0.854 |
| DOWN | 0.920 | 0.908 |

**SVM 5/5, rule 2/5.** The reviewer said two of five for the rule and four for
the SVM; the SVM is actually five of five. The sentence was wrong for both
columns.

*Action:* rewritten to the true counts, with the gap decomposition added.

### 1.2 Table I cell and caption superseded — **VERIFIED**

Cell read `0.584 / 19.3%` with a caption saying both classifiers were measured on
the ground. §V-F flies both.

*Action:* cell now `0.850 / 19.3%`; caption states both classifiers are flown and
gives the rule's 0.651. Every other occurrence of 0.584 checked: two remain, both
in §V-E and Table VI where they correctly denote the ground figure.

### 1.3 Introduction justified not flying — **VERIFIED**

The paragraph said the gesture measurement is ground-based "since what the
measurement needs from the aircraft is its codec and its optics rather than its
motion", which §V-F contradicts.

*Action:* rewritten as the justification for hover lock rather than for not
flying.

### 1.4 Trial and run counts — **VERIFIED as correct; scope now stated**

Recounted from the run directories:

| | count |
|---|---|
| campaign runs (pre-fix, non-ground) | 43 |
| hover-locked flights | 6 |
| ground-capture runs | 9 |
| raw scenario rows, all runs | 281 (270 after the validity rule) |

The abstract's 270 / 43 / 149 is correct for the campaign. The hover-locked holds
are **not** trials and are outside that count.

*Action:* §V-F states the hover-locked captures are separate from the trial count.

### 1.5 "Six deployment behaviours" — **VERIFIED as correct**

Unchanged by this pass; §V-J is a mechanism for an existing behaviour, not a new
one.

### 1.6 Standoff reported as a point against a range — **VERIFIED**

Recomputed as the same statistic, converted via the Figure 2 calibration:

| capture | bbox median (IQR) | distance (IQR) |
|---|---|---|
| SVM | 241 (225–265) | 0.85 m (0.74–0.94) |
| rule A | 234 (206–258) | 0.89 m (0.77–1.07) |
| rule L/R | 248 (230–264) | 0.81 m (0.75–0.91) |

*Action:* all three reported with medians and IQRs in both units.

### 1.7 Audit blind to prose generalisations — **VERIFIED**

Nine candidate sentences enumerated; five make a checkable quantified claim.
Recomputed:

| claim | computed | verdict |
|---|---|---|
| four of five static ≥0.92, both | 5/5 and 2/5 | **WRONG** |
| only two off-diagonal cells >0.10, ground SVM | 2 | OK |
| no off-diagonal >0.12, folds C and D | 0.119 | OK |
| detection 1.00 every static class, ground rule | min 0.995 | OK |
| 0.70 of depth-cued frames detected | 0.697 | OK |

**One of five wrong.** *Action:* audit extended with a `d_generalisations`
derivation registering counts rather than individual figures, since the numbers
in these sentences do not appear in the text and cannot be matched
term-by-term.

---

## Part 2 — depth channel

### 2.1 Abstract calls the depth channel usable while §V-J calls the same class weak — **VERIFIED**

Table VII gives SVM FORWARD 0.777 and **BACK 0.336**. §V-J's weak static class is
BACK. The abstract described the same class as both usable-as-a-pose and weak.

### 2.2 Two-defect rewrite — **applied**

Abstract, §V-I and the conclusion now state that the pose formulation removes the
motion cue's pathology outright (FORWARD 0.000 → 0.777) and in doing so exposes a
second, independent problem in BACK. 0.336 is not described as usable anywhere.
The FORWARD improvement is stated as the real result it is.

### 2.3 Depth share of the gap — **VERIFIED, reviewer's arithmetic close**

Recomputed from the logs, weighting by frame share:

| class | contribution to gap |
|---|---|
| FORWARD | +0.1132 |
| BACK | +0.0488 |
| UP | +0.0247 |
| LEFT / DOWN / RIGHT / CENTER | +0.0043 / +0.0041 / +0.0032 / +0.0004 |
| **total** | **+0.1988** |

Depth accounts for **0.162 of 0.199 = 82%**. The reviewer's 0.158 was close. The
remainder is almost entirely **UP**, not "CENTER and UP" — CENTER contributes
+0.0004.

*Action:* "almost entirely" replaced with "82% of that gap", and UP named.

---

## Part 3 — hover-lock claims

### 3.1 "The aircraft's motion does not contaminate the cue" — **VERIFIED as overclaimed**

Telemetry during the rule capture: ground velocity median **0.0** in all three
axes, p95 **0.0**. The aircraft was stationary by construction.

*Action:* narrowed to hover drift, with the telemetry stated and the actuated
case declared unmeasured.

### 3.2 Ground-vs-flight confound — **VERIFIED as unresolvable**

The ground capture wrote only `manifest.json` and `op_gesture.csv`. There is no
`decisions.csv`, so **apparent face size was never recorded** and its standoff
distribution is unknown. Conditioning is impossible.

*Action:* the claim is restated to the weaker one the data supports — flight
produced nothing resembling the identity gate's order-of-magnitude collapse, and
a protocol holding standoff fixed returned a higher figure than one that did
not.

### 3.3 "3.9 m on a single uncorrected command" — **VERIFIED as a projection**

Computed from command rate × 5.98 s during a capture with actuation suppressed.
The aircraft did not move.

*Action:* figure kept, explicitly marked a projection.

---

## Part 4 — preregistration

### 4.1 Not cited, and **never committed** — **VERIFIED, worse than stated**

`docs/PREREGISTRATION_svm_inflight.md` is absent from `main.tex` and is
**untracked in git**. Its mtime (2026-09-20 22:53) precedes the first
hover-locked flight (2026-09-21 00:25) by about 90 minutes, but with no version
control that provenance rests on a file timestamp alone.

### 4.2 Registered prediction — **DISCONFIRMED, with direction reversal**

Registered: the SVM's RIGHT collapses into BACK, as on the ground (72% of RIGHT
frames, F1 0.433).

In flight:

| cued | emitted |
|---|---|
| RIGHT | RIGHT 0.99, BACK 0.00 |
| LEFT | LEFT 0.98 |
| **BACK** | **BACK 0.34, LEFT 0.33, RIGHT 0.30** |

The prediction failed, and the confusion **reversed**: BACK is now the error
source rather than its target.

### 4.3 / 4.5 — **applied**

§V-F gains a paragraph stating what was fixed before flying, the registered
prediction, that it was disconfirmed, and both caveats: the file was not placed
under version control, and it specifies a cued protocol whose actuation
assumption was revised to the hover lock.

### 4.4 — **applied**

§V-J notes that a pose family separated by one digit should confuse in whichever
direction estimator error falls, and that the reversal is what that predicts
whereas a classifier deficiency would reproduce its direction.

---

## Part 5 — internal consistency

### 5.1 Welch t over frames — **REFUTED as stated; the real problem is larger**

The reviewer objected that 518 frames are not independent. Correct, but the
structure is worse: the split is **entirely between holds**. Rounds 0–3 read BACK
throughout (pinky 0.515–0.589), rounds 4–5 read LEFT throughout (0.848, 0.914).
No hold contains both, so a paired test is impossible and the unit is **six
holds clustered 4-vs-2**.

*Action:* statistic dropped. The effect size is retained with the between-hold
structure stated explicitly, and the reference value (genuine LEFT 0.821) carries
the argument.

### 5.2 Other frame-level statistics — swept

All remaining intervals are Wilson intervals on per-hold counts, already the
correct unit. Per-frame accuracies carry frame-level intervals and are labelled
as the optimistic figure in both Table VI and Table VII.

### 5.3 Per-class column was unconditioned recall — **VERIFIED, and materially misleading**

Recomputed:

| class | SVM recall | SVM precision | SVM F1 |
|---|---|---|---|
| LEFT | 0.980 | **0.736** | 0.841 |
| RIGHT | 0.985 | **0.685** | 0.808 |
| BACK | 0.336 | 0.859 | 0.483 |

BACK leaking into LEFT and RIGHT costs their precision roughly 0.27 and 0.31,
which recall hides entirely.

*Action:* Table VII now reports recall, precision and F1 for both arms. F1 is
commensurable with §V-I's ground figures (0.447 / 0.567).

### 5.4 Detection conditioning — **VERIFIED**

SVM FORWARD is 0.777 unconditioned and **0.858** conditioned on detection at
0.905. *Action:* both given, as Table VI does.

### 5.5 §V-I title — **VERIFIED**

*Action:* pending; the section now leads with the pose-vs-motion result and the
title still names only the frame-rate dependence.

---

## Part 6 — hedging

### 6.1 Background-vs-illumination separation — **VERIFIED as unsupported**

| group | n | pairs | range |
|---|---|---|---|
| background varies | 2 | 0.638, 0.930 | 0.638–0.930 |
| illumination varies | 4 | 0.737, 0.842, 0.966, 0.977 | 0.737–0.977 |

Near-total overlap. Mann-Whitney U = 2.0, **p = 0.533**. The smallest two-sided
p attainable at n = 2 vs 4 is **0.067**, so the design cannot reach significance
even in principle.

Scale-coverage correlations confirmed: train-covers-test **r = 0.764** (n = 12),
shared background 0.296, shared illumination −0.094.

*Action:* abstract and contributions hedged to what two against four overlapping
pairs can carry, and the scale-coverage result now leads.

### 6.2 Varga's critique is subject-independence — **VERIFIED**

Four sessions of one operator removes a session axis, not Varga's subject axis.

*Action:* §II-C states that session-independent partitioning removes a different
leakage axis, and that all figures here are within-subject and therefore upper
bounds on cross-subject performance. Limitations already notes the single
operator; not duplicated.

---

## Outstanding

- **5.5** §V-I retitle.
- **Page budget**: 11 pages, up from 9. Table VII grew by three columns and
  Part 4 added a paragraph.
- **4.1**: the preregistration should be committed to version control before
  release, so its date is verifiable by something other than a filesystem
  timestamp.

---
---

# Second pass

A second review of the text the first pass produced. Same rule: every item
recomputed from the logs before any edit, refutations recorded as such.

The published in-flight figures are scored on `emitted_gesture` — the label that
leaves arbitration and the depth stability gate, i.e. the command the aircraft
would act on — over the `n6_svm_A` capture and the `n6_rule_A` capture with its
LEFT/RIGHT holds replaced by the `n6_rule_LR` reflight. Everything below is
recomputed under that definition and reproduces Table VII exactly.

Audit after this pass: **87/87 log-derived claims reproduce** (was 74). PDF
builds clean, 0 undefined, 0 overfull, **11 pages**.

---

## Part 1 — errors introduced by the previous pass

### 1.1 Per-class contribution to the gap — **PARTIAL**

Actual frame counts per cued class, not assumed-equal:

| class | SVM frames | rule frames |
|---|---|---|
| CENTER | 743 | 814 |
| LEFT | 732 | 681 |
| RIGHT | 669 | 650 |
| UP | 761 | 724 |
| DOWN | 777 | 748 |
| FORWARD | 761 | 777 |
| BACK | 776 | 717 |

Exact decomposition, each arm at its own frame share (sums to the gap):

| class | contribution |
|---|---|
| FORWARD | +0.1132 |
| BACK | +0.0488 |
| UP | +0.0247 |
| LEFT / DOWN / RIGHT / CENTER | +0.0043 / +0.0041 / +0.0032 / +0.0004 |
| **total** | **+0.1988** |

**0.162 / 82% CONFIRMED** and robust: 81.6% at each arm's own frame shares,
81.0% at pooled shares. The reviewer's 0.037 remainder is also right.

**"Almost all of the remainder in UP" — REFUTED as written, but so is the
reviewer's replacement.** The split is not a stable quantity. At each arm's own
frame shares UP takes 67% of the remainder and CENTER 1%; at pooled shares UP
takes 54% and CENTER 42% — the reviewer's figures. The whole difference is
composition: the rule arm has 814 CENTER frames to the SVM's 743, and that 9.6%
excess cancels CENTER's recall gap (0.988 against 0.881) almost exactly. Neither
apportionment is more correct than the other.

*Action:* the headline kept with its 82%; the remainder named as UP and CENTER
with an explicit statement that their relative shares are sensitive to the frame
counts and are not separated. Both weightings registered in the audit.

### 1.2 DOWN also falls below 0.92 on the rule — **VERIFIED**

Rule recall: CENTER 0.881, LEFT 0.999, RIGHT 0.968, UP 0.854, **DOWN 0.908**.

The count "two of five" was correct — LEFT and RIGHT are the two that pass — but
the clause naming CENTER and UP read as a complete enumeration of the shortfall
and omitted DOWN.

*Action:* all three named.

### 1.3 "Exceed 0.92" was made on recall against a caption arguing for F1 — **VERIFIED**

| class | SVM recall | SVM F1 |
|---|---|---|
| CENTER | 0.988 | 0.964 |
| LEFT | 0.980 | **0.841** |
| RIGHT | 0.985 | **0.808** |
| UP | 0.999 | 0.999 |
| DOWN | **0.9202** | 0.958 |

On recall all five pass; on F1 two fail, for exactly the reason the caption
gives. DOWN is 0.9202 — it exceeds 0.92, but not at the precision the table
prints, so "exceed" overstated it.

*Action:* metric named explicitly, DOWN stated as reaching the figure, and the
two F1 shortfalls given with their cause. Both counts registered separately in
the audit, so a future revision cannot state one while meaning the other.

### 1.4 Per-hold BACK — **VERIFIED, and the published sentence is wrong**

> **Published:** "four of the six BACK holds being read BACK throughout and two
> read LEFT throughout."
> **Actual:** two BACK, two RIGHT, two LEFT.

Per-hold outcome of all six, from the log:

| round | frames | majority | distribution |
|---|---|---|---|
| 0 | 116 | BACK | BACK 105, RIGHT 10, CENTER 1 |
| 1 | 142 | BACK | BACK 105, RIGHT 28, CENTER 9 |
| 2 | 129 | **RIGHT** | RIGHT 96, BACK 27, CENTER 6 |
| 3 | 132 | **RIGHT** | RIGHT 97, BACK 24, CENTER 11 |
| 4 | 142 | LEFT | LEFT 142 |
| 5 | 115 | LEFT | LEFT 115 |

The reviewer's reasoning was exactly right: 4/6 would imply recall near 0.667
and no frames to RIGHT, and Table VII gives 0.336 with §V-E splitting BACK
0.33/0.30 between LEFT and RIGHT. Only the two LEFT holds are unanimous.

**Per-hold 0.333 against per-frame 0.336.** They do *not* diverge — BACK is the
one class where the two units agree, because its errors are sustained rather
than brief. That is worth stating, and now is.

**The two landmark comparisons have different support**, which the single
"between holds" clause obscured. The little-finger contrast (0.884 vs 0.551) is
entirely between holds: rounds 4 and 5 supply every LEFT-read frame and rounds
0–3 every correctly-read one. The **thumb contrast (0.205 vs 0.526) is within
holds** — rounds 0–3 each mix BACK and RIGHT frames — and is the better
supported of the two, not the weaker.

The definition behind those figures was not registered anywhere and had to be
recovered by matching: extension is `|tip - MCP| / |wrist - lm09|` in 2-D image
coordinates, `lm20`–`lm17` for the little finger and `lm04`–`lm02` for the
thumb. It reproduces 0.884 / 0.551 / 0.821 / 0.205 / 0.526 to three decimals.

*Action:* §V-J rewritten. Per-hold/per-frame agreement stated, the two-vs-four
hold support of each comparison stated, three cross-section consistency checks
registered.

---

## Part 2 — hedge propagated inconsistently

### 2.1 Body and conclusion still asserted what the abstract disclaims — **VERIFIED**

§V-C read "Varying the background alone costs more than varying the illumination
alone… The classifier is comparatively robust to a change of light and brittle
to a change of backdrop." The conclusion read "with varying the background
costing 0.784 against 0.880." Both flat assertions.

Recomputed: Mann-Whitney U = 2.0, **p = 0.533**; minimum attainable two-sided p
at n = 2 vs 4 is **0.067**.

*Action:* both hedged to match the abstract; the ordering reported as the
direction the matrix points in, not a resolved effect.

### 2.2 "Separates… does not resolve" — **VERIFIED as contradictory**

*Action:* "is intended to separate… but does not resolve them."

---

## Part 3 — items possibly not applied

### 3.1 Counts — **VERIFIED correct; scope now stated**

270 / 43 / 149 all reproduce. The hover-locked captures contribute **zero**
scenario trials — the last run carrying any is `n4_failsafe` — so the 84 holds
are cleanly outside the 270. The paper had not said so.

*Action:* one clause in §V-E stating it.

### 3.2 "Six deployment behaviours" — **REFUTED**

Six: stale crop, authorization hysteresis, spurious depth commands, search
coverage, battery failsafe, radio environment. Three corrected with paired
measurements: stale crop (130→4), search (60°→318°), depth (0.000→0.777). §V-J
is the mechanism behind an existing behaviour, not a seventh. Count stands.

The preregistration was not added to the contributions: at 11 pages the bullets
are the wrong place to spend lines, and §V-E already states it in full.

### 3.3 §V-J is a fourth category — **VERIFIED**

§VI's structure is "individually correct components composing badly." §V-J is
not that: the estimator and the classifier both behave as specified and the
fault is in the specification. §VI-A's taxonomy — platform-specific rate,
composition error, measurement practice — has no slot for it.

*Action:* a sentence in §VI marking §V-J as outside the shared structure, and a
fourth class added to §VI-A: a specification error, hardware-independent in the
same way, repaired by redesigning the vocabulary rather than the code.

### 3.4 `svm_infer_ms` — **VERIFIED as logged; the fix is not a table row**

The hover-locked SVM capture logs it. But Table III is built from the single
longest airborne run, which predates the column, so adding a row would mix runs.

Measured over the hover-locked captures: **SVM 0.84 ms mean** (median 0.80, p95
1.26, duty 78%), **rule 0.07 ms**, against 26.7 ms for hand detection.

*Action:* reported in §V-B as text with its source run named. The point it makes
is worth having: the 0.199 accuracy difference between the arms costs under a
millisecond, so the classifier choice is not a latency trade-off.

---

## Part 4 — cross-references

### 4.1 "The four failure modes above" — **VERIFIED**

They are enumerated two paragraphs *below*. *Action:* "set out below."

### 4.2 §III-B points only at V-D — **PARTIAL**

V-D does report both arms through the aircraft, on the ground, so the reference
is not wrong. V-E is the in-flight comparison. *Action:* both named, with "on
the ground and in flight."

### 4.3 Renumbering staleness — **REFUTED**

All 24 section cross-references swept against their surrounding prose. Every one
resolves to the section it describes. No staleness from the V-E insertion, the
V-I retitle or the V-J addition.

---

## Part 5 — introduction thesis

### 5.1 "The two differ by up to an order of magnitude" — **VERIFIED as vacuous; the sharper form is supported**

Literally true — but "up to" makes it true of any spread. The sharper version
holds:

- **image-quality-dependent:** at a *fixed* threshold, the identity gate goes
  from 0.32% to 7.3% false rejection, a factor of **23**, attributable to the
  imagery alone.
- **protocol-dependent:** gesture accuracy goes 0.997 → 0.910 → 0.783 by
  protocol, and *rises* to 0.850 in flight at matched standoff. Flight costs it
  nothing.

*Action:* restated as the reviewer proposed, with the two figures, and with the
observation that which cause dominates is itself a result.

---

## Part 6 — audit extension

### 6.1 Both shapes now covered — **VERIFIED as the gap**

None of 1.1–1.4 was catchable before: the numbers those sentences assert do not
appear in the text, so there was nothing to match.

**Shape (a), prose quantifying over a table.** Twelve such sentences enumerated;
nine make a numerically checkable claim. Before this pass, **three were wrong**
(1.1's remainder, 1.2's enumeration, 1.3's metric). Newly registered:
`svm_static_ge_092`, `svm_static_f1_ge_092`, `svm_static_min_recall`,
`rule_static_ge_092`, `rule_static_below_092`, `ground_svm_detection_rate`,
`depth_share_own`, `depth_share_pooled`, `remainder_own`.

**Shape (b), agreement across sections.** Six relationships enumerated; **one
was wrong** (1.4). New derivation `d_cross_section` registers: per-hold BACK
against per-frame BACK and their difference; the hold-majority counts; the
BACK→LEFT/RIGHT split and that it sums to the whole class; and Table I's
headline cell against Table VII's per-frame accuracy, parsed from the generated
`.tex`.

`d_inflight` now emits precision, F1, per-class per-hold recall and the hold
majorities, so the table and every sentence about it come from one derivation.

### 6.2 Metric names checked against the analysis code — **VERIFIED, one mismatch**

Every claim naming a metric was traced to what the code computes:

| claim | named | computed | verdict |
|---|---|---|---|
| §V-D ground SVM depth F1 0.447 / 0.567 | F1 | 0.4465 / 0.5672 | OK |
| §V-D rule lateral F1 0.921 / 0.937 / 0.855 | F1 | same | OK |
| §V-E prereg RIGHT F1 0.433 | F1 | 0.433 | OK |
| §V-C fold A UP F1 0.57, 59% to CENTER | F1 | 0.567, 0.586 | OK |
| §V-C fold B DOWN F1 0.33, 69% to BACK | F1 | 0.330, 0.687 | OK |
| §V-D "detected in every frame" | — | 1.000 exactly | OK |
| §V-E "all five exceed 0.92" | unnamed | recall, not F1 | **MISMATCH** |

One of seven. It is item 1.3, now fixed and registered on both metrics.

---

## Outstanding

- **Page budget: 11 pages, unchanged.** A prose-only compression pass removed
  102 words without touching a figure, hedge or finding — not enough to move a
  page. At 8,721 words, 8 tables and 3 figures this is an 11-page paper.
  Reaching 9 needs roughly 1,200 words or a table; reaching a 6-page conference
  limit means cutting results, which is a scope decision, not an editing one.
- **4.1 (first pass):** the preregistration is staged in git but still not
  committed.

---
---

# Merge pass: design exposition from the M2 report

Plan in `docs/merge_plan.md`, executed in the order it specifies. Part 0.1 went
in as its own commit before any merge edit.

Audit **95/95** (was 87). PDF **13 pages**, 0 undefined, 0 overfull. Regression
tests 9/9.

---

## The correction that went first

**§V-F's 1.7 m face-following standoff was wrong by 0.8 m, and the conclusion
drawn from it was backwards.** Committed separately as `f62dc5e` with the full
derivation; the short version is that 263 px is `sqrt(0.075 × 1280 × 720)` on a
stream that is 960×720, and the px→m step used the 0.9–2.4 m envelope that the
denser sweep had already superseded. Calibrated against the sweep itself the
target is **0.94 m**, holding band 0.90–1.00 m — near the middle of the
0.5–1.5 m envelope, not on its edge.

The claim that this was "a design parameter the offline benchmark cannot expose"
is gone: the parameter is fine. What replaced it is the calibration itself,
which *is* something the offline benchmark cannot supply, since it relates a
control constant to a physical standoff through the deployed optics.

`docs/session_log_2026-08-28.md:155` carries a dated correction in place; the
original line is untouched.

---

## What the audit now covers that it did not

| new derivation | catches |
|---|---|
| `d_follow_standoff` | the standoff figure drifting again: the code constant, both calibrations, the derived 0.94 m with its band, and the conclusion that it lies inside the envelope |
| `d_design_table` | Table III being hand-edited away from the dataclasses it is generated from — 14 parameters checked against live config |
| `d_ground_envelope_frr` | the two on-ground figures that replaced the superseded 32.0 % |

Also re-tagged: **43 claims** carried section tags that had drifted from the
manuscript even before the reorder — the in-flight claims filed under V-F while
the subsection was V-E, the language-model claims under V-J while it was V-L.
They are now recomputed from where each claim's text actually sits.

One registration was **removed**: the on-ground 32.0 % over 122 frames, because
Part 5.1 deleted that figure from the paper. A number not in the manuscript
should not be registered as a claim in it.

---

## Part 1 — what did not survive contact with the code

Everything in the plan's Part 1 table held. Three items are worth repeating
because they change what the paper says about itself:

**EMA is rule-arm only.** α = 0.35 is applied inside `RuleBasedGesture`; the SVM
receives raw landmarks. §V-E now opens with the caveat: the two arms are
compared as deployed capabilities, not as classifiers on matched input, and the
smoothing favours the rule so it cannot account for the direction of the gap.

**There is no hand-to-operator association.** `hand_detected = hand_detected_raw
and face_detected`, with `max_num_hands=1`. §III-E states the mechanism under
its own heading; Limitations cross-references it.

**The face model resolves both ways.** 13,616,099 B, PyTorch→ONNX, 3×112×112
NCHW, 512-d, 49 Conv / 34 PReLU / 12 residual Add / Flatten→Gemm→BN: InsightFace
`w600k_mbf`, the recognizer inside `buffalo_s`. The report's 159 MB names the
pack; **the paper's description was already correct** and needed no change.
§III-E gains WebFace600K and the two departures from the reference pipeline —
no landmark alignment, MediaPipe rather than RetinaFace — and §V-H names them as
a fourth candidate for the 0.32 % → 19.3 % gap, explicitly untested.

Dead config removed: `gesture_hold_s` deleted; `DeterministicConfig`'s search
defaults (5/12/5 s) annotated as unreachable with a pointer to the live values.

---

## Parts 2–7 as executed

**§III** is seven subsections: Architecture and Platform, Landmark Perception,
Two Gesture Vocabularies, Face Following, Session-Level Identity Gate,
Deterministic Arbitration, Reason-Only LLM. Eight numbered equations. Every
parameter is a Part 1 verified value. The paragraph explaining *why two
classifiers exist* is new — the paper never had it, and it is the reason
Table VI has three rows.

**Two new figures**, both generated by `build_design_figures.py`, which reads
the dataclasses rather than taking typed-in labels: an architecture block
diagram and the HFSM state diagram. If a threshold moves, the diagram moves.

**Table III** binds III to V, 15 rows, generated by `build_design_table.py`.
Its last column reads as the merge's argument: the sweep duration was wrong by a
factor of 5.6, the depth threshold produced a channel that barely functions, and
τ_on cost 16 points of false rejection — while the one parameter the paper
accused of being wrong turned out to be right.

**Reorder applied as approved.** Depth and Vocabulary Packing rose to V-F and
V-G; Identity, Stale Crops and Hysteresis shifted to V-H, V-I, V-J. All 22
section labels resolve as planned and no `\ref` dangles.

*Not in the plan:* the reorder broke §VI's opening, which named
"Sections V-G–V-I" as a contiguous range. Those three behaviours are no longer
adjacent, so the range became three explicit references — and the duplicate
sentence that listed the same three mechanisms two lines later was merged into
it.

**Cuts.** 5.1–5.8 applied. Table II folded to prose with every scenario count
kept; Table V folded into the sentence that now carries all four group means.
**Table I stays**, as decided.

**Figure 1** lost its within-session row; the caption keeps the sentence about
1.000 on the two hardest sessions. Report Figs. 3, 4, 5 and 7 were never
imported; Fig. 6 was dropped rather than compressed, since Table VIII supersedes
its interpretation and the page budget was already over.

*Not in the plan:* Part 6 left two stale references to "top row" and "bottom
row" of Figure 1 in §V-C. Both fixed.

**Part 7.** Twelve pairs identified; 9, 10 and 11 given explicit treatment as
directed. Nothing from the report's verdict sentences was imported — checked by
grep for "99.7", "confirms the system", "suitability", "buffalo", "one-shot",
"closest hand", "stable mode selection" and "deterministically": all absent.

Preserved as required: the preregistration paragraph and both its caveats, the
Varga subject-axis framing in §II-C, and the single-operator framing.

---

## Page budget: 13, not 11–12

This is the plan item that did not survive contact. The plan estimated §III at
+2.5 pages against ~1.2 pages of cuts. §III actually added **2,230 words plus
three floats** (a full-width figure, a column figure and a full-width table),
and the Part 5 cuts removed ~450 words and two tables. Net is close to +2.8
pages from a starting 11.

Three trim passes followed, removing 263 words without dropping a figure,
hedge, parameter or number — including letting the two new diagrams carry what
they show instead of restating it in prose. None moved the page count. Page 13
is currently **46 % full**, so absorbing it needs roughly 550–600 more words or
one float.

At that point the cuts stop being redundancy and start being findings, which is
your call rather than an editing decision. The obvious candidates, in the order
I would take them:

1. **Fold Table VI into prose** (~0.35 page). §V-D's text now quotes its
   headline figures anyway, and Table VII carries the comparison that matters.
2. **Drop the runtime overlay screenshot** (~0.2 page), keeping the 21-point
   skeleton. The overlay shows the operator feedback, which no claim depends on.
3. **Cut §V-C's photometry clause and the scale-coverage caveat** (~200 words).
   Both are already labelled exploratory.

Any two of those reach 12 pages. Say which and I will apply them.
