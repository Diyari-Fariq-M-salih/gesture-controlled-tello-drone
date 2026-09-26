# Merge plan: design exposition from the M2 report into the deployment study

The report explains the system; the paper measures it. This merge takes the
report's *method of explaining* and none of its claims. Where the two disagree,
the paper's measurements win.

Everything in Part 1 below was read out of the current code, not copied from
either document. Two published figures did not survive that check and are
flagged in **Part 0**.

---

## Part 0 — two published numbers that do not reproduce

### 0.1 The 1.7 m face-following standoff is wrong, and its conclusion is inverted

> **Published** (§V-F twice, and the Figure 2 caption): "its target framing
> corresponds to a 1.7~m standoff, which sits on the edge of that cliff rather
> than within the envelope."
>
> **Measured: 0.94 m, near the centre of the envelope.**

`FaceFollowConfig.target_area_frac = 0.075` is a fraction of the decoded frame
area. The envelope sweep logs both the face box and the operator's declared
standoff at eight marked distances, so the mapping is measured rather than
assumed. Over the 4,861 frames at the five marks inside the envelope
(≤ 1.75 m):

| calibration | value |
|---|---|
| median `sqrt(area_frac) × d` | 0.2579 m |
| median `bbox_h × d` | 214.4 px·m |

`sqrt(0.075) = 0.2739`, so the target standoff is `0.2579 / 0.2739` = **0.94 m**.
With the `deadband_area = 0.008` either side, the controller holds the operator
between **0.90 and 1.00 m**. The `sqrt(area_frac)` form needs no assumption
about frame width, so this does not depend on the resolution question below.

**Provenance of the error.** `docs/session_log_2026-08-28.md:155` reads
"`target_area_frac = 0.075` ≈ a 263 px face ≈ 1.7 m standoff, which sits
mid-band. The two subsystems agree by accident." Two mistakes compounded:
263 px is `sqrt(0.075 × 1280 × 720)`, a **1280×720** frame, where the Tello EDU
streams 960×720; and the px→m step used the superseded 0.9–2.4 m envelope, the
one the denser sweep later corrected to 0.5–1.5 m. The paper corrected the
envelope and carried 1.7 m across unchanged — then **reversed its sense**, from
"sits mid-band" to "sits on the edge of that cliff."

**Consequence.** The claim that the face-following standoff is a "design
parameter the offline benchmark cannot expose" does not hold: the parameter is
fine, and comfortably inside the envelope. Two sentences and a figure caption
need correcting, and Part 3's proposed table row inverts with them.

This figure was never registered in the audit, which is why it survived three
review passes. It is registered now.

### 0.2 `gesture_hold_s` is declared and never read

`ControllerConfig.gesture_hold_s = 0.35` ("keep last gesture for a moment if
hand disappears") has no reader anywhere in the tree. The behaviour it names is
actually governed by `mode_hold_s`/`hand_release_s` in the mode manager and by
`rule.reset()` on `hand_release_s`. It is not documented in Section III.

---

## Part 1 — verification of every design fact before import

| # | fact | verdict | deployed value |
|---|---|---|---|
| 1.1 | EMA smoothing | **VERIFIED, scope misstated by both documents** | α = 0.35, landmarks only, **rule arm only** |
| 1.2 | Face-follow control law | **VERIFIED; target framing REFUTED** | see below |
| 1.3 | Mode hysteresis | **VERIFIED** | hold 1.2 s, release 0.8 s / 0.8 s |
| 1.4 | Search trigger and duration | **CHANGED** | trigger 10 s, duration **28 s**, cooldown **10 s** |
| 1.5 | Battery failsafe | **VERIFIED** | ≤ 15 % |
| 1.6 | Standard-mode fallback | **PARTIAL** | largest face box ✓; hand is MediaPipe's single best, not "first" |
| 1.7 | Hand-to-operator association | **ABSENT** | no association exists |
| 1.8 | Rule thresholds | **VERIFIED** | `dir_thr` 0.10, `scale_thr` 0.18 |
| 1.9 | SVM | **VERIFIED** | RBF, C = 10.0, γ = scale, StandardScaler, 75/25 stratified, seed 42 |
| 1.10 | Throttle rates | **VERIFIED, plus a third throttle** | hands every 2nd, faces every 4th |
| 1.11 | Face recognition model | **RESOLVED — both documents partly right** | InsightFace `w600k_mbf` |

### 1.1 EMA — the asymmetry neither document states

α = 0.35 (`ControllerConfig.ema_alpha`), applied inside
`RuleBasedGesture._smooth` to the **landmark vector**, and nowhere else in the
gesture path. `TrainedClassifier.predict` receives `det.landmarks` **raw**.

So the report's eq. (2) is correct about the rule arm and wrong as a general
statement, and the paper does not mention smoothing at all. This matters for
§V-E: the two arms of the in-flight comparison do not receive the same input.
The SVM arm does still advance the rule's EMA state, because the controller
calls `rule.predict()` alongside it to keep the depth cue observable — but that
output is discarded.

A third, unrelated EMA (α = 0.25) smooths the **face box area** in
`FaceFollower`, not the control output. The report's eq. (6) puts `EMA` around
the control term; the code smooths the measurement instead.

`--no-ema` sets α = 1.0. Temporal state is reset on hand loss
(`rule.reset()` after `hand_release_s`), as the report describes. **VERIFIED.**

### 1.2 Face-following control law

```
ex = cx - W/2 ,  ey = cy - H/2 ,  ez = 0.075 - EMA_0.25(area_frac)
yaw = clip(0.20 · ex,          ±20)      deadband |ex| < 18 px
ud  = clip(-0.12 · ey,         ±40)      deadband |ey| < 18 px
fb  = clip(0.30 · (1000 · ez), ±40)      deadband |ez| < 0.008
```

Control loop rate-limited to 15 Hz; `lost_timeout_s` raised from 0.7 to **2.0 s**
by the controller; `detect_every_n` forced from 3 to **1** (see 1.10).
Target framing **0.94 m**, holding band **0.90–1.00 m** — see Part 0.1.

### 1.4 Search — the deployed value is already the retuned one

`nohuman_search_s = 10.0` matches the report. `search_duration_s = 28.0` and
`search_cooldown_s = 10.0` do not: the report specifies a 5 s sweep and 5 s
cooldown. The 28 s is the retuning §V-K reports, and it is in the shipped
default with the reasoning in a code comment (`search_yaw_cmd = 30`, measured
at ≈13°/s, so a full rotation needs 28 s; yaw is kept low deliberately because a
faster spin blurs the frame past the detector).

Section III must therefore state **both**: 5 s as specified, 28 s as deployed
after §V-K. The gap is the finding.

Note also that `DeterministicConfig`'s own dataclass defaults (5 / 12 / 5 s) are
dead — the controller always constructs it from `ControllerConfig`. Section III
quotes the live values.

### 1.6 / 1.7 Target selection, and the association that does not exist

**Face, both modes:** the largest detection by box area, as the report says.
**VERIFIED.**

**Hand:** `HandGesture(max_num_hands=1)` — MediaPipe returns at most one hand
and the controller takes it. Not "the first detected hand in the set", because
there is no set.

**Association:** the report's "only the closest hand to the authorized user's
face is used for hand gesture control" is **ABSENT from the code**. The gate is
a conjunction, not an association:

```python
hand_detected = hand_detected_raw and face_detected
```

Any hand in frame commands the aircraft while the authorized operator's face is
also in frame. This is exactly the limitation the paper already states — "the
gate establishes that an authorized operator is present, not that the commanding
hand belongs to that operator" — and the code is the reason. Section III-E must
say so plainly, because it is the paper's strongest limitation and the report
asserts the mitigation that does not exist.

### 1.10 The throttle removed in §V-G is a third one

Deployed: `hand_every_n = 2`, `face_every_n = 4`. **VERIFIED.**

Neither is what §V-G removed. `FaceFollower` carries its own `detect_every_n`,
defaulting to 3, which compounded with the controller's `face_every_n = 4` into
a real detection every 12th frame. The fix forces it to 1 and raises
`lost_timeout_s` to 2.0 for arbitration, while giving identity its own much
tighter freshness window `crop_max_age_s = 0.35 s` (the paper's Δ_max). That
separation *is* the §V-G finding, and III-E should specify all three windows.

Also undocumented in the paper: streak gating, `hand_streak_on = 2` and
`face_streak_on = 1`.

### 1.11 Face model — the discrepancy resolves, and both documents are partly right

The deployed code loads `models/third_party/arcface.onnx`. Read from the
artefact:

| property | value |
|---|---|
| size | 13,616,099 B (12.99 MiB) |
| producer | PyTorch, `torch-jit-export` |
| input | `input.1`, NCHW 3 × 112 × 112, float32 |
| output | 512-d, L2-normalised downstream |
| topology | 49 Conv, 34 PReLU, 12 residual Add, Flatten → Gemm (`fc`) → BatchNorm (`features`) |

That topology and that size are **InsightFace's `w600k_mbf`**, the recognition
model *inside* the `buffalo_s` pack: a MobileFaceNet trained on WebFace600K
under ArcFace loss. The report's "buffalo_s, 159 MB" names the whole pack
(detector + recognizer + auxiliaries); the paper's "MobileFaceNet trained under
ArcFace, exported to ONNX" describes the one file actually loaded, and is the
accurate one. **No correction needed to the paper**; III-E gains the training
set and the provenance.

**Two deployment departures the paper should state**, because they bear on the
0.32 % → 19.3 % gap:

1. **No alignment.** InsightFace's pipeline similarity-transforms the crop onto
   five landmarks before embedding. `FaceID.embed` resizes the padded MediaPipe
   box straight to 112 × 112. The embedding is being fed something its training
   distribution did not contain.
2. **Different detector.** `buffalo_s` pairs the recognizer with RetinaFace-500MF;
   the deployed system uses MediaPipe face detection for speed.

Preprocessing is `(x − 127.5)/128`, RGB, and the layout is auto-detected from
the ONNX input shape.

### 1.9 SVM, against the frozen artefact

`sha256 e34704a1…a934d4` matches `FROZEN_SVM_SHA256`.
`Pipeline([StandardScaler(), SVC(kernel='rbf', C=10.0, gamma='scale',
probability=True)])`, trained by `train_model.py` on a stratified 75/25 split
with `random_state=42`, seven classes. **VERIFIED.**

One detail for III-C: `probability=True` fits Platt scaling by internal 5-fold
CV, and the runtime takes `argmax(predict_proba)`, which is not guaranteed to
equal `predict`. The paper reports the runtime path.

---

## Part 2 — new Section III, ≈3.5 pages

| § | content | source |
|---|---|---|
| III-A | Architecture and platform: **new block diagram**, three UDP channels (8889 / 8890 / 11111, local 9013), threaded latest-frame buffer, throttling rationale. Existing hardware paragraph kept. | paper + new |
| III-B | Landmark perception: report Fig. 1 (21-point skeleton) and Fig. 2 (runtime overlay), eq. (1), EMA eq. (2) at α = 0.35 **scoped to the rule arm**, throttles 2/4, streaks 2/1, reset on loss. | report, corrected |
| III-C | Two vocabularies: rule eqs. (3)–(4) with `dir_thr` 0.10 and `scale_thr` 0.18; SVM eq. (5) with the Part 1.9 parameters; the depth stability gate. **Plus a paragraph the paper has never had: why both exist** — the rule was the prototype path and stayed as the deployed fallback, they decode different vocabularies, and that is why Table VI has three rows and §V-E two arms. Existing explicit-selection and manifest-hash paragraph kept verbatim. | report + paper + new |
| III-D | Face following: eq. (6) rewritten to the actual law (Part 1.2), gains, clips, deadbands, 15 Hz limit, and the target framing **derived to 0.94 m from the sweep calibration**. | report form, code values |
| III-E | Identity gate: enrolment N = 20 averaged (one clause noting this replaced one-shot), cosine similarity, Schmitt trigger with τ_on 0.55 / τ_off 0.45 / k = 3, the **three freshness windows** (arbitration 2.0 s, identity 0.35 s, follow 0.7 s), why detection and recognition are decoupled, the **absence of hand association**, the standard-mode fallback, and the two departures from the benchmark pipeline (no alignment, MediaPipe detector). | report + code |
| III-F | Arbitration: **new state diagram**. Six modes (`keyboard, gesture, face, search_360, hover, land`), priority order, transition conditions, hold 1.2 s / release 0.8 s, search trigger 10 s, duration **5 s specified → 28 s deployed**, battery ≤ 15 %. | report + code |
| III-G | Reason-only LLM: the report's §4.7 argument imported as design rationale — latency against the RC loop, non-determinism under similar states, unsafe suggestions under ambiguous perception — with the verdict sentences removed. Measurements stay in §V-L. | report |

Every number in III is a Part 1 value. No performance verdict is imported.

---

## Part 3 — the specification/measurement table

New Table, placed at the end of III. One row per parameter that III sets and V
measures. **The face-follow row inverts** relative to the brief, because of
Part 0.1.

| parameter | value | set in | measured in | result |
|---|---|---|---|---|
| Δ_max crop freshness | 0.35 s | III-E | V-G | background share 56 % → 20 %, authorized 41 % → 68 % |
| arbitration face hold | 2.0 s | III-E | V-G | search_360 decisions 130 → 4 |
| τ_on | 0.55 | III-E | V-F | FRR 19.3 % in flight; 35.1 % per-frame at the same threshold |
| τ_off, k | 0.45, 3 | III-E | V-H | 15.7 vs 78.5 mode switches/min; impostor 0.0 % → 5.6 % |
| enrolment N | 20 | III-E | V-F | genuine cluster 0.770 ± 0.109 |
| mode hold / release | 1.2 s / 0.8 s | III-F | V-A | 149/149 trials produced the specified behaviour |
| search duration | 5 s → 28 s | III-F | V-K | 60° of a specified 360°; retuned gives 318° over 22 episodes |
| battery threshold | 15 % | III-F | V-K | 11 consecutive landings, 91 % → 21 % |
| `scale_thr` Δs | 0.18 | III-C | V-I | depth F₁ 0.008 / 0.066; pose form reaches 0.447 / 0.567 |
| `dir_thr` | 0.10 | III-C | V-D | lateral F₁ 0.921 / 0.937 / 0.855 |
| depth streak | 2 | III-C | V-I | gated vs ungated differ by 0.004 |
| hand / face throttle | 2 / 4 | III-B | V-B | duty 50 % / 25 %, 22.3 fps |
| face-follow target | 0.075 → **0.94 m** | III-D | V-F | **inside** the 0.5–1.5 m envelope, holding band 0.90–1.00 m |

The table's point survives the correction, and arguably sharpens: of the
parameters set by reasoning rather than measurement, the search duration was
wrong by a factor of 5.6, the depth threshold produced a channel that barely
functions, and τ_on was chosen for impostor margin at a cost of 16 points of
false rejection — while the one parameter the paper accused of being wrong turns
out to be right.

Audit extension: every row checked against both the code constant and the
results-section figure it points at.

---

## Part 4 — proposed Section V order (for confirmation before applying)

Current order is chronological. Proposed order mirrors III. **Only two
subsections move**: the depth and vocabulary results rise to sit with the other
gesture-classifier results, ahead of identity.

| new | subsection | answers | was |
|---|---|---|---|
| V-A | Scenario Validation | III-F | A |
| V-B | Runtime Performance | III-A | B |
| V-C | Cross-Session Gesture Classification | III-B, III-C | C |
| V-D | Gesture Classification Through the Aircraft | III-C | D |
| V-E | Both Classifiers in Flight | III-C | E |
| **V-F** | **Depth Commands: Motion Cue versus Static Pose** | III-C | **I** |
| **V-G** | **Vocabulary Packing and Landmark Precision** | III-C | **J** |
| V-H | Identity Verification: Offline and Operational | III-E | F |
| V-I | Stale Bounding Boxes in the Identity Path | III-E | G |
| V-J | Authorization Hysteresis | III-E | H |
| V-K | Flight Behaviour and Deployment Constraints | III-F | K |
| V-L | Reason-Only Language Model | III-G | L |

Depth stays ahead of packing: depth establishes the two-defect result (FORWARD
fixed, BACK not), packing then diagnoses BACK. All cross-references are by
`\label`, so they follow the move; the audit's section tags are relabelled.

No finding is dropped and no audited claim changes.

---

## Part 5 — narration to cut

| # | passage | action | ≈words |
|---|---|---|---|
| 5.1 | §V-F "The earlier on-ground figure of 32.0 % … does not survive a larger sample" | state 16.7 % / 9.2 % over 2,792 frames; drop the superseded 32.0 % | 60 |
| 5.2 | §V-D half-column on why the rule's 0.287 is not its accuracy | one sentence; **keep** the Table VI row, since dropping it removes the evidence for the sentence | 110 |
| 5.3 | §V-C photometry paragraph | reduce to one clause: the blue-minus-red separation is 74 in A against 2, 0, −27 | 70 |
| 5.4 | §V-C three-sessions → D-closes-the-cell → factorisation-is-coarse walk | compress to the claim plus p = 0.533 / 0.067 and r = 0.764 vs 0.296, −0.094 | 260 |
| 5.5 | Table V (`tab_factors`) | fold into text | ~12 lines |
| 5.6 | Table II (`tab_scenarios`) | fold into three lines | ~14 lines |
| 5.7 | §V-L | keep only 199 calls, 5.5 % timeouts, 94 % charge, 1766 ms | 60 |
| 5.8 | §V-K battery failsafe | one sentence | 40 |

**Table I is not folded.** The brief groups it with Table II, but its
justification ("ceilings at 149/149, discriminates nothing") applies only to
Table II. Table I is the related-work comparison and is the only place the
paper's positioning is visible at a glance. Flagged for your call.

Estimated cut ≈ 600 words and 26 table lines ≈ 1.2 pages, against III growing
≈ 2.5 pages and two new figures. Net ≈ +1.5 pages before the Part 6 figure
savings.

---

## Part 6 — figures

**Import:** report Fig. 1 (21-point skeleton) → III-B; report Fig. 2 (runtime
overlay) → III-B.

**Drop:** report Fig. 3 (feature-mean sanity check — carries no claim);
report Fig. 7 (class-balance bars — the counts 2715 / 2345 / 2613 / 2784 are in
§IV-B text); report Figs. 4–5 (the 100-image and 4,678 ROC panels).

**Compress:** report Fig. 6 (full 13,410-set ROC) to a single panel in III-E or
V-H, captioned as the design-time benchmark that Table VIII supersedes.

**New:** architecture block diagram (III-A); HFSM state diagram (III-F).

**Paper Fig. 1** (`fig_sessions`, 2 × 4 confusion matrices): dropping the
within-session row halves the figure, ≈0.45 column-inches ≈ 0.25 page. The four
within-session numbers (0.997, 0.988, 1.000, 1.000) are already in the text and
the *argument* is that they are uninformative — so the row is illustrating a
non-result. **Recommend dropping it**; the caption keeps one sentence saying
the within-session protocol returned 1.000 on the two hardest sessions.

Net figure budget: −1 report figure block, −0.25 page from Fig. 1, +2 new
diagrams ≈ +0.6 page.

---

## Part 7 — design-time expectation against measurement

Twelve pairs exist. Each gets the expectation stated in III and the measurement
in V, with the failure said plainly.

| # | report expected | paper measured | where |
|---|---|---|---|
| 1 | RIGHT/DOWN confusion from "similar hand orientations under certain viewpoints" | single-digit separation below the estimator's resolution; and the confusing classes are RIGHT→BACK on the ground, BACK→LEFT/RIGHT in flight — not RIGHT/DOWN | III-C → V-G |
| 2 | "99.7 % accuracy" | 0.997 / 0.910 / 0.783; the first is leakage | III-C → V-C |
| 3 | "EER of 0.32 % confirms the system's suitability" | 19.3 % FRR in flight; 23× to imagery, the rest to the operating point | III-E → V-H |
| 4 | "logs indicate stable mode selection" | stable *because of* hysteresis: 15.7 vs 78.5 switches/min without it | III-F → V-J |
| 5 | bounded 360° search, 5 s | 60° achieved; no command produces a rotation in 5 s | III-F → V-K |
| 6 | "battery safety landing activates deterministically" | 11 consecutive landings — true, and not discriminating | III-F → V-K |
| 7 | one-shot enrolment, first face | N = 20 averaged template | III-E |
| 8 | InsightFace aligns the crop before embedding | no alignment; padded MediaPipe box resized to 112 × 112 | III-E → V-H |
| 9 | "all geometric quantities are computed on EMA-smoothed landmarks" | true of the rule, **false of the SVM**, which sees raw landmarks | III-B → V-E |
| 10 | "only the closest hand to the authorized face is used" | no association exists; the gate is a conjunction | III-E → Limitations |
| 11 | face-follow "maintains a target distance" | 0.075 → 0.94 m; the paper's own 1.7 m is wrong (Part 0.1) | III-D → V-F |
| 12 | landmarks "robust to moderate scale changes" | scale coverage is the strongest predictor of transfer, r = 0.764 | III-B → V-C |

Pairs 9, 10 and 11 are new to this pass and are the ones that most change what
the paper says about itself.

---

## Constraints honoured

No measurement changes, no reruns, no new claims. The audit must still
reproduce every existing claim; new registrations cover the Part 3 table, the
face-follow standoff and the Part 1 parameter values.
