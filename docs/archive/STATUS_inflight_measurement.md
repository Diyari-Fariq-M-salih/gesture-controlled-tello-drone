# Status: in-flight classifier measurement

Current state of the work, for review. Assumes familiarity with the repo.

## Goal

Measure gesture classification in flight, for both classifiers, on their own
vocabularies. The paper's strictest gesture figure (SVM, 0.783) is a ground
measurement with the drone on a table; every other component measured both ways
got worse airborne (identity gate 2.1x, face embedding latency 1.4x). This
closes that.

Secondary: test a prediction recorded before flying — that depth F1 rises from
0.008/0.066 (rule, motion cue) to roughly 0.447/0.567 (SVM, static pose).

## What changed in the code

### Classifier selection (done, tested)

Was: `self._trained = TrainedClassifier(...) if model_path and labels_path else None`,
falling through to the rule. No logged launch passed a model, so 43 runs flew the
rule while the paper characterised the SVM.

Now, in `src/gesture_classifier.py`:

- `--classifier {svm,rule}` mandatory, no default; argparse rejects a launch without it
- SVM arm validates existence, SHA-256 against a frozen artefact, loadability, and
  the seven-class label set — all before the aircraft is armed
- resolved arm, model path, hash, frozen-match written to a top-level `classifier`
  manifest field
- startup self-test pushes a synthetic 63-dim vector through the selected branch
  and aborts if the label shape belongs to the other one
- one `self.selection.predict(...)` call in the loop, so manifest and latency
  column cannot disagree by construction

Frozen model: `models/production/model.joblib`,
`e34704a1317101572a7604ae6ec84484c35d323637faf1a40ba11fbe88a934d4`. This is the
artefact behind the 0.783; retrained in the current sklearn from session-A
features with the same pipeline and seed as `per_session_eval.py`, scoring 0.9971
against §V-C's 0.997. The original `model-400.joblib` was pickled under sklearn
1.7.2 against a 1.5.2 runtime and warns on load.

`tests/test_classifier_selection.py` — 8/8 passing. Sweeps the archive: 52 runs,
16 with gesture frames, all rule-based.

### Arm randomisation (built, then ruled out)

`--arm-randomise` and `scripts/reveal_arms.py` exist and work. **Not used.** The
two arms need physically different gestures — pointing vs static poses — so the
operator cannot be blinded to the arm. Recorded in the preregistration as
unachievable rather than claimed.

What is randomised: which arm flies first (one coin flip), arm order across
batteries (alternating, not blocked), class order within rounds (seeded).
Not controlled: operator expectation.

### Measurement protocol — third iteration

Two earlier designs were built and discarded:

1. **Cued holds, on-screen.** Rejected: the operator cannot read a laptop while
   flying a drone in front of them.
2. **Cued holds, spoken** (`src/cue_voice.py`, Windows SAPI, works). Rejected
   for a harder reason: the drone *acts* on what it reads. A 6 s hold of FORWARD
   at rc_speed 30 travels ~3.9 m. The room is 3 m. Sustained directional holds
   are not flyable.

Current design: **free flight scored against external video.**

- operator flies naturally, gives commands however they like, moves in and out
  of frame
- phone records throughout
- `--freeflight-log` writes a dense per-frame log (`cued.csv`): cue-free, but
  carrying raw classifier output, post-gate emitted label, RC command,
  hand/face authorization, fb streak count, battery, altitude, seq, frame age
- key `m` stamps a sync marker and beeps; operator claps simultaneously. This is
  the only link between the phone's clock and the host's
- operator annotates the video into `start_s,end_s,intended` **before opening
  any log**
- `scripts/score_against_video.py` aligns on the marker and scores

The cue infrastructure (`src/cue_schedule.py`, `cue_voice.py`,
`--cue-rounds`) is retained and functional; it is unused under the current plan.

## What this measures that the cued design could not

- natural operator behaviour rather than schedule-following
- detection and the identity gate, via movement in and out of frame
- **false positives on idle hands** — `NONE` segments. The cued protocol had no
  way to measure these, and a system whose premise is not acting on the wrong
  input should be reporting them

## Preregistration

`docs/PREREGISTRATION_svm_inflight.md`, committed before any flight. Fixes the
primary metric (depth channel, per-hold), sample size (42 holds/arm; 18 needed
for depth at 80% power, 82 needed for the all-class comparison which is therefore
descriptive only), the frozen model, the exclusion rule (declared operator error
excluded, never relabelled), and a directional prediction that cuts against the
change: on the ground 72% of the SVM's RIGHT frames collapsed into BACK (F1 0.433
vs the rule's 0.937), which in flight means pointing right sends the drone
backwards.

**The preregistration is now partly stale.** It specifies per-hold scoring of a
cued schedule; the current design is segment scoring of free flight. It needs
amending before flying, and that amendment is outstanding.

## Open issues for review

1. **Self-annotation bias.** The operator annotates their own intent from video.
   Mitigation is procedural only — annotate before seeing logs, timestamp the
   file. An independent annotator would be better and is not available.
2. **Sync precision.** One clap. Frame-level alignment is not claimed; segment
   boundaries are approximate and segments should be padded.
3. **No blinding**, unavoidable as above.
4. **Unequal vocabularies.** The arms are not doing the same task. Cross-arm
   comparison is therefore about deployed capability, not classifier quality on
   matched input.
5. **Sample size under the new design is undefined.** Segments per flight depend
   on how the operator flies. The power analysis was done for cued holds and
   does not transfer directly.
6. **Actuation confound.** In free flight the drone moves in response, changing
   operator standoff — which §V-C identifies as the dominant transfer variable.
   The ground captures held distance roughly constant; this will not.

## Operator report, unverified

The operator reports all SVM gestures working in a brief informal flight test.
Not a measurement, and noted here only because it bears on the RIGHT→BACK
prediction. An informal check of this kind passed 149 times while the paper
described the wrong classifier.

## Paper state

9 pages, builds clean, 64/64 audited claims reproduce from logs. The offline EER
is attributed to I. Chaabeni and marked EXTERNAL, not derived.

Editorial decision taken this session: the classifier substitution is reported as
factual configuration in §III-B and is no longer framed as a discovery narrative.
The Discussion's self-audit passage, the abstract's disclosure sentence and the
conclusion's were removed at the author's request. Every remaining statement is
accurate — flights ran the rule, the SVM is characterised offline, both are
stated — but the paper no longer presents the mismatch as a finding.

Outstanding paper edits, pending flight results: Limitations still says "a flight
trial has never been flown with the SVM in the loop"; the abstract and conclusion
still carry 0.584 as the deployed figure.

## Immediate next step

Amend the preregistration for the free-flight design (metric, segment
definition, sample size), then fly two batteries — one per arm — with
`--freeflight-log`, phone recording, sync clap.
