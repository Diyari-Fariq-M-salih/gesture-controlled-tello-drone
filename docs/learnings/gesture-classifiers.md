# Gesture classifiers: what each one can and cannot read

Moved from `docs/COMMON_MISTAKES.md` on 2026-09-23 when that file was cut to
its top five. Wording unchanged.

## The two classifiers use different gesture vocabularies

The rule decodes the direction of the index finger; the SVM maps static hand
poses. In the captured dataset the lateral component of the index vector clears
its own threshold in under 3% of frames of *every* class — so the rule
structurally cannot emit LEFT or RIGHT from dataset poses. Cross-vocabulary
scores measure the mismatch, not accuracy. Cue by what the operator does, with
`--vocabulary pose|pointing`.

## FORWARD and BACK are motions, not poses

For the rule, they come from frame-to-frame change in hand bounding-box area, so:

- a static hold produces nothing (a still hand sits ~65x below threshold);
- a symmetric in-out motion fires the opposite class on the return;
- the cue needs a fast stroke and a slow return.

The SVM reads FORWARD and BACK as static poses instead; short paper §V-D
("Depth commands") compares the two on the same protocol.
