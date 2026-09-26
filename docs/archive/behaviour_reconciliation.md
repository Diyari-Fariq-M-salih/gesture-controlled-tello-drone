# Behaviour count reconciliation

The abstract, contributions list and conclusion said "five further behaviours,
four corrected with paired measurements." Neither number matches the sections.

## What the sections actually contain

| # | Behaviour | Section | Corrected? | Paired before/after? | The pair |
|---|---|---|---|---|---|
| 1 | Stale bounding boxes in the identity path | V-E | yes | **yes** | background cluster 56% → 20% of frames; authorized share 41% → 68%; `search_360` decisions 130 → 4; `face` decisions 72 → 82 |
| 2 | Authorization chatter at the threshold | V-F | yes | **yes** | mode-switch rate 78.5 → 15.7 per minute; the failed trial replayed through Eq. (1) gives 13 authorized frames of 15 rather than 5 |
| 3 | Search sweep covering 60° not 360° | V-H | yes | **yes** | median coverage 60° over 5 episodes → 318° over 22 |
| 4 | Frame-rate-dependent depth commands | V-G | no | no | measured only: 4 of 64 cued-lateral frames (6.2%) carry an uncommanded depth component |
| 5 | Radio environment dependence | V-H | no | no | characterised across venues; no correction attempted |
| 6 | Reason-only language model behaviour | V-I | no | no | characterised; the architectural exclusion predates the measurement |

**Six behaviours. Three corrected with genuine paired measurements.**

The battery failsafe ladder (V-H) is deliberately excluded from the count: it
validates designed behaviour across eleven consecutive activations rather than
identifying a behaviour that needed finding.

## Why "four corrected" was wrong

The fourth item counted was most likely the depth-cue finding, which is analysed
and explained but never fixed — the unnormalised `delta_t` is still in
`RuleBasedGesture.predict()`. Section V-G says so plainly. Counting it as
corrected overstates the work in the direction that a reviewer checks first.

## Wording applied

**Abstract.** "Six further behaviours are characterised from the logs, three of
them corrected with paired measurements."

**Contributions.** "Characterisation of six deployment behaviours that appear
only in flight, three of them corrected and reported with paired measurements,
together with in-flight ablations of the identity gate and of authorization
hysteresis."

**Conclusion.** "Six further behaviours were quantified from the logs and three
corrected with paired measurements, and in-flight ablation establishes that
authorization hysteresis reduces the mode-switch rate fivefold."

All three are in `paper/main.tex` as of this pass.
