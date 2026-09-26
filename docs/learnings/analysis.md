# Analysis pitfalls

## Correlations here are small-n

Moved from `docs/COMMON_MISTAKES.md` on 2026-09-23. Wording unchanged.

The scale-coverage result is r = 0.764 over **12 ordered pairs**, with the
predictor chosen after seeing the matrix. It is reported as exploratory. Do not
promote it.

## The unit of analysis for cued captures is the hold

The operator is cued once and forms the gesture once, so frames within a hold are
not independent. Report per-hold figures alongside per-frame ones (short paper
Table III gives both, with Wilson intervals).

## Declared operator error is excluded, never relabelled

An inverted or mis-formed hold that the operator declared at capture time is
dropped with `analyse_operational_gesture --exclude CLASS:ROUNDS` (e.g.
`LEFT:0-2`), and the decision is recorded in the output. Its frames
are never reassigned to the class the operator actually formed.
