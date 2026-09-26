# Hand-face association

Binds the commanding hand to the verified operator, closing the paper's stated
limitation ("the gate establishes presence, not provenance"). Design by
I. Chaabeni (`Ilyes-branch`, 2026-09-22/23); ported to `tello_gesture_py/src/hand_association.py`
on 2026-09-23. **Off by default**: the paper's flights ran without it.

## How it decides

1. **Face → skeleton.** MediaPipe Pose runs on the frame; the skeleton whose eye
   and mouth points fall ≥60% inside the verified face box is the operator's.
2. **Skeleton → hand.** A hand is accepted if its wrist/thumb/index/pinky sit
   within `assoc_dist_thresh` shoulder-widths of the *target* arm's matching
   points, and clearly nearer that arm than the other. A confident handedness
   label that contradicts the arm vetoes it.
3. **Between pose runs** (every 20 fresh hand detections) the hand is carried by
   bounding-box IoU; a lost track re-runs pose on the same frame.

It fails closed: no qualifying hand means no gesture, and no gesture mode.

## The arm must actually be in shot

Pose returns all 33 landmarks every time and **invents** an out-of-frame arm,
placing it where the visible hand is, so a hand with no arm in shot matched its
own guessed arm. Pose's visibility score does not catch this: on the operator's
accepted calibration frames the elbow was outside the image in 22% of cases yet
had a median visibility of 0.94.

So by default (`assoc_require_arm`) the commanding arm's **elbow and wrist must
lie inside the image with visibility >= 0.5** at every pose check. That keeps 76%
of the operator's close-range calibration gestures; the rest have the elbow
below the frame, which is the case the rule exists to reject. `--no-arm-check`
restores the original behaviour.

## Mirroring decides which arm is read

MediaPipe **Hands** labels handedness as if the image were a mirrored selfie;
MediaPipe **Pose** labels the arms it sees. On an unflipped frame they disagree,
and `mirrored` tells the associator which convention applies.

- The **Tello feed is not flipped**: `mirrored=False` (the default).
- `webcam_demo --flip` flips the image *and* sets `mirrored=True` together.

The original controller set `mirrored=True` on the unflipped Tello feed, which
swaps the arm read: with `target_side='right'` it accepted the **left** hand.

## Which arm: set it to the operator's habit

On 267 calibration frames (the operator's own dataset captures, unflipped), the
operator gestures with the **left** hand:

| target / mirrored | accepted |
|---|---|
| right / False | 2% |
| **left / False** | **84%** |
| right / True | 84% (two errors cancel) |
| left / True | 2% |

So this operator flies with `--target-side left`. The default stays `right`.

## Threshold

`assoc_dist_thresh` is 0.7, from the same frames: 0.5 admits 85% of the
operator's gesturing hands, 0.7 admits 91%, and larger values gain ~1 point.
FORWARD is the hardest class (median distance 0.50). A bystander's hand measured
~4.2 on one photo; **the bystander margin has not been measured with two people.**

Regenerate: `python -m tello_gesture_py.scripts.association_calibration`
(writes `outputs/metrics/association_calibration.json`).

## Test it

```bash
python -m tello_gesture_py.scripts.webcam_demo --associate-hands --target-side left --run-id webcam_association
python -m tello_gesture_py.src.main --classifier svm --associate-hands --target-side left --run-id drone_association
```

The webcam overlay draws every landmark the decision reads: the face box and
Pose's eye/mouth points, all hands (accepted green, rejected red), and the
commanding arm with each joint's visibility; a joint failing the check is red,
and an off-frame one is pinned to the edge. The same overlay is in both the
webcam harness and the flight window: `v` toggles it, and `b` runs pose on every
frame instead of every 20 hand detections (~2.5 s at typical frame rates), at a
cost in frame rate. Start with them on via `--debug-overlay` / `--live-pose`;
toggles are logged as events in `perf_events.csv`. Each run logs `assoc_mode`, `assoc_ok`, `assoc_reason`, `assoc_d` per frame
in `perf.csv` and `decisions.csv`, and pose latency as `assoc_pose_ms`.
