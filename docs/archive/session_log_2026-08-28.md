# Session log — 28 August 2026

Day 1 of the arXiv sprint. Location: campus (eduroam). One battery.

**Headline:** three of eight scenarios closed with N=20 per arm, two real defects
found and fixed with before/after data, and the operating envelope measured. No
usable flight data — the 2.4 GHz link is not reliable enough to fly safely here.

128 scenario trials, 63,164 logged frames, 23 runs.

---

## 1. Environment was broken before anything else

The project had **no working Python environment**. Every conda env directory
(`tello_drone_mk1`, `tellogest`, `mp_gestures`, …) was an empty shell, and the
only interpreter present was 3.14, where mediapipe no longer ships the legacy
`mp.solutions` API that every perception wrapper here depends on.

Rebuilt as `.venv` on **CPython 3.12.3** with pinned versions, verified
`mp.solutions.hands` and `face_detection` both construct. `requirements.txt` now
pins exact versions and declares `onnxruntime` and `requests`, which the code
imports but never listed.

**Python 3.12 is a hard requirement.** 3.13+ has no legacy solutions API.

---

## 2. Defects found and fixed

### 2.1 Every run overwrote the previous one

All four log files were fixed paths opened with `"w"`. Four days of flights would
have left only the last battery. Now every launch writes
`outputs/runs/<timestamp>_<run-id>/` with a `manifest.json` recording config, git
SHA, library versions, host and ablations.

### 2.2 Stage latencies were diluted by skipped frames

`StageTimer.mark()` recorded a duration on every frame, including frames where a
throttled stage never ran. Reported latency was therefore the average of real
detections and near-zero no-ops.

`mark(stage, ran=...)` now records a sample only when the stage executed, and
emits a `<stage>_ran` column so the duty cycle is reported alongside.

Effect on the manuscript's numbers, measured on the Tello video path:

| Stage | Manuscript | Actual (conditional) |
|---|---:|---:|
| Hand detection | 12.6 ms | **24.4 ms** |
| Face detection | 0.47 ms | **6.2 ms** |

The 0.47 ms figure was ~13x understated and physically implausible for MediaPipe.
Both manuscript numbers were also measured on a webcam, not the drone.

### 2.3 Identity crops were taken from stale bounding boxes

`crop_face()` gated on `face_detected()`, which deliberately stays true for 2 s
to keep mode arbitration stable across stream gaps. Meanwhile real detection ran
only every ~12th frame, because the controller throttled `observe()` every 4
frames *and* `FaceFollower` had its own internal `detect_every_n = 3`. Between
detections, ArcFace was being handed a crop from a box the face had already left.

Symptom: a **bimodal** FaceID score — a genuine cluster at 0.770 (sd 0.109) and a
garbage cluster at 0.193 (sd 0.033), the latter being background embeddings.

Fixed by giving identity its own freshness window (`crop_max_age_s = 0.35`) and a
60 px minimum, and by removing the double throttle. Arbitration's 2 s hold is
untouched — it is correct for mode stability and only wrong for identity.

| | Before | After |
|---|---:|---:|
| Garbage cluster | 56% of frames | **20%** |
| Authorized at thr 0.55 | 41% | **68%** |
| `search_360` decisions | 130 | **4** |
| `face` mode decisions | 72 | **82** |

The `search_360` collapse is the clearest evidence: the drone had been repeatedly
concluding nobody was present.

### 2.4 No hysteresis on the authorization decision

The mode manager has careful hysteresis. The identity gate did not — it was a
bare `score >= threshold` re-evaluated per frame. A score oscillating around 0.55
made `face_auth` chatter, starving the arbitration layer of the stable input it
was designed around.

Found by investigating the single failed trial in scenario 4 (see §3).

Fixed with a Schmitt trigger: enter at `cosine_thr` (0.55), release only below
`release_thr` (0.45) and only after `release_frames` (3) consecutive frames.
Reversible with `--no-auth-hysteresis` on both harnesses.

---

## 3. Results

### 3.1 Scenario trials — 3 of 8 closed, both arms

Webcam harness. Same perception, gating and mode manager as the drone; only
actuation is absent. These three scenarios make claims purely about *which mode
was selected*, so they are complete, not partial.

| Scenario | Bare threshold | Hysteresis |
|---|---:|---:|
| unauthorized gesture rejected | 20/20 | **20/20** |
| gesture preempts face | 19/20 | **20/20** |
| hysteresis prevents flicker | 20/20 | **21/21** |

Wilson 95% CI on 20/20 is [84%, 100%].

### 3.2 The hysteresis effect, and its cost

Frame-level, inside scenario-4 trial windows:

| | Bare | Hysteresis |
|---|---:|---:|
| Authorized frames | 75.3% | **91.5%** |
| Authorization flips | 14 | **4** |

Replaying the exact score sequence from the one failed trial through both gates:
authorized frames **5/15 → 13/15**, longest run **4 → 7**. The hand was detected
from t+2.08 s but `hand_auth` stayed false; the bare gate only recovered at
t+4.20 s. With hysteresis it would have preempted at ~2.1 s, matching the 2.00 s
median of the 19 successful trials.

**Cost, measured not assumed.** In-trial impostor authorization rose
**0.0% → 5.6%**, one frame reaching 0.547. All 20 trials still rejected. The
latch clears in a median of 3 frames ≈ 1.5 s; tightening `release_thr` does not
speed this up because the binding constraint is the consecutive-frame count.

**Do not over-claim.** 19/20 → 20/20 is a single trial (Fisher p = 1.0). The real
evidence is the frame-level stability change and the replay. N=20 per arm cannot
detect a small increase in false accepts; the 5.6% frame-level leak is the honest
signal there.

### 3.3 Operating envelope

Authorization against face size, self-enrolled, drone camera:

| Face height | ≈ Distance | Authorized |
|---|---|---:|
| ≤150 px | ≥3.0 m | 12% |
| 150–180 px | 2.5–3.0 m | 27% |
| 180–280 px | 1.6–2.4 m | 75–100% |
| 280–480 px | 0.9–1.6 m | 88–100% |
| >480 px | <0.9 m | **0%** |

**Usable range ≈ 0.9–2.4 m.** Two walls, both real. The far wall is score
degradation with resolution, *not* the 60 px crop floor — the smallest face
observed was 130 px, so that floor never engaged. The near wall is the padded
crop clipping at the frame edge: getting closer makes it worse.

Face-follow targets `target_area_frac = 0.075` ≈ a 263 px face ≈ 1.7 m standoff,
which sits mid-band. The two subsystems agree by accident.

> **Correction, 2026-09-21. Both numbers in the line above are wrong.** 263 px is
> `sqrt(0.075 × 1280 × 720)`, but the Tello EDU streams **960×720**, which gives a
> 228 px box. The px→m step then used the 0.9–2.4 m envelope from this session,
> which the denser sweep of 2026-09-20 superseded with 0.5–1.5 m. Calibrated
> against that sweep — median `sqrt(area_frac)·d` = 0.2579 m over the 4,861 frames
> at marks inside the envelope — the target is **0.94 m**, with a 0.90–1.00 m
> holding band from the ±0.008 deadband. The conclusion survives in substance (the
> follower does drive to a standoff the gate admits) but not as stated: §V-F of the
> paper had carried 1.7 m forward and inverted it into "sits on the edge of that
> cliff", which is now corrected. Registered in `audit_claims.d_follow_standoff`
> so it cannot drift again. The original line is left as written.

Caveat: two buckets have n=2. The shape is unambiguous, the edges are provisional.

### 3.4 Camera domain gap — a result worth reporting

Same code, same operator, same threshold, comparable face size (~180 px):

| Condition | Authorized | Garbage cluster |
|---|---:|---:|
| Webcam | **98.9%** | 0.0% |
| Tello camera | 68% | 20% |

With the offline still-image evaluation reporting 0.32% EER, that is a
three-point degradation curve: **0.32% EER → 99% webcam → 68% drone.** Offline
face-verification benchmarks substantially overstate operational performance
through a compressed drone video pipeline. Nobody in the related work reports
this because nobody measures it.

Caveat: lighting and distance were not controlled between conditions. Indicative,
not a clean ablation. A matched-distance side-by-side would cost one battery.

### 3.5 LLM explanations — early negative result

490 explanations logged. 7% hard errors. Among the rest, with battery at 46%:

> "Battery warning: battery <= 15"
> "the drone's battery level is below the minimum requirement of 15 for hovering"
> "The drone's camera is on its main frame, indicating no need for battery management"

The system prompt explicitly forbids inventing battery warnings below 15. It
invented them anyway. Latency mean 1766 ms, p95 4002 ms — hitting the 4 s
timeout. The most common output was a verbatim echo of the deterministic reason
string.

Points clearly toward the template baseline winning on correctness and latency.

---

## 4. The blocker: RF

Flight was attempted and abandoned.

| Run | Frames | fps | Longest gap | Reopens |
|---|---:|---:|---:|---:|
| Ground stream test | 3130 | **21.8** | 0.78 s | **0** |
| In flight | 1379 | **8.4** | 18.2 s | **48** |

The ground test was clean. In flight, on the same link in the same room, the
stream collapsed — and worse, **the command channel timed out while airborne**:

```
takeoff: True ok
emergency: False timeout    <- emergency stop did not reach the drone
takeoff:   False timeout
emergency: True ok          <- succeeded only on retry
```

Land and emergency are the only ways to stop the aircraft. Both were unreliable.
Flying was stopped on safety grounds, not data grounds.

Cause is almost certainly 2.4 GHz saturation from campus APs, compounded in
flight by motor EMI, higher power draw and a moving antenna. The Tello cannot
select a channel.

**Also note:** the 2 s stall-timeout fix, which worked perfectly on the ground,
made things worse in the air — 48 reopens in 283 s, each restarting the decoder
and waiting for the next SPS/PPS keyframe (the `non-existing PPS` / `no frame!`
storm). Needs a higher threshold and a cooldown before the next flight attempt.

**Next attempt: a different RF environment (home, no eduroam).** The ground test
proves the pipeline is sound when the link is.

---

## 5. State

**Done**

- Environment rebuilt and pinned; `docs/flight_protocol.md` written
- Run-scoped outputs with provenance manifests on both harnesses
- `tello_check` pre-flight tool (SSID, route, SDK, battery, telemetry)
- Scenarios 2, 4, 5 closed, both hysteresis arms, N=20 each
- Operating envelope, camera domain gap, LLM negative result
- Four defects fixed, three with before/after data

**Open**

- Scenarios 1, 3, 6, 7 in the air (N=20 each) — blocked on RF
- Scenario 8 via `--battery-land-pct` parameterisation
- Gesture sessions B and C (webcam, unblocked, highest value remaining)
- Spoof / hand-binding test (webcam, unblocked)
- Face verification through the drone camera
- Analysis scripts: cross-run aggregation with Wilson CIs, ablation comparison,
  LLM consistency, cross-session runner
- **No face-verification evaluation script exists in the repo** — the 0.32% EER
  cannot be reproduced from the artifact

**Manuscript, not yet started**

- Revert the July 8 Limitations edits (claims flight trials that did not happen)
- Port the newer `.docx` content into `main.tex`, un-anonymise for arXiv
- Replace the latency paragraph with measured numbers

---

## 6. For the paper

Things learned today that belong in it:

1. Stage-conditioned latency with duty cycles, not diluted means.
2. Identity needs a tighter freshness window than arbitration — one hold time
   does not serve both.
3. Authorization needs hysteresis of its own; mode-level hysteresis protects a
   signal already corrupted upstream.
4. The offline→webcam→drone degradation curve.
5. Usable standoff range as a measured envelope, with both near and far walls.
6. RF reliability as a real deployment constraint: 21.8 fps on the ground,
   8.4 fps and command timeouts in flight, same room.
