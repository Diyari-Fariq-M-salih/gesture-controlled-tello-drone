# Where things live and how data flows

## Layout
| path | what |
|---|---|
| `tello_gesture_py/src/` | deployed stack: perception, identity gate, arbitration |
| `tello_gesture_py/scripts/` | pre-flight check, webcam demo, capture harnesses, analysis, flash-latency test, run catalogue |
| `tello_gesture_py/src/gestures/` | evaluation, audit, table and figure generators |
| `outputs/runs/<stamp>_<type>_<what>/` | one directory per launch, with manifest; contents never edited |
| `outputs/run_renames.csv` | old -> new directory names from the 2026-09-25 rename; manifests keep the name logged at the time |
| `outputs/metrics/`, `outputs/paper/` | evaluation results; generated long-paper tables |
| `outputs/sampling_density/` | the 100/250/400-per-class training runs |
| `outputs/legacy_2026-03/` | the original pipeline's outputs, kept for `src/utils/reporting.py` |
| `data/processed/session{A..D}{,_features}.csv` | per-session images list and landmark features; `data/sessions.json` their conditions; `data/raw/` frames (ignored) |
| `models/production/model.joblib` | the frozen SVM; `models/experiments/` earlier fits; `models/mediapipe/` the pose model for association |
| `models/third_party/arcface.onnx` | face embedding, not redistributed (see README) |
| `paper/` | `short.tex` (arXiv), `main.tex` (long), `tables/`, `figures/` |
| `paper/arxiv-v1/` | the exact source submitted to arXiv (2609.25511); frozen, never edited |
| `tello_gesture_py/requirements-lock.txt` | exact versions of the working `.venv` |

## The loop (`src/`)
```
video_stream.py, latest_frame.py  UDP 11111 H.264 -> single-slot buffer (seq, timestamp)
hand_gesture.py                   MediaPipe Hands -> 21 landmarks, every 2nd frame
gesture_classifier.py             select(): --classifier svm|rule, validated before arming
  gesture_logic.py                  RuleBasedGesture (index direction + area change)
                                    DepthStabilityGate (holds FORWARD/BACK until repeated)
  model_classifier.py               TrainedClassifier (RBF-SVM, frozen)
face_follow.py                    MediaPipe face, every 4th frame; crop freshness; P control
face_id.py                        ONNX MobileFaceNet embedding, cosine, Schmitt trigger
hand_association.py               opt-in: pose skeleton binds the hand to the verified face
association_overlay.py            debug drawing of hands, face, skeleton (keys v, b)
mode_manager.py                   FSM: failsafe > gesture > face > search > hover; LLMReasoner (log only)
controller.py                     the loop; RC out on UDP 8889; telemetry in on 8890
session_recorder.py               opt-in (--record): the window as session.mp4, real-time, seq-stamped
```
`config.py` holds every threshold and ablation switch; Table II is generated from it.
`run_context.py` writes the run directory and manifest. `perf_logger.py` logs a
stage's latency only on frames where it ran, deduplicated by sequence number.

## Short paper (arXiv) to source
| artefact | produced by |
|---|---|
| Table II | `build_design_table --short` from `config.py` |
| Tables I, III, IV | hand-written in `paper/tables/`; no generator; checked only where `audit_claims` registers a claim |
| Fig. 1 | `build_design_figures` (TikZ) |
| Fig. 3 | `fig_envelope_distance` from `stream_identity.csv` |
| Fig. 4 | `build_offline_eer_figure`; data external (I. Chaabeni) |
| §V-A | `per_session_eval`, `loso_eval`, `analyse_operational_gesture` |
| §V-C | `envelope_eval` |

Long-paper mapping: `docs/learnings/long-paper-sources.md`.

## Run directory
```
manifest.json        config, git SHA, library versions, host, harness, classifier
telemetry.csv        drone state             (absent => webcam or ground harness)
decisions.csv        mode, command, gesture, faceid score, bbox, per change
perf.csv             per-frame stage latency, with <stage>_ran columns
perf_events.csv      rc_send, llm_explanation, video_reopen, failsafe
scenarios.csv        manual trial outcomes
cued.csv             cued holds, one row per frame   (--cue-rounds)
session.mp4          the flight window             (--record; not committed)
flash_latency.csv    camera-to-host delay per flash (scripts/flash_latency)
op_gesture.csv       cued gesture capture     (operational_gesture_eval)
stream_identity.csv  cued distance capture    (stream_capture)
```
Identify a run by `manifest.harness`, not its tag. An empty directory is a launch
that never reached the drone.

Names: `<YYYYmmdd-HHMMSS>_<type>_<what>[_aborted]`. Type is `drone` (flight stack),
`webcam`, `capture` (ground harness) or `latency`; `nN` in `<what>` is the study
stage the run belongs to; `_aborted` marks a run the operator declared aborted,
which is kept and excluded. The timestamp prefix is what `src/gestures/evidence.py`
uses to fix the v1 evidence set; paper scripts select runs by exact name.

## Sessions
`data/sessions.json` gives illumination, background and standoff per session;
`loso_eval` groups transfer pairs by what changes. A–C were backfilled; D was
recorded at capture. `nominal_distance_m` is null for A–C.
