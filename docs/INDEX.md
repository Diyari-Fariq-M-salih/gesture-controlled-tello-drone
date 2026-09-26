# Docs index

One line per document. Read only what the task needs.

## Start here
- [QUICK_START.md](QUICK_START.md) — every command: fly, capture, analyse, regenerate tables and figures, build the paper
- [ARCHITECTURE.md](ARCHITECTURE.md) — where things live, the control loop, which script produces each paper result
- [COMMON_MISTAKES.md](COMMON_MISTAKES.md) — the top five pitfalls that already cost real time

## Operating the drone
- [flight_protocol.md](flight_protocol.md) — flight-day checklist: every flight, keys, the four run types, bench check, after each battery
- [RUNBOOK.md](RUNBOOK.md) — operator runbook for the three ground captures (cued gesture, authorization envelope, session D), props off
- [KEYS.md](KEYS.md) — keystroke sequence for those three captures; referenced from `webcam_demo.py` and `controller.py`
- [capture_protocol_v2.md](capture_protocol_v2.md) — design of the same three captures: what each measures and why

## Study record
- [PREREGISTRATION_svm_inflight.md](PREREGISTRATION_svm_inflight.md) — preregistered in-flight SVM-vs-rule comparison and frozen-model SHA-256. **Never edit; add dated amendments only.**

## Learnings (one topic each)
- [learnings/environment.md](learnings/environment.md) — why CPython 3.12 is required (mediapipe `mp.solutions`), rebuilding `.venv`
- [learnings/llm-reasoner.md](learnings/llm-reasoner.md) — the reason-only local LLM: why it cannot actuate, install, usage
- [learnings/tello-platform.md](learnings/tello-platform.md) — Tello EDU (TLW004) hardware and SDK notes
- [learnings/gesture-classifiers.md](learnings/gesture-classifiers.md) — the rule and the SVM read different vocabularies; FORWARD/BACK as motion vs pose
- [learnings/analysis.md](learnings/analysis.md) — small-n correlations, the hold as the unit of analysis, excluding operator error
- [learnings/hand-association.md](learnings/hand-association.md) — binding the commanding hand to the verified operator: mirroring, which arm, threshold calibration
- [learnings/latency.md](learnings/latency.md) — what the latency columns measure, and the camera-to-host delay they leave out
- [learnings/long-paper-sources.md](learnings/long-paper-sources.md) — which script produces each section of the 13-page `paper/main.tex`

## Reference
- [reports/state-of-the-art.pdf](reports/state-of-the-art.pdf) — literature survey
- [assets/](assets/) — README GIFs, HUD and data-collection screenshots, hand landmarks, hardware photo, poster

## Finished work (not needed day to day)
- [archive/](archive/) — finished and superseded docs: the merge plan, the external-review response, the behaviour-count reconciliation, the in-flight measurement status, the 2026-08-28 session log, the old command list, old package layout, older LLM write-up, the original dataset how-to, ARO journal review notes, the reverted threshold-tuning patch (2026-09-23), and [archive/legacy-code/](archive/legacy-code/) (the pre-refactor gestures scripts and an unused utils module)
