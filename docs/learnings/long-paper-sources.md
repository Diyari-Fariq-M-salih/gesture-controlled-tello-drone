# Long paper (`paper/main.tex`): section to source

Moved from `docs/ARCHITECTURE.md` on 2026-09-23. Section letters are the
13-page archival version's, not the arXiv short paper's.

| section | produced by | reads |
|---|---|---|
| V-A scenarios | `build_paper_tables.py` | `scenarios.csv`, `decisions.csv` |
| V-B runtime | `build_paper_tables.py` | `perf.csv` |
| V-C cross-session | `per_session_eval.py`, `loso_eval.py` | `data/processed/*_features.csv` |
| V-D identity | `envelope_eval.py`, `fig_envelope_distance.py` | `stream_identity.csv` |
| V-E operational gesture | `analyse_operational_gesture.py` | `op_gesture.csv` |
| V-F stale crops | `build_paper_figures.py` | `drone_bench` vs `drone_bench-2` decisions |
| V-G hysteresis | `audit_claims.d_hysteresis_trial` | `webcam_d1-arbitration` scenarios |
| V-H depth, search, radio | `audit_claims`, `analyse_operational_gesture` | decisions, telemetry |
| V-J language model | `audit_claims.d_llm_denominators` | `decisions.csv`, `perf_events.csv` |
