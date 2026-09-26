<div align="center">

# A Deployment Study of Identity-Gated Drone Gesture Control

**Diyari Mohammed Salih**<sup>1</sup> · **Ilyes Chaabeni**<sup>1</sup> · **Naïma Aït Oufroukh**<sup>2</sup>

<sup>1</sup>M2 Smart Aerospace and Autonomous Systems, Université Paris-Saclay &nbsp;·&nbsp;
<sup>2</sup>IBISC Laboratory, Université Paris-Saclay

[![arXiv](https://img.shields.io/badge/arXiv-2609.25511-b31b1b.svg)](https://arxiv.org/abs/2609.25511)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg)](docs/learnings/environment.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<img src="docs/assets/face_follow.gif" width="48%" alt="Face following"> <img src="docs/assets/search_mode.gif" width="48%" alt="Search mode">

</div>

**IGate** is an identity-gated control stack for a DJI Tello EDU: gesture commands
and face tracking are admitted only while an enrolled operator is verified. The
paper's contribution is the evaluation: every component is scored both by the
offline metric conventionally reported for it and by the rate it achieves through
the drone's own video pipeline, in flight.

## Results

| | offline | through the drone's video |
|---|---|---|
| Face verification | 0.32% equal error rate | 19.3% false rejection in flight (7.3% at the offline threshold) |
| Gesture classification (RBF-SVM) | 0.997 within-session, 0.910 held-out session | 0.783 on the ground, 0.850 in flight |

- 270 logged trials, 149 of them flown. Operator authorization is usable between
  0.5 and 1.5 m.
- In flight, under one hover-locked protocol, the RBF-SVM reaches 0.850 against
  0.651 for the geometric rule; 82% of that gap is on the depth channel.
- Six behaviours appeared only in flight; three were corrected and are reported
  with paired measurements.

Every number above is rederived from the logs in this repository by
`audit_claims.py` (see [Reproducing the paper](#reproducing-the-paper)), except
the offline 0.32%, which comes from a separate still-image benchmark.

## How it works

```
Tello video (UDP) ──> hands, face ──> identity gate ──> arbitration ──> RC commands (UDP)
                         │              (face embedding,     (failsafe > gesture >
                         │               cosine, hysteresis)  face follow > search > hover)
                         └──> gesture classifier (RBF-SVM or geometric rule)
```

Perception, identity and arbitration all run on the host; the drone only streams
video and receives RC commands. A local language model writes one-sentence
explanations to the log but cannot reach the command path. Full map:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

```bash
py -3.12 -m venv .venv            # CPython 3.12 is required by mediapipe
.venv\Scripts\activate
pip install -r tello_gesture_py/requirements.txt
```

`requirements-lock.txt` pins the exact versions the paper's runs used.

Two model files are needed that this repository does not redistribute:

- **Face embedding**: InsightFace's `w600k_mbf.onnx` (MobileFaceNet, from the
  `buffalo_s` model pack) saved as `models/third_party/arcface.onnx`. InsightFace's
  pretrained models are licensed for non-commercial research use.
- **Language model (optional)**: [Ollama](https://ollama.com) with
  `qwen2.5:0.5b-instruct`; see [`docs/learnings/llm-reasoner.md`](docs/learnings/llm-reasoner.md).
  Run with `--no-llm` to skip it.

The gesture SVM (`models/production/model.joblib`) and the pose model used by
hand-face association (`models/mediapipe/`) are included.

## Usage

Join the drone's `TELLO-XXXXXX` Wi-Fi, then:

```bash
python -m tello_gesture_py.scripts.tello_check                       # link and battery
python -m tello_gesture_py.src.main --classifier svm --run-id flight    # or --classifier rule
```

`--classifier` is required. Press `p` facing the camera to enrol the operator.

| key | action | key | action |
|---|---|---|---|
| `t` / `l` | take off / land | `e` | **emergency stop** |
| `p` / `o` | enrol / clear the operator | `q` | quit (lands first) |
| `w a s d` `r f` `j k` | manual override while flying | `h` | key help on screen |
| `v` | landmark overlay | `i` | show / hide the LLM text |

No drone? `python -m tello_gesture_py.scripts.webcam_demo` runs the same perception,
gating and arbitration on a webcam.

**Hand-face association** (opt-in, beyond the paper): accepts only a hand at the
end of the verified operator's own arm, so a bystander's hand cannot command.
Add `--associate-hands --target-side left|right`; see
[`docs/learnings/hand-association.md`](docs/learnings/hand-association.md).

Every launch writes `outputs/runs/<timestamp>_<tag>/`: a manifest (configuration,
Git commit, model hash) and per-frame logs of telemetry, decisions and latency.
Run names read `<time>_<type>_<what>`, for example `20260921-002538_drone_n6-hover-cued-svm`.
The development history before 2026-09-26 is archived separately, so the commit
hash in older manifests (`git_sha`) does not resolve in this repository.
`--record` also saves the flight window as `session.mp4` in real time, each frame
stamped with the stream sequence number that `perf.csv` logs (`--record-raw` adds
the clean camera frames). Videos show faces and are not committed.

## Reproducing the paper

```bash
python -m tello_gesture_py.src.gestures.audit_claims      # rederives every number; exits 0 only if all reproduce
```

The run logs behind the paper (through 2026-09-22), the per-session landmark
features and the frozen classifier are all in this repository. The raw capture
images are not, since they show the operator's face. Commands to regenerate every
table and figure and to build the paper are in
[`docs/QUICK_START.md`](docs/QUICK_START.md).

The paper's LaTeX source is in [`paper/`](paper/): `short.tex` is the arXiv paper
and `main.tex` a longer archival version. [`paper/arxiv-v1/`](paper/arxiv-v1/) is
the exact source submitted to arXiv.

## Repository layout

```
tello_gesture_py/   src/ the flight stack · scripts/ capture and analysis tools · tests/
data/               per-session landmark features (sessions A-D) and capture conditions
models/             the frozen gesture SVM, earlier fits, the pose model
outputs/            runs/ per-launch logs · metrics/ evaluation results
paper/              LaTeX source, generated tables and figures, arxiv-v1/
docs/               protocols, runbooks, learnings; start at docs/INDEX.md
```

## Citation

```bibtex
@article{mohammedsalih2026igate,
  title   = {A Deployment Study of Identity-Gated Drone Gesture Control},
  author  = {Mohammed Salih, Diyari and Chaabeni, Ilyes and A{\"\i}t Oufroukh, Na{\"\i}ma},
  journal = {arXiv preprint arXiv:2609.25511},
  year    = {2026}
}
```

## Acknowledgements

We thank the M2 Smart Aerospace and Autonomous Systems program at Université
Paris-Saclay.

## License

Code is released under the [MIT License](LICENSE). Third-party models keep their
own licenses: MediaPipe (Apache-2.0) and InsightFace (non-commercial research).
