# Python environment

The previous Python environment on this machine was gone (empty conda env
folders). A pinned one now lives in `.venv`, built on **CPython 3.12**.

```bash
.venv\Scripts\activate
```

Python 3.12 is not optional: mediapipe's legacy `mp.solutions` API — used by
`hand_gesture.py`, `face_follow.py` and every dataset script — is absent from
the mediapipe builds published for 3.13+. To rebuild from scratch:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r tello_gesture_py/requirements.txt
```
