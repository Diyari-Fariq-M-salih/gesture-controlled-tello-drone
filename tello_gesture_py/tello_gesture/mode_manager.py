from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

MODES = ("keyboard", "gesture", "face", "search_360", "hover", "land")


def _norm(m: str) -> str:
    return (m or "").strip().lower()


@dataclass
class DeterministicConfig:
    battery_land_pct: int = 15

    nohuman_search_s: float = 10.0
    search_duration_s: float = 5.0
    search_cooldown_s: float = 10.0

    mode_hold_s: float = 1.2
    hand_release_s: float = 0.8
    face_release_s: float = 0.8


class DeterministicModeManager:
    """
    Inputs are expected to be identity-gated upstream:
      hand_detected: TRUE only when authorized face is present AND hand is present (your gating)
      face_detected: TRUE only when authorized face is present (your FaceID lock)

    Behavior:
      - battery low -> land
      - gesture > face (with hysteresis)
      - if no human for >= nohuman_search_s -> search_360
      - search_360 exits immediately if hand/face reappears
      - else hover
    """

    def __init__(self, cfg: Optional[DeterministicConfig] = None):
        self.cfg = cfg or DeterministicConfig()
        self.mode: str = "hover"
        self.reason: str = ""

        self._mode_enter_ts: float = time.time()
        self._search_enter_ts: float = 0.0
        self._next_search_allowed_ts: float = 0.0

    def _set_mode(self, mode: str, reason: str):
        m = _norm(mode)
        if m not in MODES:
            m = "hover"
            reason = f"Invalid mode -> hover (got '{mode}')"
        if m != self.mode:
            self._mode_enter_ts = time.time()
        self.mode = m
        self.reason = reason

        if m == "search_360":
            self._search_enter_ts = self._mode_enter_ts

    def tick(self, state: Dict[str, Any]) -> Tuple[str, str]:
        now = time.time()

        hand = bool(state.get("hand_detected", False))
        face = bool(state.get("face_detected", False))

        t_hand = float(state.get("time_since_hand_s", 999.0))
        t_face = float(state.get("time_since_face_s", 999.0))
        t_any = float(state.get("time_since_any_seen_s", 999.0))

        # Battery safety
        bat = state.get("battery", None)
        try:
            if bat is not None and float(bat) <= float(self.cfg.battery_land_pct):
                self._set_mode("land", f"Safety: battery {bat}% <= {self.cfg.battery_land_pct}% -> land")
                return self.mode, self.reason
        except Exception:
            pass

        # Search_360: exit early if reacquired
        if self.mode == "search_360":
            if hand:
                self._set_mode("gesture", "Search: hand_detected -> gesture (exit search)")
                return self.mode, self.reason
            if face:
                self._set_mode("face", "Search: face_detected -> face (exit search)")
                return self.mode, self.reason

            if (now - self._search_enter_ts) >= float(self.cfg.search_duration_s):
                self._set_mode("hover", f"Autonomy: search done ({self.cfg.search_duration_s:.0f}s) -> hover")
                self._next_search_allowed_ts = now + float(self.cfg.search_cooldown_s)
            return self.mode, self.reason

        time_in_mode = now - self._mode_enter_ts

        # Hold gesture mode unless hand truly gone
        if self.mode == "gesture":
            if time_in_mode < self.cfg.mode_hold_s:
                return self.mode, self.reason
            if t_hand <= self.cfg.hand_release_s:
                return self.mode, self.reason

        # Hold face mode unless face truly gone; gesture preempts face immediately
        if self.mode == "face":
            if hand:
                self._set_mode("gesture", "Perception: hand_detected -> gesture (preempts face)")
                return self.mode, self.reason

            if time_in_mode < self.cfg.mode_hold_s:
                return self.mode, self.reason
            if t_face <= self.cfg.face_release_s:
                return self.mode, self.reason

        # Priority: gesture > face
        if hand:
            self._set_mode("gesture", "Perception: hand_detected -> gesture (priority)")
            return self.mode, self.reason

        if face:
            self._set_mode("face", "Perception: face_detected -> face")
            return self.mode, self.reason

        # No human -> search trigger
        if t_any >= float(self.cfg.nohuman_search_s) and now >= self._next_search_allowed_ts:
            self._set_mode("search_360", f"Autonomy: no human for {t_any:.1f}s -> search_360")
            return self.mode, self.reason

        self._set_mode("hover", "Autonomy: no intent -> hover")
        return self.mode, self.reason


# -------------------------
# LLM Reasoner (Reason Only)
# -------------------------

@dataclass
class LLMReasonConfig:
    enabled: bool = True
    model: str = "qwen2.5:0.5b-instruct"
    url: str = "http://127.0.0.1:11434/api/chat"
    decision_hz: float = 1.0
    timeout_s: float = 4.0


class LLMReasoner:
    def __init__(self, cfg: Optional[LLMReasonConfig] = None):
        self.cfg = cfg or LLMReasonConfig()
        self._last_reason: str = ""
        self._last_tick_ts: float = 0.0

        self._lock = threading.Lock()
        self._latest_payload: Optional[Dict[str, Any]] = None

        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop_event.set()
        self._request_event.set()
        self._worker.join(timeout=1.0)

    def get_reason(self) -> str:
        return self._last_reason

    def tick(self, payload: Dict[str, Any]) -> str:
        if not self.cfg.enabled:
            self._last_reason = ""
            return self._last_reason

        now = time.time()
        min_dt = 1.0 / max(float(self.cfg.decision_hz), 0.05)
        if (now - self._last_tick_ts) < min_dt:
            return self._last_reason

        self._last_tick_ts = now
        with self._lock:
            self._latest_payload = dict(payload)
        self._request_event.set()
        return self._last_reason

    def _worker_loop(self):
        while not self._stop_event.is_set():
            self._request_event.wait()
            self._request_event.clear()
            if self._stop_event.is_set():
                break

            with self._lock:
                st = dict(self._latest_payload) if self._latest_payload else None
            if not st:
                continue

            self._last_reason = self._call_llm(st)

    def _call_llm(self, payload: Dict[str, Any]) -> str:
        system = (
            "You are generating a SHORT explanation for a drone action.\n"
            "You do NOT control the drone.\n"
            "Return ONLY JSON: {\"reason\": \"...\"}\n"
            "The reason must be 1 short sentence and must match the given command/mode.\n"
            "Do not invent battery warnings unless battery <= 15.\n"
        )

        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)},
            ],
            "stream": False,
            "format": "json",
        }

        try:
            import requests

            r = requests.post(self.cfg.url, json=body, timeout=self.cfg.timeout_s)
            r.raise_for_status()
            msg = r.json().get("message", {}).get("content", "")
            data = json.loads(msg) if isinstance(msg, str) else msg
            reason = str(data.get("reason", "")).strip()
            return reason or "LLM: (no reason)"
        except Exception as e:
            return f"LLM error ({type(e).__name__})"
