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
class LLMConfig:
    enabled: bool = True
    model: str = "qwen2.5:0.5b-instruct"
    url: str = "http://127.0.0.1:11434/api/chat"

    # Decision cadence
    decision_hz: float = 1.0
    timeout_s: float = 4.0

    # Safety
    battery_land_pct: int = 15

    # Mode stability
    mode_lock_s: float = 0.4  # shorter, more reactive

    # HARD CONSTRAINTS IN CODE (robotics-style):
    # If hand/face is detected, we force those modes immediately.
    enforce_perception_priority: bool = True


class LLMModeManager:
    """
    Non-blocking LLM controller with hard constraints:
      - tick(state) is cheap, never blocks
      - worker thread calls Ollama occasionally
      - OPTIONAL: hard force gesture/face when hand/face detected (recommended)
    """

    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()
        self.mode: str = "hover"
        self._last_reason: str = ""
        self._last_decision_ts: float = 0.0

        self._lock_until: float = 0.0

        self._state_lock = threading.Lock()
        self._latest_state: Optional[Dict[str, Any]] = None
        self._request_event = threading.Event()
        self._stop_event = threading.Event()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop_event.set()
        self._request_event.set()
        self._worker.join(timeout=1.0)

    def get(self) -> Tuple[str, str]:
        return self.mode, self._last_reason

    def _set_mode(self, new_mode: str, reason: str):
        nm = _norm(new_mode)
        if nm not in MODES:
            nm = "hover"
            reason = f"Invalid mode -> hover (got '{new_mode}')"
        self.mode = nm
        self._last_reason = reason
        self._lock_until = time.time() + float(self.cfg.mode_lock_s)

    def tick(self, state: Dict[str, Any]) -> Tuple[str, str]:
        """
        Main-thread tick. Applies hard constraints first (optional), then schedules LLM.
        Returns current (mode, reason) immediately.
        """
        if not self.cfg.enabled:
            return self.mode, self._last_reason

        now = time.time()

        # --- Hard safety: battery (instant, cannot be overridden) ---
        bat = state.get("battery", None)
        try:
            if bat is not None and float(bat) <= float(self.cfg.battery_land_pct):
                if self.mode != "land":
                    self._set_mode("land", f"Hard safety: battery {bat}% <= {self.cfg.battery_land_pct}%")
                return self.mode, self._last_reason
        except Exception:
            pass

        # --- Hard perception constraints (robotics constraint layer) ---
        if self.cfg.enforce_perception_priority:
            # These are hard, because your demo expects it.
            if bool(state.get("hand_detected", False)):
                if self.mode != "gesture":
                    self._set_mode("gesture", "Constraint: hand_detected -> gesture")
                return self.mode, self._last_reason

            if bool(state.get("face_detected", False)):
                if self.mode != "face":
                    self._set_mode("face", "Constraint: face_detected -> face")
                return self.mode, self._last_reason

        # Push latest state for worker (LLM decides only when nothing is detected)
        with self._state_lock:
            self._latest_state = dict(state)

        # Schedule LLM call
        min_dt = 1.0 / max(float(self.cfg.decision_hz), 0.05)
        if (now - self._last_decision_ts) >= min_dt:
            self._last_decision_ts = now
            self._request_event.set()

        return self.mode, self._last_reason

    def _worker_loop(self):
        while not self._stop_event.is_set():
            self._request_event.wait()
            self._request_event.clear()
            if self._stop_event.is_set():
                break

            with self._state_lock:
                st = dict(self._latest_state) if self._latest_state else None
            if not st:
                continue

            # Respect lock
            if time.time() < self._lock_until:
                continue

            mode, reason = self._call_llm(st)
            self._set_mode(mode, reason)

    def _call_llm(self, state: Dict[str, Any]) -> Tuple[str, str]:
        system = (
            "Return ONLY JSON: {\"next_mode\": str, \"reason\": str}\n"
            f"next_mode must be one of {list(MODES)}.\n\n"
            "Decision context:\n"
            "- hand_detected and face_detected may be false here.\n"
            "- Decide between hover vs search_360 vs land.\n\n"
            "Hard rules:\n"
            "- If battery <= 15: next_mode MUST be \"land\".\n"
            "- Else if time_since_any_seen_s >= 10: next_mode SHOULD be \"search_360\".\n"
            "- Else: next_mode SHOULD be \"hover\".\n"
            "Reason must be short.\n"
        )

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(state)},
            ],
            "stream": False,
            "format": "json",
        }

        try:
            import requests

            r = requests.post(self.cfg.url, json=payload, timeout=self.cfg.timeout_s)
            r.raise_for_status()
            msg = r.json().get("message", {}).get("content", "")
            data = json.loads(msg) if isinstance(msg, str) else msg

            nm = _norm(data.get("next_mode", "hover"))
            reason = str(data.get("reason", "")).strip() or "LLM: no reason"

            if nm not in MODES:
                return "hover", f"LLM invalid '{nm}' -> hover"
            return nm, reason

        except Exception as e:
            return self.mode, f"LLM error ({type(e).__name__})"
