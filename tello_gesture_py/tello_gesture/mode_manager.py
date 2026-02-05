from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# LLM is allowed to output ONLY these modes:
MODES = ("keyboard", "gesture", "face", "search_360", "hover", "land")


def _norm_mode(m: str) -> str:
    m = (m or "").strip().lower()
    aliases = {
        "follow": "face",
        "face_follow": "face",
        "gesture_control": "gesture",
        "search": "search_360",
        "spin": "search_360",
        "search_360": "search_360",
        "hover": "hover",
        "land": "land",
        "keyboard": "keyboard",
        "gesture": "gesture",
        "face": "face",
    }
    return aliases.get(m, m)


@dataclass
class LLMConfig:
    enabled: bool = True
    ollama_url: str = "http://127.0.0.1:11434/api/chat"
    model: str = "qwen2.5:0.5b-instruct"

    # Slower = smoother video + fewer timeouts
    decision_hz: float = 0.5  # call LLM every 2 seconds
    timeout_s: float = 3.5

    # Hard safety
    battery_land_pct: int = 15


class LLMModeChooser:
    """Ollama local JSON mode chooser. Returns (next_mode, reason)."""

    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()

    def choose(self, state: Dict[str, Any]) -> Tuple[str, str]:
        if not self.cfg.enabled:
            return _norm_mode(state.get("mode", "gesture")), "LLM disabled"

        # Hard rule: low battery -> land (LLM cannot override)
        bat = state.get("battery", None)
        try:
            if bat is not None and float(bat) <= float(self.cfg.battery_land_pct):
                return "land", f"Hard safety: battery <= {self.cfg.battery_land_pct}%"
        except Exception:
            pass

        system = (
            "You are a drone SAFETY mode selector.\n"
            "Return ONLY JSON with keys: next_mode, reason.\n"
            f"next_mode MUST be one of: {list(MODES)}.\n"
            "Be conservative: if uncertain choose 'hover'.\n"
            "\n"
            "PRIORITY:\n"
            "1) key (explicit user control)\n"
            "2) safety (battery)\n"
            "3) deterministic rules\n"
            "4) otherwise choose best safe mode\n"
            "\n"
            "Key mapping:\n"
            "- 'l' -> land\n"
            "- '1' -> keyboard\n"
            "- '2' -> gesture\n"
            "- '3' -> face\n"
            "\n"
            "Core policy:\n"
            "- If hand_detected is true, prefer next_mode='gesture' (unless key overrides).\n"
            "- Else if face_detected is true, prefer next_mode='face' (unless key overrides).\n"
            "- Only choose 'hover' if neither face_detected nor hand_detected.\n"
            "\n"
            "Deterministic rules:\n"
            "- If current mode is 'gesture' and time_since_hand_s >= 2.5: prefer 'face' if face_detected else 'hover'.\n"
            "- If current mode is 'face' and time_since_face_s >= 10: choose 'search_360'.\n"
        )

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
            ],
            "stream": False,
            "format": "json",
        }

        try:
            import requests

            r = requests.post(self.cfg.ollama_url, json=payload, timeout=self.cfg.timeout_s)
            r.raise_for_status()
            msg = r.json().get("message", {}).get("content", "")

            data = json.loads(msg) if isinstance(msg, str) else msg

            next_mode = _norm_mode(str(data.get("next_mode", "hover")))
            reason = str(data.get("reason", "")).strip() or "LLM: no reason"

            if next_mode not in MODES:
                return "hover", f"LLM invalid next_mode '{next_mode}' -> hover"

            return next_mode, reason

        except Exception as e:
            return "hover", f"LLM error -> hover ({type(e).__name__}: {e})"


@dataclass
class ModeManager:
    mode: str = "keyboard"
    llm: Optional[LLMModeChooser] = None
    _last_llm_ts: float = 0.0

    def set_mode(self, new_mode: str):
        nm = _norm_mode(new_mode)
        if nm not in MODES:
            return
        self.mode = nm

    def is_mode(self, name: str) -> bool:
        return self.mode == _norm_mode(name)

    def maybe_update_from_llm(self, state: Dict[str, Any]) -> Tuple[str, str]:
        """Returns (mode_after, reason). If no tick, returns (mode, '')."""
        if self.llm is None or not getattr(self.llm.cfg, "enabled", False):
            return self.mode, ""

        hz = max(float(self.llm.cfg.decision_hz), 0.1)
        min_dt = 1.0 / hz
        now = time.time()
        if (now - self._last_llm_ts) < min_dt:
            return self.mode, ""
        self._last_llm_ts = now

        st = dict(state)
        st["mode"] = self.mode

        next_mode, reason = self.llm.choose(st)

        next_mode = _norm_mode(next_mode)
        if next_mode in MODES and next_mode != self.mode:
            self.mode = next_mode

        return self.mode, reason
