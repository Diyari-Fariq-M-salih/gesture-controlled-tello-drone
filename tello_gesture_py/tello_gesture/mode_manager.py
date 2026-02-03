from dataclasses import dataclass

MODES = ("keyboard", "gesture", "face")

@dataclass
class ModeManager:
    mode: str = "keyboard"

    def set_mode(self, new_mode: str):
        if new_mode not in MODES:
            return
        self.mode = new_mode

    def is_mode(self, name: str) -> bool:
        return self.mode == name
