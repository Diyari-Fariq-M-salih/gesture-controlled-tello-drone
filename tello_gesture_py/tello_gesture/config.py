from dataclasses import dataclass

@dataclass
class ControllerConfig:
    # timing
    ui_dt: float = 1.0 / 30.0      # ~30 Hz UI loop
    rc_dt: float = 1.0 / 10.0      # ~10 Hz RC send loop

    # RC control
    rc_speed: int = 30
    rc_deadband: int = 6
    rc_limit: int = 100

    # gesture thresholds
    dir_thr: float = 0.10          # index vector threshold (normalized)
    scale_thr: float = 0.18        # forward/back by hand scale change
    gesture_hold_s: float = 0.35   # keep last gesture for a moment if hand disappears
    ema_alpha: float = 0.35        # landmark smoothing

    # telemetry logging
    log_hz: float = 1.0
    log_path: str = "telemetry_log.csv"

    # tello
    tello_ip: str = "192.168.10.1"
    cmd_port: int = 8889
    state_port: int = 8890
    video_port: int = 11111
    local_cmd_port: int = 9012
