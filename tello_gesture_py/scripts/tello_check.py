"""Fast pre-flight check: is the Tello reachable, and what is the battery?

Standard library only -- no OpenCV, no MediaPipe -- so it starts instantly and
cannot be confused by a perception-stack problem. Run it before every flight,
and any time the controller reports a connection error.

    python -m tello_gesture_py.scripts.tello_check
    python -m tello_gesture_py.scripts.tello_check --watch    # keep printing telemetry

Diagnoses the three failures that actually happen:

  WinError 10051 / no route  -> the PC is not associated with the Tello's
                                access point. Windows silently leaves an AP
                                that has no internet, so this recurs mid-session
                                unless the adapter is pinned to TELLO-XXXXXX.
  timeout on "command"       -> associated, but the drone is not answering:
                                still booting, already in another app's session,
                                or the battery is too flat to bring up the radio.
  low battery                -> the Tello refuses takeoff under roughly 10% and
                                drops the link entirely when nearly empty.
"""

import argparse
import platform
import select
import socket
import subprocess
import time

TELLO_IP = "192.168.10.1"
CMD_PORT = 8889
STATE_PORT = 8890
# MUST match ControllerConfig.local_cmd_port. The Tello sends SDK replies to
# the client endpoint it latched onto, not to the source port of the packet
# it is answering -- so if the checker runs from a different port, the
# controller's handshake is received but its reply goes to the checker's
# port and times out. The two tools therefore share a port and must not run
# at the same time.
LOCAL_PORT = 9013


def local_route_ok() -> bool:
    """True if this host currently has an address on the Tello's subnet."""
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.settimeout(0.5)
        probe.connect((TELLO_IP, CMD_PORT))
        addr = probe.getsockname()[0]
        probe.close()
        return addr.startswith("192.168.10.")
    except Exception:
        return False


def current_ssid() -> str:
    if platform.system() != "Windows":
        return "?"
    try:
        out = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=6,
        ).stdout
        for line in out.splitlines():
            s = line.strip()
            if s.lower().startswith("ssid") and not s.lower().startswith("bssid"):
                return s.split(":", 1)[1].strip() or "(none)"
    except Exception:
        pass
    return "?"


def ask(sock: socket.socket, msg: str, timeout_s: float = 5.0):
    try:
        sock.sendto(msg.encode(), (TELLO_IP, CMD_PORT))
    except OSError as e:
        return None, f"send failed: {e}"
    r, _, _ = select.select([sock], [], [], timeout_s)
    if not r:
        return None, "timeout"
    try:
        data, _ = sock.recvfrom(2048)
        return data.decode(errors="ignore").strip(), None
    except Exception as e:
        return None, str(e)


def read_state(seconds: float = 2.0) -> dict:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("", STATE_PORT))
    except OSError:
        s.close()
        return {"_error": f"UDP {STATE_PORT} busy (controller still running?)"}
    s.settimeout(1.0)
    latest, end = {}, time.time() + seconds
    while time.time() < end:
        try:
            data, _ = s.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            break
        for kv in data.decode(errors="ignore").split(";"):
            if ":" in kv:
                k, v = kv.split(":", 1)
                try:
                    latest[k] = float(v)
                except ValueError:
                    pass
    s.close()
    return latest


def verdict(bat) -> str:
    if bat is None:
        return ""
    if bat >= 60:
        return "  -> full circuit flight"
    if bat >= 35:
        return "  -> short flight, expect ~2-3 laps"
    if bat >= 20:
        return "  -> failsafe test only, do not start a trial run"
    return "  -> CHARGE. Too low to fly; the link drops when nearly empty."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="keep printing telemetry until Ctrl+C")
    args = ap.parse_args()

    print(f"Wi-Fi SSID : {current_ssid()}")

    if not local_route_ok():
        print("\nNO ROUTE to 192.168.10.1 -- this PC is not on the Tello's network.")
        print("  1. Power the drone on and wait for the light to blink.")
        print("  2. Join Wi-Fi TELLO-XXXXXX.")
        print("  3. Windows leaves APs with no internet on its own. In")
        print("     Wi-Fi properties for TELLO-XXXXXX, disable 'Connect")
        print("     automatically when this network is in range' for your home")
        print("     network, or turn the other adapters off for the session.")
        return 2

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", LOCAL_PORT))
    sock.setblocking(False)

    resp, err = ask(sock, "command", 6.0)
    if resp is None:
        print(f"\nSDK handshake FAILED ({err}).")
        print("  Route exists but the drone is not answering on 8889.")
        print("  - Close the Tello mobile app; it holds an exclusive session.")
        print("  - Power-cycle the drone and retry after the light blinks.")
        print("  - A nearly flat battery brings up the radio but not the SDK.")
        sock.close()
        return 3
    print(f"SDK        : {resp}")

    for q, label in (("battery?", "Battery"), ("sdk?", "SDK ver"), ("wifi?", "Wi-Fi SNR")):
        val, err = ask(sock, q, 4.0)
        print(f"{label:<11}: {val if val is not None else f'({err})'}"
              + (verdict(float(val)) if label == "Battery" and val and val.isdigit() else ""))
    sock.close()

    st = read_state(2.0)
    if "_error" in st:
        print(f"Telemetry  : {st['_error']}")
    elif st:
        print(f"Telemetry  : bat={st.get('bat')}  height={st.get('h')}cm  "
              f"tof={st.get('tof')}cm  temp={st.get('templ')}-{st.get('temph')}C")
    else:
        print(f"Telemetry  : nothing on UDP {STATE_PORT} (firewall?)")

    if args.watch:
        print("\nwatching -- Ctrl+C to stop")
        try:
            while True:
                s = read_state(1.0)
                print(f"  bat={s.get('bat')}  h={s.get('h')}  tof={s.get('tof')}", flush=True)
        except KeyboardInterrupt:
            print("\nstopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
