# Latency: what the logs measure and what they leave out

## The logged columns are host time
| column | from | to |
|---|---|---|
| `frame_age_ms` | the decoder thread hands a frame to the single-slot buffer | the control loop takes it |
| `camera_to_command_ms` | the loop takes the frame | the RC command string is built, before the UDP send |
| `<stage>_ms` | start of one stage | its end; logged only when `<stage>_ran` is 1 |

The two intervals are disjoint (`controller.py` starts the stage timer after frame
age is taken), so host time is their sum. The LLM call, HUD drawing, display and
recording are outside both.

## What no log can show
Exposure, H.264 encoding on the aircraft, the Wi-Fi hop, FFmpeg buffering and
decode happen before a frame reaches the host, and the RC response after the
command leaves it. The stream carries no capture time and the aircraft's clock is
not the host's, so this part cannot be recovered from the logs.

## Measuring the camera path: `scripts/flash_latency.py`
The drone sits on the ground, props off, facing the laptop screen. The screen
switches black/white at random intervals; the switch and the first frame showing
it are both timed on the host clock. The deployed perception load runs on every
frame, and the video URL is the controller's, so decoding competes for the CPU as
it does in flight. `--selftest <ms>` replaces the drone with a synthetic camera
of known delay; at 120 ms it reads 137-140 ms, the delay plus half a frame
interval, as it should.

`20260924-182345_latency_flash`: 60/60 flashes seen, camera-to-host median
**604 ms**, p95 650 ms, range 558-664 ms, no drift over the run, 22.2 fps. The
host adds ~4 ms of frame age and ~19-25 ms of processing, so camera to command
is about 0.63 s, of which the host is a few percent.

Limits: the figure includes the screen's own response (about one refresh), so it
is an upper bound on the camera path; it was measured on the ground, so in-flight
Wi-Fi may add to it.

## Measuring command-to-motion: `scripts/yaw_latency.py`
While hovering, short alternating yaw pulses are sent and the first sign of the
turn is timed twice: in the returning video (global image shift by phase
correlation; the whole loop, command out and video back) and in the heading of
the state packets (command-to-motion plus the state link, whole degrees at
~10 Hz). Loop minus the flash-test camera path gives command-to-motion; both
include the same frame quantisation, so it cancels. Self-test with a simulated
150 ms response behind a 600 ms video path: loop 775 ms detected, 730 ms
extrapolated (true 750), heading 265 ms (expected ~255).
