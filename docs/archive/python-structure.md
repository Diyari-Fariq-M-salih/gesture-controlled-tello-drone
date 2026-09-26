python/
drone/
tello_client.py # connect, takeoff/land, send_rc_control
safety.py # timeouts, max speeds, geofence-ish constraints
vision/
camera_stream.py # reads tello video frames
hands.py # mediapipe hands -> 21 landmarks
face.py # face detector -> bbox
tracker.py # CSRT (or other tracker) wrapper
ml/
gesture_static.py # your current frame-based classifier
gesture_lstm.py # sequence model + smoothing + confidence
dataset_tools.py # record sequences, save, label
app/
state.py # shared state, mode, telemetry snapshot
controller.py # maps “intent” -> drone rc commands
web/
dashboard_streamlit.py or dashboard_flask.py
video_mjpeg.py / ws.py
