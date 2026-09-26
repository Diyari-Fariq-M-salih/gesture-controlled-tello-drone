## Hardware Platform — DJI Tello EDU (TLW004)

### Model Identification

- **Manufacturer:** DJI
- **Product Line:** Tello EDU
- **Model Code:** TLW004
- **Drone Type:** Indoor quadrotor (educational / research)

---

### Physical Characteristics

- **Weight:** ~87 g (including battery)
- **Dimensions:** 98 × 92.5 × 41 mm
- **Frame:** Lightweight plastic
- **Configuration:** Quadcopter with fixed-pitch propellers

---

### Flight Performance

- **Maximum Flight Time:** ~13 minutes
- **Maximum Speed:** ~8 m/s
- **Maximum Flight Distance:** ~100 m (Wi-Fi dependent)
- **Maximum Altitude:** ~10 m (software-limited)
- **Hovering Accuracy:**
  - Vision positioning: ±0.1 m
  - Height control: ±0.1 m

---

### Sensors

- **Downward Vision System**
  - Optical flow sensor
  - Infrared depth sensor (altitude estimation)
- **IMU**
  - 6-axis inertial measurement unit (accelerometer + gyroscope)
- **Barometer**
  - Used for altitude stabilization
- **Camera**
  - 5 MP CMOS sensor
  - Video streaming: 720p @ 30 fps
  - Field of View: ~82.6°

> **Note:** Raw IMU and depth data are not directly accessible; sensor fusion is handled internally by the drone.

---

### Communication

- **Protocol:** Wi-Fi (2.4 GHz)
- **Operational Range:** ~100 m (open space)
- **Latency:** Suitable for real-time control and video streaming
- **Supported Interfaces:**
  - DJI Tello SDK
  - MATLAB Support Package for Ryze Tello
  - Python SDKs (e.g., Tello / TelloPy)
  - ROS (via community wrappers)

---

### Control Capabilities

The Tello EDU supports **high-level command-based control**, including:

- Takeoff and landing
- Hovering
- Ascending and descending
- Translational motion (forward, backward, left, right)
- Rotational motion (clockwise and counter-clockwise)
- Predefined movement distances and angles

> Low-level motor control and direct PID tuning are not accessible.

---

### Programming & SDK Support

- **Official SDK:** DJI Tello SDK 2.0
- **Supported Programming Environments:**
  - MATLAB
  - Python
  - Scratch (educational features)
- **Tello EDU Exclusive Features:**
  - Multi-drone (swarm) control
  - Mission Pad support
  - Extended SDK commands

---

### Suitability for This Project

The DJI Tello EDU (TLW004) is well suited for this project due to:

- Onboard camera streaming required for gesture recognition
- Stable indoor flight using vision-based positioning
- Full compatibility with MATLAB image processing workflows
- High-level control commands enabling safe mission execution
- Lightweight design, reducing risk during indoor experimentation

---

### Limitations

- No obstacle avoidance sensors
- Limited access to raw sensor data
- Wi-Fi latency can affect fast control loops
- Not designed for outdoor or high-speed autonomous flight

# all data here is subject to change!
