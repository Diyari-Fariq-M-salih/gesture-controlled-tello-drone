# 

## 🛰️ System Logic & Control Modes

This project implements a multi-modal control system for the DJI Tello, utilizing real-time computer vision to toggle between user-specific authorization and general interaction.      

Due to the use of optimized and small ML models, the project **CAN/CANNOT?** run in real-time on an onboard compute (like Raspberry Pi or Jetson Orin Nano). It has been successfully  tested with a Raspberry Pi 4 with 8GB of RAM.        
 
With the tello drone, the PC (or onboard computer) runs all the algorithms and the drone only sends data and receives commands. However, the project can be used with any other drone by modifying the communication layer.     

### 🔐 Face Recognition & Authorization

The system's behavior is governed by the **Face Recognition** toggle.

* **Enabled (Authorized Mode):** Upon startup, the drone scans the first face it encounters and stores its embedding as the **Authorized User**. All subsequent detections categorize individuals as "Authorized" or "Intruder."
* **Disabled (Standard Mode):** The drone operates on a "first-come, first-served" basis, prioritizing the most prominent targets in the frame.

---

### 🕹️ Operational Modes

| Mode | Face Recognition ON (Authorized) | Face Recognition OFF (Standard) |
| --- | --- | --- |
| **Gesture** | Only accepts gestures from the **Authorized User**. If multiple hands exist, it selects the one closest to the authorized face. | Accepts gestures from the first detected hand in the frame. |
| **Face Follow** | Tracks **only** the Authorized User. If lost, the drone performs a 360° yaw search. Lands if the user isn't found after two rotations. | Tracks the "primary" face (the detection with the largest bounding box area). |
| **Keyboard** | Full manual teleoperation. Authorized user detection remains active for logging/UI. | Full manual teleoperation. |
---

### 🛠️ Safety & Fail-safes

* **Authorized Gesture Timeout:** In Gesture Mode (Recognition ON), if the authorized user is not detected within a specific window, the drone will enter a stationary hover before initiating an auto-landing sequence for safety.
* **Search Protocol:** The 360° yaw rotation ensures the drone doesn't drift aimlessly if the subject moves out of the field of view (FOV).

---

## Installation
**Install dependencies**        
Python libraries versions should be compatible. It is recommended to create a virtual environment and install the libraries. Versions in `requirements.txt` might be old.       

```sh
git clone https://github.com/Diyari-Fariq-M-salih/gesture-controlled-tello-drone.git
```
**For Linux/macOS:**
```sh
cd gesture-controlled-tello-drone
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```
**For Windows:**
```sh
dir gesture-controlled-tello-drone
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Then finally, download models required for face detection and recognition:
```sh
python -m face_recognition.download_models
```



## Run
The drone uses UDP to communicate with your PC/laptop via Wifi. Note that all the algorithms are run on your PC, and the drone only sends data and receives commands.   

The steps to run the project are as follows:  
1. Power on drone
2. Connect PC to Tello Wi-Fi (TELLO-XXXXXX)
3. Run:
```sh
cd tello_gesture_py
python -m tello_gesture.main --model model.joblib --labels tello_gesture/gestures/labels_example.json
```
If you have any connection issues (especialy on Windows), try disabling UDP firewall.       

After a test is ended, logs for telemetry data are automatically saved to `telemetry_logs.csv` in the root directory:

- Logged at 1 Hz
- Includes battery, height, yaw, and velocities

**Communication Layer and Saving Telemetry Data:**

- UDP commands (8889)
- Telemetry listener (8890)
- Video stream (11111)
- MediaPipe hands landmarks
- Controller loop:
  - Gesture-based XYZ translation (lr/fb/ud via RC)
  - Keyboard fallback
  - 1 Hz telemetry logging to CSV

## Usage
The face recognizer (authorization mode) can be enabled/disabled in the config file `config.py`:
```py
recognize_faces: bool = True
```

### Flight

| Key     | Action             |
| ------- | ------------------ |
| `t`     | Takeoff            |
| `l`     | Land               |
| `e`     | Emergency stop     |
| `q`     | Quit (lands first) |

### Mode Selection
The available modes are: `keyboard`, `gesture`, `face`, `search_360`, `hover` and `land`.   

The mode selection is automatic. Keyboard mode (teleoperation) is not available in this branch, but will be included soon for safety.   
The modes are selected as follows, if face recognition is enabled:
- If battery level <= 15%, perform landing.   
- If authorized face detected and hand detected -> gesture mode
- If authorized face detected but no hand detected -> face following.     
- If no authorized face was detected -> hover. If still no detection after some time (`nohuman_search_s=10s`) -> 360° search mode: inplace yaw rotation untill authorized face was detected.    

## Safety Tips

- Test function before sending takeoff command (moving the drone with hands)
- Train gestures before flying
- Use `l` to land after ending a test
- Be ready to press `e` for emergency stop
