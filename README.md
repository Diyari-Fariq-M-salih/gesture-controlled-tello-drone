# Gesture-Based Drone Control Project

## Project Description

### Overview

The objective of this project is to design and implement a **gesture-based control system for an indoor drone**.  
The drone operates in a semi-autonomous mode, taking off, hovering, and executing predefined movement commands based on **visual gestures performed by a human operator**.

The system relies on **image processing techniques** to detect and classify gestures, which are then mapped to drone commands such as ascending, descending, looping, or following a circular trajectory.  
The drone maintains safe and stable behavior throughout operation, including fallback procedures when operator input is unavailable.

The project is primarily developed in **MATLAB**, as recommended, using a **Tello mini-drone equipped with a camera**. In parallel, an exploratory implementation using **Python and ROS** may be evaluated to compare performance, modularity, and development complexity.

---

## System Behavior

1. The drone takes off and enters a **hovering state**
2. The onboard camera continuously captures visual input
3. The system waits for valid operator gestures
4. Recognized gestures are translated into predefined commands
5. The drone executes the selected mission
6. If gestures are no longer detected, the system switches to a **safe fallback behavior**

---

## Work Distribution

The project is developed by **four team members**, divided into **two subgroups**:

### Subgroup A — Localization & Movement
- Drone localization and positioning
- Motion primitives and trajectory execution
- Control logic and safety handling

### Subgroup B — Gesture-Based Control
- Gesture vocabulary definition
- Image processing and gesture recognition
- Command mapping

---

## Development Environments

- **MATLAB** (exploratory)
- **Python + ROS** 

---

## Version Control Workflow

- Feature-based Git branches
- Independent development and documentation
- Final merge and result comparison

---

## Proposed Project Structure

```
project-root/
├── README.md
├── LICENSE
├── docs/
├── matlab/
├── python_ros/
├── tests/
└── results/
```

---

## Authors

- **Author 1** — Localization and Movement Control Nguyen Viet Khanh 
- **Author 2** — Localization and Movement Control ouchaouir khalid
- **Author 3** — Gesture Recognition and Image Processing  Mohammed-salih Diyari
- **Author 4** — Gesture Recognition and Image Processing  Ilyes Chaabeni

---

## License

MIT License
