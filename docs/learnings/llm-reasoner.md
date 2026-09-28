# Local LLM Reasoner — Setup, Usage, and Lessons Learned

This project uses a **local Large Language Model (LLM)** strictly for **reasoning and explanation**.

The LLM:
- Runs fully offline
- Does NOT control the drone
- Explains *why* the system behaves the way it does

This document explains:
1. What the LLM is and does
2. Why we chose this model
3. The problems we faced when the LLM tried to control the drone
4. How to install and run the LLM (step by step)
5. How it integrates safely with the drone system

---

## 1. LLM Used

The system uses a **local, lightweight LLM** running fully offline:

- **Model:** Qwen 2.5 – 1.5B (Instruct); the arXiv v1 runs used the 0.5B
- **Identifier:** `qwen2.5:1.5b-instruct` (`--llm-model` to change)
- **Model size:** ~1.5 billion parameters, 1 GB
- **Runtime:** Ollama
- **Execution:** Local CPU inference (no GPU required)
- **Interface:** HTTP (`/api/chat`)
- **Network:** Fully offline (works over Tello Wi-Fi)

---

## 2. What the LLM Does (Current Design)

The LLM operates in **reason-only mode**.

It:
- Observes the current system state
- Receives the active control mode and RC command
- Produces a short, human-readable explanation
- Logs reasoning alongside telemetry
- Displays explanations on the video overlay

Example output:
> “No face detected for several seconds, initiating search rotation.”

---

## 3. What the LLM Explicitly Does NOT Do

For safety and stability, the LLM:

❌ Does NOT:
- Fly the drone  
- Send RC commands  
- Select control modes  
- Override deterministic logic  
- Run inside the control loop  

✅ All flight behavior is:
- Deterministic
- Rule-based
- Reproducible
- Identical even if the LLM is disabled

---

## 4. Why This Model Was Chosen

This model is a deliberate engineering tradeoff:

### Advantages
- ✅ Runs on CPU-only hardware
- ✅ Fast enough for real-time reasoning
- ✅ Fully offline (no internet required)
- ✅ Lightweight and stable
- ✅ Sufficient for short explanations

### Limitations
- ❌ Not suitable for safety-critical or real-time control

That limitation directly motivated the **reason-only design**.

---

## 5. What We Tried First (And Why It Failed)

Initially, we attempted to let a **local LLM influence control decisions**.

This caused serious problems.

### What Went Wrong

- The LLM reacted to **noisy perception** (false hands, partial faces)
- Outputs were **non-deterministic**
- Small timing changes caused different commands
- Conflicting intents appeared:
  - search → backward → up → hover
- The drone:
  - Spun unexpectedly
  - Drifted backward
  - Oscillated between behaviors
- Debugging became nearly impossible:
  - No clear cause-and-effect
  - No reproducibility
  - No single source of truth

In short:

> The drone stopped behaving like a control system  
> and started behaving like a confused conversation.

---

## 6. Final Design Rule

**Deterministic code flies the drone.  
The LLM explains the drone.**

---

## 7. Installing the LLM

### Step 1: Install Ollama
https://ollama.com

Verify:
```
ollama --version
```

### Step 2: Download the model
```
ollama pull qwen2.5:1.5b-instruct
```

### Step 3: Test
```
ollama run qwen2.5:1.5b-instruct
```

---

## 8. Running with the Project
```
python -m tello_gesture_py.src.main --classifier svm --run-id <tag>
```
The reasoner starts with the controller when Ollama is running; `--no-llm` skips it
and `i` hides its text on the HUD.

---

## 9. Takeaway

LLMs are excellent for explanation and transparency, but unsafe for direct control.

This project demonstrates a safe integration strategy for live robotics systems.

## Choosing the model (2026-09-27)
`scripts/llm_bench.py` replays logged decision moments from every flight through
each model with the controller's exact request. 40 moments, 8 per mode, laptop
otherwise idle (in flight perception shares the CPU):

| model | valid | names the mode | invented battery warning | median | size |
|---|---|---|---|---|---|
| qwen2.5:0.5b (v1) | 78-93% | 55-72% | 23-42% | 1.4-1.6 s | 0.4 GB |
| **qwen2.5:1.5b** | 100% | 95-97% | 7% | 0.8-0.9 s | 1.0 GB |
| phi4-mini | 100% | 95% | 5% | 1.05 s | 2.5 GB |
| llama3.2:3b | 100% | 82-90% | 5-7% | 0.8-1.0 s | 2.0 GB |
| qwen2.5:3b | 100% | 90-93% | 33-35% | 1.1-1.4 s | 1.9 GB |
| gemma3:1b | 20-25% | 10-15% | - | 1.1 s | 0.8 GB |

The v1 timeouts were Ollama unloading the idle model: reloading takes 2-8 s
against a 4 s timeout. The request now keeps the model loaded (`keep_alive`) and
the controller loads it at start; no model timed out afterwards. "Names the mode"
and "invented battery warning" are keyword checks; read `llm_bench.csv` before
quoting them. Both 1.5B and phi4-mini often repeat the controller's own reason
verbatim: correct, but not much of an explanation.
