# LLM Reasoner — Summary & Lessons Learned

## What the LLM Is (Now)

The LLM in this system is **reason-only**.

It:

- Observes the current system state
- Produces a human-readable explanation of behavior
- Logs and displays its reasoning for transparency

It **does not control the drone**.

All flight behavior is handled by deterministic, rule-based logic.

---

## What We Tried (And Why It Failed)

Initially, we experimented with using a **local LLM to directly influence control decisions**  
(e.g. selecting motion directions or modes based on perception input).

This turned out to be a **bad idea**.

### What Went Wrong

- The LLM reacted to **noisy perception** (false hands, partial faces)
- It produced **non-deterministic outputs** for the same state
- Small timing changes caused **different commands each frame**
- Conflicting intents were generated (e.g. _“search” → “backward” → “up”_)
- The drone oscillated, spun, or drifted unpredictably
- Debugging became nearly impossible:
  - No clear causal chain
  - No reproducible failures
  - No stable “source of truth”

In short:

> The drone stopped behaving like a control system and started behaving like a confused conversation.

---

## Why This Was a Problem (Especially for Safety)

LLMs are:

- Probabilistic
- Stateless unless carefully constrained
- Sensitive to prompt wording and timing

Real-time drone control requires:

- Determinism
- Bounded latency
- Predictable failure modes

Mixing the two **directly** created unsafe and unexplainable behavior.

---

## The Fix: Separation of Concerns

We redesigned the system around a strict rule:

> **LLMs may explain decisions, but never make them.**

The final architecture:

- Deterministic logic:
  - Handles perception gating
  - Selects modes
  - Generates RC commands
- LLM:
  - Observes the same state
  - Explains _why_ a decision happened
  - Logs reasoning for humans

Disabling the LLM produces **identical flight behavior**.

---

## Why This Is Better

- Stable flight
- Fully reproducible behavior
- Clear debugging path
- Human-readable explanations
- Zero safety risk from the LLM

And most importantly:

> We can finally trust the drone again.

---

## LLM Used

The system uses a **local, lightweight Large Language Model (LLM)** running fully offline:

- **Model:** Qwen 2.5 – 0.5B (Instruct)
- **Identifier:** `qwen2.5:0.5b-instruct`
- **Model size:** ~0.5 billion parameters
- **Runtime:** Ollama
- **Execution:** Local CPU inference (no GPU)
- **Interface:** HTTP (`/api/chat`)
- **Network:** Fully offline (operates over Tello Wi-Fi)

---

## Role in the System

The LLM operates in **reason-only mode**.

It:

- Observes system state and decisions
- Produces short, human-readable explanations
- Logs and overlays reasoning for transparency

It **does not**:

- Control the drone
- Select control modes
- Send RC commands
- Interact with the control loop

All flight behavior remains deterministic and rule-based.

---

## Why This Model Was Chosen

This model represents a deliberate engineering tradeoff:

- ✅ Runs reliably on CPU-only hardware
- ✅ Fast enough for real-time reasoning alongside video and control
- ✅ Fully offline (no internet dependency)
- ✅ Sufficient for short, constrained explanations

Limitations:

- ❌ Not suitable for safety-critical or real-time control  
  (which directly motivated the move to a strict reason-only role)

---

## Takeaway

Using **Qwen 2.5 – 0.5B (Instruct)** as a local, offline, reason-only LLM provides
explainability and debugging insight **without compromising control stability or safety**.

This separation proved essential after early attempts to involve the LLM in control
resulted in unstable and non-deterministic behavior.

## Takeaway

LLMs are powerful tools for **interpretation**, **explanation**, and **analysis**.

They are **not suitable** as real-time control authorities in safety-critical systems.

This project demonstrates a practical and safe way to integrate LLMs **without sacrificing control reliability**.

---
