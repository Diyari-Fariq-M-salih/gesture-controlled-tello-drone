# Content Review Notes for ARO Journal Submission

**Paper reviewed:** *Face-Authorized Gesture Control and Explainable Mode Arbitration for Tello Drones*  
**Target journal:** ARO – The Scientific Journal of Koya University  
**Review focus:** Scientific/content readiness, not formatting or paper style.

---

## Overall Assessment

The topic is relevant and potentially publishable: a real-time indoor drone-control system combining hand-gesture teleoperation, face-following, identity-aware authorization, deterministic mode arbitration, and reason-only LLM explanations.

However, the current manuscript reads more like a strong course/project report than a mature journal article. The main weakness is not the writing style, but the gap between the paper’s claims and the evidence currently provided.

The core idea is good, but the manuscript should not be submitted in its current form without strengthening the experimental validation.

---

## 1. Main Content Problem

The paper presents an interesting system integration:

- DJI Tello drone
- MediaPipe hand landmarks
- SVM gesture classifier
- Face-recognition gate using ArcFace-style embeddings
- Deterministic mode manager
- Reason-only local LLM for explanations

This is useful, but the research novelty is not yet sharply defined.

At the moment, the paper sounds like:

> We combined several existing modules into a working drone-control stack.

For a journal article, it needs to sound more like:

> Existing gesture-controlled drone systems often lack identity-aware authorization and deterministic arbitration between gesture, tracking, safety, and search modes. This work evaluates whether a lightweight, identity-gated arbitration stack can provide reliable indoor interaction on low-cost drone hardware.

---

## 2. Biggest Issue: Experimental Evidence Is Too Limited

The abstract and conclusion claim that the system is:

- real-time,
- stable,
- interpretable,
- predictable,
- suitable for indoor drone interaction.

But most of the quantitative results are only about:

- gesture-classification accuracy,
- face-recognition EER,
- ROC/AUC,
- dataset-size effects.

These are useful, but they do not fully validate the complete drone-control system.

### Missing Evidence

The paper should include real system-level experiments such as:

- number of real drone flight trials,
- flight duration,
- crash or emergency-stop rate,
- gesture-command success rate during flight,
- face-following success rate,
- target-loss recovery rate,
- mode-switching correctness,
- false mode switches,
- search behavior success rate,
- latency from camera frame to command,
- FPS and control-loop frequency,
- command rate to the Tello,
- battery failsafe demonstration.

Without these, reviewers may say that the paper evaluates classifiers but not the full drone-control system.

---

## 3. Gesture Dataset Validation Is Too Weak

The reported SVM performance is very high:

| Cap per class | Test samples | Accuracy | Macro F1 |
|---:|---:|---:|---:|
| 100 | 175 | 0.9486 | 0.9503 |
| 250 | 438 | 0.9840 | 0.9840 |
| 400 | 679 | 0.9971 | 0.9971 |

This is promising, but the paper correctly notes that train/test splitting was done within the same recording sessions. Therefore, the results mainly show **in-session generalization**, not robust real-world generalization.

### Needed Improvements

Add at least one of the following:

- cross-session evaluation: train on one day/session, test on another;
- cross-user evaluation: test on people not included in training;
- different lighting conditions;
- different backgrounds;
- different camera distances;
- different camera angles;
- tests while the drone is actually moving.

Without this, the 99.7% accuracy may look inflated.

---

## 4. Face-Recognition Evaluation Is Promising but Overclaimed

The paper reports:

- 76 positive images of the authorized user,
- around 13,000+ negative/intruder images,
- final EER of 0.32%,
- ROC AUC around 0.99.

This looks strong, but the validation has weaknesses.

### Problems

- The positive set is small: only 76 images.
- Only one authorized person appears to be evaluated.
- Negative images from LFW/Kaggle may not match the real drone-camera environment.
- The threshold seems to be selected using the evaluation data itself.
- No independent validation set is described.
- No spoofing tests are included.
- No printed-photo or phone-screen replay attack is tested.
- No test with lookalikes or similar faces.
- No real multi-person indoor experiment is reported.

### Recommended Reframing

Avoid saying or implying that this is a strong security system.

Use safer language:

- “identity-aware authorization”
- “operator-selection gate”
- “biometric gating for session-level control”
- “face-verification-based access filtering”

Avoid overusing:

- “secure”
- “security layer”
- “biometric security”
- “intruder prevention”

unless the threat model is clearly limited.

---

## 5. One-Shot Enrollment Is a Security Weakness

The paper states that the first detected face at startup becomes the authorized user.

This is practical for a demo, but weak from a security perspective. If the wrong person is first in view, they become the authorized operator.

### Add a Threat Model

The paper should explicitly say:

- The system prevents accidental control by bystanders after enrollment.
- It does not prevent malicious enrollment.
- It does not prevent spoofing attacks.
- It does not prevent replay attacks using a phone screen or printed photo.
- It does not implement liveness detection.
- It is intended as a session-level operator-selection mechanism, not a complete biometric-security solution.

### Possible Future Improvements

- manual enrollment confirmation,
- password or keyboard confirmation,
- stored authorized profile,
- liveness detection,
- anti-spoofing model,
- multi-frame enrollment,
- identity confirmation before takeoff.

---

## 6. LLM Contribution Is Interesting but Not Yet Validated

The reason-only LLM design is a strong engineering choice because the LLM does not control the drone. This avoids nondeterministic actuation and keeps flight commands deterministic.

However, the paper does not evaluate the LLM explanations.

### Add Evidence

The paper should include one or more of the following:

- sample table of deterministic decisions and generated explanations;
- explanation consistency score;
- percentage of explanations matching the deterministic reason;
- latency overhead of LLM explanation generation;
- examples of good and bad LLM explanations;
- comparison between deterministic reason string and LLM output;
- small human readability/usefulness evaluation.

### Otherwise

If no LLM evaluation is added, reduce the importance of the LLM in the title and claims.

Possible title adjustment:

> Face-Authorized Gesture Control and Deterministic Mode Arbitration for Tello Drones

instead of emphasizing “Explainable Mode Arbitration.”

---

## 7. Mode Manager Is Strong but Needs Validation

The hierarchical finite state machine is one of the best contributions of the paper.

It includes:

- battery-triggered landing,
- gesture-over-face priority,
- authorization filtering,
- hysteresis,
- bounded search behavior,
- cooldown after search,
- hover/fallback behavior.

However, it is described more than tested.

### Recommended Scenario-Test Table

Add a table like this:

| Scenario | Expected behavior | Observed behavior | Result |
|---|---|---|---|
| Authorized face + valid gesture | Gesture preempts face-following | Report result | Pass/fail |
| Authorized face, no gesture | Face-following active | Report result | Pass/fail |
| Unauthorized face + hand | Ignore command | Report result | Pass/fail |
| No face/no hand for >10 s | Search behavior starts | Report result | Pass/fail |
| Target reacquired during search | Return to tracking/control | Report result | Pass/fail |
| Battery ≤ 15% | Immediate landing | Report result | Pass/fail |
| Intermittent gesture detection | Hysteresis prevents flicker | Report result | Pass/fail |

Even 20–30 structured scenario trials would make the system validation much stronger.

---

## 8. Real-Time Claims Need Quantitative Support

The paper repeatedly says the system is real-time, lightweight, and suitable for real-time applications.

These claims need measurements.

### Add a Runtime Table

Suggested table:

| Component | Rate / latency |
|---|---:|
| Camera stream FPS | value |
| MediaPipe hand detection latency | value |
| Face detection latency | value |
| Face embedding latency | value |
| SVM inference latency | value |
| Mode-manager update rate | value |
| RC command output rate | value |
| LLM explanation latency | value |
| End-to-end camera-to-command latency | value |

This would make the real-time claim much more defensible.

---

## 9. Reproducibility Details Are Missing

The paper says the pipeline is reproducible, but it does not provide enough exact details.

### Add Technical Details

Include:

- host computer specifications;
- operating system;
- Python version;
- MediaPipe version;
- OpenCV version;
- scikit-learn version;
- SVM kernel, C, gamma, class weighting;
- train/test split ratio;
- random seed;
- exact number of raw samples per class;
- exact number of usable samples per class;
- face-recognition model name;
- embedding dimension;
- threshold-selection method;
- Tello SDK command frequency;
- EMA parameters;
- control gains;
- deadbands;
- hysteresis hold time;
- search duration;
- cooldown duration;
- battery threshold.

---

## 10. Ethical and Privacy Statement Is Missing

Because the paper uses face recognition and human-face datasets, it needs an ethics/privacy section.

### Add a Short Statement Covering

- consent for custom face images;
- usage conditions of LFW/Kaggle datasets;
- whether embeddings are stored permanently or only in volatile memory;
- privacy limitations of face recognition;
- demographic-bias limitations;
- indoor drone safety precautions;
- no use of face recognition for identification beyond operator gating.

Suggested wording:

> The face-recognition component is used only for session-level operator gating. The enrolled embedding is stored in volatile memory during the session and is not used for persistent identification. Custom face images were collected with consent. Public face datasets were used only for offline evaluation. The system does not include liveness detection or anti-spoofing, and therefore should not be interpreted as a complete biometric-security mechanism.

---

## 11. Discussion Section Needs Stronger Limitations

The current future-work section mentions that the closest hand to the authorized face may not necessarily belong to that person. This is an excellent limitation and should be expanded.

### Add More Limitations

- one authorized user only;
- no anti-spoofing/liveness detection;
- limited positive face dataset;
- gesture data collected in controlled conditions;
- no cross-user or cross-session validation;
- no obstacle avoidance;
- no robust indoor localization;
- no evaluation under drone vibration/motion blur;
- Tello has no obstacle-avoidance sensors;
- face-hand association is heuristic;
- LLM explanations are not formally evaluated.

---

## 12. Conclusion Should Be More Careful

The conclusion currently claims that the system ensures predictable flight logic and stable behavior. That may be true, but the results section does not yet provide enough quantitative flight evidence.

### Safer Conclusion Wording

Use something like:

> The results demonstrate reliable in-session gesture classification and promising face-verification performance, while runtime logs indicate correct execution of the designed arbitration rules in preliminary indoor trials.

This is more defensible than claiming full stability or general robustness.

---

## 13. ARO-Specific Warning

ARO’s author guidance emphasizes originality, robust data, and clear methodology. The paper should therefore strengthen the experimental evidence before submission.

Also, ARO’s current author guide states that text generated from AI, machine learning, or similar algorithmic tools is not allowed in papers submitted to the journal. Therefore, these notes should be used only as revision guidance. The final manuscript should be genuinely written, checked, and approved by the authors.

ARO also uses double-blind review, so the submitted manuscript should not contain author-identifying information in the review version.

---

## 14. Minimum Fixes Before Submission

Before submitting, do at least the following:

1. Add real flight/scenario tests for the complete system.
2. Add latency, FPS, and control-loop measurements.
3. Add cross-session gesture evaluation.
4. Reframe face recognition as identity-aware gating, not strong security.
5. Add ethics/privacy and dataset-use statement.
6. Add LLM explanation examples or reduce the LLM claim.
7. Add mode-manager scenario validation.
8. Add exact reproducibility parameters.
9. Clarify the novelty and research gap.
10. Tone down unsupported stability/security claims.

---

## 15. Suggested Stronger Contribution Statement

A stronger contribution statement could be:

> This paper presents a lightweight identity-aware human-drone interaction stack for indoor Tello drone operation. The system combines static hand-gesture teleoperation, face-following, session-level face-verification gating, and deterministic hierarchical mode arbitration. Unlike gesture-only control systems, the proposed design explicitly separates perception, authorization, safety arbitration, and explanation. All actuation decisions are produced by deterministic logic, while a local LLM is used only for post-hoc explanation logging. Experiments evaluate gesture classification, face-verification performance, and structured runtime scenarios demonstrating arbitration behavior under authorized, unauthorized, target-loss, and safety-critical conditions.

---

## 16. Suggested Additional Results Tables

### Table A: Runtime Performance

| Module | Metric | Value |
|---|---|---:|
| Video stream | FPS | TBD |
| Hand detection | latency/frame | TBD |
| Face detection | latency/frame | TBD |
| Face embedding | latency/frame | TBD |
| Gesture SVM | inference latency | TBD |
| Mode manager | update rate | TBD |
| RC command output | command rate | TBD |
| LLM explanation | latency/decision | TBD |
| Full loop | camera-to-command latency | TBD |

### Table B: Scenario-Level System Tests

| Test case | Trials | Successes | Success rate |
|---|---:|---:|---:|
| Authorized gesture accepted | TBD | TBD | TBD |
| Unauthorized gesture rejected | TBD | TBD | TBD |
| Face-following activates correctly | TBD | TBD | TBD |
| Gesture preempts face-following | TBD | TBD | TBD |
| Hysteresis prevents flicker | TBD | TBD | TBD |
| Search starts after target loss | TBD | TBD | TBD |
| Target reacquired after search | TBD | TBD | TBD |
| Battery failsafe landing | TBD | TBD | TBD |

### Table C: LLM Explanation Evaluation

| Deterministic state/reason | Command issued | LLM explanation | Correct? |
|---|---|---|---|
| Authorized hand gesture LEFT detected | yaw/left command | TBD | yes/no |
| Authorized face centered | hover/track | TBD | yes/no |
| No target for >10 s | search | TBD | yes/no |
| Battery ≤ 15% | land | TBD | yes/no |

---

## Final Recommendation

Do not submit the manuscript exactly as it is.

The core system is interesting and the paper has potential, but the current evidence is not strong enough for a journal article. The manuscript should be revised to include system-level flight/scenario validation, real-time measurements, stronger reproducibility details, clearer novelty, and more careful claims about security, stability, and explainability.

With those changes, the paper would become much closer to a serious journal submission rather than a polished project report.
