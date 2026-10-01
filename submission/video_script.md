# 3-minute demo script (screen recording + voiceover)

**Title options**
1. My cancer-risk model kept making the same mistake, so I gave it memory
2. Agent memory with Hindsight: from stateless classifier to surveillance assistant
3. How a clinician's "no" became memory instead of getting lost
4. Same model, same data, different decision: what memory changes
5. Building a patient-history agent on Hindsight

## 0:00 Intro (30s)
Say: "Hi, I'm <NAME>. I built an agent that wraps a pancreatic-cancer Random Forest with Hindsight memory, so it remembers each patient's history and clinician feedback."
Show: repo README, then `src/agent.py`.

## 0:30 The problem (30s)
Show: `python src/demo.py`, scroll to **STATELESS RANDOM FOREST**, patient P-2071.
Say: "The same pancreatitis-flare pattern triggers PDAC escalation at visit 2 and again at visit 4. The clinician already explained it. The model has no way to know."

## 1:00 Live demo (2 min)
1. **P-1042 (memory mode)**: point at visit 2 trend lines: "Every reading is 'Benign', but REG1B is up 67%. Memory-backed: shorten follow-up. Stateless: routine."
2. **P-2071**: show `record_feedback` in `agent.py`. Say: "The clinician's overrule is retained to Hindsight tagged `patient:P-2071`."
3. Visit 4 output: "Same alarm, but recall brings back the pancreatitis note. The agent flags for clinician review instead of blindly escalating."
4. Show `src/memory.py` `HindsightMemory.retain / recall`. If running against Hindsight Cloud, show the bank in the Hindsight UI.

## 3:00 Wrap (30s)
Say: "The surprise was how little code memory needs, and how much it changed the hand-off. Caveats: synthetic data, research prototype, human stays in the loop."
