"""
demo.py
========
Side-by-side: the stateless Random Forest vs. the same model wrapped in the
memory-backed agent, over two synthetic surveillance patients.

    python demo.py                       # offline fallback memory
    HINDSIGHT_BASE_URL=... HINDSIGHT_API_KEY=... python demo.py   # real Hindsight

Patient values are synthetic, on the same scale as the project's synthetic
training data. Nothing here is medical advice.
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from agent import SurveillanceAgent  # noqa: E402
from memory import get_memory  # noqa: E402

BASE = dict(age=64, sex="M", patient_cohort="Cohort1", sample_origin="BPTB",
            creatinine=0.9, REG1A=20.0)


def visit(reg, tff, ca, ly):
    return dict(BASE, REG1B=reg, TFF1=tff, plasma_CA19_9=ca, LYVE1=ly)


# P-1042: every single reading looks "borderline", but the trajectory is not.
P1042 = [visit(3, 9, 8, 1.4), visit(5, 16, 14, 2.0), visit(8, 30, 22, 2.6),
         visit(12, 50, 35, 3.4), visit(17, 70, 46, 4.6)]

# P-2071: transient false alarm at visit 2 (pancreatitis flare), same pattern at visit 4.
P2071 = [visit(3, 9, 8, 1.4), visit(12, 50, 35, 3.4), visit(3.1, 9.5, 8.5, 1.4),
         visit(11, 48, 33, 3.2)]


def run(use_memory: bool, memory):
    agent = SurveillanceAgent(memory, use_memory=use_memory)
    out = []
    for i, s in enumerate(P1042):
        out.append(("P-1042", agent.review_visit("P-1042", s)))
    for i, s in enumerate(P2071):
        b = agent.review_visit("P-2071", s)
        out.append(("P-2071", b))
        if i == 1:  # clinician reviews the escalation and overrules it
            agent.record_feedback(
                "P-2071", "OVERRULED escalation: acute pancreatitis flare, markers normalised on repeat. "
                "Transient REG1B/TFF1 elevation during flares is expected for this patient.",
                confounder=True)
    return out


if __name__ == "__main__":
    store = Path(__file__).resolve().parent.parent / "data" / "demo_memory.json"
    store.unlink(missing_ok=True)
    memory = get_memory(store)
    print(f"Memory backend: {memory.name}\n")
    for label, use in (("STATELESS RANDOM FOREST", False), ("MEMORY-BACKED AGENT", True)):
        print("=" * 70, f"\n{label}\n" + "=" * 70)
        current = None
        for pid, b in run(use, memory):
            if pid != current:
                print(f"\n-- patient {pid}")
                current = pid
            print(b.render())
    store.unlink(missing_ok=True)
