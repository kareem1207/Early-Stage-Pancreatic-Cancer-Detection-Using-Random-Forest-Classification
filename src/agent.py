"""
agent.py
=========
SurveillanceAgent -- a memory-backed layer on top of the Random Forest.

The classifier alone judges each urine sample in isolation. Real surveillance
of a high-risk patient is longitudinal: what matters is how REG1B / TFF1 /
CA19-9 move across visits, and what the clinician already learned about this
patient (e.g. "CA19-9 runs high when he has cholestasis").

Per visit the agent:
  1. RECALLs prior visits + clinician notes for the patient from memory
  2. runs the Random Forest on the new sample (predict.py, unchanged)
  3. compares against the patient's previous values (trend) and applies any
     recalled confounder warnings
  4. RETAINs the visit, and later the clinician's feedback, so the next visit
     starts with more context

Structured numbers (for trend maths) are kept in a small per-patient ledger;
Hindsight holds the narrative memory (visits, notes, feedback) that recall and
reflect operate on. `use_memory=False` gives the stateless baseline.
"""

from dataclasses import dataclass, field

import pandas as pd

from predict import predict

MARKERS = ["REG1B", "TFF1", "plasma_CA19_9", "LYVE1"]
RISE_ALERT = 0.25  # >=25% rise vs previous visit flags a marker


@dataclass
class Briefing:
    visit_no: int
    predicted: str
    p_pdac: float
    trend_flags: list = field(default_factory=list)
    recalled: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    action: str = ""

    def render(self) -> str:
        lines = [f"Visit {self.visit_no}: RF says {self.predicted} (P(PDAC)={self.p_pdac:.2f})"]
        lines += [f"  trend: {t}" for t in self.trend_flags]
        lines += [f"  memory: {w}" for w in self.warnings]
        lines.append(f"  suggested action: {self.action}")
        return "\n".join(lines)


class SurveillanceAgent:
    def __init__(self, memory, use_memory: bool = True):
        self.memory = memory
        self.use_memory = use_memory
        self.ledger: dict[str, list[dict]] = {}

    @staticmethod
    def _tag(pid): return [f"patient:{pid}"]

    def review_visit(self, pid: str, sample: dict, note: str = "") -> Briefing:
        history = self.ledger.setdefault(pid, [])
        visit_no = len(history) + 1
        res = predict(pd.DataFrame([sample])).iloc[0]
        p_pdac = float(res["probability_PDAC"])
        b = Briefing(visit_no, res["predicted_diagnosis"], p_pdac)

        if self.use_memory:
            b.recalled = self.memory.recall(
                f"prior visits, clinician notes and confounders for patient {pid}", self._tag(pid))
            if history:
                prev = history[-1]
                for m in MARKERS:
                    if prev[m] > 0 and (sample[m] - prev[m]) / prev[m] >= RISE_ALERT:
                        b.trend_flags.append(
                            f"{m} up {100*(sample[m]-prev[m])/prev[m]:.0f}% since visit {prev['visit']} "
                            f"({prev[m]:g} -> {sample[m]:g})")
                if len(history) >= 2 and all(h["p_pdac"] < p_pdac for h in history[-2:]):
                    b.trend_flags.append("P(PDAC) has risen across 3 consecutive visits")
            if p_pdac >= 0.3:  # only surface confounder notes when the model is actually alarmed
                b.warnings = [t for t in b.recalled if "CONFOUNDER" in t]

        b.action = self._action(b)
        history.append({"visit": visit_no, "p_pdac": p_pdac, **{m: sample[m] for m in MARKERS}})

        if self.use_memory:
            self.memory.retain(
                f"Visit {visit_no} for patient {pid}: " +
                ", ".join(f"{m}={sample[m]:g}" for m in MARKERS) +
                f". Random Forest: {b.predicted} (P(PDAC)={p_pdac:.2f})." +
                (f" Note: {note}" if note else ""),
                self._tag(pid), context="surveillance visit")
        return b

    def record_feedback(self, pid: str, feedback: str, confounder: bool = False) -> None:
        """Clinician feedback becomes memory the next visit's recall can use."""
        if not self.use_memory:
            return
        label = "CONFOUNDER" if confounder else "CLINICIAN FEEDBACK"
        self.memory.retain(f"{label} for patient {pid}: {feedback}", self._tag(pid),
                           context="clinician feedback")

    @staticmethod
    def _action(b: Briefing) -> str:
        if b.warnings and b.p_pdac < 0.9:
            return "flag for clinician: known confounder on record -- review before escalating"
        if b.predicted == "PDAC" or b.p_pdac >= 0.5:
            return "escalate: refer for imaging (CT/MRI or EUS)"
        if b.trend_flags:
            return "shorten follow-up interval; repeat sample in 4-6 weeks"
        return "routine follow-up"
