# My Random Forest Said "Cancer" Twice. Hindsight Remembered It Was Pancreatitis.

The second time the classifier flagged patient P-2071 as PDAC, it was wrong for exactly the same reason as the first time. A clinician had already told the system why. The system had simply forgotten.

That is the problem I set out to fix: a good stateless model wrapped in a workflow that has no memory.

## What I built

Surveillance programs for high-risk patients (family history, new-onset diabetes, cystic lesions) don't make one decision. They make a series of small ones across months. I had a [Random Forest classifier](https://github.com/kareem1207/early-stage-pancreatic-cancer-detection-using-random-forest-classification) that takes urinary biomarkers (REG1B, TFF1, LYVE1, CA19-9, creatinine, age, sex) and predicts Control / Benign / PDAC. It scores every sample as if the patient had never walked in before.

So I put an agent around it, backed by [Hindsight](https://github.com/vectorize-io/hindsight), an open-source agent memory system. Per visit the agent:

1. **Recalls** what memory holds about this patient: past visits and clinician feedback.
2. Runs the Random Forest on the new sample, unchanged.
3. Compares against the previous visit and applies any recalled confounder notes.
4. **Retains** the visit. When a clinician reviews an alert, their verdict is retained too.

One Hindsight bank per clinic. Every fact is tagged `patient:<id>`, so recall stays scoped to the patient in front of you. I'd recommend reading the [Hindsight docs](https://hindsight.vectorize.io/) before designing your own banks; the mission and tagging choices matter more than I expected.

## The through-line: feedback is data the model never sees

The Random Forest can't learn from a clinician's "no, that was pancreatitis". It's trained offline. But that sentence is exactly the kind of knowledge that makes the next alert better. [Agent memory](https://vectorize.io/what-is-agent-memory) is where it belongs.

Retaining feedback is one call:

```python
def record_feedback(self, pid, feedback, confounder=False):
    label = "CONFOUNDER" if confounder else "CLINICIAN FEEDBACK"
    self.memory.retain(f"{label} for patient {pid}: {feedback}",
                       self._tag(pid), context="clinician feedback")
```

And the Hindsight-backed store is thin:

```python
def retain(self, content, tags, context=""):
    self.client.retain(bank_id=BANK_ID, content=content,
                       context=context or None, tags=tags)

def recall(self, query, tags=None, limit=20):
    resp = self.client.recall(bank_id=BANK_ID, query=query, tags=tags,
                              tags_match="any", budget="mid")
    return [r.text for r in resp.results[:limit]]
```

At the next visit the agent recalls, and only surfaces confounder notes when the model is actually alarmed (P(PDAC) ≥ 0.3), because a warning on a clean result is just noise.

## Before and after

Two synthetic patients. Same model, same samples, two modes.

**P-1042**: every reading is individually unremarkable.

| Visit | Stateless RF | With memory |
|---|---|---|
| 2 | Benign, routine follow-up | Benign, but REG1B +67%, TFF1 +78% vs. visit 1: **shorten follow-up** |
| 3 | Benign, routine follow-up | Same markers still climbing, P(PDAC) up three visits in a row: **shorten follow-up** |
| 4 | PDAC, escalate | PDAC, escalate |

The memory-backed agent asks for a repeat sample two visits before the classifier alone would raise anything.

**P-2071**: a pancreatitis flare at visit 2 spikes the markers. The model says PDAC (0.87) and escalates. The clinician overrules and the agent retains that. At visit 4 the same pattern appears (0.79). The stateless model escalates again. The agent instead reports: *"CONFOUNDER on record: acute pancreatitis flare... flag for clinician, review before escalating."*

Note what it does *not* do: it doesn't suppress the alert or override the model. It routes the decision to a human with the relevant history attached.

## What went wrong

- **My first version warned on every visit.** Visit 3 of P-2071 was a clean Control result, yet the confounder note was shown. Gating on model alarm level fixed it.
- **Trend maths and memory are different jobs.** Recall returns text facts. Parsing numbers back out of prose is fragile, so the agent keeps a small structured ledger for arithmetic and uses Hindsight for narrative, cross-visit context. I'd rather split the responsibilities than force one store to do both.
- **The data is synthetic.** The classifier was trained on data generated to match the schema of a published urinary-biomarker dataset (Debernardi et al., 2020). The numbers show the *mechanism*, not clinical performance. Don't read them as a claim about detection accuracy.
- **A remembered confounder is a risk as well as a help.** If a note is stale, the agent could downplay a real cancer. That's why it only ever flags for review and never suppresses an escalation above 0.9.

## Takeaways

1. **Put memory where the model can't learn.** Human feedback after deployment is the cleanest example.
2. **Tag memories by entity from day one.** Per-patient scoping turned recall from "search everything" into "what do I know about this person".
3. **Keep structured numbers structured.** Use memory for narrative context, not arithmetic.
4. **Gate what you surface.** Recalling something isn't a reason to show it.
5. **Keep the human in the loop.** Memory should sharpen the hand-off, not replace the clinician.

The code is in the [repository](https://github.com/kareem1207/early-stage-pancreatic-cancer-detection-using-random-forest-classification); run `python src/demo.py` to reproduce both patients.

*This is a research prototype, not a diagnostic device.*
