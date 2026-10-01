"""
memory.py
==========
Memory backends for the surveillance agent.

    HindsightMemory  -- real Hindsight (https://github.com/vectorize-io/hindsight)
                        via the `hindsight-client` SDK. Used when
                        HINDSIGHT_BASE_URL is set (Hindsight Cloud or a local
                        server). One memory *bank* per clinic; every fact is
                        tagged with `patient:<id>` so recall stays per-patient.
    LocalMemory      -- tiny JSON-file stand-in with the same interface, so the
                        demo and tests run offline. It does naive keyword
                        recall only -- it exists to keep the code runnable,
                        not to imitate Hindsight's retrieval quality.

Both expose retain / recall / reflect, mirroring Hindsight's API.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

BANK_ID = "pdac-surveillance-clinic"
BANK_MISSION = (
    "I support a clinical team running pancreatic-cancer surveillance for "
    "high-risk patients. I remember each patient's biomarker history, the "
    "model's past risk calls, and what clinicians confirmed or overruled, and "
    "I use that to spot trends and to learn this clinic's recurring pitfalls "
    "(e.g. confounders that inflate a marker)."
)


class LocalMemory:
    """Offline fallback. Keyword-overlap recall over a JSON file."""

    name = "local-json (offline fallback)"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.items = json.loads(self.path.read_text()) if self.path.exists() else []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, indent=2))

    def retain(self, content: str, tags: list[str], context: str = "") -> None:
        self.items.append(
            {"text": content, "tags": tags, "context": context,
             "at": datetime.now(timezone.utc).isoformat()}
        )
        self._save()

    def recall(self, query: str, tags: list[str] | None = None, limit: int = 20) -> list[str]:
        words = {w.lower().strip(".,?") for w in query.split() if len(w) > 3}
        scored = []
        for it in self.items:
            if tags and not set(tags) & set(it["tags"]):
                continue
            overlap = len(words & {w.lower().strip(".,") for w in it["text"].split()})
            scored.append((overlap, it["at"], it["text"]))
        scored.sort(reverse=True)
        return [t for _, _, t in scored[:limit]]

    def reflect(self, query: str, tags: list[str] | None = None) -> str:
        facts = self.recall(query, tags, limit=8)
        return "Relevant memories:\n- " + "\n- ".join(facts) if facts else ""


class HindsightMemory:
    """Thin wrapper over the Hindsight SDK (sync client)."""

    def __init__(self, base_url: str, api_key: str | None = None):
        from hindsight_client import Hindsight

        self.client = Hindsight(base_url=base_url, api_key=api_key)
        self.name = f"Hindsight ({base_url})"
        try:
            self.client.create_bank(bank_id=BANK_ID, name="PDAC surveillance clinic",
                                    mission=BANK_MISSION)
        except Exception:
            pass  # bank already exists

    def retain(self, content: str, tags: list[str], context: str = "") -> None:
        self.client.retain(bank_id=BANK_ID, content=content, context=context or None, tags=tags)

    def recall(self, query: str, tags: list[str] | None = None, limit: int = 20) -> list[str]:
        resp = self.client.recall(bank_id=BANK_ID, query=query, tags=tags,
                                  tags_match="any", budget="mid")
        return [r.text for r in resp.results[:limit]]

    def reflect(self, query: str, tags: list[str] | None = None) -> str:
        resp = self.client.reflect(bank_id=BANK_ID, query=query, tags=tags,
                                   tags_match="any", budget="low")
        return resp.text


def get_memory(local_path: Path):
    """Hindsight if configured, otherwise the offline fallback."""
    url = os.environ.get("HINDSIGHT_BASE_URL")
    if url:
        return HindsightMemory(url, os.environ.get("HINDSIGHT_API_KEY"))
    return LocalMemory(local_path)
