from __future__ import annotations

import json
import logging
from openai import OpenAI
from config.settings import get_settings

logger = logging.getLogger(__name__)


def answer_query(scenario_id: str, event_t: int, question: str) -> str:
    settings = get_settings()
    client   = OpenAI(api_key=settings.openai_api_key)
    path = settings.annotated_log_dir / f"{scenario_id}.annotated.json"
    if not path.exists():
        path = settings.raw_log_dir / f"{scenario_id}.raw.json"
    if not path.exists():
        raise FileNotFoundError(f"No log found for {scenario_id}")
    with open(path) as f:
        data = json.load(f)
    events = data.get("events", [])
    window = [e for e in events if abs(e["t"] - event_t) <= 5]
    context = json.dumps([{
        "t":             e["t"],
        "sender":        e["sender"],
        "receiver":      e["receiver"],
        "message":       e["message"][:300],
        "anomaly_score": e.get("tom", {}).get("anomaly_score", 0),
        "narrative":     e.get("tom", {}).get("narrative", ""),
        "beliefs":       e.get("tom", {}).get("beliefs", []),
    } for e in window], indent=2)
    response = client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": (
                "You are an expert analyst for a multi-agent AI security dashboard. "
                "Help users understand what happened in agent conversations and why. "
                "Be concise and ground answers in the data provided."
            )},
            {"role": "user", "content": (
                f"Scenario: {scenario_id}\n"
                f"Attack type: {data.get('attack_type', 'unknown')}\n"
                f"Topology: {data.get('topology', 'unknown')}\n\n"
                f"Question about t={event_t}: {question}\n\n"
                f"Context:\n{context}"
            )},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content or "No answer generated."
