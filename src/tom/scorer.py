from __future__ import annotations

import math
from tom.schema import Event, ToMAnnotation


def compute_anomaly_score(
    current: ToMAnnotation,
    prior: ToMAnnotation | None,
) -> float:
    """
    Blend the LLM-assigned anomaly score with a signal derived from
    trust score deltas between consecutive timesteps.

    This gives a more robust score that:
      - stays close to the LLM score when there is no prior context
      - amplifies the score when trust changes sharply
      - dampens it when beliefs are stable

    Returns a float in [0.0, 1.0].
    """
    llm_score = current.anomaly_score

    if prior is None or not prior.beliefs or not current.beliefs:
        return llm_score

    # build lookup: target_agent → trust_score
    prior_trust  = {b.target_agent: b.trust_score for b in prior.beliefs}
    current_trust = {b.target_agent: b.trust_score for b in current.beliefs}

    deltas = []
    for agent, trust in current_trust.items():
        if agent in prior_trust:
            deltas.append(abs(trust - prior_trust[agent]))

    if not deltas:
        return llm_score

    # mean absolute trust delta — large swings amplify the anomaly score
    mean_delta = sum(deltas) / len(deltas)

    # sigmoid-blend: weight LLM score 70%, trust delta signal 30%
    delta_signal = 1.0 - math.exp(-3.0 * mean_delta)   # 0→0, 0.33→0.63, 1→0.95
    blended = 0.70 * llm_score + 0.30 * delta_signal

    return round(min(1.0, max(0.0, blended)), 4)


def enrich_flags(events: list[Event]) -> list[Event]:
    """
    Post-processing pass: add rule-based anomaly flags that the LLM may miss.
    Operates on the full event list after annotation.
    """
    # build per-agent trust history
    trust_history: dict[str, list[float]] = {}

    for event in events:
        for belief in event.tom.beliefs:
            key = f"{event.sender}->{belief.target_agent}"
            trust_history.setdefault(key, []).append(belief.trust_score)

    # flag sharp trust collapses (>0.3 drop in one step)
    for event in events:
        for belief in event.tom.beliefs:
            key = f"{event.sender}->{belief.target_agent}"
            history = trust_history.get(key, [])
            idx = len([
                e for e in events[:events.index(event)]
                if any(b.target_agent == belief.target_agent for b in e.tom.beliefs)
            ])
            if idx > 0 and idx < len(history):
                delta = history[idx - 1] - history[idx]
                if delta > 0.30 and "trust_collapse" not in event.tom.anomaly_flags:
                    event.tom.anomaly_flags.append("trust_collapse")
                if delta < -0.30 and "trust_inflation" not in event.tom.anomaly_flags:
                    event.tom.anomaly_flags.append("trust_inflation")

    return events