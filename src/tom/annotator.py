from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from config.settings import get_settings
from tom.prompts import get_prompts
from tom.schema import (
    BeliefAboutAgent,
    ScenarioLog,
    ToMAnnotation,
)
from tom.scorer import compute_anomaly_score, enrich_flags

logger = logging.getLogger(__name__)


class ToMAnnotator:
    """
    Post-hoc ToM annotator.

    Reads a raw ScenarioLog, calls the LLM once per event to infer
    belief states, merges the ToM layer back into each Event, then
    writes the annotated log to logs/annotated/.

    Usage:
        annotator = ToMAnnotator()
        annotated = annotator.annotate(raw_log)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client   = OpenAI(api_key=self.settings.openai_api_key)
        self.prompts  = get_prompts(self.settings.tom_prompt_version)

    # ── Public ────────────────────────────────────────────────────────────────

    def annotate(self, log: ScenarioLog) -> ScenarioLog:
        logger.info(
            "Annotating %s  (%d events)", log.scenario_id, len(log.events)
        )

        prior_annotation: ToMAnnotation | None = None

        for i, event in enumerate(tqdm(log.events, desc=log.scenario_id)):
            # context window: last N events before this one
            window_start = max(0, i - self.settings.tom_context_window)
            window_events = log.events[window_start:i]

            tom = self._annotate_event(
                event=event,
                window_events=window_events,
                log=log,
                prior_annotation=prior_annotation,
            )
            event.tom = tom
            prior_annotation = tom

        # rule-based enrichment pass over the full annotated list
        log.events = enrich_flags(log.events)
        log.tom_annotated = True

        self._save_annotated(log)
        logger.info("Annotation complete: %s", log.scenario_id)
        return log

    def annotate_from_file(self, raw_path: Path) -> ScenarioLog:
        """Convenience: load raw JSON, annotate, return annotated log."""
        with open(raw_path) as f:
            data = json.load(f)
        log = ScenarioLog.model_validate(data)
        return self.annotate(log)

    # ── Private ───────────────────────────────────────────────────────────────

    def _annotate_event(
        self,
        event,
        window_events,
        log: ScenarioLog,
        prior_annotation: ToMAnnotation | None,
    ) -> ToMAnnotation:
        other_agents = [
            s.agent_id for s in log.agent_specs
            if s.agent_id != event.sender
        ]

        # serialise window for the prompt
        window_json = json.dumps(
            [
                {
                    "t":        e.t,
                    "sender":   e.sender,
                    "receiver": e.receiver,
                    "message":  e.message[:500],   # truncate long messages
                }
                for e in window_events
            ],
            indent=2,
        )
        current_json = json.dumps(
            {
                "t":        event.t,
                "sender":   event.sender,
                "receiver": event.receiver,
                "message":  event.message[:500],
            },
            indent=2,
        )

        user_prompt = self.prompts["user"].format(
            sender_id=event.sender,
            other_agents=", ".join(other_agents),
            topology=log.topology.value,
            attack_type=log.attack_type.value,
            window_size=self.settings.tom_context_window,
            events_json=window_json,
            current_event_json=current_json,
        )

        raw_response = self._call_llm(user_prompt)
        tom = self._parse_response(raw_response, event.sender, other_agents)

        # blend LLM score with trust-delta signal
        tom.anomaly_score = compute_anomaly_score(tom, prior_annotation)
        tom.tom_prompt_version = self.settings.tom_prompt_version
        tom.annotated_at = datetime.now(timezone.utc).isoformat()

        return tom

    def _call_llm(self, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.tom_model,
                messages=[
                    {"role": "system", "content": self.prompts["system"]},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,    # low temp for consistent structured output
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or "{}"
        except Exception as exc:
            logger.warning("LLM call failed: %s — using empty annotation", exc)
            return "{}"

    def _parse_response(
        self,
        raw: str,
        sender: str,
        other_agents: list[str],
    ) -> ToMAnnotation:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response")
            return ToMAnnotation()

        beliefs = []
        for b in data.get("beliefs", []):
            try:
                beliefs.append(BeliefAboutAgent(
                    target_agent  = b.get("target_agent", "unknown"),
                    inferred_goal = b.get("inferred_goal", ""),
                    trust_score   = float(b.get("trust_score", 0.5)),
                    confidence    = float(b.get("confidence", 0.5)),
                    notes         = b.get("notes"),
                ))
            except Exception:
                continue

        # ensure every other agent has at least a default belief entry
        covered = {b.target_agent for b in beliefs}
        for agent in other_agents:
            if agent not in covered:
                beliefs.append(BeliefAboutAgent(
                    target_agent  = agent,
                    inferred_goal = "unknown — insufficient context",
                    trust_score   = 0.5,
                    confidence    = 0.1,
                ))

        return ToMAnnotation(
            beliefs        = beliefs,
            anomaly_score  = float(data.get("anomaly_score", 0.0)),
            anomaly_flags  = data.get("anomaly_flags", []),
            narrative      = data.get("narrative", ""),
        )

    def _save_annotated(self, log: ScenarioLog) -> None:
        self.settings.ensure_dirs()
        out_path = (
            self.settings.annotated_log_dir
            / f"{log.scenario_id}.annotated.json"
        )
        with open(out_path, "w") as f:
            f.write(log.model_dump_json(indent=2))
        logger.info("Annotated log saved: %s", out_path)