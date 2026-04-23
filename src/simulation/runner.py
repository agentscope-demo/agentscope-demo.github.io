from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attacks.base import BaseAttack, MessageContext
from attacks import get_attack
from simulation.agents import AgentFactory
from simulation.topologies import get_topology
from tom.schema import (
    AttackType,
    Event,
    GraphSnapshot,
    ScenarioLog,
    Topology,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """
    Orchestrates a single scenario run end-to-end:

      1. Build N agents via AgentFactory
      2. Wire into the requested topology
      3. Register the attack middleware
      4. Monkey-patch AutoGen's message flow to capture + inject per event
      5. Run the conversation
      6. Write ScenarioLog to logs/raw/{scenario_id}.raw.json

    Usage:
        runner = ScenarioRunner(
            topology=Topology.STAR,
            attack_type=AttackType.COLLUSION,
            n_agents=8,
            task="research_synthesis",
        )
        log = runner.run()
    """

    def __init__(
        self,
        topology: Topology,
        attack_type: AttackType,
        n_agents: int,
        task: str,
        attack_config: dict | None = None,
        scenario_id: str | None = None,
    ) -> None:
        self.settings     = get_settings()
        self.topology     = topology
        self.attack_type  = attack_type
        self.n_agents     = n_agents
        self.task         = task
        self.attack_config = attack_config or {}

        self.scenario_id  = scenario_id or (
            f"{topology.value}__{attack_type.value}__{task}__{n_agents}a"
        )

        self._events: list[Event]  = []
        self._t: int               = 0
        self._attack: BaseAttack | None = None

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> ScenarioLog:
        logger.info(
            "Starting scenario: %s  (topology=%s, attack=%s, n=%d)",
            self.scenario_id, self.topology.value,
            self.attack_type.value, self.n_agents,
        )

        # 1. Build agents
        factory = AgentFactory(
            model=self.settings.model,
            api_key=self.settings.openai_api_key,
            temperature=self.settings.temperature,
        )
        agents     = factory.build(self.n_agents)
        user_proxy = factory.build_user_proxy()
        specs      = factory.specs

        # 2. Build attack middleware
        self._attack = get_attack(self.attack_type, self.attack_config)
        if hasattr(self._attack, "register_agents"):
            self._attack.register_agents([a.name for a in agents])

        # 3. Wire topology
        topo = get_topology(
            topology=self.topology,
            agents=agents,
            user_proxy=user_proxy,
            max_turns=self.settings.max_turns,
        )
        static_edges = topo.edges()

        # 4. Patch agents to intercept messages
        self._patch_agents(agents, static_edges)

        # 5. Resolve task prompt
        task_prompt = self._resolve_task(self.task)

        # 6. Run
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            topo.start(task_prompt)
        except Exception as exc:
            logger.error("Run failed: %s", exc, exc_info=True)
            raise
        finished_at = datetime.now(timezone.utc).isoformat()

        # 7. Build and save ScenarioLog
        scenario_log = ScenarioLog(
            scenario_id=self.scenario_id,
            topology=self.topology,
            attack_type=self.attack_type,
            n_agents=self.n_agents,
            task=self.task,
            agent_specs=specs,
            events=self._events,
            tom_annotated=False,
            run_started_at=started_at,
            run_finished_at=finished_at,
        )
        self._save_raw(scenario_log)

        logger.info(
            "Scenario complete: %s  (%d events captured)",
            self.scenario_id, len(self._events),
        )
        return scenario_log

    # ── Message interception ──────────────────────────────────────────────────

    def _patch_agents(
        self,
        agents,
        static_edges: list[tuple[str, str]],
    ) -> None:
        """
        Monkey-patch each agent's generate_reply to intercept every message,
        run it through the attack middleware, and log an Event.
        """
        runner_ref = self   # capture for closure

        for agent in agents:
            original_generate_reply = agent.generate_reply

            def make_patched(ag, orig_fn):
                def patched_generate_reply(messages=None, sender=None, **kwargs):
                    # call the real LLM first
                    reply = orig_fn(messages=messages, sender=sender, **kwargs)

                    if reply is None:
                        return reply

                    sender_name   = sender.name if sender else "unknown"
                    receiver_name = ag.name
                    content       = reply if isinstance(reply, str) else str(reply)

                    # build context for attack middleware
                    context = MessageContext(
                        t=runner_ref._t,
                        sender=sender_name,
                        receiver=receiver_name,
                        all_agents=[a.name for a in agents],
                        topology=runner_ref.topology.value,
                        prior_events=[
                            e.model_dump()
                            for e in runner_ref._events[-10:]
                        ],
                    )

                    # run attack injection
                    injected_content, attack_meta = runner_ref._attack.inject(
                        content, context
                    )

                    # build graph snapshot
                    graph = GraphSnapshot(
                        active_edges=static_edges,
                        agent_roles={
                            spec.agent_id: spec.role
                            for spec in runner_ref._get_specs(agents)
                        },
                        topology_snapshot=runner_ref._adjacency(agents, static_edges),
                    )

                    # log the event
                    event = Event(
                        t=runner_ref._t,
                        event_id=str(uuid.uuid4()),
                        sender=sender_name,
                        receiver=receiver_name,
                        message=injected_content,
                        reasoning=None,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        attack=attack_meta,
                        graph=graph,
                    )
                    runner_ref._events.append(event)
                    runner_ref._t += 1

                    # return the (possibly modified) content
                    return injected_content

                return patched_generate_reply

            agent.generate_reply = make_patched(agent, original_generate_reply)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_task(self, task_name: str) -> str:
        """Load task prompt from scenarios.yaml or fall back to task_name as raw prompt."""
        import yaml
        yaml_path = Path(__file__).parent.parent / "config" / "scenarios.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            for t in cfg.get("tasks", []):
                if t["name"] == task_name and t.get("enabled", True):
                    return t["prompt"]
        # fallback — use task_name as the literal prompt
        return task_name

    def _get_specs(self, agents) -> list:
        """Lazy re-fetch of specs — safe to call during patched callbacks."""
        from tom.schema import AgentRole, AgentSpec
        specs = []
        for a in agents:
            role_map = {
                "orchestrator": AgentRole.ORCHESTRATOR,
                "planner":      AgentRole.PLANNER,
                "writer":       AgentRole.WRITER,
                "critic":       AgentRole.CRITIC,
                "monitor":      AgentRole.MONITOR,
                "researcher":   AgentRole.RESEARCHER,
                "executor":     AgentRole.EXECUTOR,
            }
            role_hint = a.system_message.split("\n")[0].lower()
            role = next(
                (r for k, r in role_map.items() if k in role_hint),
                AgentRole.RESEARCHER,
            )
            specs.append(AgentSpec(agent_id=a.name, role=role, model=self.settings.model))
        return specs

    def _adjacency(
        self,
        agents,
        edges: list[tuple[str, str]],
    ) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = {a.name: [] for a in agents}
        for src, dst in edges:
            if src in adj:
                adj[src].append(dst)
        return adj

    def _save_raw(self, log: ScenarioLog) -> None:
        self.settings.ensure_dirs()
        out_path = self.settings.raw_log_dir / f"{log.scenario_id}.raw.json"
        with open(out_path, "w") as f:
            f.write(log.model_dump_json(indent=2))
        logger.info("Raw log saved: %s", out_path)