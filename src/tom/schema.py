from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class Topology(str, Enum):
    FULLY_CONNECTED = "fully_connected"
    STAR            = "star"
    RING            = "ring"
    HIERARCHICAL    = "hierarchical"


class AttackType(str, Enum):
    BASELINE  = "baseline"
    COLLUSION = "collusion"
    MITM      = "mitm"
    DECEPTION = "deception"


class AgentRole(str, Enum):
    PLANNER      = "planner"
    RESEARCHER   = "researcher"
    WRITER       = "writer"
    CRITIC       = "critic"
    EXECUTOR     = "executor"
    MONITOR      = "monitor"
    ORCHESTRATOR = "orchestrator"


class InjectionType(str, Enum):
    COLLUSION_TOKEN = "collusion_token"
    MITM_REWRITE    = "mitm_rewrite"
    DECEPTION       = "deception"
    NONE            = "none"


# ── ToM sub-models ────────────────────────────────────────────────────────────

class BeliefAboutAgent(BaseModel):
    """What the sender believes about one other agent at this timestep."""
    target_agent:  str
    inferred_goal: str
    trust_score:   float = Field(..., ge=0.0, le=1.0)
    confidence:    float = Field(..., ge=0.0, le=1.0)
    notes:         str | None = None


class ToMAnnotation(BaseModel):
    """Full ToM annotation for one event — written post-hoc by tom/annotator.py."""
    beliefs:            list[BeliefAboutAgent] = Field(default_factory=list)
    anomaly_score:      float                  = Field(0.0, ge=0.0, le=1.0)
    anomaly_flags:      list[str]              = Field(default_factory=list)
    narrative:          str                    = ""
    tom_prompt_version: str                    = "v1"
    annotated_at:       str | None             = None


# ── Attack sub-model ──────────────────────────────────────────────────────────

class AttackMetadata(BaseModel):
    """Written in real-time by the attack middleware during simulation."""
    is_injected:      bool          = False
    injection_type:   InjectionType = InjectionType.NONE
    original_message: str | None   = None
    attack_targets:   list[str]    = Field(default_factory=list)


# ── Graph sub-model ───────────────────────────────────────────────────────────

class GraphSnapshot(BaseModel):
    """Topology state at this timestep — used by the dashboard graph view."""
    active_edges:      list[tuple[str, str]]    = Field(default_factory=list)
    agent_roles:       dict[str, AgentRole]     = Field(default_factory=dict)
    topology_snapshot: dict[str, list[str]]     = Field(default_factory=dict)


# ── Core event ────────────────────────────────────────────────────────────────

class Event(BaseModel):
    """
    One agent turn — the atomic unit of every log file.

    Three layers written by three different modules:
      - core fields     → simulation/logger.py   (real-time)
      - attack          → attacks/*.py            (real-time)
      - tom             → tom/annotator.py        (post-hoc)
    """
    t:          int
    event_id:   str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender:     str
    receiver:   str   # agent_id | "broadcast" | "orchestrator"
    message:    str
    reasoning:  str | None              = None
    tool_calls: list[dict[str, Any]]   = Field(default_factory=list)
    timestamp:  str | None             = None

    # populated by attack middleware
    attack: AttackMetadata = Field(default_factory=AttackMetadata)

    # populated by tom/annotator.py
    tom: ToMAnnotation = Field(default_factory=ToMAnnotation)

    # populated by simulation/logger.py
    graph: GraphSnapshot = Field(default_factory=GraphSnapshot)


# ── Agent spec ────────────────────────────────────────────────────────────────

class AgentSpec(BaseModel):
    """Static description of one agent — stored in the scenario envelope."""
    agent_id:      str
    role:          AgentRole
    model:         str = "gpt-4o-mini"
    system_prompt: str = ""


# ── Scenario envelope ─────────────────────────────────────────────────────────

class ScenarioLog(BaseModel):
    """
    One complete run — serialised to:
      logs/raw/{scenario_id}.raw.json           after simulation
      logs/annotated/{scenario_id}.annotated.json  after tom annotation
    """
    scenario_id:      str
    topology:         Topology
    attack_type:      AttackType
    n_agents:         int
    task:             str
    agent_specs:      list[AgentSpec] = Field(default_factory=list)
    events:           list[Event]     = Field(default_factory=list)
    tom_annotated:    bool            = False
    run_started_at:   str | None      = None
    run_finished_at:  str | None      = None


# ── Scenarios index ───────────────────────────────────────────────────────────

class ScenarioMeta(BaseModel):
    """
    Lightweight entry in logs/scenarios.index.json.
    The API loads this index at startup — no events, just metadata.
    """
    scenario_id:    str
    topology:       Topology
    attack_type:    AttackType
    n_agents:       int
    task:           str
    tom_annotated:  bool
    event_count:    int
    raw_path:       str
    annotated_path: str | None = None


class ScenariosIndex(BaseModel):
    scenarios: list[ScenarioMeta] = Field(default_factory=list)