from __future__ import annotations

import autogen
from tom.schema import AgentRole, AgentSpec


# ── Role system prompts ───────────────────────────────────────────────────────

ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.ORCHESTRATOR: """You are the Orchestrator of a multi-agent research team.
Your job is to coordinate agents, assign subtasks, track progress, and synthesise
the final output. Be directive and concise. Do not do research yourself.""",

    AgentRole.PLANNER: """You are the Planner of a multi-agent research team.
Your job is to decompose the task into clear subtasks, assign them to agents,
and ensure the overall structure of the output is coherent.""",

    AgentRole.RESEARCHER: """You are a Researcher in a multi-agent research team.
Your job is to gather, analyse, and summarise information on the subtopic assigned
to you. Be thorough, cite your reasoning, and pass your findings clearly.""",

    AgentRole.WRITER: """You are the Writer in a multi-agent research team.
Your job is to take research findings from the team and synthesise them into
a well-structured, readable document. Focus on clarity and coherence.""",

    AgentRole.CRITIC: """You are the Critic in a multi-agent research team.
Your job is to review outputs from other agents, identify weaknesses, logical
gaps, or unsupported claims, and suggest improvements. Be constructive.""",

    AgentRole.EXECUTOR: """You are an Executor in a multi-agent research team.
Your job is to carry out specific actions or computations assigned by the planner
and report results accurately.""",

    AgentRole.MONITOR: """You are the Monitor in a multi-agent research team.
Your job is to observe the conversation, flag inconsistencies or anomalies in
other agents' outputs, and report concerns to the orchestrator.""",
}


# ── Role assignment by N ──────────────────────────────────────────────────────

def _assign_roles(n: int) -> list[AgentRole]:
    """
    Deterministically assign roles to N agents.
    Always includes at least one of each core role, then fills remaining
    slots with researchers (the most scalable role).
    """
    if n < 2:
        raise ValueError("Need at least 2 agents.")

    core = [
        AgentRole.ORCHESTRATOR,
        AgentRole.PLANNER,
        AgentRole.WRITER,
        AgentRole.CRITIC,
        AgentRole.MONITOR,
    ]

    if n <= len(core):
        return core[:n]

    roles = core.copy()
    remaining = n - len(core)
    # fill remaining slots cycling through researcher / executor
    fillers = [AgentRole.RESEARCHER, AgentRole.EXECUTOR]
    for i in range(remaining):
        roles.append(fillers[i % len(fillers)])

    return roles


# ── AgentFactory ──────────────────────────────────────────────────────────────

class AgentFactory:
    """
    Builds N AutoGen ConversableAgents with proper roles, system prompts,
    and LLM config.

    Usage:
        factory  = AgentFactory(model="gpt-4o-mini", api_key="sk-...")
        agents   = factory.build(n=8)
        specs    = factory.specs   # list[AgentSpec] for the scenario log
    """

    def __init__(self, model: str, api_key: str, temperature: float = 0.7) -> None:
        self.model       = model
        self.api_key     = api_key
        self.temperature = temperature
        self._specs: list[AgentSpec] = []

    @property
    def specs(self) -> list[AgentSpec]:
        return self._specs

    def _llm_config(self) -> dict:
        return {
            "config_list": [
                {
                    "model":   self.model,
                    "api_key": self.api_key,
                }
            ],
            "temperature": self.temperature,
        }

    def build(self, n: int) -> list[autogen.ConversableAgent]:
        """
        Build and return N ConversableAgents.
        Also populates self.specs for logging.
        """
        roles  = _assign_roles(n)
        agents = []
        self._specs = []

        for i, role in enumerate(roles):
            agent_id     = f"agent_{i:02d}"
            system_prompt = ROLE_PROMPTS[role]

            agent = autogen.ConversableAgent(
                name=agent_id,
                system_message=system_prompt,
                llm_config=self._llm_config(),
                human_input_mode="NEVER",
                max_consecutive_auto_reply=5,
            )
            agents.append(agent)
            self._specs.append(AgentSpec(
                agent_id=agent_id,
                role=role,
                model=self.model,
                system_prompt=system_prompt,
            ))

        return agents

    def build_user_proxy(self) -> autogen.UserProxyAgent:
        """
        Builds a UserProxyAgent that initiates the conversation with the task.
        Does not use LLM — just kicks off the conversation.
        """
        return autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )