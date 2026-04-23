from __future__ import annotations

import autogen
from simulation.topologies.base import BaseTopology


class RingTopology(BaseTopology):
    """
    Sequential pipeline — each agent hands off to the next.
    Agent N-1 wraps back to Agent 0 to close the ring.
    Implemented as a series of two-agent chats chained together.
    Each agent receives the prior agent's output as its input.
    """

    def build(self) -> None:
        # pre-build the ordered chain — ring closes at the end
        self._chain: list[autogen.ConversableAgent] = self.agents[:]

    def start(self, task: str) -> list[dict]:
        all_messages: list[dict] = []
        current_message = task

        for i, agent in enumerate(self._chain):
            next_agent = self._chain[(i + 1) % len(self._chain)]

            # each hop is a fresh two-agent chat
            proxy = autogen.UserProxyAgent(
                name=f"relay_{i:02d}",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
            )
            result = proxy.initiate_chat(
                agent,
                message=current_message,
                max_turns=2,
            )
            hop_messages = result.chat_history if hasattr(result, "chat_history") else []
            for m in hop_messages:
                m["_topology_hop"] = i
                m["_next_agent"]   = next_agent.name
            all_messages.extend(hop_messages)

            # pass this agent's last reply as input to the next
            replies = [m for m in hop_messages if m.get("role") == "assistant"]
            if replies:
                current_message = replies[-1]["content"]

        return all_messages

    def edges(self) -> list[tuple[str, str]]:
        ids = self.agent_ids()
        return [(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]