from __future__ import annotations

import autogen
from simulation.topologies.base import BaseTopology


class StarTopology(BaseTopology):
    """
    Hub-and-spoke topology.
    Agent 0 (Orchestrator) is the hub. All communication goes through it.
    Implemented as a GroupChat where only the hub can initiate exchanges,
    using round_robin with orchestrator always included.
    """

    def build(self) -> None:
        self._hub = self.agents[0]   # orchestrator
        self._spokes = self.agents[1:]

        self._group_chat = autogen.GroupChat(
            agents=self.agents,
            messages=[],
            max_round=self.max_turns,
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False,
        )
        self._manager = autogen.GroupChatManager(
            groupchat=self._group_chat,
            llm_config=self._hub.llm_config,
        )

    def start(self, task: str) -> list[dict]:
        self.user_proxy.initiate_chat(
            self._manager,
            message=task,
        )
        return self._group_chat.messages

    def edges(self) -> list[tuple[str, str]]:
        hub_id = self._hub.name
        return [(hub_id, spoke.name) for spoke in self._spokes]