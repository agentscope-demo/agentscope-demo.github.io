from __future__ import annotations

import autogen
from simulation.topologies.base import BaseTopology


class FullyConnectedTopology(BaseTopology):
    """
    All agents participate in a single GroupChat.
    Every agent can speak to every other agent.
    AutoGen's GroupChat manages turn-taking via its speaker_selection_method.
    """

    def build(self) -> None:
        self._group_chat = autogen.GroupChat(
            agents=self.agents,
            messages=[],
            max_round=self.max_turns,
            speaker_selection_method="auto",
            allow_repeat_speaker=True,
        )
        self._manager = autogen.GroupChatManager(
            groupchat=self._group_chat,
            llm_config=self.agents[0].llm_config,
        )

    def start(self, task: str) -> list[dict]:
        self.user_proxy.initiate_chat(
            self._manager,
            message=task,
        )
        return self._group_chat.messages

    def edges(self) -> list[tuple[str, str]]:
        ids = self.agent_ids()
        return [
            (a, b)
            for i, a in enumerate(ids)
            for b in ids[i + 1:]
        ]