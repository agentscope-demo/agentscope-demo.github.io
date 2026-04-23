from __future__ import annotations

from abc import ABC, abstractmethod

import autogen


class BaseTopology(ABC):
    """
    Abstract base for all topology implementations.

    Each topology wires a list of AutoGen agents into a conversation
    pattern and exposes a start() method that kicks off the run.

    The topology is responsible for:
      - setting up GroupChat / nested chat / sequential handoffs
      - returning the manager or initiating agent for the runner to call
    """

    def __init__(
        self,
        agents: list[autogen.ConversableAgent],
        user_proxy: autogen.UserProxyAgent,
        max_turns: int = 30,
    ) -> None:
        self.agents     = agents
        self.user_proxy = user_proxy
        self.max_turns  = max_turns

    @abstractmethod
    def build(self) -> None:
        """Wire agents into the topology. Called once before start()."""
        ...

    @abstractmethod
    def start(self, task: str) -> list[dict]:
        """
        Initiate the conversation with the task prompt.
        Returns the raw AutoGen message history as a list of dicts.
        """
        ...

    def agent_ids(self) -> list[str]:
        return [a.name for a in self.agents]

    def edges(self) -> list[tuple[str, str]]:
        """
        Return the static edge list for this topology.
        Used to populate GraphSnapshot.active_edges.
        Subclasses override for dynamic edge sets.
        """
        return []