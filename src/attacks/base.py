from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tom.schema import AttackMetadata, AttackType, InjectionType


@dataclass
class MessageContext:
    """
    Everything the attack middleware knows about the current message.
    Passed to inject() so each attack can make context-aware decisions.
    """
    t:             int                  # current timestep
    sender:        str                  # sending agent id
    receiver:      str                  # receiving agent id
    all_agents:    list[str]            = field(default_factory=list)
    topology:      str                  = "unknown"
    prior_events:  list[dict[str, Any]] = field(default_factory=list)  # last N events


class BaseAttack(ABC):
    """
    Abstract base for all attack implementations.

    Each attack must implement:
      - attack_type  : AttackType enum value
      - inject()     : given a message + context, return (modified_message, metadata)

    The runner calls inject() on every message. For baseline this is a passthrough.
    The returned AttackMetadata is written directly into the Event.attack field.
    """

    attack_type: AttackType

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}

    @abstractmethod
    def inject(
        self,
        message: str,
        context: MessageContext,
    ) -> tuple[str, AttackMetadata]:
        """
        Core injection method.

        Args:
            message:  the original message content
            context:  runtime context (timestep, agents, topology, history)

        Returns:
            (modified_message, metadata)
            - modified_message: what the receiver actually sees
            - metadata:         AttackMetadata written into the event log
        """
        ...

    def _passthrough(self, message: str) -> tuple[str, AttackMetadata]:
        """Helper for no-op cases — returns message unchanged, is_injected=False."""
        return message, AttackMetadata(
            is_injected=False,
            injection_type=InjectionType.NONE,
        )

    def should_inject(self, context: MessageContext) -> bool:
        """
        Gate: only inject after inject_from_turn (configurable per attack).
        Subclasses can override for more complex logic.
        """
        inject_from = self.config.get("inject_from_turn", 0)
        return context.t >= inject_from

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"