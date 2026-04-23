from __future__ import annotations

from tom.schema import AttackMetadata, AttackType, InjectionType
from attacks.base import BaseAttack, MessageContext


class BaselineAttack(BaseAttack):
    """
    Control condition — no injection whatsoever.
    Every message passes through unchanged with is_injected=False.
    Used for the baseline scenario in the HCI study (Condition A control).
    """

    attack_type = AttackType.BASELINE

    def inject(
        self,
        message: str,
        context: MessageContext,
    ) -> tuple[str, AttackMetadata]:
        return self._passthrough(message)