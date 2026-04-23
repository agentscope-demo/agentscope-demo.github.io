from __future__ import annotations

from typing import Type

from attacks.base import BaseAttack
from attacks.baseline import BaselineAttack
from attacks.collusion import CollusionAttack
from attacks.mitm import MITMAttack
from attacks.deception import DeceptionAttack
from tom.schema import AttackType


# ── Registry ──────────────────────────────────────────────────────────────────
# Single source of truth — add new attacks here and they're available everywhere.

ATTACK_REGISTRY: dict[AttackType, Type[BaseAttack]] = {
    AttackType.BASELINE:  BaselineAttack,
    AttackType.COLLUSION: CollusionAttack,
    AttackType.MITM:      MITMAttack,
    AttackType.DECEPTION: DeceptionAttack,
}


def get_attack(attack_type: AttackType, config: dict | None = None) -> BaseAttack:
    """
    Factory function — returns a configured attack instance.

    Usage:
        attack = get_attack(AttackType.COLLUSION, {"colluding_agents_fraction": 0.25})
    """
    cls = ATTACK_REGISTRY.get(attack_type)
    if cls is None:
        raise ValueError(f"Unknown attack type: {attack_type!r}. "
                         f"Available: {list(ATTACK_REGISTRY.keys())}")
    return cls(config=config)


__all__ = [
    "BaseAttack",
    "BaselineAttack",
    "CollusionAttack",
    "MITMAttack",
    "DeceptionAttack",
    "ATTACK_REGISTRY",
    "get_attack",
]