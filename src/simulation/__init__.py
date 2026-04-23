from __future__ import annotations

from typing import Type

from simulation.topologies.base import BaseTopology
from simulation.topologies.fully_connected import FullyConnectedTopology
from simulation.topologies.star import StarTopology
from simulation.topologies.ring import RingTopology
from simulation.topologies.hierarchical import HierarchicalTopology
from tom.schema import Topology


TOPOLOGY_REGISTRY: dict[Topology, Type[BaseTopology]] = {
    Topology.FULLY_CONNECTED: FullyConnectedTopology,
    Topology.STAR:            StarTopology,
    Topology.RING:            RingTopology,
    Topology.HIERARCHICAL:    HierarchicalTopology,
}


def get_topology(
    topology: Topology,
    agents,
    user_proxy,
    max_turns: int = 30,
) -> BaseTopology:
    cls = TOPOLOGY_REGISTRY.get(topology)
    if cls is None:
        raise ValueError(f"Unknown topology: {topology!r}")
    instance = cls(agents=agents, user_proxy=user_proxy, max_turns=max_turns)
    instance.build()
    return instance


__all__ = [
    "BaseTopology",
    "FullyConnectedTopology",
    "StarTopology",
    "RingTopology",
    "HierarchicalTopology",
    "TOPOLOGY_REGISTRY",
    "get_topology",
]