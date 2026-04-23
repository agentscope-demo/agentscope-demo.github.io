from __future__ import annotations

import autogen
from simulation.topologies.base import BaseTopology


class HierarchicalTopology(BaseTopology):
    """
    Manager tree — depth 2 for N<=8, depth 3 for N>8.

    Structure:
      - Agent 0: top-level manager (Orchestrator)
      - Agents 1..M: mid-level managers (Planners)
      - Remaining agents: leaf workers partitioned under mid-managers

    Implemented using AutoGen nested GroupChats:
      - Top manager runs a GroupChat with mid-managers
      - Each mid-manager runs a sub-GroupChat with its leaf workers
    """

    def build(self) -> None:
        n = len(self.agents)
        self._top_manager = self.agents[0]

        if n <= 4:
            # flat two-level: 1 manager + rest as direct reports
            self._mid_managers = []
            self._leaf_groups: dict[str, list[autogen.ConversableAgent]] = {
                self._top_manager.name: self.agents[1:]
            }
        else:
            # two mid-managers, split remaining agents between them
            self._mid_managers = self.agents[1:3]
            leaves = self.agents[3:]
            mid = len(leaves) // 2
            self._leaf_groups = {
                self._mid_managers[0].name: leaves[:mid],
                self._mid_managers[1].name: leaves[mid:],
            }

    def start(self, task: str) -> list[dict]:
        all_messages: list[dict] = []

        # ── leaf-level chats ─────────────────────────────────────────────────
        leaf_summaries: dict[str, str] = {}
        for manager_name, leaves in self._leaf_groups.items():
            if not leaves:
                continue
            manager = next(a for a in self.agents if a.name == manager_name)
            sub_chat = autogen.GroupChat(
                agents=[manager] + leaves,
                messages=[],
                max_round=max(4, self.max_turns // 3),
                speaker_selection_method="round_robin",
            )
            sub_mgr = autogen.GroupChatManager(
                groupchat=sub_chat,
                llm_config=manager.llm_config,
            )
            proxy = autogen.UserProxyAgent(
                name=f"proxy_{manager_name}",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
            )
            proxy.initiate_chat(sub_mgr, message=task)
            msgs = sub_chat.messages
            for m in msgs:
                m["_hierarchy_level"] = "leaf"
                m["_manager"] = manager_name
            all_messages.extend(msgs)

            # summarise the leaf group's output for the top manager
            replies = [m["content"] for m in msgs if m.get("role") == "assistant"]
            leaf_summaries[manager_name] = replies[-1] if replies else ""

        # ── top-level chat ───────────────────────────────────────────────────
        summary_input = (
            f"Original task: {task}\n\nSub-team reports:\n"
            + "\n\n".join(
                f"[{mgr}]: {summary}"
                for mgr, summary in leaf_summaries.items()
            )
        )
        top_participants = [self._top_manager] + self._mid_managers
        top_chat = autogen.GroupChat(
            agents=top_participants,
            messages=[],
            max_round=max(4, self.max_turns // 3),
            speaker_selection_method="round_robin",
        )
        top_mgr = autogen.GroupChatManager(
            groupchat=top_chat,
            llm_config=self._top_manager.llm_config,
        )
        top_proxy = autogen.UserProxyAgent(
            name="proxy_top",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )
        top_proxy.initiate_chat(top_mgr, message=summary_input)
        for m in top_chat.messages:
            m["_hierarchy_level"] = "top"
        all_messages.extend(top_chat.messages)

        return all_messages

    def edges(self) -> list[tuple[str, str]]:
        edges = []
        top_id = self._top_manager.name
        for mid in self._mid_managers:
            edges.append((top_id, mid.name))
        for manager_name, leaves in self._leaf_groups.items():
            for leaf in leaves:
                edges.append((manager_name, leaf.name))
        if not self._mid_managers:
            for agent in self.agents[1:]:
                edges.append((top_id, agent.name))
        return edges