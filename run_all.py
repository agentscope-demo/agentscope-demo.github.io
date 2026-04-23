from __future__ import annotations

"""
run_all.py  —  Master script for the MAS-ToM project.

Full matrix: 4 topologies × 4 attacks × 3 agent counts × 4 tasks = 192 scenarios

Each scenario:
  1. Runs the AutoGen simulation  → logs/raw/{id}.raw.json
  2. Runs the ToM annotator       → logs/annotated/{id}.annotated.json
  3. Updates logs/scenarios.index.json

Usage:
    python run_all.py                          # full matrix
    python run_all.py --topology star --attack baseline --n 4 --task research_synthesis
    python run_all.py --skip-existing          # skip completed scenarios
    python run_all.py --no-annotate            # simulation only
    python run_all.py --annotate-only          # annotate existing raw logs
"""

import argparse
import json
import logging
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, "src")

from config.settings import get_settings
from simulation.runner import ScenarioRunner
from tom.annotator import ToMAnnotator
from tom.logger import setup_logging
from tom.schema import (
    AttackType,
    ScenarioMeta,
    ScenariosIndex,
    Topology,
)

logger = logging.getLogger(__name__)


# ── Full matrix ───────────────────────────────────────────────────────────────

TOPOLOGIES   = [Topology.STAR, Topology.RING, Topology.HIERARCHICAL]
ATTACKS      = [AttackType.BASELINE, AttackType.COLLUSION, AttackType.MITM, AttackType.DECEPTION]
AGENT_COUNTS = [4, 8]
TASKS        = ["research_synthesis", "code_review", "qa_pipeline"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAS-ToM scenario runner")
    p.add_argument("--topology",      type=str, default=None,
                   choices=[t.value for t in Topology])
    p.add_argument("--attack",        type=str, default=None,
                   choices=[a.value for a in AttackType])
    p.add_argument("--n",             type=int, default=None)
    p.add_argument("--task",          type=str, default=None,
                   choices=TASKS)
    p.add_argument("--no-annotate",   action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--annotate-only", action="store_true")
    return p.parse_args()


# ── Index helpers ─────────────────────────────────────────────────────────────

def load_index(path: Path) -> ScenariosIndex:
    if path.exists():
        return ScenariosIndex.model_validate_json(path.read_text())
    return ScenariosIndex()


def save_index(index: ScenariosIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(index.model_dump_json(indent=2))


def upsert_meta(index: ScenariosIndex, meta: ScenarioMeta) -> None:
    for i, existing in enumerate(index.scenarios):
        if existing.scenario_id == meta.scenario_id:
            index.scenarios[i] = meta
            return
    index.scenarios.append(meta)


# ── Single scenario ───────────────────────────────────────────────────────────

def run_scenario(
    topology:      Topology,
    attack:        AttackType,
    n_agents:      int,
    task:          str,
    annotate:      bool,
    skip_existing: bool,
    annotator:     ToMAnnotator | None,
    settings,
    index:         ScenariosIndex,
) -> None:
    scenario_id = f"{topology.value}__{attack.value}__{task}__{n_agents}a"
    raw_path    = settings.raw_log_dir       / f"{scenario_id}.raw.json"
    ann_path    = settings.annotated_log_dir / f"{scenario_id}.annotated.json"

    # ── simulation ────────────────────────────────────────────────────────────
    if skip_existing and raw_path.exists():
        logger.info("SKIP simulation (exists): %s", scenario_id)
    else:
        logger.info("═" * 64)
        logger.info("RUNNING: %s", scenario_id)
        logger.info("═" * 64)
        t0 = time.time()
        try:
            runner = ScenarioRunner(
                topology=topology,
                attack_type=attack,
                n_agents=n_agents,
                task=task,
                scenario_id=scenario_id,
            )
            log = runner.run()
            logger.info(
                "Simulation done in %.1fs — %d events",
                time.time() - t0, len(log.events),
            )
        except Exception as exc:
            logger.error("Simulation FAILED %s: %s", scenario_id, exc, exc_info=True)
            return

    # ── annotation ────────────────────────────────────────────────────────────
    if annotate and annotator is not None:
        if skip_existing and ann_path.exists():
            logger.info("SKIP annotation (exists): %s", scenario_id)
        elif not raw_path.exists():
            logger.warning("Cannot annotate — raw log missing: %s", raw_path)
        else:
            t0 = time.time()
            try:
                annotated = annotator.annotate_from_file(raw_path)
                logger.info(
                    "Annotation done in %.1fs — %d events",
                    time.time() - t0, len(annotated.events),
                )
            except Exception as exc:
                logger.error("Annotation FAILED %s: %s", scenario_id, exc, exc_info=True)
                return

    # ── update index ──────────────────────────────────────────────────────────
    if raw_path.exists():
        with open(raw_path) as f:
            raw_data = json.load(f)
        meta = ScenarioMeta(
            scenario_id    = scenario_id,
            topology       = topology,
            attack_type    = attack,
            n_agents       = n_agents,
            task           = task,
            tom_annotated  = ann_path.exists(),
            event_count    = len(raw_data.get("events", [])),
            raw_path       = str(raw_path),
            annotated_path = str(ann_path) if ann_path.exists() else None,
        )
        upsert_meta(index, meta)
        save_index(index, settings.scenarios_index_path)
        logger.info("Index updated: %d scenarios total", len(index.scenarios))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    settings = get_settings()
    settings.ensure_dirs()

    setup_logging(
        level    = settings.log_level,
        log_file = settings.log_dir / "run_all.log",
    )

    logger.info("MAS-ToM run_all starting")
    logger.info("Model: %s | Log dir: %s", settings.model, settings.log_dir)

    topologies   = [Topology(args.topology)]   if args.topology else TOPOLOGIES
    attacks      = [AttackType(args.attack)]   if args.attack   else ATTACKS
    agent_counts = [args.n]                    if args.n        else AGENT_COUNTS
    tasks        = [args.task]                 if args.task     else TASKS

    total = len(topologies) * len(attacks) * len(agent_counts) * len(tasks)
    logger.info(
        "Matrix: %d topologies × %d attacks × %d agent counts × %d tasks = %d scenarios",
        len(topologies), len(attacks), len(agent_counts), len(tasks), total,
    )

    annotator = ToMAnnotator() if not args.no_annotate else None
    index     = load_index(settings.scenarios_index_path)
    completed = 0
    failed    = 0

    # ── annotate-only mode ────────────────────────────────────────────────────
    if args.annotate_only:
        assert annotator, "--annotate-only requires annotation enabled"
        for raw_path in sorted(settings.raw_log_dir.glob("*.raw.json")):
            try:
                annotator.annotate_from_file(raw_path)
                completed += 1
            except Exception as exc:
                logger.error("Failed %s: %s", raw_path.name, exc)
                failed += 1
        logger.info("Annotate-only done: %d ok / %d failed", completed, failed)
        return

    # ── full matrix ───────────────────────────────────────────────────────────
    for i, (topology, attack, n_agents, task) in enumerate(
        product(topologies, attacks, agent_counts, tasks), start=1
    ):
        logger.info("Progress: %d / %d", i, total)
        try:
            run_scenario(
                topology       = topology,
                attack         = attack,
                n_agents       = n_agents,
                task           = task,
                annotate       = not args.no_annotate,
                skip_existing  = args.skip_existing,
                annotator      = annotator,
                settings       = settings,
                index          = index,
            )
            completed += 1
        except Exception as exc:
            logger.error("Scenario failed: %s", exc, exc_info=True)
            failed += 1

    logger.info("═" * 64)
    logger.info(
        "DONE: %d completed / %d failed / %d total",
        completed, failed, total,
    )
    logger.info("Index: %s", settings.scenarios_index_path)


if __name__ == "__main__":
    main()