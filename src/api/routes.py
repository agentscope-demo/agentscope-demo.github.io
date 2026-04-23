from __future__ import annotations

import json
import logging
from flask import Blueprint, jsonify, request
from config.settings import get_settings
from tom.schema import ScenariosIndex

bp     = Blueprint("api", __name__, url_prefix="/api")
logger = logging.getLogger(__name__)


@bp.route("/scenarios", methods=["GET"])
def list_scenarios():
    s = get_settings()
    if not s.scenarios_index_path.exists():
        return jsonify({"scenarios": []})
    index = ScenariosIndex.model_validate_json(
        s.scenarios_index_path.read_text()
    )
    return jsonify(index.model_dump())


@bp.route("/scenario/<scenario_id>", methods=["GET"])
def get_scenario(scenario_id: str):
    s      = get_settings()
    mode   = request.args.get("mode", "annotated")
    log_dir = s.annotated_log_dir if mode == "annotated" else s.raw_log_dir
    suffix  = ".annotated.json"   if mode == "annotated" else ".raw.json"
    path    = log_dir / f"{scenario_id}{suffix}"
    if not path.exists():
        path = s.raw_log_dir / f"{scenario_id}.raw.json"
        if not path.exists():
            return jsonify({"error": f"Not found: {scenario_id}"}), 404
    with open(path) as f:
        data = json.load(f)
    return jsonify(data)


@bp.route("/scenario/<scenario_id>/events", methods=["GET"])
def get_events(scenario_id: str):
    s      = get_settings()
    mode   = request.args.get("mode", "annotated")
    start  = int(request.args.get("start", 0))
    limit  = int(request.args.get("limit", 50))
    log_dir = s.annotated_log_dir if mode == "annotated" else s.raw_log_dir
    suffix  = ".annotated.json"   if mode == "annotated" else ".raw.json"
    path    = log_dir / f"{scenario_id}{suffix}"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    with open(path) as f:
        data = json.load(f)
    events = data.get("events", [])
    return jsonify({
        "scenario_id": scenario_id,
        "total":       len(events),
        "start":       start,
        "limit":       limit,
        "events":      events[start: start + limit],
    })


@bp.route("/query", methods=["POST"])
def query():
    body        = request.get_json()
    scenario_id = body.get("scenario_id")
    event_t     = body.get("event_t", 0)
    question    = body.get("question", "")
    if not scenario_id or not question:
        return jsonify({"error": "scenario_id and question required"}), 400
    from api.query_engine import answer_query
    try:
        ans = answer_query(scenario_id, event_t, question)
        return jsonify({"answer": ans})
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
