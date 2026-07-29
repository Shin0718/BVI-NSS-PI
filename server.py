"""Serve the local BVI-SAS web interface and simulation API."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import random
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean, stdev
from urllib.parse import unquote, urlparse

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
ENGINE_DIR = BASE_DIR / "BVI-SAS"
MPL_CACHE_DIR = BASE_DIR / ".matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
MPL_CACHE_DIR.mkdir(exist_ok=True)


def _load_engine_cli():
    """Load the BVI-SAS command-line module from the hyphenated source folder."""
    if str(ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_DIR))

    spec = importlib.util.spec_from_file_location("bvisas_cli", ENGINE_DIR / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load BVI-SAS/main.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE_CLI = None
SIMULATION_LOCK = threading.Lock()


DEFAULT_FUNCTION_PARAMS = {
    "obstacle": {
        "recognition_rate": 0.90,
        "small_obstacle_rate": 0.40,
        "category_trigger_prob": 0.40,
        "false_alarm_rate": 0.03,
        "miss_rate": None,
        "action_duration": None,
        "preparation_duration": None,
        "feedback": {"modality": "auditory", "duration": 0.2, "frequency": 1.0, "competition": 1.0},
    },
    "terrain": {
        "recognition_rate": 0.75,
        "category_trigger_prob": 0.05,
        "false_alarm_rate": 0.05,
        "miss_rate": None,
        "action_duration": None,
        "preparation_duration": None,
        "feedback": {"modality": "auditory", "duration": 0.2, "frequency": 1.0, "competition": 1.0},
    },
    "pedestrian": {
        "recognition_rate": 0.85,
        "category_trigger_prob": None,
        "false_alarm_rate": 0.04,
        "miss_rate": None,
        "action_duration": None,
        "preparation_duration": None,
        "feedback": {"modality": "auditory", "duration": 0.2, "frequency": 1.0, "competition": 1.15},
    },
    "vehicle": {
        "recognition_rate": 0.88,
        "category_trigger_prob": None,
        "false_alarm_rate": 0.03,
        "miss_rate": None,
        "action_duration": None,
        "preparation_duration": None,
        "feedback": {"modality": "auditory", "duration": 0.2, "frequency": 1.0, "competition": 1.2},
    },
    "guidance": {
        "recognition_rate": 0.82,
        "category_trigger_prob": None,
        "false_alarm_rate": 0.03,
        "miss_rate": None,
        "action_duration": None,
        "preparation_duration": None,
        "targets": ["curb", "wall", "railing", "tactile_guidance", "landmark"],
        "feedback": {"modality": "auditory", "duration": 0.2, "frequency": 1.0, "competition": 0.85},
    },
    "other": {
        "recognition_rate": 0.72,
        "category_trigger_prob": None,
        "false_alarm_rate": 0.05,
        "miss_rate": None,
        "action_duration": None,
        "preparation_duration": None,
        "feedback": {"modality": "auditory", "duration": 0.2, "frequency": 1.0, "competition": 1.0},
    },
}

SLOT_SCOPE_FALLBACK = {
    "start-road": "segment",
    "mid-road": "intersection",
    "turn-crossing": "segment",
    "end-crossing": "segment",
}


def _bounded_int(value, default, low, high):
    """Convert a value to an integer constrained to a small local-run range."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _safe_stdev(values):
    """Return standard deviation for two or more values, otherwise zero."""
    return stdev(values) if len(values) >= 2 else 0.0


def _risk_label(risk_mean):
    """Translate model risk scores into UI labels."""
    if risk_mean >= 0.55:
        return "High"
    if risk_mean >= 0.28:
        return "Moderate"
    return "Low"


def _bounded_float(value, default, low=0.0, high=1.0):
    """Convert a payload value to a bounded float."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _designer_level_factor(value, *, low=0.65, medium=1.0, high=1.45):
    text = str(value or "").strip().lower()
    if text in {"low", "none", "unfamiliar"}:
        return low
    if text in {"high", "continuous", "familiar"}:
        return high
    return medium


def _device_trust_value(payload):
    user_profile = payload.get("user_profile", {})
    raw = user_profile.get("device_trust", 0.65)
    if isinstance(raw, (int, float)):
        return _bounded_float(raw, 0.65)
    return {"low": 0.35, "medium": 0.65, "high": 0.85}.get(
        str(raw).strip().lower(), 0.65
    )


def _normalize_function_key(key):
    key = str(key or "").strip().lower()
    aliases = {
        "obstacle_detection": "obstacle",
        "terrain_detection": "terrain",
        "pedestrian_detection": "pedestrian",
        "vehicle_approach_detection": "vehicle",
        "guidance_object_detection": "guidance",
        "other_information_detection": "other",
    }
    return aliases.get(key, key)


def _normalize_trigger_key(key):
    key = str(key or "").strip().lower()
    return {"none": "always_on", "5": "always_on", "": "always_on"}.get(key, key)


def _compile_route_template(payload):
    """Compile front-end route_template slots into scope-based intervention rules."""
    route_template = payload.get("route_template") or {}
    slots = route_template.get("slots") or []
    compiled = []

    for index, slot in enumerate(slots):
        slot_id = str(slot.get("slot_id") or slot.get("id") or f"slot_{index}")
        scope = str(slot.get("scope") or SLOT_SCOPE_FALLBACK.get(slot_id, "segment")).strip().lower()
        if scope not in {"segment", "intersection"}:
            scope = SLOT_SCOPE_FALLBACK.get(slot_id, "segment")
        functions = [
            _normalize_function_key(item)
            for item in slot.get("functions", [])
            if _normalize_function_key(item) in DEFAULT_FUNCTION_PARAMS
        ]
        triggers = [
            _normalize_trigger_key(item)
            for item in slot.get("triggers", [])
        ]
        triggers = [item for item in triggers if item]
        if not functions:
            continue
        if not triggers:
            triggers = ["always_on"]
        compiled.append(
            {
                "slot_id": slot_id,
                "scope": scope,
                "functions": functions,
                "triggers": triggers,
            }
        )

    if compiled:
        return compiled

    return [
        {
            "slot_id": "default_segment_obstacle",
            "scope": "segment",
            "functions": ["obstacle"],
            "triggers": ["obstacle"],
        }
    ]


def _function_params(payload, function_key):
    """Read per-function parameters, falling back to calibrated prototype defaults."""
    parameters = payload.get("device_parameters") or {}
    raw = parameters.get(function_key) or parameters.get(f"{function_key}_detection") or {}
    defaults = json.loads(json.dumps(DEFAULT_FUNCTION_PARAMS[function_key]))
    defaults["recognition_rate"] = _bounded_float(
        raw.get("recognition_rate"), defaults["recognition_rate"]
    )
    if "small_obstacle_rate" in defaults:
        defaults["small_obstacle_rate"] = _bounded_float(
            raw.get("small_obstacle_rate"), defaults["small_obstacle_rate"]
        )
    if "category_trigger_prob" in defaults and raw.get("category_trigger_prob") is not None:
        defaults["category_trigger_prob"] = _bounded_float(
            raw.get("category_trigger_prob"), 0.0
        )
    defaults["false_alarm_rate"] = _bounded_float(
        raw.get("false_alarm_rate"), defaults["false_alarm_rate"]
    )
    if raw.get("miss_rate") is not None:
        defaults["miss_rate"] = _bounded_float(raw.get("miss_rate"), 0.0)
    if "action_duration" in defaults and raw.get("action_duration") is not None:
        defaults["action_duration"] = _bounded_float(
            raw.get("action_duration"), 0.0, low=0.0, high=30.0
        )
    if "preparation_duration" in defaults and raw.get("preparation_duration") is not None:
        defaults["preparation_duration"] = _bounded_float(
            raw.get("preparation_duration"), 0.0, low=0.0, high=30.0
        )
    feedback = raw.get("feedback") or {}
    defaults["feedback"]["modality"] = str(
        feedback.get("modality", defaults["feedback"]["modality"])
    ).strip().lower()
    defaults["feedback"]["duration"] = _bounded_float(
        feedback.get("duration"), defaults["feedback"]["duration"], low=0.0, high=10.0
    )
    defaults["feedback"]["frequency"] = _bounded_float(
        feedback.get("frequency"), defaults["feedback"]["frequency"], low=0.0, high=20.0
    )
    defaults["feedback"]["competition"] = _bounded_float(
        feedback.get("competition"), defaults["feedback"]["competition"], low=0.0, high=5.0
    )
    if "targets" in defaults:
        target_aliases = {
            "curb": "curb",
            "路缘": "curb",
            "wall": "wall",
            "墙": "wall",
            "railing": "railing",
            "栏杆": "railing",
            "tactile": "tactile_guidance",
            "tactile_guidance": "tactile_guidance",
            "blind_path": "tactile_guidance",
            "盲道": "tactile_guidance",
            "提示路面": "tactile_guidance",
            "landmark": "landmark",
            "地标": "landmark",
        }
        raw_targets = raw.get("targets")
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        if isinstance(raw_targets, list):
            normalized_targets = []
            for target in raw_targets:
                target_key = target_aliases.get(str(target).strip().lower()) or target_aliases.get(str(target).strip())
                if target_key and target_key not in normalized_targets:
                    normalized_targets.append(target_key)
            defaults["targets"] = normalized_targets
    return defaults


def _extract_familiarity(payload):
    """Map the UI familiarity control to the model's binary familiarity input."""
    user_profile = payload.get("user_profile", {})
    familiarity = str(user_profile.get("familiarity", "Familiar")).strip().lower()
    return 0 if familiarity == "unfamiliar" else 1


def _report_url(file_path):
    """Convert a generated report path into a local URL."""
    if not file_path:
        return None
    try:
        path = Path(file_path).resolve()
        relative = path.relative_to(BASE_DIR)
    except (OSError, ValueError):
        return None
    return "/" + relative.as_posix()


def _summarize_single_run(summary):
    """Convert a single simulation report into the API response shape."""
    result = summary.get("result", {})
    statistics = summary.get("statistics", {})
    risk_mean = float(statistics.get("risk", {}).get("mean", 0.0))

    return {
        "runs": 1,
        "success_rate": 1.0 if result.get("reached_goal") else 0.0,
        "mean_cognitive_load": float(statistics.get("actr_iw", {}).get("mean", 0.0)),
        "high_load_episodes": int(result.get("actr_iw_high_count", 0)),
        "risk_level": _risk_label(risk_mean),
        "risk_mean": risk_mean,
        "stop_probe_count": int(result.get("stop_probe_count", 0)),
        "total_steps": int(result.get("total_steps", 0)),
        "gate_passed_count": int(result.get("gate_passed_count", 0)),
        "map_image_url": _report_url(summary.get("map_image")),
        "actr_chart_url": _report_url(summary.get("actr_overview_chart_image")),
    }


def _summarize_monte_carlo(summary):
    """Convert a Monte Carlo report into the API response shape."""
    aggregate = summary.get("aggregate", {})
    rows = summary.get("run_details", [])
    risk_mean = float(aggregate.get("risk_mean", 0.0))
    high_load_counts = []
    map_image_url = None
    actr_chart_url = None

    for row in rows:
        json_path = row.get("summary_json")
        if not json_path:
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                run_summary = json.load(handle)
            high_load_counts.append(
                int(run_summary.get("result", {}).get("actr_iw_high_count", 0))
            )
            map_image_url = map_image_url or _report_url(run_summary.get("map_image"))
            actr_chart_url = actr_chart_url or _report_url(
                run_summary.get("actr_overview_chart_image")
            )
        except OSError:
            continue

    return {
        "runs": int(summary.get("runs", len(rows) or 1)),
        "success_rate": float(aggregate.get("goal_reach_rate", 0.0)),
        "mean_cognitive_load": float(aggregate.get("actr_iw_mean", 0.0)),
        "high_load_episodes": round(mean(high_load_counts), 2)
        if high_load_counts
        else 0,
        "risk_level": _risk_label(risk_mean),
        "risk_mean": risk_mean,
        "stop_probe_count": float(aggregate.get("stop_probe_mean", 0.0)),
        "total_steps": float(aggregate.get("total_steps_mean", 0.0)),
        "gate_passed_count": float(aggregate.get("gate_passed_mean", 0.0)),
        "total_steps_std": float(aggregate.get("total_steps_std", 0.0)),
        "stop_probe_std": float(aggregate.get("stop_probe_std", 0.0)),
        "risk_std": float(aggregate.get("risk_std", 0.0)),
        "cognitive_load_std": float(aggregate.get("actr_iw_std", 0.0)),
        "map_image_url": map_image_url,
        "actr_chart_url": actr_chart_url,
    }


def _engine_modules():
    """Return mutable module dictionaries used by the imported simulation engine."""
    cli = _engine_cli()
    modules = [cli.run_simulation.__globals__]
    for name in ("config", "actr_setup", "environment"):
        module = sys.modules.get(name)
        if module is not None:
            modules.append(module.__dict__)
    return modules


def _engine_cli():
    """Load the legacy engine only when the legacy runner is explicitly used."""
    global ENGINE_CLI
    if ENGINE_CLI is None:
        ENGINE_CLI = _load_engine_cli()
    return ENGINE_CLI


def _set_override(overrides, name, value):
    """Store a runtime parameter override when the value is available."""
    if value is not None:
        overrides[name] = value


def _level_factor(value, *, low=0.65, medium=1.0, high=1.45):
    """Map common UI levels to numeric multipliers."""
    text = str(value or "").strip().lower()
    if text in {"low", "weak", "subtle", "quiet", "short", "fast", "none"}:
        return low
    if text in {"high", "strong", "prominent", "long", "slow", "continuous"}:
        return high
    return medium


def _find_named(items, name):
    """Find a named row in a list of UI control records."""
    target = str(name).strip().lower()
    for item in items or []:
        if str(item.get("name", "")).strip().lower() == target:
            return item
    return {}


def _enabled_factor(item, *, off=0.0, low=0.65, medium=1.0, high=1.45):
    """Map enabled/reliability table rows to probability multipliers."""
    if not item or not item.get("enabled", False):
        return off
    return _level_factor(item.get("value"), low=low, medium=medium, high=high)


def _range_value(items, name, default=2):
    """Read a one-to-three slider value from the UI payload."""
    item = _find_named(items, name)
    try:
        return int(item.get("value", default))
    except (TypeError, ValueError):
        return default


def _range_enabled(items, name, default=True):
    """Read a checkbox value from the UI payload."""
    item = _find_named(items, name)
    value = item.get("enabled")
    return default if value is None else bool(value)


def _detection_rate(item, rates, default=0.0):
    """Map an enabled High/Medium/Low detection row to an explicit rate."""
    if not item or not item.get("enabled", False):
        return 0.0
    level = str(item.get("value", "Medium")).strip().lower()
    return rates.get(level, default)


def _parse_lat_lon(value):
    """Parse a latitude-longitude text field."""
    try:
        lat_text, lon_text = str(value).split(",", 1)
        return float(lat_text.strip()), float(lon_text.strip())
    except (TypeError, ValueError):
        return None


def _build_environment_loader(start_point, goal_point):
    """Build a route loader that uses the UI start and goal coordinates."""
    cli = _engine_cli()
    original_loader = cli.run_simulation.__globals__.get("load_environment")
    environment_module = sys.modules.get("environment")

    if original_loader is None or environment_module is None:
        return None

    start_lat_lon = _parse_lat_lon(start_point)
    goal_lat_lon = _parse_lat_lon(goal_point)
    if start_lat_lon is None and goal_lat_lon is None:
        return original_loader

    def load_environment_from_ui(center_point=None, dist=500):
        import osmnx as ox

        center = start_lat_lon or center_point or (-33.8688, 151.2093)
        graph, default_start, default_goal, _ = original_loader(center_point=center, dist=dist)

        start_node = default_start
        goal_node = default_goal
        if start_lat_lon is not None:
            start_node = ox.nearest_nodes(graph, X=start_lat_lon[1], Y=start_lat_lon[0])
        if goal_lat_lon is not None:
            goal_node = ox.nearest_nodes(graph, X=goal_lat_lon[1], Y=goal_lat_lon[0])

        route_phases = environment_module.build_route_phases(graph, start_node, goal_node)
        return graph, start_node, goal_node, route_phases

    return load_environment_from_ui


def _has_designer_slots(payload):
    return bool((payload.get("route_template") or {}).get("slots"))


def _designer_slot_summary(payload):
    """Translate the new route-template UI into legacy model parameter factors."""
    slots = _compile_route_template(payload) if _has_designer_slots(payload) else []
    summary = {
        "slots": slots,
        "function_scopes": {key: set() for key in DEFAULT_FUNCTION_PARAMS},
        "trigger_scopes": {key: set() for key in ["obstacle", "vehicle", "pedestrian", "landmark", "surface", "always_on"]},
        "always_on": False,
        "auditory_weight": 0.0,
        "tactile_weight": 0.0,
        "manual_weight": 0.0,
        "competition_sum": 0.0,
        "frequency_max": 0.0,
        "false_alarm_sum": 0.0,
    }
    for slot in slots:
        scope = slot["scope"]
        triggers = set(slot["triggers"])
        for trigger in triggers:
            summary["trigger_scopes"].setdefault(trigger, set()).add(scope)
        if "always_on" in triggers:
            summary["always_on"] = True
        for function_key in slot["functions"]:
            summary["function_scopes"].setdefault(function_key, set()).add(scope)
            params = _function_params(payload, function_key)
            feedback = params["feedback"]
            active_duration = (params.get("action_duration") or 0.0) + (params.get("preparation_duration") or 0.0)
            duration_factor = max(0.2, feedback["duration"]) + 0.15 * active_duration
            weight = (
                max(0.1, params["recognition_rate"])
                * max(0.1, feedback["competition"])
                * max(0.2, duration_factor)
            )
            modality = feedback["modality"]
            if modality == "tactile":
                summary["tactile_weight"] += weight
            elif modality in {"traction", "牵引力"}:
                summary["manual_weight"] += weight
            else:
                summary["auditory_weight"] += weight
            summary["competition_sum"] += feedback["competition"]
            summary["frequency_max"] = max(summary["frequency_max"], feedback["frequency"])
            summary["false_alarm_sum"] += params["false_alarm_rate"]
    return summary


def _scope_factor(scopes, scope):
    return 1.0 if scope in scopes else 0.0


def _designer_recognition(payload, function_key, *, small=False):
    params = _function_params(payload, function_key)
    if small and function_key == "obstacle":
        return params.get("category_trigger_prob") if params.get("category_trigger_prob") is not None else params.get("small_obstacle_rate", params["recognition_rate"])
    return params["recognition_rate"]


def _apply_designer_slot_overrides(overrides, payload, *, traffic_factor, crowd_factor, tactile_factor, trust_gap):
    """Map new front-end route_template slots onto the existing BVI-SAS override layer."""
    if not _has_designer_slots(payload):
        return

    summary = _designer_slot_summary(payload)
    function_scopes = summary["function_scopes"]
    trigger_scopes = summary["trigger_scopes"]

    obstacle_scopes = function_scopes.get("obstacle", set()) & (
        trigger_scopes.get("obstacle", set()) | trigger_scopes.get("always_on", set())
    )
    terrain_scopes = function_scopes.get("terrain", set()) & (
        trigger_scopes.get("surface", set()) | trigger_scopes.get("always_on", set())
    )
    pedestrian_scopes = function_scopes.get("pedestrian", set()) & (
        trigger_scopes.get("pedestrian", set()) | trigger_scopes.get("always_on", set())
    )
    vehicle_scopes = function_scopes.get("vehicle", set()) & (
        trigger_scopes.get("vehicle", set()) | trigger_scopes.get("always_on", set())
    )
    active_trigger_scopes = set()
    for scopes in trigger_scopes.values():
        active_trigger_scopes |= scopes
    guidance_scopes = function_scopes.get("guidance", set()) & active_trigger_scopes
    other_scopes = function_scopes.get("other", set()) & (
        trigger_scopes.get("landmark", set()) | trigger_scopes.get("always_on", set())
    )

    obstacle_rate = _designer_recognition(payload, "obstacle")
    small_obstacle_rate = _designer_recognition(payload, "obstacle", small=True)
    obstacle_enabled = 1.0 if obstacle_scopes else 0.0
    terrain_enabled = 1.0 if terrain_scopes else 0.0
    vehicle_segment = _scope_factor(vehicle_scopes, "segment")
    vehicle_intersection = _scope_factor(vehicle_scopes, "intersection")
    pedestrian_segment = _scope_factor(pedestrian_scopes, "segment")
    pedestrian_intersection = _scope_factor(pedestrian_scopes, "intersection")
    guidance_params = _function_params(payload, "guidance")
    guidance_targets = set(guidance_params.get("targets") or [])
    guidance_strength = _designer_recognition(payload, "guidance") if guidance_scopes else 0.0
    other_strength = _designer_recognition(payload, "other") * 0.8 if other_scopes else 0.0
    curb_guidance = guidance_strength if "curb" in guidance_targets else 0.0
    wall_guidance = guidance_strength if "wall" in guidance_targets else 0.0
    railing_guidance = guidance_strength if "railing" in guidance_targets else 0.0
    tactile_guidance = guidance_strength if "tactile_guidance" in guidance_targets else 0.0
    landmark_guidance = guidance_strength if "landmark" in guidance_targets else 0.0

    _set_override(overrides, "CANE_OBSTACLE_PROB", 0.00179 * obstacle_rate * obstacle_enabled)
    _set_override(overrides, "CANE_CURB_PROB", 0.02731 * max(small_obstacle_rate * obstacle_enabled, terrain_enabled * tactile_factor * 0.35, curb_guidance))
    _set_override(overrides, "CANE_WALL_PROB", 0.00507 * wall_guidance)
    _set_override(overrides, "CANE_RAILING_PROB", 0.00377 * railing_guidance)
    _set_override(overrides, "SOUND_VEHICLE_APPROACH_PROB", 0.00142 * vehicle_segment * _designer_recognition(payload, "vehicle") * traffic_factor)
    _set_override(overrides, "SOUND_VEHICLE_APPROACH_CROSSING_PROB", 0.00574 * vehicle_intersection * _designer_recognition(payload, "vehicle") * traffic_factor)
    _set_override(overrides, "SOUND_HORN_PROB", 0.000167 * max(vehicle_segment, vehicle_intersection) * traffic_factor)
    _set_override(overrides, "SOUND_REVERSE_BEEP_PROB", 0.000251 * max(vehicle_segment, vehicle_intersection) * traffic_factor)
    _set_override(overrides, "HUMAN_ACTIVITY_PROB", 0.01036 * pedestrian_segment * _designer_recognition(payload, "pedestrian") * crowd_factor)
    _set_override(overrides, "CROSSING_HUMAN_ACTIVITY_PROB", 0.02402 * pedestrian_intersection * _designer_recognition(payload, "pedestrian") * crowd_factor)
    landmark_strength = max(landmark_guidance, other_strength)
    _set_override(overrides, "LANDMARK_TRIGGER_PROB_MIN", 0.006 * landmark_strength)
    _set_override(overrides, "LANDMARK_TRIGGER_PROB_MAX", 0.030 * landmark_strength)

    if terrain_enabled or tactile_guidance:
        surface_distribution = {
            "flat_road": max(0.05, 0.78 - 0.14 * tactile_factor - 0.10 * tactile_guidance),
            "uneven_natural": 0.012 if terrain_enabled else 0.006,
            "slope_surface": 0.014 if terrain_enabled else 0.010,
            "height_drop": 0.010 if terrain_enabled else 0.006,
            "tactile_guidance": max(0.0, 0.16 * tactile_factor + 0.18 * tactile_guidance),
        }
        total_surface = sum(surface_distribution.values()) or 1.0
        _set_override(
            overrides,
            "SURFACE_PROBABILITY_DISTRIBUTION",
            {key: value / total_surface for key, value in surface_distribution.items()},
        )

    total_channel = max(
        0.01,
        summary["auditory_weight"] + summary["tactile_weight"] + summary["manual_weight"] + 0.20 + 0.25 * trust_gap,
    )
    _set_override(overrides, "ACTR_AUDITORY_SHARE", summary["auditory_weight"] / total_channel)
    _set_override(overrides, "ACTR_TACTILE_SHARE", summary["tactile_weight"] / total_channel)
    _set_override(overrides, "ACTR_MANUAL_SHARE", (summary["manual_weight"] + 0.20 + 0.25 * trust_gap) / total_channel)
    if summary["always_on"]:
        _set_override(overrides, "NAV_CYCLE_STEPS", max(2, int(5 - min(3, summary["frequency_max"]))))
    _set_override(
        overrides,
        "ATTENTION_GATED_CENTRAL_DANGER_BOOST",
        0.20 + 0.04 * min(3.0, summary["competition_sum"]) + 0.25 * min(1.0, summary["false_alarm_sum"]),
    )
    _set_override(overrides, "DESIGNER_COMPILED_SLOTS", summary["slots"])
    _set_override(overrides, "DESIGNER_GUIDANCE_TARGETS", sorted(guidance_targets))


def _derive_model_overrides(payload):
    """Map designer UI controls to model constants used by BVI-SAS."""
    overrides = {}
    scenario = payload.get("scenario", {})
    simulation = payload.get("simulation", {})
    device = payload.get("device", {})
    user_profile = payload.get("user_profile", {})
    feedback = payload.get("feedback", {})
    walking = payload.get("walking_response", {})
    duration = payload.get("reference_duration", {})
    cognitive = payload.get("cognitive_impact", [])

    references = device.get("references", [])
    modalities = feedback.get("modalities", [])
    timing = feedback.get("timing", [])

    scenario_factor = {
        "daily commute": 1.0,
        "campus navigation": 0.9,
        "street crossing": 1.35,
        "shopping mall": 1.15,
        "transit station": 1.3,
        "hospital or public building": 0.95,
        "unfamiliar outdoor route": 1.45,
    }.get(str(scenario.get("type", "")).strip().lower(), 1.0)
    traffic_factor = _level_factor(scenario.get("traffic_density"), low=0.45, medium=1.0, high=1.8)
    crowd_factor = _level_factor(scenario.get("crowd_density"), low=0.55, medium=1.0, high=1.7)
    tactile_factor = _level_factor(scenario.get("tactile_paving"), low=0.0, medium=1.0, high=1.8)

    surface_distribution = {
        "flat_road": max(0.05, 0.86 - 0.18 * tactile_factor),
        "uneven_natural": 0.006 * scenario_factor,
        "slope_surface": 0.010 * scenario_factor,
        "height_drop": 0.012 * scenario_factor,
        "tactile_guidance": max(0.0, 0.12 * tactile_factor),
    }
    total_surface = sum(surface_distribution.values()) or 1.0
    _set_override(
        overrides,
        "SURFACE_PROBABILITY_DISTRIBUTION",
        {key: value / total_surface for key, value in surface_distribution.items()},
    )
    _set_override(overrides, "MAX_STEPS", int(200 * scenario_factor))

    obstacle_rate = _detection_rate(
        _find_named(references, "Obstacle"),
        {"low": 0.60, "medium": 0.75, "high": 0.90},
        default=0.75,
    )
    small_ground_rate = _detection_rate(
        _find_named(references, "Small ground obstacle"),
        {"low": 0.20, "medium": 0.40, "high": 0.60},
        default=0.40,
    )
    vehicle_factor = _enabled_factor(_find_named(references, "Vehicle approach"))
    crossing_factor = _enabled_factor(_find_named(references, "Crossing"))
    landmark_factor = _enabled_factor(_find_named(references, "Landmark"))
    text_factor = _enabled_factor(_find_named(references, "Text or sign"))
    entrance_factor = _enabled_factor(_find_named(references, "Building entrance"))
    device_trust_level = str(user_profile.get("device_trust", "Medium")).strip().lower()
    device_trust = {"low": 0.35, "medium": 0.65, "high": 0.85}.get(
        device_trust_level, 0.65
    )
    trust_gap = max(0.0, 1.0 - device_trust)

    _set_override(overrides, "CANE_OBSTACLE_PROB", 0.00179 * obstacle_rate)
    _set_override(overrides, "CANE_CURB_PROB", 0.02731 * max(small_ground_rate, tactile_factor * 0.35))
    _set_override(overrides, "CANE_WALL_PROB", 0.00507 * obstacle_rate)
    _set_override(overrides, "CANE_RAILING_PROB", 0.00377 * obstacle_rate)
    _set_override(overrides, "SOUND_VEHICLE_APPROACH_PROB", 0.00142 * vehicle_factor * traffic_factor)
    _set_override(overrides, "SOUND_VEHICLE_APPROACH_CROSSING_PROB", 0.00574 * vehicle_factor * crossing_factor * traffic_factor)
    _set_override(overrides, "SOUND_HORN_PROB", 0.000167 * traffic_factor * vehicle_factor)
    _set_override(overrides, "SOUND_REVERSE_BEEP_PROB", 0.000251 * traffic_factor * vehicle_factor)
    _set_override(overrides, "HUMAN_ACTIVITY_PROB", 0.01036 * crowd_factor)
    _set_override(overrides, "CROSSING_HORN_PROB", 0.00313 * crossing_factor * traffic_factor)
    _set_override(overrides, "CROSSING_HUMAN_ACTIVITY_PROB", 0.02402 * crossing_factor * crowd_factor)

    landmark_strength = max(landmark_factor, text_factor * 0.8, entrance_factor * 0.9)
    _set_override(overrides, "LANDMARK_TRIGGER_PROB_MIN", 0.006 * landmark_strength)
    _set_override(overrides, "LANDMARK_TRIGGER_PROB_MAX", 0.030 * landmark_strength)
    _set_override(overrides, "LANDMARK_DECAY_RATE", {"Weak": 0.70, "Normal": 0.82, "Strong": 0.94}.get(duration.get("landmark_persistence"), 0.82))

    vibration = _find_named(modalities, "Vibration")
    beep = _find_named(modalities, "Beep")
    speech = _find_named(modalities, "Speech")
    spatial_audio = _find_named(modalities, "Spatial audio")
    tactile_share = 0.40 * _enabled_factor(vibration, low=0.55, medium=1.0, high=1.35)
    auditory_share = 0.20 * _enabled_factor(beep, low=0.6, medium=1.0, high=1.25)
    auditory_share += 0.20 * _enabled_factor(speech, low=0.65, medium=1.0, high=1.35)
    auditory_share += 0.15 * _enabled_factor(spatial_audio, low=0.65, medium=1.0, high=1.35)
    manual_share = (
        0.20
        + (0.15 if walking.get("require_cane_confirmation") == "Yes" else 0.0)
        + 0.25 * trust_gap
    )
    total_share = max(0.01, auditory_share + tactile_share + manual_share)
    _set_override(overrides, "ACTR_AUDITORY_SHARE", auditory_share / total_share)
    _set_override(overrides, "ACTR_TACTILE_SHARE", tactile_share / total_share)
    _set_override(overrides, "ACTR_MANUAL_SHARE", manual_share / total_share)

    frequency = _range_value(timing, "Warning Frequency", 2)
    alert_interval = _range_value(timing, "Alert Interval", 6)
    activation = str(feedback.get("activation_mode", "Event-triggered")).strip().lower()
    nav_cycle = {1: 10, 2: 5, 3: 2}.get(frequency, 5)
    if activation == "always on":
        nav_cycle = max(2, nav_cycle - 2)
    elif activation == "manual request":
        nav_cycle = max(nav_cycle, 12)
    elif activation == "periodic update":
        nav_cycle = max(3, int(alert_interval))
    _set_override(overrides, "NAV_CYCLE_STEPS", nav_cycle)
    critical_override = _range_enabled(timing, "Critical Override", True)
    _set_override(overrides, "SEEV_GATE_ADAPTIVE_THRESHOLD_ENABLED", critical_override)
    _set_override(overrides, "ATTENTION_GATED_CENTRAL_DANGER_BOOST", 0.28 if critical_override else 0.08)

    alert_duration = duration.get("alert_duration", "Medium")
    vehicle_duration = duration.get("vehicle_alert_persistence", "Medium")
    cooldown = duration.get("repeated_alert_cooldown", "Medium")
    _set_override(overrides, "LANDMARK_EPISODE_STEPS_MIN", {"Short": 3, "Medium": 5, "Long": 8}.get(alert_duration, 5))
    _set_override(overrides, "LANDMARK_EPISODE_STEPS_MAX", {"Short": 5, "Medium": 9, "Long": 14}.get(alert_duration, 9))
    _set_override(overrides, "VEHICLE_APPROACH_MIN_STEPS", {"Short": 1, "Medium": 2, "Long": 4}.get(vehicle_duration, 2))
    _set_override(overrides, "VEHICLE_APPROACH_MAX_STEPS", {"Short": 3, "Medium": 5, "Long": 8}.get(vehicle_duration, 5))
    _set_override(overrides, "LANDMARK_REFRACTORY_STEPS_MIN", {"Short": 8, "Medium": 18, "Long": 30}.get(cooldown, 18))
    _set_override(overrides, "LANDMARK_REFRACTORY_STEPS_MAX", {"Short": 18, "Medium": 35, "Long": 55}.get(cooldown, 35))

    default_response = str(walking.get("default_obstacle_response", "")).strip().lower()
    high_risk_response = str(walking.get("high_risk_crossing_response", "")).strip().lower()
    recovery = walking.get("recovery_after_alert", "Normal")
    _set_override(overrides, "UTILITY_DANGER_RESPONSE", 10.5 if default_response == "stop and probe" else 8.5)
    _set_override(overrides, "UTILITY_DEFAULT_FORWARD", 5.0 if default_response == "continue" else 4.0)
    _set_override(overrides, "UTILITY_SAFETY_CRITICAL", 12.8 if high_risk_response == "wait" else 11.4)
    _set_override(
        overrides,
        "PROBE_RELIEF_RATIO",
        (0.42 if walking.get("require_cane_confirmation") == "Yes" else 0.22)
        + 0.16 * trust_gap,
    )
    _set_override(overrides, "ACTR_LOAD_RESUME_THRESHOLD", {"Fast": 5.8, "Normal": 5.0, "Slow": 4.4}.get(recovery, 5.0))
    _set_override(overrides, "LOOMING_RESUME_THRESHOLD", {"Fast": 0.16, "Normal": 0.10, "Slow": 0.06}.get(recovery, 0.10))

    trust = _range_value(cognitive, "Device Trust", {"low": 1, "medium": 2, "high": 3}.get(device_trust_level, 2))
    information_load = _range_value(cognitive, "Information Load", 2)
    auditory_interference = _range_value(cognitive, "Auditory Interference", 1)
    false_alarm = _range_value(cognitive, "False Alarm Impact", 2)
    missed_alert = _range_value(cognitive, "Missed Alert Risk", 2)
    _set_override(overrides, "MEMORY_ACTIVE_RETRIEVAL_TH", {1: 0.10, 2: 0.07, 3: 0.04}.get(trust, 0.07))
    _set_override(overrides, "DEVICE_TRUST", device_trust)
    _set_override(overrides, "ACTR_IW_HIGH_THRESHOLD", {1: 7.0, 2: 6.0, 3: 5.0}.get(information_load, 6.0))
    _set_override(overrides, "ACTR_CENTRAL_WEIGHT", {1: 1.6, 2: 2.0, 3: 2.5}.get(information_load, 2.0))
    _set_override(overrides, "ACTR_AUDITORY_SHARE", overrides["ACTR_AUDITORY_SHARE"] * {1: 0.85, 2: 1.0, 3: 1.2}.get(auditory_interference, 1.0))
    _set_override(overrides, "ACTR_ERROR_BOOST", {1: 1.5, 2: 2.0, 3: 2.8}.get(false_alarm, 2.0))
    _set_override(overrides, "RISK_ERROR_COEF", {1: 0.55, 2: 0.75, 3: 1.0}.get(missed_alert, 0.75))

    _apply_designer_slot_overrides(
        overrides,
        payload,
        traffic_factor=traffic_factor,
        crowd_factor=crowd_factor,
        tactile_factor=tactile_factor,
        trust_gap=trust_gap,
    )

    loader = _build_environment_loader(simulation.get("start_point"), simulation.get("goal_point"))
    if loader is not None:
        _set_override(overrides, "load_environment", loader)

    return overrides


@contextlib.contextmanager
def _temporary_model_overrides(payload):
    """Temporarily apply UI-derived parameters to the loaded engine modules."""
    overrides = _derive_model_overrides(payload)
    modules = _engine_modules()
    original_values = []

    for module_dict in modules:
        for name, value in overrides.items():
            if name in module_dict:
                original_values.append((module_dict, name, module_dict[name]))
                module_dict[name] = value

    try:
        yield overrides
    finally:
        for module_dict, name, value in reversed(original_values):
            module_dict[name] = value


def _json_safe_parameters(parameters):
    """Convert runtime parameter overrides into a JSON-safe response object."""
    safe_values = {}
    for name, value in parameters.items():
        if callable(value):
            safe_values[name] = "<runtime route loader>"
        else:
            safe_values[name] = value
    return safe_values


def _save_ui_run_config(report_path, payload, applied_parameters, result):
    """Persist the designer payload and output that produced a simulation report."""
    report_path = Path(report_path)
    timestamp = report_path.stem.replace("sim_summary_", "")
    config_path = report_path.with_name(f"sim_ui_config_{timestamp}.json")
    io_report_path = report_path.with_name(f"sim_io_report_{timestamp}.md")
    safe_applied_parameters = _json_safe_parameters(applied_parameters)
    config = {
        "summary_report": str(report_path),
        "io_report": str(io_report_path),
        "received_config": payload,
        "applied_parameters": safe_applied_parameters,
        "output_preview": result,
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    with io_report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# BVI-SAS UI Input/Output Report\n\n")
        handle.write(f"- Summary report: `{report_path}`\n")
        handle.write(f"- UI config JSON: `{config_path}`\n\n")
        handle.write("## Output Preview\n\n")
        for key, value in result.items():
            handle.write(f"- `{key}`: {value}\n")
        handle.write("\n## Received Frontend Config\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        handle.write("\n```\n\n")
        handle.write("## Applied Legacy Parameter Overrides\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(safe_applied_parameters, ensure_ascii=False, indent=2))
        handle.write("\n```\n")
    return config_path, io_report_path


def run_simulation_from_payload(payload):
    """Run the existing BVI-SAS engine with the new designer parameter mapping."""
    return run_legacy_simulation_from_payload(payload)


def run_legacy_simulation_from_payload(payload):
    """Run the existing simulation engine from a web request payload."""
    simulation = payload.get("simulation", {})
    runs = _bounded_int(simulation.get("runs"), default=1, low=1, high=3)
    seed = _bounded_int(simulation.get("seed"), default=42, low=1, high=999999)
    familiarity = _extract_familiarity(payload)

    log_buffer = io.StringIO()
    with SIMULATION_LOCK:
        with _temporary_model_overrides(payload) as applied_parameters:
            with contextlib.redirect_stdout(log_buffer):
                if runs > 1:
                    cli = _engine_cli()
                    _, summary_path = cli.run_monte_carlo(
                        familiarity=familiarity,
                        mc_runs=runs,
                        seed_start=seed,
                    )
                    with open(summary_path, "r", encoding="utf-8") as handle:
                        summary = json.load(handle)
                    result = _summarize_monte_carlo(summary)
                    report_path = summary_path
                else:
                    random.seed(seed)
                    np.random.seed(seed)
                    cli = _engine_cli()
                    _, _, summary_path = cli.run(familiarity=familiarity)
                    with open(summary_path, "r", encoding="utf-8") as handle:
                        summary = json.load(handle)
                    result = _summarize_single_run(summary)
                    report_path = summary_path
            config_path, io_report_path = _save_ui_run_config(
                report_path, payload, applied_parameters, result
            )

    return {
        "ok": True,
        "result": result,
        "report_path": str(report_path),
        "config_path": str(config_path),
        "io_report_path": str(io_report_path),
        "applied_parameters": _json_safe_parameters(applied_parameters),
        "engine_log": log_buffer.getvalue()[-4000:],
        "received_config": payload,
    }


class BviSasRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler for static files and the simulation endpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _send_json(self, status_code, payload):
        """Write a JSON response."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_report_file(self, *, include_body):
        """Serve a generated report file from the reports directory."""
        path = unquote(urlparse(self.path).path)
        file_path = (BASE_DIR / path.lstrip("/")).resolve()
        reports_dir = (BASE_DIR / "reports").resolve()
        try:
            file_path.relative_to(reports_dir)
        except ValueError:
            self.send_error(403, "Forbidden")
            return
        if not file_path.is_file():
            self.send_error(404, "File not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(file_path)))
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()
        if include_body:
            with open(file_path, "rb") as handle:
                self.copyfile(handle, self.wfile)

    def do_GET(self):
        """Serve the web app and generated report images."""
        path = unquote(urlparse(self.path).path)
        if path.startswith("/reports/"):
            self._send_report_file(include_body=True)
            return

        super().do_GET()

    def do_HEAD(self):
        """Serve report metadata for generated images."""
        path = unquote(urlparse(self.path).path)
        if path.startswith("/reports/"):
            self._send_report_file(include_body=False)
            return

        super().do_HEAD()

    def do_POST(self):
        """Handle API requests from the local web interface."""
        path = urlparse(self.path).path
        if path != "/simulate":
            self._send_json(404, {"ok": False, "error": "Unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8") or "{}")
            response = run_simulation_from_payload(payload)
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
            self._send_json(500, response)
            return

        self._send_json(200, response)


def main():
    """Start the local product prototype server."""
    port = int(os.environ.get("BVI_SAS_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), BviSasRequestHandler)
    print(f"BVI-SAS local site: http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop the server.")
    server.serve_forever()


if __name__ == "__main__":
    main()
