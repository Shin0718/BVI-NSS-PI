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
from urllib.parse import urlparse


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


ENGINE_CLI = _load_engine_cli()
SIMULATION_LOCK = threading.Lock()


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


def _extract_familiarity(payload):
    """Map the UI familiarity control to the model's binary familiarity input."""
    user_profile = payload.get("user_profile", {})
    familiarity = str(user_profile.get("familiarity", "Familiar")).strip().lower()
    return 0 if familiarity == "unfamiliar" else 1


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
    }


def _summarize_monte_carlo(summary):
    """Convert a Monte Carlo report into the API response shape."""
    aggregate = summary.get("aggregate", {})
    rows = summary.get("run_details", [])
    risk_mean = float(aggregate.get("risk_mean", 0.0))
    high_load_counts = []

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
    }


def _engine_modules():
    """Return mutable module dictionaries used by the imported simulation engine."""
    modules = [ENGINE_CLI.run_simulation.__globals__]
    for name in ("config", "actr_setup", "environment"):
        module = sys.modules.get(name)
        if module is not None:
            modules.append(module.__dict__)
    return modules


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


def _parse_lat_lon(value):
    """Parse a latitude-longitude text field."""
    try:
        lat_text, lon_text = str(value).split(",", 1)
        return float(lat_text.strip()), float(lon_text.strip())
    except (TypeError, ValueError):
        return None


def _build_environment_loader(start_point, goal_point):
    """Build a route loader that uses the UI start and goal coordinates."""
    original_loader = ENGINE_CLI.run_simulation.__globals__.get("load_environment")
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


def _derive_model_overrides(payload):
    """Map designer UI controls to model constants used by BVI-SAS."""
    overrides = {}
    scenario = payload.get("scenario", {})
    simulation = payload.get("simulation", {})
    device = payload.get("device", {})
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

    obstacle_factor = _enabled_factor(_find_named(references, "Obstacle"))
    vehicle_factor = _enabled_factor(_find_named(references, "Vehicle approach"))
    crossing_factor = _enabled_factor(_find_named(references, "Crossing"))
    landmark_factor = _enabled_factor(_find_named(references, "Landmark"))
    text_factor = _enabled_factor(_find_named(references, "Text or sign"))
    people_factor = _enabled_factor(_find_named(references, "People or crowd"))
    entrance_factor = _enabled_factor(_find_named(references, "Building entrance"))

    device_factor = {
        "smart cane": (1.2, 0.7, 0.8, 0.8),
        "obstacle sensor": (1.6, 0.8, 0.5, 0.4),
        "wearable ai navigation": (1.1, 1.5, 1.3, 1.4),
        "gps guidance": (0.4, 0.6, 1.4, 1.2),
        "ai scene description": (0.5, 0.7, 0.9, 1.6),
        "remote visual assistance": (0.7, 0.8, 1.0, 1.5),
        "indoor landmark navigation": (0.5, 0.3, 0.8, 1.8),
    }.get(str(device.get("type", "")).strip().lower(), (1.0, 1.0, 1.0, 1.0))

    _set_override(overrides, "CANE_OBSTACLE_PROB", 0.00179 * obstacle_factor * device_factor[0])
    _set_override(overrides, "CANE_CURB_PROB", 0.02731 * max(obstacle_factor, tactile_factor * 0.6))
    _set_override(overrides, "CANE_WALL_PROB", 0.00507 * obstacle_factor)
    _set_override(overrides, "CANE_RAILING_PROB", 0.00377 * obstacle_factor)
    _set_override(overrides, "SOUND_VEHICLE_APPROACH_PROB", 0.00142 * vehicle_factor * traffic_factor * device_factor[1])
    _set_override(overrides, "SOUND_VEHICLE_APPROACH_CROSSING_PROB", 0.00574 * vehicle_factor * crossing_factor * traffic_factor)
    _set_override(overrides, "SOUND_HORN_PROB", 0.000167 * traffic_factor * vehicle_factor)
    _set_override(overrides, "SOUND_REVERSE_BEEP_PROB", 0.000251 * traffic_factor * vehicle_factor)
    _set_override(overrides, "SOUND_HUMAN_ACTIVITY_PROB", 0.01036 * crowd_factor * max(people_factor, 0.2))
    _set_override(overrides, "CROSSING_HORN_PROB", 0.00313 * crossing_factor * traffic_factor)
    _set_override(overrides, "CROSSING_HUMAN_ACTIVITY_PROB", 0.02402 * crossing_factor * crowd_factor)

    landmark_strength = max(landmark_factor, text_factor * 0.8, entrance_factor * 0.9) * device_factor[3]
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
    manual_share = 0.20 + (0.15 if walking.get("require_cane_confirmation") == "Yes" else 0.0)
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
    _set_override(overrides, "PROBE_RELIEF_RATIO", 0.42 if walking.get("require_cane_confirmation") == "Yes" else 0.22)
    _set_override(overrides, "ACTR_LOAD_RESUME_THRESHOLD", {"Fast": 5.8, "Normal": 5.0, "Slow": 4.4}.get(recovery, 5.0))
    _set_override(overrides, "LOOMING_RESUME_THRESHOLD", {"Fast": 0.16, "Normal": 0.10, "Slow": 0.06}.get(recovery, 0.10))

    trust = _range_value(cognitive, "Perceived Trust", 2)
    information_load = _range_value(cognitive, "Information Load", 2)
    auditory_interference = _range_value(cognitive, "Auditory Interference", 1)
    false_alarm = _range_value(cognitive, "False Alarm Impact", 2)
    missed_alert = _range_value(cognitive, "Missed Alert Risk", 2)
    _set_override(overrides, "MEMORY_ACTIVE_RETRIEVAL_TH", {1: 0.10, 2: 0.07, 3: 0.04}.get(trust, 0.07))
    _set_override(overrides, "ACTR_IW_HIGH_THRESHOLD", {1: 7.0, 2: 6.0, 3: 5.0}.get(information_load, 6.0))
    _set_override(overrides, "ACTR_CENTRAL_WEIGHT", {1: 1.6, 2: 2.0, 3: 2.5}.get(information_load, 2.0))
    _set_override(overrides, "ACTR_AUDITORY_SHARE", overrides["ACTR_AUDITORY_SHARE"] * {1: 0.85, 2: 1.0, 3: 1.2}.get(auditory_interference, 1.0))
    _set_override(overrides, "ACTR_ERROR_BOOST", {1: 1.5, 2: 2.0, 3: 2.8}.get(false_alarm, 2.0))
    _set_override(overrides, "RISK_ERROR_COEF", {1: 0.55, 2: 0.75, 3: 1.0}.get(missed_alert, 0.75))

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


def run_simulation_from_payload(payload):
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
                    _, summary_path = ENGINE_CLI.run_monte_carlo(
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
                    _, _, summary_path = ENGINE_CLI.run(familiarity=familiarity)
                    with open(summary_path, "r", encoding="utf-8") as handle:
                        summary = json.load(handle)
                    result = _summarize_single_run(summary)
                    report_path = summary_path

    return {
        "ok": True,
        "result": result,
        "report_path": str(report_path),
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
