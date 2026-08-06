"""Generate tabular and visual reports from simulation outputs.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import csv
import json
import os
from datetime import datetime
from statistics import mean as _stat_mean, stdev as _stat_stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

try:
    from .config import REPORT_DIR
except ImportError:
    from config import REPORT_DIR

MAP_INTERP_STEP_METERS = 8.0
ACTR_REPORT_HIGH_THRESHOLD = 6.0
ACTR_REPORT_RESUME_THRESHOLD = 5.0
DEVICE_FUNCTION_KEYS = ("obstacle", "terrain", "pedestrian", "vehicle", "guidance", "other")
DEVICE_OUTPUT_SUFFIXES = (
    "detection_probability",
    "detected",
    "alert",
    "alert_active",
    "missed",
    "target",
)
DEVICE_NULL = "null"

PROFILE_FIELDS_LOWER = [
    ("user_id", "USER_ID", "default"),
    ("familiarity_level", "FAMILIARITY_LEVEL", 0.5),
    ("expertise_proxy", "EXPERTISE_PROXY", 0.8),
    ("landmark_expectancy_bonus", "LANDMARK_EXPECTANCY_BONUS", 0.55),
    ("sound_source_threshold", "SOUND_SOURCE_THRESHOLD", 0.4),
    ("d", "D", 0.5),
    ("mas", "MAS", 1.5),
    ("rt", "RT", -2.0),
    ("ans", "ANS", 0.2),
]


def _safe_stdev(values):
    """Handle safe stdev behavior."""
    return _stat_stdev(values) if len(values) >= 2 else 0.0


def _safe_pct(cnt, total):
    """Handle safe pct behavior."""
    return (cnt / total * 100.0) if total else 0.0


def _to_bool(value):
    """Handle to bool behavior."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _edge_length(graph, from_node, to_node):
    """Handle edge length behavior."""
    edge_data = graph.get_edge_data(from_node, to_node)
    if edge_data is None:
        return 1.0

    if isinstance(edge_data, dict) and "length" in edge_data:
        return max(0.1, float(edge_data.get("length", 1.0)))

    lengths = []
    if isinstance(edge_data, dict):
        for attrs in edge_data.values():
            if isinstance(attrs, dict):
                lengths.append(float(attrs.get("length", 1.0)))

    if lengths:
        return max(0.1, min(lengths))
    return 1.0


def _interpolate_points(start_xy, end_xy, distance_m, step_m):
    """Handle interpolate points behavior."""
    if distance_m <= 0:
        return [start_xy, end_xy]
    segments = max(1, int(distance_m // step_m))
    points = []
    for idx in range(segments + 1):
        ratio = idx / segments
        x = start_xy[0] + (end_xy[0] - start_xy[0]) * ratio
        y = start_xy[1] + (end_xy[1] - start_xy[1]) * ratio
        points.append((x, y))
    return points


def _render_actr_charts(sim_log, event_log, report_dir, ts):
    """Handle render actr charts behavior."""
    if not sim_log:
        return {}

    steps_arr = [r["step"] for r in sim_log]
    risk_dbn_arr = [float(r.get("risk_prob", 0.0)) for r in sim_log]
    risk_actr_arr = [
        float(r.get("actr_risk_signal", r.get("risk_prob", 0.0))) for r in sim_log
    ]
    action_arr = [r["next_action"] for r in sim_log]
    iw_arr = [r.get("actr_iw_total", 0.0) for r in sim_log]
    wave_arr = [r.get("actr_wave", 0.0) for r in sim_log]

    landmark_steps = [e["step"] for e in event_log if e["type"] == "landmark_match"]
    landmark_trigger_steps = [
        e["step"] for e in event_log if e["type"] == "landmark_trigger"
    ]
    gate_steps = [e["step"] for e in event_log if e["type"] == "gate_passed"]
    iw_high_steps = [e["step"] for e in event_log if e["type"] == "actr_iw_high"]

    action_colors = {
        "move_direct": "#4CAF50",
        "stop_and_probe": "#FF9800",
        "wait_at_red": "#F44336",
    }

    fig1, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        dpi=150,
        gridspec_kw={"height_ratios": [5, 1.2, 1.2]},
        sharex=True,
    )
    fig1.patch.set_facecolor("#FAFAFA")

    ax = axes[0]
    ax.set_facecolor("#F5F5F5")
    ax.axhspan(
        ACTR_REPORT_HIGH_THRESHOLD,
        max(
            ACTR_REPORT_HIGH_THRESHOLD,
            max(iw_arr) if iw_arr else ACTR_REPORT_HIGH_THRESHOLD,
        ),
        color="#FFCDD2",
        alpha=0.35,
        zorder=0,
        label="_nolegend_",
    )
    ax.axhline(
        ACTR_REPORT_HIGH_THRESHOLD,
        color="#E53935",
        linewidth=0.9,
        linestyle="--",
        alpha=0.85,
        label=f"ACT-R high threshold {ACTR_REPORT_HIGH_THRESHOLD:.1f}",
    )
    ax.axhline(
        ACTR_REPORT_RESUME_THRESHOLD,
        color="#F57C00",
        linewidth=0.8,
        linestyle="-.",
        alpha=0.75,
        label=f"ACT-R resume threshold {ACTR_REPORT_RESUME_THRESHOLD:.1f}",
    )

    ax.plot(
        steps_arr,
        iw_arr,
        color="#5D4037",
        linewidth=1.0,
        alpha=0.85,
        zorder=3,
        label="ACT-R IW(t)",
    )
    ax.plot(
        steps_arr,
        wave_arr,
        color="#1565C0",
        linewidth=1.1,
        alpha=0.92,
        zorder=4,
        label="ACT-R W_ave",
    )

    lm_iw = [iw_arr[s - 1] for s in landmark_steps if 0 < s <= len(iw_arr)]
    lm_trigger_iw = [
        iw_arr[s - 1] for s in landmark_trigger_steps if 0 < s <= len(iw_arr)
    ]
    gt_iw = [iw_arr[s - 1] for s in gate_steps if 0 < s <= len(iw_arr)]
    iw_high_vals = [iw_arr[s - 1] for s in iw_high_steps if 0 < s <= len(iw_arr)]
    ax.scatter(
        landmark_steps[: len(lm_iw)],
        lm_iw,
        s=16,
        marker="v",
        color="#66BB6A",
        zorder=6,
        label="Landmark active steps",
    )
    ax.scatter(
        landmark_trigger_steps[: len(lm_trigger_iw)],
        lm_trigger_iw,
        s=38,
        marker="D",
        color="#1B5E20",
        zorder=7,
        label="Landmark trigger events",
    )
    ax.scatter(
        gate_steps[: len(gt_iw)],
        gt_iw,
        s=22,
        marker="*",
        color="#AB47BC",
        zorder=6,
        label="Attention gate passed",
    )
    ax.scatter(
        iw_high_steps[: len(iw_high_vals)],
        iw_high_vals,
        s=18,
        marker="x",
        color="#BF360C",
        zorder=6,
        label=f"IW high events (≥{ACTR_REPORT_HIGH_THRESHOLD:.0f})",
    )

    ax.set_ylabel("ACT-R Load", fontsize=9)
    ax.set_ylim(0, max(6.5, (max(iw_arr) if iw_arr else 0.0) * 1.10))
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.75)
    ax.set_title(f"BVI ACT-R Load Overview ({len(steps_arr)} steps)", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    ax2 = axes[1]
    ax2.set_facecolor("#F5F5F5")
    ax2.fill_between(
        steps_arr,
        risk_dbn_arr,
        color="#FFCDD2",
        alpha=0.55,
        linewidth=0,
        label="DBN risk",
    )
    ax2.plot(
        steps_arr,
        risk_dbn_arr,
        color="#C62828",
        linewidth=0.75,
        alpha=0.85,
        label="DBN risk",
    )
    ax2.plot(
        steps_arr,
        risk_actr_arr,
        color="#1565C0",
        linewidth=0.95,
        alpha=0.9,
        label="ACT-R risk",
    )
    ax2.set_ylabel("Risk Signal", fontsize=8)
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.legend(fontsize=7, loc="upper right", framealpha=0.8)

    ax3 = axes[2]
    ax3.set_facecolor("#F0F0F0")
    action_idx = {"move_direct": 1, "stop_and_probe": 2, "wait_at_red": 3}
    action_labels = {1: "move_direct", 2: "stop_probe", 3: "wait_red"}
    for step, action in zip(steps_arr, action_arr):
        y = action_idx.get(action, 0)
        color = action_colors.get(action, "#9E9E9E")
        ax3.bar(step, 1, bottom=y - 1, width=1.0, color=color, alpha=0.75, linewidth=0)
    ax3.set_yticks([0.5, 1.5, 2.5])
    ax3.set_yticklabels([action_labels.get(i + 1, "") for i in range(3)], fontsize=7)
    ax3.set_ylim(0, 3)
    ax3.set_xlabel("Step", fontsize=9)

    from matplotlib.patches import Patch

    legend_patches = [
        Patch(color=c, alpha=0.75, label=a) for a, c in action_colors.items()
    ]
    ax3.legend(
        handles=legend_patches, fontsize=7, loc="upper right", framealpha=0.8, ncol=2
    )

    plt.tight_layout(h_pad=0.4)
    actr_overview_path = os.path.join(report_dir, f"sim_actr_dashboard_{ts}.png")
    fig1.savefig(actr_overview_path, dpi=180, bbox_inches="tight")
    plt.close(fig1)

    return {
        "actr_overview": actr_overview_path,
    }


def _is_truthy(value):
    """Handle is truthy behavior."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _as_float(row, *keys, default=0.0):
    """Handle as float behavior."""
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            try:
                return float(row.get(key))
            except (TypeError, ValueError):
                continue
    return float(default)


def _as_int(row, *keys, default=0):
    """Handle as int behavior."""
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            try:
                return int(float(row.get(key)))
            except (TypeError, ValueError):
                continue
    return int(default)


def _row_guidance_flags(row):
    """Return route-reference flags from tactile/cane/landmark state."""
    guidance_text = " ".join(
        str(row.get(key, ""))
        for key in (
            "cane_guidance_type",
            "dominant_cane_type",
            "matched_landmark",
            "surface_type",
        )
    ).lower()
    matched_landmark = str(row.get("matched_landmark", "none")).strip().lower()
    surface_type = str(row.get("surface_type", "")).strip().lower()

    tactile = (
        "tactile" in guidance_text
        or surface_type == "tactile_guidance"
        or row.get("surface_cn") == "盲道"
    )
    wall = "wall" in guidance_text or "墙" in guidance_text
    railing = "railing" in guidance_text or "栏杆" in guidance_text
    landmark = (
        matched_landmark not in {"", "none", "nan"}
        or _is_truthy(row.get("landmark_triggered", False))
        or _is_truthy(row.get("landmark_episode_active", False))
    )
    route = (
        tactile
        or wall
        or railing
        or landmark
        or _is_truthy(row.get("cane_guidance_present", False))
        or _is_truthy(row.get("spatial_anchored", False))
    )
    return {
        "tactile_paving": tactile,
        "wall": wall,
        "railing": railing,
        "route": route,
        "landmark": landmark,
        "none": not route and not landmark,
    }


def _row_reference_present(row):
    """Return whether any navigation reference is available on this step."""
    flags = _row_guidance_flags(row)
    return any(
        flags[key] for key in ("tactile_paving", "wall", "railing", "route", "landmark")
    )


def _device_csv_value(row, field):
    value = row.get(field, DEVICE_NULL)
    if value is None or value == "":
        return DEVICE_NULL
    return value


def _write_single_simulation_data_csv(sim_log, report_dir, ts, run_id=1, seed=""):
    """Write a compact per-step CSV for intervention-effect analysis.

    Environmental events and device outputs are deliberately separated:
    ``environment_*`` records what exists in the simulated world, while
    ``device_*`` records whether the assistive device detects and alerts on it.
    Device configuration must never be interpreted as changing the underlying
    environmental event itself.
    """
    csv_path = os.path.join(report_dir, f"single_simulation_data_{ts}.csv")
    fieldnames = [
        "run",
        "seed",
        "step",
        "simulation_time",
        "segment_id",
        "intersection_state",
        "light_state",
        "crossing_subphase",
        "surface_type",
        "crowd_density",
        "traffic_density",
        "reference_tactile_paving",
        "reference_wall",
        "reference_railing",
        "reference_landmark",
        "reference_none",
        "environment_vehicle_approach",
        "environment_horn",
        "environment_pedestrian",
        "environment_obstacle",
        "environment_obstacle_type",
        "environment_terrain_event",
        "environment_terrain_type",
        "environment_guidance_object",
        "environment_guidance_type",
        "environment_other",
        "environment_other_type",
        "device_alert_active",
        "device_alert_source",
        "device_alert_modality",
        "device_auditory_demand",
        "device_tactile_demand",
        "device_manual_demand",
        *[
            f"device_{function_key}_{suffix}"
            for function_key in DEVICE_FUNCTION_KEYS
            for suffix in DEVICE_OUTPUT_SUFFIXES
        ],
        "risk",
        "IW",
        "W_ave",
        "move_direct",
        "stop_and_probe",
        "wait",
        "position_change",
        "reference_anchored",
        "guidance_absent_steps",
        "memory_retrieval",
        "load_auditory",
        "load_tactile",
        "load_manual",
        "load_central",
        "load_memory",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sim_log:
            flags = _row_guidance_flags(row)
            action = str(row.get("next_action", ""))
            edge_from = row.get("edge_from", "")
            edge_to = row.get("edge_to", "")
            segment_id = (
                f"{edge_from}->{edge_to}"
                if edge_from not in (None, "") and edge_to not in (None, "")
                else str(row.get("position", ""))
            )
            if _is_truthy(row.get("crossing_active", False)):
                intersection_state = "crossing"
            elif _is_truthy(row.get("at_intersection", False)):
                intersection_state = "intersection"
            else:
                intersection_state = "segment"

            position_change = _as_float(row, "step_travel_m", "step_len_m", default=0.0)
            dominant_cane = str(row.get("dominant_cane_type", "none")).strip().lower()
            environment_obstacle = _is_truthy(row.get("cane_obstacle", False))
            environment_obstacle_type = str(
                row.get("environment_obstacle_type", row.get("obstacle_type", ""))
            ).strip().lower()
            if environment_obstacle and environment_obstacle_type in {"", "none", "nan"}:
                environment_obstacle_type = (
                    dominant_cane if dominant_cane == "obstacle" else "obstacle"
                )
            if not environment_obstacle:
                environment_obstacle_type = "none"

            writer.writerow(
                {
                    "run": row.get("run", run_id),
                    "seed": row.get("seed", seed),
                    "step": row.get("step", ""),
                    "simulation_time": row.get("sim_time", ""),
                    "segment_id": segment_id,
                    "intersection_state": intersection_state,
                    "light_state": row.get("light_state", ""),
                    "crossing_subphase": row.get("crossing_subphase", ""),
                    "surface_type": row.get("surface_type", ""),
                    "crowd_density": row.get("crowd_density", ""),
                    "traffic_density": row.get("traffic_density", ""),
                    "reference_tactile_paving": int(flags["tactile_paving"]),
                    "reference_wall": int(flags["wall"]),
                    "reference_railing": int(flags["railing"]),
                    "reference_landmark": int(flags["landmark"]),
                    "reference_none": int(flags["none"]),
                    "environment_vehicle_approach": int(
                        _is_truthy(row.get("vehicle_approach", False))
                        or str(row.get("dominant_sound_type", "")) == "vehicle_approach"
                    ),
                    "environment_horn": int(
                        _is_truthy(row.get("snd_horn", False))
                        or str(row.get("dominant_sound_type", "")) == "horn"
                    ),
                    "environment_pedestrian": int(
                        _is_truthy(row.get("human_activity_triggered", False))
                        or _is_truthy(row.get("human_activity", False))
                    ),
                    "environment_obstacle": int(environment_obstacle),
                    "environment_obstacle_type": environment_obstacle_type,
                    "environment_terrain_event": int(
                        _is_truthy(row.get("surface_change", False))
                        or _is_truthy(row.get("just_entered_intersection", False))
                    ),
                    "environment_terrain_type": (
                        "intersection"
                        if _is_truthy(row.get("just_entered_intersection", False))
                        else row.get("surface_type", "none")
                        if _is_truthy(row.get("surface_change", False))
                        else "none"
                    ),
                    "environment_guidance_object": int(
                        _is_truthy(row.get("landmark_triggered", False))
                        or _is_truthy(row.get("cane_guidance_present", False))
                    ),
                    "environment_guidance_type": (
                        "landmark"
                        if _is_truthy(row.get("landmark_triggered", False))
                        else row.get("cane_guidance_type", "none")
                        if _is_truthy(row.get("cane_guidance_present", False))
                        else "none"
                    ),
                    "environment_other": int(
                        _is_truthy(row.get("snd_horn", False))
                        or _is_truthy(row.get("snd_reverse_beep", False))
                    ),
                    "environment_other_type": (
                        "horn"
                        if _is_truthy(row.get("snd_horn", False))
                        else "reverse_beep"
                        if _is_truthy(row.get("snd_reverse_beep", False))
                        else "none"
                    ),
                    "device_alert_active": row.get("device_alert_active", 0),
                    "device_alert_source": row.get("device_alert_source", "none"),
                    "device_alert_modality": row.get("device_alert_modality", "none"),
                    "device_auditory_demand": row.get("device_auditory_demand", 0.0),
                    "device_tactile_demand": row.get("device_tactile_demand", 0.0),
                    "device_manual_demand": row.get("device_manual_demand", 0.0),
                    **{
                        f"device_{function_key}_{suffix}": _device_csv_value(
                            row, f"device_{function_key}_{suffix}"
                        )
                        for function_key in DEVICE_FUNCTION_KEYS
                        for suffix in DEVICE_OUTPUT_SUFFIXES
                    },
                    "risk": row.get("actr_risk_signal", row.get("risk_prob", "")),
                    "IW": row.get("actr_iw_total", ""),
                    "W_ave": row.get("actr_wave", ""),
                    "move_direct": int(action == "move_direct"),
                    "stop_and_probe": int(action == "stop_and_probe"),
                    "wait": int(action.startswith("wait")),
                    "position_change": round(position_change, 6),
                    "reference_anchored": int(
                        _is_truthy(row.get("spatial_anchored", False))
                        or _row_reference_present(row)
                    ),
                    "guidance_absent_steps": row.get("guidance_absent_steps", ""),
                    "memory_retrieval": int(_is_truthy(row.get("actr_memory_active", False))),
                    "load_auditory": row.get("actr_iw_auditory", ""),
                    "load_tactile": row.get("actr_iw_tactile", ""),
                    "load_manual": row.get("actr_iw_manual", ""),
                    "load_central": row.get("actr_iw_central", ""),
                    "load_memory": row.get("actr_iw_memory", ""),
                }
            )
    return csv_path


def _csv_float(row, key, default=0.0):
    """Read a numeric value from a compact report row."""
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _csv_true(row, key):
    """Read a boolean-like value from a compact report row."""
    return _is_truthy(row.get(key, False))


def _csv_is_null(value):
    return value is None or str(value).strip().lower() in {"", "null", "none", "nan"}


def _onset_indices(rows, predicate):
    """Return episode onsets instead of counting every active step as an event."""
    indices = []
    previous = False
    for idx, row in enumerate(rows):
        current = bool(predicate(row))
        if current and not previous:
            indices.append(idx)
        previous = current
    return indices


def _response_metrics(rows, event_indices, window=5):
    """Calculate response probability and delay after event onsets."""
    if not event_indices:
        return 0, 0.0, 0.0
    response_count = 0
    delays = []
    for idx in event_indices:
        end = min(len(rows), idx + max(1, int(window)) + 1)
        previous_active = False
        if idx > 0:
            previous_active = (
                _csv_true(rows[idx - 1], "stop_and_probe")
                or _csv_true(rows[idx - 1], "wait")
            )
        for j in range(idx, end):
            current_active = (
                _csv_true(rows[j], "stop_and_probe")
                or _csv_true(rows[j], "wait")
            )
            if current_active and not previous_active:
                response_count += 1
                t0 = _csv_float(rows[idx], "simulation_time", idx)
                t1 = _csv_float(rows[j], "simulation_time", j)
                delays.append(max(0.0, t1 - t0))
                break
            previous_active = current_active
    probability = response_count / len(event_indices)
    delay_mean = _stat_mean(delays) if delays else 0.0
    return response_count, probability, delay_mean


def _compact_scenario_specs(rows):
    """Define the small set of contexts needed for intervention evaluation."""
    return [
        (
            "overall",
            lambda row: True,
            lambda row: (
                _csv_true(row, "environment_obstacle")
                or _csv_true(row, "environment_vehicle_approach")
                or _csv_true(row, "environment_horn")
            ),
        ),
        (
            "obstacle",
            lambda row: _csv_true(row, "environment_obstacle"),
            lambda row: _csv_true(row, "environment_obstacle"),
        ),
        (
            "small_obstacle",
            lambda row: _csv_true(row, "environment_obstacle")
            and any(
                token in str(row.get("environment_obstacle_type", "")).lower()
                for token in ("small", "ground", "low")
            ),
            lambda row: _csv_true(row, "environment_obstacle"),
        ),
        (
            "vehicle_approach",
            lambda row: _csv_true(row, "environment_vehicle_approach"),
            lambda row: _csv_true(row, "environment_vehicle_approach"),
        ),
        (
            "intersection",
            lambda row: str(row.get("intersection_state", "")).lower()
            in {"intersection", "crossing"},
            lambda row: str(row.get("intersection_state", "")).lower()
            in {"intersection", "crossing"},
        ),
        (
            "high_crowd",
            lambda row: str(row.get("crowd_density", "")).strip().lower() == "high",
            lambda row: str(row.get("crowd_density", "")).strip().lower() == "high",
        ),
        (
            "no_reference",
            lambda row: _csv_true(row, "reference_none"),
            lambda row: _csv_true(row, "reference_none"),
        ),
    ]


def write_scenario_summary_from_csv(csv_path, output_path=None, context=None):
    """Create one concise scenario-level analysis file from a per-step CSV.

    The output intentionally contains only the values needed to evaluate
    intervention direction, effect size, contextual boundaries, benefit-cost
    trade-offs, and later mechanism diagnostics.
    """
    context = dict(context or {})
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if output_path is None:
        source = os.path.basename(str(csv_path))
        suffix = source.replace("single_simulation_data_", "").rsplit(".csv", 1)[0]
        output_path = os.path.join(os.path.dirname(str(csv_path)), f"scenario_summary_{suffix}.csv")

    fieldnames = [
        "run",
        "seed",
        "condition",
        "function_config",
        "model_variant",
        "scenario",
        "total_steps",
        "environment_event_count",
        "obstacle_event_count",
        "device_detected_count",
        "device_alert_count",
        "device_detection_rate",
        *[
            f"device_{function_key}_{metric}"
            for function_key in DEVICE_FUNCTION_KEYS
            for metric in ("event_count", "detected_count", "alert_count", "detection_rate")
        ],
        "response_probability",
        "response_delay_mean_s",
        "stop_probe_rate_per100steps",
        "green_crossing_probe_reorientation_ratio",
        "risk_actr_mean",
        "workload_mean",
        "workload_peak",
        "high_load_ratio",
    ]

    output_rows = []
    for scenario, selector, trigger in _compact_scenario_specs(rows):
        selected = [row for row in rows if selector(row)]
        if scenario != "overall" and not selected:
            continue

        # Use onsets from the full time series so episode boundaries are retained.
        event_indices = _onset_indices(rows, trigger)
        if scenario != "overall":
            event_indices = [idx for idx in event_indices if selector(rows[idx])]

        _, response_probability, response_delay = _response_metrics(rows, event_indices)
        total_steps = len(selected)
        obstacle_event_indices = _onset_indices(
            rows,
            lambda row: _csv_true(row, "environment_obstacle") and selector(row),
        )
        obstacle_event_count = len(obstacle_event_indices)
        device_detected_count = len(
            _onset_indices(
                rows,
                lambda row: selector(row)
                and _csv_true(row, "device_obstacle_detected"),
            )
        )
        device_alert_count = len(
            _onset_indices(
                rows,
                lambda row: selector(row)
                and _csv_true(row, "device_obstacle_alert"),
            )
        )
        device_detection_rate = (
            device_detected_count / obstacle_event_count if obstacle_event_count else 0.0
        )
        function_event_columns = {
            "obstacle": "environment_obstacle",
            "terrain": "environment_terrain_event",
            "pedestrian": "environment_pedestrian",
            "vehicle": "environment_vehicle_approach",
            "guidance": "environment_guidance_object",
            "other": "environment_other",
        }
        function_metrics = {}
        for function_key, event_column in function_event_columns.items():
            present = any(
                not _csv_is_null(row.get(f"device_{function_key}_detected"))
                for row in rows
            )
            if not present:
                function_metrics.update(
                    {
                        f"device_{function_key}_event_count": DEVICE_NULL,
                        f"device_{function_key}_detected_count": DEVICE_NULL,
                        f"device_{function_key}_alert_count": DEVICE_NULL,
                        f"device_{function_key}_detection_rate": DEVICE_NULL,
                    }
                )
                continue
            event_count = len(
                _onset_indices(
                    rows,
                    lambda row, column=event_column: selector(row) and _csv_true(row, column),
                )
            )
            detected_count = len(
                _onset_indices(
                    rows,
                    lambda row, key=function_key: selector(row)
                    and _csv_true(row, f"device_{key}_detected"),
                )
            )
            alert_count = len(
                _onset_indices(
                    rows,
                    lambda row, key=function_key: selector(row)
                    and _csv_true(row, f"device_{key}_alert"),
                )
            )
            function_metrics.update(
                {
                    f"device_{function_key}_event_count": event_count,
                    f"device_{function_key}_detected_count": detected_count,
                    f"device_{function_key}_alert_count": alert_count,
                    f"device_{function_key}_detection_rate": round(
                        detected_count / event_count if event_count else 0.0, 6
                    ),
                }
            )
        stop_probe_count = sum(
            1 for row in selected if _csv_true(row, "stop_and_probe")
        )
        # During green-light crossing, ``stop_and_probe`` represents probing,
        # confirmation, disorientation correction, or reorientation performed
        # while the agent continues crossing. It must not be reported as a
        # literal stopping rate. This is a reporting-only reclassification and
        # does not alter the ACT-R decision or movement mechanism.
        green_crossing_rows = [
            row
            for row in selected
            if str(row.get("intersection_state", "")).strip().lower() == "crossing"
            and str(row.get("light_state", "")).strip().lower() == "green"
        ]
        green_crossing_probe_reorientation_count = sum(
            1
            for row in green_crossing_rows
            if _csv_true(row, "stop_and_probe")
        )
        green_crossing_probe_reorientation_ratio = (
            green_crossing_probe_reorientation_count / len(green_crossing_rows)
            if green_crossing_rows
            else 0.0
        )
        # ``risk`` in the concise per-step file is the ACT-R decision risk
        # signal, not the raw DBN posterior. Keep that meaning explicit here.
        risk_actr_values = [_csv_float(row, "risk") for row in selected] or [0.0]
        workload_values = [_csv_float(row, "IW") for row in selected] or [0.0]
        high_load_count = sum(
            1 for value in workload_values if value >= ACTR_REPORT_HIGH_THRESHOLD
        )

        output_rows.append(
            {
                "run": context.get("run", rows[0].get("run", "") if rows else ""),
                "seed": context.get("seed", rows[0].get("seed", "") if rows else ""),
                "condition": context.get("condition", "control"),
                "function_config": context.get("function_config", "baseline"),
                "model_variant": context.get("model_variant", "full"),
                "scenario": scenario,
                "total_steps": total_steps,
                "environment_event_count": len(event_indices),
                "obstacle_event_count": obstacle_event_count,
                "device_detected_count": device_detected_count,
                "device_alert_count": device_alert_count,
                "device_detection_rate": round(device_detection_rate, 6),
                **function_metrics,
                "response_probability": round(response_probability, 6),
                "response_delay_mean_s": round(response_delay, 6),
                "stop_probe_rate_per100steps": round(
                    (stop_probe_count / total_steps * 100.0) if total_steps else 0.0,
                    6,
                ),
                "green_crossing_probe_reorientation_ratio": round(
                    green_crossing_probe_reorientation_ratio, 6
                ),
                "risk_actr_mean": round(_stat_mean(risk_actr_values), 6),
                "workload_mean": round(_stat_mean(workload_values), 6),
                "workload_peak": round(max(workload_values), 6),
                "high_load_ratio": round(
                    high_load_count / total_steps if total_steps else 0.0, 6
                ),
            }
        )

    with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return str(output_path)

def _count_true(rows, *keys):
    """Handle count true behavior."""
    return sum(
        1 for row in rows if any(_is_truthy(row.get(key, False)) for key in keys)
    )


def _count_action(rows, action_name):
    """Handle count action behavior."""
    return sum(1 for row in rows if row.get("next_action") == action_name)


def _mean_for(rows, *keys, default=0.0):
    """Handle mean for behavior."""
    vals = [_as_float(row, *keys, default=default) for row in rows]
    return _stat_mean(vals) if vals else 0.0


def _max_for(rows, *keys, default=0.0):
    """Handle max for behavior."""
    vals = [_as_float(row, *keys, default=default) for row in rows]
    return max(vals) if vals else 0.0


def _streak_lengths(flags):
    """Handle streak lengths behavior."""
    streaks = []
    cur = 0
    for flag in flags:
        if flag:
            cur += 1
        elif cur:
            streaks.append(cur)
            cur = 0
    if cur:
        streaks.append(cur)
    return streaks


def _event_after_rate(rows, trigger_fn, response_fn, window=3):
    """Handle event after rate behavior."""
    trigger_indices = [idx for idx, row in enumerate(rows) if trigger_fn(row)]
    if not trigger_indices:
        return 0.0
    hits = 0
    for idx in trigger_indices:
        end = min(len(rows), idx + window + 1)
        if any(response_fn(rows[j]) for j in range(idx, end)):
            hits += 1
    return hits / len(trigger_indices)


def _mean_drop_after(rows, trigger_fn, value_keys, window=3):
    """Handle mean drop after behavior."""
    drops = []
    for idx, row in enumerate(rows):
        if not trigger_fn(row):
            continue
        before = _as_float(row, *value_keys)
        end = min(len(rows), idx + window + 1)
        after_values = [_as_float(rows[j], *value_keys) for j in range(idx + 1, end)]
        if after_values:
            drops.append(before - min(after_values))
    return _stat_mean(drops) if drops else 0.0


def _scenario_label(key):
    """Handle scenario label behavior."""
    labels = {
        "intersection": "S1 路口/过街场景",
        "tactile_guidance": "S2 盲道/触觉引导场景",
        "flat_road": "S3 平整人行道场景",
        "uneven_natural": "S4 不平整自然路面场景",
        "slope_surface": "S5 坡道场景",
        "height_drop": "S6 高度落差场景",
        "overall": "整体汇总",
    }
    return labels.get(key, key)


def _rows_for_scenario(sim_log, scenario_key):
    """Handle rows for scenario behavior."""
    if scenario_key == "overall":
        return list(sim_log)
    if scenario_key == "intersection":
        return [row for row in sim_log if _is_truthy(row.get("crossing_active", False))]
    return [row for row in sim_log if row.get("surface_type") == scenario_key]


def _summarize_typical_outputs(rows):
    """Handle summarize typical outputs behavior."""
    total = len(rows)
    if total == 0:
        return None

    probe_flags = [row.get("next_action") == "stop_and_probe" for row in rows]
    high_load_flags = [
        _as_float(row, "actr_iw_total", "actr_wave") >= ACTR_REPORT_HIGH_THRESHOLD
        for row in rows
    ]
    overloaded_flags = [
        str(row.get("imaginal_load_state", row.get("load_state", ""))) == "overloaded"
        for row in rows
    ]
    reference_absent_flags = [
        not _is_truthy(row.get("spatial_anchored", False)) for row in rows
    ]
    probe_streaks = _streak_lengths(probe_flags)
    high_load_streaks = _streak_lengths(high_load_flags)
    ref_absent_streaks = _streak_lengths(reference_absent_flags)

    hazard_fn = lambda row: (
        _as_float(row, "risk_prob") >= 0.6
        or _as_float(row, "actr_risk_signal", "risk_prob") >= 0.6
        or row.get("risk_label") == "high"
        or row.get("actr_risk_label") == "high"
        or row.get("dominant_sound_type")
        in {"vehicle_approach", "horn", "reverse_beep"}
        or row.get("dominant_cane_type") in {"obstacle", "curb", "wall", "railing"}
    )
    vehicle_fn = lambda row: row.get(
        "dominant_sound_type"
    ) == "vehicle_approach" or _is_truthy(row.get("vehicle_approach", False))
    response_fn = lambda row: row.get("next_action") in {
        "stop_and_probe",
        "wait_at_red",
    }
    nav_fn = lambda row: _is_truthy(row.get("nav_announcement", False)) or _is_truthy(
        row.get("actr_nav_announcement", False)
    )

    action_counts = {}
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    for row in rows:
        action = row.get("next_action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        risk_label = row.get("actr_risk_label", row.get("risk_label", "low"))
        risk_counts[risk_label] = risk_counts.get(risk_label, 0) + 1

    sim_times = [_as_float(row, "sim_time") for row in rows if "sim_time" in row]
    total_time = (max(sim_times) - min(sim_times)) if len(sim_times) >= 2 else 0.0

    return {
        "total_steps": total,
        "total_time_s": total_time,
        "mean_step_time_s": total_time / total if total_time > 0 else 0.0,
        "mean_risk_prob": _mean_for(rows, "risk_prob"),
        "peak_risk_prob": _max_for(rows, "risk_prob"),
        "mean_actr_risk": _mean_for(rows, "actr_risk_signal", "risk_prob"),
        "peak_actr_risk": _max_for(rows, "actr_risk_signal", "risk_prob"),
        "mean_seev_salience": _mean_for(rows, "salience", "seev_salience"),
        "peak_seev_salience": _max_for(rows, "salience", "seev_salience"),
        "hazard_response_rate": _event_after_rate(rows, hazard_fn, response_fn),
        "stop_after_hazard_rate": _event_after_rate(
            rows, hazard_fn, lambda row: row.get("next_action") == "stop_and_probe"
        ),
        "cane_relief_effect": _mean_drop_after(
            rows,
            lambda row: row.get("dominant_cane_type", "none") != "none"
            or _is_truthy(row.get("cane_guidance", False)),
            ("actr_risk_signal", "risk_prob"),
        ),
        "landmark_relief_effect": _mean_drop_after(
            rows,
            lambda row: row.get("matched_landmark", "none") != "none",
            ("actr_risk_signal", "risk_prob"),
        ),
        "probe_count": _count_action(rows, "stop_and_probe"),
        "probe_rate": _count_action(rows, "stop_and_probe") / total,
        "mean_probe_duration_steps": (
            _stat_mean(probe_streaks) if probe_streaks else 0.0
        ),
        "median_probe_duration_steps": (
            sorted(probe_streaks)[len(probe_streaks) // 2] if probe_streaks else 0.0
        ),
        "max_probe_duration_steps": max(probe_streaks) if probe_streaks else 0.0,
        "mean_load_drop_after_probe": _mean_drop_after(
            rows,
            lambda row: row.get("next_action") == "stop_and_probe",
            ("actr_iw_total", "actr_wave"),
        ),
        "mean_spatial_wm_load": _mean_for(rows, "actr_iw_memory", "spatial_wm_load"),
        "peak_spatial_wm_load": _max_for(rows, "actr_iw_memory", "spatial_wm_load"),
        "memory_retrieval_count": _count_true(
            rows, "actr_memory_active", "memory_retrieval_active"
        ),
        "memory_retrieval_rate": _count_true(
            rows, "actr_memory_active", "memory_retrieval_active"
        )
        / total,
        "reference_absent_streak_mean": (
            _stat_mean(ref_absent_streaks) if ref_absent_streaks else 0.0
        ),
        "reference_absent_streak_max": (
            max(ref_absent_streaks) if ref_absent_streaks else 0.0
        ),
        "mean_landmark_anchor": _mean_for(rows, "landmark_bonus", "landmark_anchor"),
        "landmark_trigger_count": _count_true(rows, "landmark_triggered"),
        "landmark_active_step_count": sum(
            1 for row in rows if row.get("matched_landmark", "none") != "none"
        ),
        "landmark_active_step_rate": sum(
            1 for row in rows if row.get("matched_landmark", "none") != "none"
        )
        / total,
        "mean_landmark_episode_steps": (
            sum(1 for row in rows if row.get("matched_landmark", "none") != "none")
            / max(1, _count_true(rows, "landmark_triggered"))
        ),
        "landmark_match_count": sum(
            1 for row in rows if row.get("matched_landmark", "none") != "none"
        ),
        "vehicle_approach_count": sum(1 for row in rows if vehicle_fn(row)),
        "vehicle_response_rate": _event_after_rate(rows, vehicle_fn, response_fn),
        "stop_after_vehicle_rate": _event_after_rate(
            rows, vehicle_fn, lambda row: row.get("next_action") == "stop_and_probe"
        ),
        "mean_looming_boost_peak": _max_for(
            rows, "looming_boost", "vehicle_looming_boost"
        ),
        "nav_announcement_count": sum(1 for row in rows if nav_fn(row)),
        "mean_wm_peak_after_nav": _mean_drop_after(
            rows, nav_fn, ("actr_iw_total", "actr_wave")
        )
        * -1.0,
        "overload_to_probe_delay_steps": _event_after_rate(
            rows,
            lambda row: str(row.get("imaginal_load_state", row.get("load_state", "")))
            == "overloaded",
            lambda row: row.get("next_action") == "stop_and_probe",
        ),
        "mean_actr_load": _mean_for(rows, "actr_iw_total"),
        "peak_actr_load": _max_for(rows, "actr_iw_total"),
        "mean_actr_wave": _mean_for(rows, "actr_wave"),
        "high_load_step_rate": sum(high_load_flags) / total,
        "overload_streak_mean": (
            _stat_mean(high_load_streaks) if high_load_streaks else 0.0
        ),
        "overload_streak_max": max(high_load_streaks) if high_load_streaks else 0.0,
        "stop_count": _count_action(rows, "stop_and_probe")
        + _count_action(rows, "wait_at_red"),
        "stop_rate": (
            _count_action(rows, "stop_and_probe") + _count_action(rows, "wait_at_red")
        )
        / total,
        "gate_passed_count": _count_true(rows, "gate_passed"),
        "spatial_anchored_rate": _count_true(rows, "spatial_anchored") / total,
        "action_distribution": action_counts,
        "risk_distribution": risk_counts,
    }


def _fmt(value, digits=4):
    """Handle fmt behavior."""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)



def generate_report(
    sim_log,
    event_log,
    profile,
    steps,
    start_node,
    goal_node,
    current_position,
    max_steps,
    graph=None,
    initial_production_utilities=None,
    report_dir=REPORT_DIR,
):
    """Handle generate report behavior."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reached_goal = str(current_position) == str(goal_node)

    iw_values = [r.get("actr_iw_total", 0.0) for r in sim_log] or [0.0]
    wave_values = [r.get("actr_wave", 0.0) for r in sim_log] or [0.0]
    risk_values_dbn = [float(r.get("risk_prob", 0.0)) for r in sim_log] or [0.0]
    risk_values_actr = [
        float(r.get("actr_risk_signal", r.get("risk_prob", 0.0))) for r in sim_log
    ] or [0.0]
    pri_values = [r["net_priority"] for r in sim_log] or [0.0]
    sal_values = [r["salience"] for r in sim_log] or [0.0]
    int_values = [r["intensity"] for r in sim_log] or [0.0]
    risk_abs_diff = [abs(a - b) for a, b in zip(risk_values_dbn, risk_values_actr)] or [
        0.0
    ]

    gate_passed_count = sum(1 for r in sim_log if r["gate_passed"])
    landmark_trigger_count = sum(
        1 for r in sim_log if _is_truthy(r.get("landmark_triggered", False))
    )
    landmark_active_step_count = sum(
        1 for r in sim_log if r["matched_landmark"] != "none"
    )
    landmark_match_count = landmark_active_step_count
    guidance_reference_step_count = sum(
        1 for r in sim_log if _row_guidance_flags(r)["route"]
    )
    reference_active_step_count = sum(1 for r in sim_log if _row_reference_present(r))
    spatial_anchored_count = sum(1 for r in sim_log if r.get("spatial_anchored", False))
    stop_probe_count = sum(1 for r in sim_log if r["next_action"] == "stop_and_probe")
    green_crossing_rows = [
        r
        for r in sim_log
        if _is_truthy(r.get("crossing_active", False))
        and str(r.get("light_state", "")).strip().lower() == "green"
    ]
    green_crossing_probe_reorientation_count = sum(
        1 for r in green_crossing_rows if r.get("next_action") == "stop_and_probe"
    )
    green_crossing_probe_reorientation_ratio = (
        green_crossing_probe_reorientation_count / len(green_crossing_rows)
        if green_crossing_rows
        else 0.0
    )
    actr_iw_high_count = sum(
        1 for r in sim_log if r.get("actr_iw_total", 0.0) >= ACTR_REPORT_HIGH_THRESHOLD
    )
    actr_iw_resume_count = sum(
        1
        for r in sim_log
        if r.get("actr_iw_total", 0.0) >= ACTR_REPORT_RESUME_THRESHOLD
    )

    landmark_stats = {}
    landmark_trigger_stats = {}
    for r in sim_log:
        lm = r["matched_landmark"]
        if lm != "none":
            landmark_stats.setdefault(lm, []).append(r["landmark_bonus"])
        if _is_truthy(r.get("landmark_triggered", False)):
            trigger_lm = lm if lm != "none" else "generic"
            landmark_trigger_stats.setdefault(trigger_lm, []).append(
                r["landmark_bonus"]
            )

    risk_dist_dbn = {"low": 0, "medium": 0, "high": 0}
    risk_dist_actr = {"low": 0, "medium": 0, "high": 0}
    for r in sim_log:
        risk_dist_dbn[r.get("risk_label", "low")] += 1
        risk_dist_actr[r.get("actr_risk_label", r.get("risk_label", "low"))] += 1

    action_dist = {}
    for r in sim_log:
        action_dist[r["next_action"]] = action_dist.get(r["next_action"], 0) + 1

    _has_sim_time = sim_log and "sim_time" in sim_log[0]
    if _has_sim_time:
        _sim_times = [r["sim_time"] for r in sim_log]
        total_sim_time = _sim_times[-1] if _sim_times else 0.0
        _action_time: dict = {}
        for _i, _r in enumerate(sim_log):
            _delta = _r["sim_time"] - (_sim_times[_i - 1] if _i > 0 else 0.0)
            _act = _r["next_action"]
            if _act not in _action_time:
                _action_time[_act] = {"count": 0, "total_s": 0.0}
            _action_time[_act]["count"] += 1
            _action_time[_act]["total_s"] = round(
                _action_time[_act]["total_s"] + _delta, 6
            )
    else:
        total_sim_time = 0.0
        _action_time = {}

    pos_state_counts = {
        "node_normal": 0,
        "node_crossing": 0,
        "edge_crossing": 0,
        "edge_normal": 0,
    }
    for r in sim_log:
        at_n = r.get("at_node", True)
        ca = r.get("crossing_active", False)
        if at_n and not ca:
            pos_state_counts["node_normal"] += 1
        elif at_n and ca:
            pos_state_counts["node_crossing"] += 1
        elif not at_n and ca:
            pos_state_counts["edge_crossing"] += 1
        else:
            pos_state_counts["edge_normal"] += 1

    stop_probe_state_counts = {
        "node_normal": 0,
        "node_crossing": 0,
        "edge_crossing": 0,
        "edge_normal": 0,
    }
    for r in sim_log:
        if r.get("next_action") != "stop_and_probe":
            continue
        at_n = r.get("at_node", True)
        ca = r.get("crossing_active", False)
        if at_n and not ca:
            stop_probe_state_counts["node_normal"] += 1
        elif at_n and ca:
            stop_probe_state_counts["node_crossing"] += 1
        elif not at_n and ca:
            stop_probe_state_counts["edge_crossing"] += 1
        else:
            stop_probe_state_counts["edge_normal"] += 1

    def _infer_probe_source(row):
        """Handle infer probe source behavior."""
        selected = str(row.get("actr_selected_production", "")) or ""
        if selected and selected != "none":
            return selected

        action_source = str(row.get("action_source", ""))
        if action_source == "actr_context_cue":
            cue_summary = (
                f"risk_band={row.get('tick_signal_risk_band', 'low')},"
                f"iw_high={row.get('tick_signal_iw_high', 'no')},"
                f"prev_action={row.get('tick_signal_prev_action', 'none')},"
                f"reference={row.get('tick_signal_reference_now', 'no')}"
            )
            return f"actr_context_cue({cue_summary})"
        if action_source == "actr_bookkeeping":
            return (
                f"actr_bookkeeping(phase={row.get('imaginal_overload_phase','none')}/"
            )
            f"{row.get('imaginal_reference_phase','present')}/{row.get('imaginal_safety_phase','none')})"
        if action_source == "actr_production_competition":
            return "actr_production_competition"

        if (
            row.get("crossing_active", False)
            and row.get("dominant_cane_type") == "obstacle"
        ):
            return "crossing_obstacle_alert"
        if row.get("just_entered_intersection"):
            return "cue_just_entered_crossing_probe"
        if (
            row.get("crossing_active", False)
            and row.get("dominant_cane_type") == "none"
        ):
            return "crossing_guidance_lost"

        if row.get("dominant_sound_type") == "vehicle_approach":
            return (
                "react_vehicle_approach_at_crossing"
                if row.get("crossing_active")
                else "react_vehicle_approach_on_sidewalk"
            )
        if row.get("dominant_sound_type") == "horn":
            return (
                "react_horn_at_crossing"
                if row.get("crossing_active")
                else "react_horn_on_sidewalk"
            )
        if row.get("dominant_sound_type") == "reverse_beep":
            return (
                "react_reverse_beep_at_crossing"
                if row.get("crossing_active")
                else "react_reverse_beep_on_sidewalk"
            )
        if row.get("dominant_cane_type") == "obstacle" or row.get(
            "cane_obstacle", False
        ):
            return "react_cane_obstacle_bottom_up"

        if row.get("imaginal_load_state", row.get("load_state")) == "overloaded":
            return "predict_goal_high_load"
        if (
            row.get("imaginal_load_state", row.get("load_state")) == "normal"
            and row.get(
                "imaginal_risk", row.get("actr_risk_label", row.get("risk_label"))
            )
            == "high"
        ):
            return "predict_goal_high_risk"
        if (
            int(row.get("guidance_absent_steps", 0)) >= 10
            and row.get("imaginal_load_state", row.get("load_state")) == "normal"
        ):
            return "probe_when_spatial_lost"

        if (
            row.get("seev_attention_gated") == "yes"
            and row.get("seev_attention_source") == "sound"
            and row.get("seev_salience_band") == "high"
        ):
            return "attend_gated_sound_high"
        if (
            row.get("seev_attention_gated") == "yes"
            and row.get("seev_attention_source") == "tactile"
            and row.get("seev_salience_band") == "high"
        ):
            return "attend_gated_tactile_high"

        return "unknown_probe_source"

    probe_source_counts = {}
    for r in sim_log:
        if r.get("next_action") != "stop_and_probe":
            continue
        src = _infer_probe_source(r)
        probe_source_counts[src] = probe_source_counts.get(src, 0) + 1

    csv_path = os.path.join(report_dir, f"sim_data_{ts}.csv")
    if sim_log:
        fieldnames = list(sim_log[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sim_log)

    module_csv_path = os.path.join(report_dir, f"sim_module_data_{ts}.csv")
    module_fields = [
        "step",
        "actr_iw_auditory",
        "actr_iw_tactile",
        "actr_iw_manual",
        "actr_iw_central",
        "actr_iw_memory",
        "actr_auditory_active",
        "actr_tactile_active",
        "actr_manual_active",
        "actr_central_active",
        "actr_memory_active",
        "actr_auditory_error",
        "actr_tactile_error",
        "actr_manual_error",
        "actr_central_error",
        "actr_memory_error",
        "actr_dt_auditory",
        "actr_dt_tactile",
        "actr_dt_manual",
        "actr_dt_central",
        "actr_dt_memory",
    ]
    with open(module_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=module_fields)
        writer.writeheader()
        for row in sim_log:
            writer.writerow({field: row.get(field, 0.0) for field in module_fields})

    single_simulation_csv_path = _write_single_simulation_data_csv(
        sim_log, report_dir, ts
    )
    scenario_summary_csv_path = write_scenario_summary_from_csv(
        single_simulation_csv_path
    )

    summary = {
        "timestamp": ts,
        "user_profile": profile,
        "network": {
            "start": str(start_node),
            "goal": str(goal_node),
        },
        "result": {
            "total_steps": steps,
            "reached_goal": reached_goal,
            "gate_passed_count": gate_passed_count,
            "spatial_anchored_count": spatial_anchored_count,
            "landmark_trigger_count": landmark_trigger_count,
            "landmark_active_step_count": landmark_active_step_count,
            "landmark_match_count": landmark_match_count,
            "guidance_reference_step_count": guidance_reference_step_count,
            "reference_active_step_count": reference_active_step_count,
            "stop_probe_count": stop_probe_count,
            "green_crossing_probe_reorientation_count": green_crossing_probe_reorientation_count,
            "green_crossing_probe_reorientation_ratio": round(
                green_crossing_probe_reorientation_ratio, 6
            ),
            "actr_iw_high_count": actr_iw_high_count,
            "actr_iw_resume_count": actr_iw_resume_count,
        },
        "report_metric_labels_zh": {
            "green_crossing_probe_reorientation_ratio": "绿灯穿越阶段探测与重新定向行为比例"
        },
        "statistics": {
            "actr_iw": {
                "mean": round(_stat_mean(iw_values), 4),
                "std": round(_safe_stdev(iw_values), 4),
                "min": round(min(iw_values), 4),
                "max": round(max(iw_values), 4),
            },
            "actr_wave": {
                "mean": round(_stat_mean(wave_values), 4),
                "std": round(_safe_stdev(wave_values), 4),
                "min": round(min(wave_values), 4),
                "max": round(max(wave_values), 4),
            },
            "risk": {
                "mean": round(_stat_mean(risk_values_actr), 4),
                "std": round(_safe_stdev(risk_values_actr), 4),
            },
            "risk_dbn": {
                "mean": round(_stat_mean(risk_values_dbn), 4),
                "std": round(_safe_stdev(risk_values_dbn), 4),
            },
            "risk_actr": {
                "mean": round(_stat_mean(risk_values_actr), 4),
                "std": round(_safe_stdev(risk_values_actr), 4),
            },
            "risk_alignment": {
                "mae": round(_stat_mean(risk_abs_diff), 4),
                "std_abs_diff": round(_safe_stdev(risk_abs_diff), 4),
            },
            "net_priority": {
                "mean": round(_stat_mean(pri_values), 4),
                "std": round(_safe_stdev(pri_values), 4),
            },
            "salience": {
                "mean": round(_stat_mean(sal_values), 4),
                "std": round(_safe_stdev(sal_values), 4),
            },
            "intensity": {
                "mean": round(_stat_mean(int_values), 4),
                "std": round(_safe_stdev(int_values), 4),
            },
        },
        "risk_distribution": risk_dist_actr,
        "risk_distribution_dbn": risk_dist_dbn,
        "risk_distribution_actr": risk_dist_actr,
        "action_distribution": action_dist,
        "landmark_match_stats": {
            k: {"count": len(v), "mean_bonus": round(_stat_mean(v), 4)}
            for k, v in landmark_stats.items()
        },
        "landmark_trigger_stats": {
            k: {"count": len(v), "mean_bonus": round(_stat_mean(v), 4)}
            for k, v in landmark_trigger_stats.items()
        },
        "key_events_count": len(event_log),
    }

    env_types = [
        "intersection",
        "tactile_guidance",
        "flat_road",
        "uneven_natural",
        "slope_surface",
        "height_drop",
    ]
    env_stats = {}
    for env_type in env_types:
        env_rows = [
            r
            for r in sim_log
            if r.get("surface_type") == env_type
            or (env_type == "intersection" and r.get("crossing_active"))
        ]
        if env_rows:
            lm_active_steps = sum(
                1 for r in env_rows if r["matched_landmark"] != "none"
            )
            reference_active_steps = sum(
                1 for r in env_rows if _row_reference_present(r)
            )
            guidance_reference_steps = sum(
                1 for r in env_rows if _row_guidance_flags(r)["route"]
            )
            lm_triggers = sum(
                1 for r in env_rows if _is_truthy(r.get("landmark_triggered", False))
            )
            error_vals = [float(r.get("actr_pm_error", 0.0)) for r in env_rows]
            env_stats[env_type] = {
                "total_steps": len(env_rows),
                "landmark_triggered": lm_triggers,
                "landmark_active_steps": lm_active_steps,
                "landmark_rate": round(
                    lm_active_steps / len(env_rows) if env_rows else 0.0, 4
                ),
                "reference_active_steps": reference_active_steps,
                "reference_rate": round(
                    reference_active_steps / len(env_rows) if env_rows else 0.0, 4
                ),
                "guidance_reference_steps": guidance_reference_steps,
                "guidance_reference_rate": round(
                    guidance_reference_steps / len(env_rows) if env_rows else 0.0, 4
                ),
                "landmark_trigger_rate": round(
                    lm_triggers / len(env_rows) if env_rows else 0.0, 4
                ),
                "error_mean": round(_stat_mean(error_vals), 4),
                "error_std": round(_safe_stdev(error_vals), 4),
            }
    summary["environment_schema_stats"] = env_stats

    module_specs = [
        ("auditory", "听觉"),
        ("tactile", "触觉(感知)"),
        ("manual", "执行(manual)"),
        ("central", "中央"),
        ("memory", "记忆"),
    ]
    module_stats = {}
    for key, label in module_specs:
        a_values = [float(r.get(f"actr_{key}_active", 0.0)) for r in sim_log] or [0.0]
        e_values = [float(r.get(f"actr_{key}_error", 0.0)) for r in sim_log] or [0.0]
        dt_values = [float(r.get(f"actr_dt_{key}", 0.0)) for r in sim_log] or [0.0]
        iw_values_mod = [float(r.get(f"actr_iw_{key}", 0.0)) for r in sim_log] or [0.0]
        module_stats[key] = {
            "label": label,
            "A_mean": round(_stat_mean(a_values), 4),
            "E_mean": round(_stat_mean(e_values), 4),
            "dt_mean_s": round(_stat_mean(dt_values), 4),
            "IW_mean": round(_stat_mean(iw_values_mod), 4),
        }
    summary["module_stats"] = module_stats
    summary["module_data_csv"] = module_csv_path
    summary["single_simulation_data_csv"] = single_simulation_csv_path
    summary["scenario_summary_csv"] = scenario_summary_csv_path
    if initial_production_utilities:
        sorted_utilities = sorted(
            initial_production_utilities.items(),
            key=lambda kv: -float(kv[1]),
        )
        summary["initial_production_utilities"] = {
            "total_count": len(initial_production_utilities),
            "by_priority": [
                {"production": name, "utility": round(float(u), 4)}
                for name, u in sorted_utilities
            ],
        }

    json_path = os.path.join(report_dir, f"sim_summary_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    map_path = None
    try:
        map_path = _render_step_map(graph, sim_log, report_dir, ts)
    except Exception as error:
        print(f"地图生成失败（不影响报告输出）: {error}")

    actr_chart_paths = {}
    try:
        actr_chart_paths = _render_actr_charts(sim_log, event_log, report_dir, ts) or {}
    except Exception as error:
        print(f"ACT-R 负荷图生成失败（不影响报告输出）: {error}")

    if map_path or actr_chart_paths:
        if map_path:
            summary["map_image"] = map_path
        if actr_chart_paths.get("actr_overview"):
            summary["actr_overview_chart_image"] = actr_chart_paths.get("actr_overview")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[报告已生成]:")
    print(f"   CSV 数据: {csv_path}")
    print(f"   单轮模拟数据CSV: {single_simulation_csv_path}")
    print(f"   情境汇总CSV: {scenario_summary_csv_path}")
    print(f"   模块CSV: {module_csv_path}")
    print(f"   JSON 摘要: {json_path}")
    if map_path:
        print(f"   地图图片: {map_path}")
    if actr_chart_paths.get("actr_overview"):
        print(f"   ACT-R总览图: {actr_chart_paths['actr_overview']}")
    # Keep a three-item return tuple for backward compatibility.
    # The first item used to be a Markdown path and is now intentionally None.
    return None, csv_path, json_path
