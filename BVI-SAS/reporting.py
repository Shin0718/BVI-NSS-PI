"""Generate tabular, visual, and markdown reports from simulation outputs.

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


def _write_single_simulation_data_csv(sim_log, report_dir, ts, run_id=1, seed=""):
    """Write the compact per-step CSV requested for single-run analysis."""
    csv_path = os.path.join(report_dir, f"single_simulation_data_{ts}.csv")
    fieldnames = [
        "run",
        "seed",
        "step",
        "simulation_time",
        "segment_id",
        "intersection_state",
        "surface_type",
        "crowd_density",
        "traffic_density",
        "reference_tactile_paving",
        "reference_wall",
        "reference_railing",
        "reference_route",
        "reference_landmark",
        "reference_none",
        "vehicle_approach",
        "horn",
        "pedestrian",
        "obstacle",
        "obstacle_type",
        "risk",
        "net_priority",
        "gate_passed",
        "IW",
        "W_ave",
        "move_direct",
        "stop_and_probe",
        "wait",
        "walking_speed",
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

            step_dt = _as_float(row, "actr_step_dt", default=0.0)
            position_change = _as_float(row, "step_travel_m", "step_len_m", default=0.0)
            walking_speed = position_change / step_dt if step_dt > 0 else 0.0
            dominant_cane = str(row.get("dominant_cane_type", "none")).strip().lower()
            obstacle = _is_truthy(row.get("cane_obstacle", False)) or dominant_cane == "obstacle"
            obstacle_type = dominant_cane if obstacle and dominant_cane else "none"

            writer.writerow(
                {
                    "run": row.get("run", run_id),
                    "seed": row.get("seed", seed),
                    "step": row.get("step", ""),
                    "simulation_time": row.get("sim_time", ""),
                    "segment_id": segment_id,
                    "intersection_state": intersection_state,
                    "surface_type": row.get("surface_type", ""),
                    "crowd_density": row.get("crowd_density", ""),
                    "traffic_density": row.get("traffic_density", ""),
                    "reference_tactile_paving": int(flags["tactile_paving"]),
                    "reference_wall": int(flags["wall"]),
                    "reference_railing": int(flags["railing"]),
                    "reference_route": int(flags["route"]),
                    "reference_landmark": int(flags["landmark"]),
                    "reference_none": int(flags["none"]),
                    "vehicle_approach": int(
                        _is_truthy(row.get("vehicle_approach", False))
                        or str(row.get("dominant_sound_type", "")) == "vehicle_approach"
                    ),
                    "horn": int(
                        _is_truthy(row.get("snd_horn", False))
                        or str(row.get("dominant_sound_type", "")) == "horn"
                    ),
                    "pedestrian": int(
                        _is_truthy(row.get("human_activity_triggered", False))
                        or _is_truthy(row.get("human_activity", False))
                    ),
                    "obstacle": int(obstacle),
                    "obstacle_type": obstacle_type,
                    "risk": row.get("actr_risk_signal", row.get("risk_prob", "")),
                    "net_priority": row.get("net_priority", ""),
                    "gate_passed": int(_is_truthy(row.get("gate_passed", False))),
                    "IW": row.get("actr_iw_total", ""),
                    "W_ave": row.get("actr_wave", ""),
                    "move_direct": int(action == "move_direct"),
                    "stop_and_probe": int(action == "stop_and_probe"),
                    "wait": int(action.startswith("wait")),
                    "walking_speed": round(walking_speed, 6),
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


def _write_typical_outputs_md(sim_log, profile, report_dir, ts):
    """Handle write typical outputs md behavior."""
    scenario_keys = ["overall"]
    summaries = {"overall": _summarize_typical_outputs(list(sim_log))}

    md_path = os.path.join(report_dir, f"sim_typical_outputs_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as md:
        md.write("# 高维机制变量典型输出仿真报告\n\n")
        md.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        md.write(
            "本文件用于后续与实测观察数据对齐。当前版本只输出关键校准量，并按三类数据组织：①第一视角视频中的外部事件频率，②第一视角视频中的行为反应与地标变量，③NASA-TLX认知负荷问卷。地标统计已改为两层口径：`地标触发次数`表示新识别出几个地标事件，`地标触发步数`表示这些地标持续作为定位参照的步数。\n\n"
        )

        md.write("## 1. 用户画像\n\n")
        md.write("| 参数 | 值 |\n|---|---|\n")
        for key, value in profile.items():
            md.write(f"| `{key}` | {value} |\n")
        md.write("\n")

        md.write("## 2. 三类关键校准数据定义\n\n")
        md.write(
            "本报告只保留关键校准数据，按现有实测材料分为三类：第一视角视频中的外部事件频率、第一视角视频中的行为反应与地标参照、NASA-TLX 认知负荷与总体表现。地标指标已经拆分为“触发次数”和“触发步数”，不再把每一步持续参照误读为新地标。\n\n"
        )
        md.write("| 校准数据类 | 主要校准目标 | 主要观测方式 |\n|---|---|---|\n")
        md.write(
            "| 第1类：视频-外部事件频率 | 声音、车辆、盲杖接触、路面、路口等外部输入概率 | 对第一视角视频逐步或逐片段标注事件，换算为每步概率或片段占比 |\n"
        )
        md.write(
            "| 第2类：视频-行为反应与地标参照 | 停步、探测、车辆响应、地标触发次数、地标持续步数 | 标注停下/恢复、减速、探杖、方向修正、地标识别或明显定位确认片段 |\n"
        )
        md.write(
            "| 第3类：NASA-TLX认知负荷 | ACT-R负荷、风险权重、参照缓释、无参照记忆压力 | 将问卷分数按路线、路段或事件窗口与仿真均值/峰值/高负荷比例对齐 |\n\n"
        )

        md.write("### 2.1 关键校准参数清单\n\n")
        md.write("| 参数 | 当前值/公式 | 数据类别 | 如何从现有数据观察 |\n")
        md.write("|---|---:|---|---|\n")
        key_params = [
            (
                "AVG_STEP_METERS",
                "0.48",
                "第1类/视频-时空行为",
                "路线距离除以可见步数，优先用直线路段",
            ),
            (
                "BVI_WALKING_SPEED",
                "0.558",
                "第1类/视频-时空行为",
                "路段距离除以通过时间，普通路段和路口分开复核",
            ),
            (
                "CANE_OBSTACLE_PROB",
                "0.021",
                "第1类/视频-外部事件",
                "探杖碰撞/绕避障碍事件数除以非路口步数",
            ),
            (
                "SOUND_VEHICLE_APPROACH_PROB",
                "0.06",
                "第1类/视频-外部事件",
                "非路口明显车辆逼近事件数除以非路口步数",
            ),
            (
                "SOUND_VEHICLE_APPROACH_CROSSING_PROB",
                "0.65",
                "第1类/视频-外部事件",
                "路口片段车辆逼近声出现步数或短事件数除以路口步数",
            ),
            (
                "LANDMARK_TRIGGER_PROB_MIN/MAX",
                "0.00015 / 0.00100",
                "第2类/视频-地标",
                "熟悉街道中新识别地标次数；目标通常约 3 次左右",
            ),
            (
                "LANDMARK_EPISODE_STEPS_MIN/MAX",
                "4 / 6",
                "第2类/视频-地标",
                "地标识别后稳定行走/不再确认的持续步数；总触发步数目标约 12-18 步",
            ),
            (
                "LANDMARK_REFRACTORY_STEPS_MIN/MAX",
                "90 / 150",
                "第2类/视频-地标",
                "相邻两个地标识别事件之间的步数间隔，拉长它可以明显减少重复触发",
            ),
            (
                "VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK",
                "0.40",
                "第2类/视频-行为",
                "非路口车声后停步/减速的命中与误报",
            ),
            (
                "PROBE_RELIEF_RATIO",
                "0.40",
                "第2类+第3类",
                "停探测后恢复速度，以及对应NASA-TLX下降",
            ),
            (
                "SEEV_PRIORITY_SCALE",
                "4.00",
                "第2类+第3类",
                "稳定乘法SEEV整体缩放；gate整体偏低则提高，误触发过多则降低",
            ),
            (
                "SEEV_TERM_FLOOR",
                "0.05",
                "第3类/NASA-TLX",
                "乘法项下限补偿；一般固定，防止任一项略低导致Priority塌陷",
            ),
            (
                "SEEV_EFFORT_WEIGHT",
                "0.75",
                "第3类/NASA-TLX",
                "Effort抑制强度；高负荷下注意门控过少则降低，过度响应则提高",
            ),
            (
                "SEEV_SALIENCE_RISK_BOOST_WEIGHT",
                "0.20",
                "第3类/NASA-TLX",
                "DBN风险只通过SEEV显著性影响ACT-R，用视频危险事件锚定NASA-TLX升高",
            ),
            (
                "ACTR_RISK_CANE_GUIDANCE_RELIEF",
                "0.10",
                "第2类+第3类",
                "盲杖引导线索后停探测减少和NASA-TLX压力下降幅度",
            ),
            (
                "ACTR_RISK_LANDMARK_RELIEF",
                "0.08",
                "第2类+第3类",
                "地标识别后停探测减少和NASA-TLX压力下降幅度",
            ),
            (
                "MEMORY_ACTIVE_ABSENT_STEPS_TH",
                "11",
                "第2类+第3类",
                "连续无参照多久后出现停探测/回头确认/路线回忆",
            ),
            (
                "NAV_WM_PEAK",
                "0.25",
                "第3类/NASA-TLX",
                "导航提示后短时停顿、注意转移和NASA-TLX升高",
            ),
        ]
        for param, value, data_class, proxy in key_params:
            md.write(f"| `{param}` | {value} | {data_class} | {proxy} |\n")
        md.write("\n")

        summary = summaries.get("overall") or {}
        proxy_notes = {
            "total_steps": "视频：路线完成总步数；可由脚步声、画面节律或IMU辅助标注",
            "total_time_s": "视频：从开始行走到到达目标的总时长",
            "mean_step_time_s": "视频：总时长/总步数，普通路段与路口可分开复核",
            "mean_actr_load": "NASA-TLX：总体mental demand/effort与仿真平均负荷对齐",
            "peak_actr_load": "NASA-TLX：最高压力片段与仿真峰值负荷对齐",
            "mean_actr_wave": "NASA-TLX：整段主观负荷与平滑负荷均值对齐",
            "mean_actr_risk": "视频+NASA-TLX：危险片段比例和主观风险感对齐",
            "peak_actr_risk": "视频：车辆/障碍/路口等高危窗口中的风险峰值",
            "probe_count": "视频：原地停下探测次数",
            "probe_rate": "视频：原地探测步数/总步数",
            "mean_probe_duration_steps": "视频：每次停下探测持续步数或持续时间折算步数",
            "median_probe_duration_steps": "视频：典型单次探测持续步数",
            "max_probe_duration_steps": "视频：最长一次探测持续步数",
            "mean_load_drop_after_probe": "NASA-TLX+视频：探测后紧张度/停顿后恢复幅度",
            "stop_rate": "视频：停步或等待步数占总步数比例",
            "stop_count": "视频：stop_and_probe + wait_at_red 总次数",
            "vehicle_approach_count": "视频音轨：车辆逼近声事件次数，路口和路段分开更好",
            "vehicle_response_rate": "视频：车辆声出现后3步内停步/等待/明显减速比例",
            "stop_after_vehicle_rate": "视频：车辆声出现后3步内停步比例",
            "mean_looming_boost_peak": "视频：车辆逼近影响持续时间；用于校准LOOMING_BOOST_PEAK/DECAY",
            "landmark_trigger_count": "视频：新识别/确认地标的次数；熟悉街道目标通常约3-4次",
            "landmark_active_step_count": "视频：地标识别后持续稳定定位的总步数；熟悉街道目标约20-30步",
            "landmark_active_step_rate": "视频：地标参照持续步数/总步数",
            "mean_landmark_episode_steps": "视频：每次地标识别后的平均持续影响步数",
            "landmark_relief_effect": "视频+NASA-TLX：地标后风险、停步和负荷下降幅度",
            "reference_absent_streak_mean": "视频：连续无盲道/墙/路缘/地标参照的平均步数",
            "reference_absent_streak_max": "视频：最长连续无参照步数；常对应迷失/方向修正",
            "memory_retrieval_count": "视频：路线回忆、停下确认方向、明显寻找参照的次数代理",
            "memory_retrieval_rate": "视频：显式回忆/找参照步数比例",
            "mean_spatial_wm_load": "NASA-TLX：memory demand/effort 与空间记忆通道均值对齐",
            "peak_spatial_wm_load": "NASA-TLX：无参照或迷失片段的记忆负荷峰值",
            "spatial_anchored_rate": "视频：有盲道/墙/路缘/地标参照支持的步数比例",
            "high_load_step_rate": "NASA-TLX：高负荷评分对应的高IW步数比例",
            "overload_streak_mean": "NASA-TLX+视频：连续紧张/停顿/犹豫片段平均长度",
            "overload_streak_max": "NASA-TLX+视频：最长连续高负荷片段",
            "gate_passed_count": "视频：明显注意转移事件数，如转头、停顿、探杖、对声源反应",
            "hazard_response_rate": "视频：危险线索后3步内发生停步/等待/探测的比例",
            "stop_after_hazard_rate": "视频：危险线索后3步内停步比例",
            "cane_relief_effect": "视频+NASA-TLX：盲杖引导物出现后风险/停步/负荷下降幅度",
            "nav_announcement_count": "视频/音轨：导航语音播报次数",
            "mean_wm_peak_after_nav": "NASA-TLX+视频：导航播报后短窗口停顿或负荷升高幅度",
            "overload_to_probe_delay_steps": "视频：高负荷征兆出现后转入探测的延迟步数代理",
            "mean_risk_prob": "视频：外部危险事件密度与DBN风险均值对齐",
            "peak_risk_prob": "视频：最危险事件窗口与DBN风险峰值对齐",
            "mean_seev_salience": "视频：注意转移/明显警觉事件密度与显著性均值对齐",
            "peak_seev_salience": "视频：突然声源/障碍/路口窗口与显著性峰值对齐",
        }
        md.write("## 3. 整体总览指标\n\n")
        overview_cols = [
            ("total_steps", "总步数"),
            ("total_time_s", "总时间(s)"),
            ("mean_actr_load", "ACT-R负荷均值"),
            ("peak_actr_load", "ACT-R负荷峰值"),
            ("probe_count", "probe次数"),
            ("stop_rate", "停步/等待比例"),
            ("vehicle_approach_count", "车辆逼近次数"),
            ("landmark_trigger_count", "地标触发次数"),
            ("landmark_active_step_count", "地标触发步数"),
            ("mean_landmark_episode_steps", "单次地标平均持续步数"),
        ]
        md.write("| 指标 | 仿真值 | 如何从实测观察 |\n|---|---:|---|\n")
        for metric, label in overview_cols:
            md.write(
                f"| {label} | {_fmt(summary.get(metric, 0.0))} | {proxy_notes.get(metric, '按视频或问卷统计后与仿真对齐')} |\n"
            )
        md.write("\n")

        sections = [
            (
                "4. 第1类：视频-外部事件频率",
                [
                    ("vehicle_approach_count", "车辆逼近事件次数"),
                    ("mean_risk_prob", "平均 DBN 风险概率"),
                    ("peak_risk_prob", "DBN 风险概率峰值"),
                    ("mean_seev_salience", "平均 SEEV 显著性"),
                    ("peak_seev_salience", "SEEV 显著性峰值"),
                ],
            ),
            (
                "5. 第2类：视频-行为反应与地标参照",
                [
                    ("probe_count", "probe 次数"),
                    ("probe_rate", "probe 步数比例"),
                    ("mean_probe_duration_steps", "单次 probe 平均持续步数"),
                    ("stop_count", "停步/等待次数"),
                    ("stop_rate", "停步/等待比例"),
                    ("vehicle_response_rate", "车辆事件后响应比例"),
                    ("stop_after_vehicle_rate", "车辆事件后停步比例"),
                    ("landmark_trigger_count", "地标触发次数"),
                    ("landmark_active_step_count", "地标触发步数"),
                    ("landmark_active_step_rate", "地标触发步数比例"),
                    ("mean_landmark_episode_steps", "单次地标平均持续步数"),
                    ("landmark_relief_effect", "地标识别后风险下降幅度"),
                    ("reference_absent_streak_mean", "平均连续失去参照步数"),
                    ("reference_absent_streak_max", "最大连续失去参照步数"),
                    ("spatial_anchored_rate", "有参照支持步数比例"),
                ],
            ),
            (
                "6. 第3类：NASA-TLX认知负荷与总体表现",
                [
                    ("mean_actr_load", "ACT-R 瞬时负荷均值"),
                    ("peak_actr_load", "ACT-R 瞬时负荷峰值"),
                    ("mean_actr_wave", "ACT-R 平滑负荷均值"),
                    ("high_load_step_rate", "高负荷步数比例"),
                    ("overload_streak_mean", "平均连续高负荷步数"),
                    ("overload_streak_max", "最大连续高负荷步数"),
                    ("mean_spatial_wm_load", "平均空间/记忆通道负荷"),
                    ("peak_spatial_wm_load", "空间/记忆通道负荷峰值"),
                    ("memory_retrieval_count", "显式记忆检索次数"),
                    ("mean_load_drop_after_probe", "probe 后负荷平均下降幅度"),
                    ("cane_relief_effect", "盲杖线索后风险下降幅度"),
                    ("gate_passed_count", "注意门控通过次数"),
                ],
            ),
        ]

        for title, metrics in sections:
            md.write(f"## {title}\n\n")
            md.write("| 指标 | 仿真值 | 如何观察/统计 |\n")
            md.write("|---|---:|---|\n")
            for metric, label in metrics:
                md.write(
                    f"| {label} | {_fmt(summary.get(metric, 0.0))} | {proxy_notes.get(metric, '按视频或问卷统计后与仿真对齐')} |\n"
                )
            md.write("\n")

        md.write("## 7. 风险状态与动作分布\n\n")
        if summary:
            md.write("### 风险状态分布\n\n")
            md.write("| 风险状态 | 步数 | 占比 |\n|---|---:|---:|\n")
            for risk_level, count in sorted(summary["risk_distribution"].items()):
                md.write(
                    f"| {risk_level} | {count} | {_safe_pct(count, summary['total_steps']):.1f}% |\n"
                )
            md.write("\n### 动作分布\n\n")
            md.write("| 动作 | 步数 | 占比 |\n|---|---:|---:|\n")
            for action, count in sorted(
                summary["action_distribution"].items(), key=lambda item: -item[1]
            ):
                md.write(
                    f"| {action} | {count} | {_safe_pct(count, summary['total_steps']):.1f}% |\n"
                )
            md.write("\n")

        md.write("## 8. 三类数据的关键校准建议\n\n")
        md.write(
            "本轮不建议校准所有参数，只校准会直接影响可观测输出的关键参数。推荐顺序如下：\n\n"
        )
        md.write(
            "1. **先用视频校准外部事件频率**：路面类型、车辆声、喇叭、人声、盲杖障碍和引导物概率。先让输入事件频率接近真实视频，否则后续行为和NASA-TLX都会被错误输入带偏。\n"
        )
        md.write(
            "2. **再用视频校准行为反应与地标参照**：重点看 `probe_count`、`mean_probe_duration_steps`、`vehicle_response_rate`、`landmark_trigger_count`、`landmark_active_step_count`。熟悉街道中地标建议目标为触发约 3–4 次，持续约 20–30 步。\n"
        )
        md.write(
            "3. **最后用NASA-TLX校准认知负荷权重**：用整段或事件窗口的 NASA-TLX mental demand / effort / frustration 与 `mean_actr_load`、`peak_actr_load`、`high_load_step_rate`、`mean_spatial_wm_load` 对齐。\n\n"
        )
        md.write(
            "地标相关参数现在应按两个指标校准：`LANDMARK_TRIGGER_PROB_MIN/MAX` 控制新地标出现次数；`LANDMARK_EPISODE_STEPS_MIN/MAX` 控制每次地标影响持续步数；`LANDMARK_REFRACTORY_STEPS_MIN/MAX` 控制两个地标事件之间的间隔。若当前地标次数偏高，优先增大 `LANDMARK_REFRACTORY_STEPS_MIN/MAX`，其次降低 `LANDMARK_TRIGGER_SCALE`。\n"
        )

    return md_path


def _render_step_map(graph, sim_log, report_dir, ts):
    """Handle render step map behavior."""
    if graph is None or not sim_log:
        return None

    node_lookup = {str(node): node for node in graph.nodes}
    control_nodes = []

    first_node = node_lookup.get(str(sim_log[0].get("position")))
    if first_node is not None:
        control_nodes.append(first_node)

    for row in sim_log:
        if not _to_bool(row.get("at_node", True)):
            continue
        if not _to_bool(row.get("at_intersection", False)):
            continue
        node = node_lookup.get(str(row.get("position")))
        if node is None:
            continue
        if not control_nodes or control_nodes[-1] != node:
            control_nodes.append(node)

    last_node = node_lookup.get(str(sim_log[-1].get("position")))
    if last_node is not None and (not control_nodes or control_nodes[-1] != last_node):
        control_nodes.append(last_node)

    if len(control_nodes) < 2:
        return None

    ordered_points = []
    for idx in range(len(control_nodes) - 1):
        node_a = control_nodes[idx]
        node_b = control_nodes[idx + 1]
        xy_a = (float(graph.nodes[node_a]["x"]), float(graph.nodes[node_a]["y"]))
        xy_b = (float(graph.nodes[node_b]["x"]), float(graph.nodes[node_b]["y"]))

        try:
            segment_path = nx.shortest_path(graph, node_a, node_b, weight="length")
        except Exception:
            segment_path = [node_a, node_b]

        segment_points = []
        for seg_i in range(len(segment_path) - 1):
            from_node = segment_path[seg_i]
            to_node = segment_path[seg_i + 1]
            from_xy = (
                float(graph.nodes[from_node]["x"]),
                float(graph.nodes[from_node]["y"]),
            )
            to_xy = (float(graph.nodes[to_node]["x"]), float(graph.nodes[to_node]["y"]))
            seg_len = _edge_length(graph, from_node, to_node)
            interp = _interpolate_points(
                from_xy, to_xy, seg_len, MAP_INTERP_STEP_METERS
            )
            if segment_points:
                segment_points.extend(interp[1:])
            else:
                segment_points.extend(interp)

        if not segment_points:
            segment_points = [xy_a, xy_b]

        if ordered_points:
            ordered_points.extend(segment_points[1:])
        else:
            ordered_points.extend(segment_points)

    if not ordered_points:
        return None

    fig, ax = plt.subplots(figsize=(9, 9), dpi=180)
    ax.set_facecolor("white")
    for edge_u, edge_v in graph.edges():
        node_u = graph.nodes[edge_u]
        node_v = graph.nodes[edge_v]
        ax.plot(
            [node_u["x"], node_v["x"]],
            [node_u["y"], node_v["y"]],
            color="#D0D0D0",
            linewidth=0.6,
            alpha=0.9,
            zorder=1,
        )

    xs = [point[0] for point in ordered_points]
    ys = [point[1] for point in ordered_points]

    ax.plot(xs, ys, color="#d62728", linewidth=1.8, alpha=0.85, zorder=4)
    ax.scatter(xs, ys, s=8, c="#1f77b4", alpha=0.8, edgecolors="none", zorder=5)

    control_xs = [float(graph.nodes[node]["x"]) for node in control_nodes]
    control_ys = [float(graph.nodes[node]["y"]) for node in control_nodes]
    ax.scatter(
        control_xs,
        control_ys,
        s=24,
        c="#111111",
        alpha=0.9,
        edgecolors="white",
        linewidths=0.4,
        zorder=6,
    )
    for idx, (x, y) in enumerate(zip(control_xs, control_ys), start=1):
        ax.text(x, y, str(idx), fontsize=6, color="#111111", zorder=7)

    ax.scatter(
        [xs[0]],
        [ys[0]],
        s=60,
        c="#2ca02c",
        marker="*",
        edgecolors="black",
        linewidths=0.5,
        zorder=7,
    )
    ax.scatter(
        [xs[-1]],
        [ys[-1]],
        s=70,
        c="#ff7f0e",
        marker="X",
        edgecolors="black",
        linewidths=0.5,
        zorder=7,
    )
    ax.set_title("BVI Simulation Step Map", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    map_path = os.path.join(report_dir, f"sim_map_{ts}.png")
    fig.savefig(map_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return map_path


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
            "actr_iw_high_count": actr_iw_high_count,
            "actr_iw_resume_count": actr_iw_resume_count,
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

    typical_outputs_md_path = None
    try:
        typical_outputs_md_path = _write_typical_outputs_md(
            sim_log, profile, report_dir, ts
        )
        summary["typical_outputs_markdown"] = typical_outputs_md_path
    except Exception as error:
        print(f"典型变量 Markdown 报告生成失败（不影响原报告输出）: {error}")

    json_path = os.path.join(report_dir, f"sim_summary_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(report_dir, f"sim_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as md:
        md.write("# BVI 态势感知模拟报告\n\n")
        md.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        md.write("## 1. 模型配置\n\n")
        md.write("### 1.1 用户画像\n\n")
        md.write("| 参数 | 值 |\n|------|-----|\n")
        for lower_key, upper_key, default in PROFILE_FIELDS_LOWER:
            value = profile.get(upper_key, profile.get(lower_key, default))
            md.write(f"| {upper_key} | {value} |\n")

        md.write("\n### 1.2 网络环境\n\n")
        md.write(f"- **起点**: {start_node}\n")
        md.write(f"- **终点**: {goal_node}\n")
        md.write(f"- **最大步数**: {max_steps}\n\n")

        md.write("### 1.3 ACT-R 亚符号参数\n\n")
        md.write("| 参数 | 值 | 说明 |\n|------|-----|------|\n")
        md.write(
            f"| MAS | {profile.get('MAS', profile.get('mas', 1.5))} | 关联强度（spreading activation） |\n"
        )
        md.write(f"| RT | {profile.get('RT', profile.get('rt', -2.0))} | 检索阈值 |\n")
        md.write(
            f"| ANS | {profile.get('ANS', profile.get('ans', 0.2))} | 瞬时噪声 |\n"
        )

        if initial_production_utilities:
            md.write("\n### 1.4 产生式优先级先验（本模型不做学习，单次出行内固定）\n\n")
            md.write(
                f"- 配置初始 utility 的产生式数: **{len(initial_production_utilities)}**\n"
            )
            md.write(
                "- 优先级由 (familiarity, expertise) 映射决定，越高越倾向 fire。\n\n"
            )

            sorted_top = sorted(
                initial_production_utilities.items(),
                key=lambda kv: -float(kv[1]),
            )[:20]
            md.write("| 产生式 | 初始 Utility |\n")
            md.write("|--------|-------------|\n")
            for name, util in sorted_top:
                md.write(f"| {name} | {float(util):.3f} |\n")

        md.write("## 2. 模拟结果概要\n\n")
        md.write("| 指标 | 值 |\n|------|-----|\n")
        md.write(f"| 总步数 | {steps} |\n")
        md.write(f"| 是否到达目标 | {'是' if reached_goal else '否'} |\n")
        md.write(f"| DBN风险均值 | {_stat_mean(risk_values_dbn):.4f} |\n")
        md.write(f"| ACT-R风险均值 | {_stat_mean(risk_values_actr):.4f} |\n")
        md.write(f"| 风险通道偏差MAE | {_stat_mean(risk_abs_diff):.4f} |\n")
        md.write(
            f"| 注意门控通过次数 | {gate_passed_count} / {steps} ({_safe_pct(gate_passed_count, steps):.1f}%) |\n"
        )
        md.write(
            f"| 有参照支持的步数（地标+盲杖引导+盲道） | {spatial_anchored_count} / {steps} ({_safe_pct(spatial_anchored_count, steps):.1f}%) |\n"
        )
        md.write(
            f"| 导航参照存在步数（盲道/墙/栏杆/路线/地标） | {reference_active_step_count} / {steps} ({_safe_pct(reference_active_step_count, steps):.1f}%) |\n"
        )
        md.write(
            f"| 其中：引导物/路线参照步数 | {guidance_reference_step_count} / {steps} ({_safe_pct(guidance_reference_step_count, steps):.1f}%) |\n"
        )
        md.write(f"| 其中：音频语义地标触发次数 | {landmark_trigger_count} |\n")
        md.write(
            f"| 其中：音频语义地标触发步数 | {landmark_active_step_count} / {steps} ({_safe_pct(landmark_active_step_count, steps):.1f}%) |\n"
        )
        md.write(f"| 停止探测次数 | {stop_probe_count} |\n")
        md.write(f"| ACT-R 高瞬时负荷次数 | {actr_iw_high_count} |\n")

        md.write("## 3. 核心指标统计\n\n")
        md.write(
            "| 指标 | 均值 | 标准差 | 最小值 | 最大值 |\n|------|------|--------|--------|--------|\n"
        )
        for name, vals in [
            ("ACT-R 瞬时负荷 IW", iw_values),
            ("ACT-R 平均负荷 W_ave", wave_values),
            ("风险概率(DBN)", risk_values_dbn),
            ("风险信号(ACT-R)", risk_values_actr),
            ("净优先级", pri_values),
            ("突显度", sal_values),
            ("声音强度", int_values),
        ]:
            md.write(
                f"| {name} | {_stat_mean(vals):.4f} | {_safe_stdev(vals):.4f} | {min(vals):.4f} | {max(vals):.4f} |\n"
            )
        md.write("\n")

        md.write("### 3.2 风险一致性（DBN通道 vs ACT-R决策通道）\n\n")
        md.write("| 指标 | DBN通道 | ACT-R决策通道 |\n")
        md.write("|------|---------|----------------|\n")
        md.write(
            f"| 均值 | {_stat_mean(risk_values_dbn):.4f} | {_stat_mean(risk_values_actr):.4f} |\n"
        )
        md.write(
            f"| 标准差 | {_safe_stdev(risk_values_dbn):.4f} | {_safe_stdev(risk_values_actr):.4f} |\n"
        )
        md.write(
            f"| 最小值 | {min(risk_values_dbn):.4f} | {min(risk_values_actr):.4f} |\n"
        )
        md.write(
            f"| 最大值 | {max(risk_values_dbn):.4f} | {max(risk_values_actr):.4f} |\n"
        )
        md.write(
            f"| 双通道偏差 MAE(|DBN-ACT-R|) | {summary['statistics']['risk_alignment']['mae']:.4f} | {summary['statistics']['risk_alignment']['mae']:.4f} |\n"
        )
        md.write("\n")

        md.write("### 3.3 模块级 A/E/持续时间/IW（均值）\n\n")
        md.write("| 模块 | A均值 | E均值 | 持续时间均值(ms) | IW均值 |\n")
        md.write("|------|------|------|-----------------|--------|\n")
        for key in ["auditory", "tactile", "manual", "central", "memory"]:
            st = module_stats.get(key, {})
            md.write(
                f"| {st.get('label', key)} | {st.get('A_mean', 0.0):.4f} | {st.get('E_mean', 0.0):.4f} | "
                f"{st.get('dt_mean_s', 0.0) * 1000:.2f} | {st.get('IW_mean', 0.0):.4f} |\n"
            )
        md.write("\n")

        md.write("## 4. 风险等级分布\n\n")
        md.write("### 4.1 DBN风险等级\n\n")
        md.write("| 等级 | 步数 | 占比 |\n|------|------|------|\n")
        for level in ["low", "medium", "high"]:
            cnt = risk_dist_dbn[level]
            md.write(f"| {level} | {cnt} | {_safe_pct(cnt, steps):.1f}% |\n")
        md.write("\n")

        md.write("### 4.2 ACT-R风险等级（决策通道）\n\n")
        md.write("| 等级 | 步数 | 占比 |\n|------|------|------|\n")
        for level in ["low", "medium", "high"]:
            cnt = risk_dist_actr[level]
            md.write(f"| {level} | {cnt} | {_safe_pct(cnt, steps):.1f}% |\n")
        md.write("\n")

        md.write("## 5. 决策动作分布\n\n")
        md.write("| 动作 | 步数 | 占比 |\n|------|------|------|\n")
        for act, cnt in sorted(action_dist.items(), key=lambda x: -x[1]):
            md.write(f"| {act} | {cnt} | {_safe_pct(cnt, steps):.1f}% |\n")
        md.write("\n")

        md.write("## 6. 仿真时间统计\n\n")
        if _has_sim_time and total_sim_time > 0:
            _avg_step_s = total_sim_time / steps if steps else 0.0
            md.write("| 指标 | 值 |\n|------|-----|\n")
            md.write(f"| 总仿真时间 | {total_sim_time:.3f} s |\n")
            md.write(f"| 总步数 | {steps} |\n")
            md.write(f"| 每步平均时长 | {_avg_step_s * 1000:.1f} ms |\n")
            md.write("\n**各动作占用时间**\n\n")
            md.write("| 动作 | 步数 | 累计时间 (s) | 平均时长 (ms) | 占总时间比 |\n")
            md.write("|------|------|-------------|--------------|------------|\n")
            for _act, _info in sorted(
                _action_time.items(), key=lambda x: -x[1]["total_s"]
            ):
                _avg_ms = (
                    (_info["total_s"] / _info["count"] * 1000)
                    if _info["count"]
                    else 0.0
                )
                _pct = _safe_pct(_info["total_s"], total_sim_time)
                md.write(
                    f"| {_act} | {_info['count']} | {_info['total_s']:.3f} | {_avg_ms:.1f} | {_pct:.1f}% |\n"
                )
        else:
            md.write("*（当前记录不含 sim_time 字段，请升级后重新运行模拟）*\n")
        md.write("\n")

        md.write("## 8. 环境 Schema 与地标触发率（熟悉度模型验证）\n\n")
        md.write(
            "根据环境条件概率结构（env_schema.py）的预测，在不同环境下地标触发和错误负荷的分布。\n"
        )
        md.write(
            "验证假设：高熟悉度 × 高 P(landmark|environment) → 地标激活强 → 触发率高，错误负荷低\n\n"
        )

        if env_stats:
            md.write(
                "| 环境类型 | 总步数 | 地标触发次数 | 语义地标步数 | 导航参照步数 | 导航参照占比 | 平均错误负荷 | 错误负荷标差 |\n"
            )
            md.write(
                "|---------|--------|----------|----------|----------|---------|----------|----------|\n"
            )
            for env_type in sorted(env_stats.keys()):
                stats = env_stats[env_type]
                md.write(
                    f"| {env_type} | {stats['total_steps']} | {stats['landmark_triggered']} | "
                    f"{stats['landmark_active_steps']} | {stats['reference_active_steps']} | "
                    f"{stats['reference_rate']:.1%} | "
                    f"{stats['error_mean']:.4f} | {stats['error_std']:.4f} |\n"
                )
            md.write("\n")
            md.write("**解释**:\n")
            md.write(
                "- **地标触发次数**: 新识别出一个地标事件的次数，适合与实测“几次认出地标”对齐\n"
            )
            md.write(
                "- **语义地标步数**: 仅统计 matched_landmark，不包含盲道、墙、栏杆、路线等引导参照\n"
            )
            md.write(
                "- **导航参照步数/占比**: 统计盲道、墙、栏杆、路线、语义地标任一参照存在的步数，更适合当前设备介入 UI\n"
            )
            md.write(
                "- **错误负荷**: ACT-R 感知-运动通道的平均错误负荷指标（范围不限定在 [0,1]）\n"
            )
            md.write(
                "- **高熟悉度预期**: 熟悉度高的BVI在熟悉环境下触发率 ↑，错误负荷 ↓\n\n"
            )
        else:
            md.write("*（当前记录未包含环境类型或surface_type字段）*\n\n")

        md.write("## 9. 位置状态分布\n\n")
        md.write(
            "位置状态由 `at_node`（是否在节点上）和 `crossing_active`（是否处于路口阶段）共同决定。\n\n"
        )
        md.write("| 状态 | 说明 | 步数 | 占比 |\n|------|------|------|------|\n")
        state_labels = [
            ("node_normal", "at_node=T, crossing=F", "普通节点（路段端点，非路口）"),
            ("node_crossing", "at_node=T, crossing=T", "路口节点（等灯 / 开始穿越）"),
            ("edge_crossing", "at_node=F, crossing=T", "边内推进中（路口穿越阶段）"),
            ("edge_normal", "at_node=F, crossing=F", "边内推进中（正常路段）"),
        ]
        for key, cond, desc in state_labels:
            cnt = pos_state_counts[key]
            md.write(f"| {desc} | `{cond}` | {cnt} | {_safe_pct(cnt, steps):.1f}% |\n")
        md.write("\n")

        md.write("### 9.1 stop_and_probe 发生在哪些位置状态\n\n")
        md.write("| 状态 | stop_and_probe步数 | 占 stop_and_probe 比例 |\n")
        md.write("|------|-------------------|----------------------|\n")
        for key, _, desc in state_labels:
            cnt = stop_probe_state_counts[key]
            md.write(f"| {desc} | {cnt} | {_safe_pct(cnt, stop_probe_count):.1f}% |\n")
        md.write("\n")
        md.write("```\n")
        md.write("    title stop_and_probe 发生位置状态分布\n")
        for key, _, desc in state_labels:
            md.write(f'    "{desc}" : {stop_probe_state_counts[key]}\n')
        md.write("```\n\n")

        md.write("### 9.2 哪个产生式（或外部门控）引发了 probe\n\n")
        md.write(
            "| 触发来源 | stop_and_probe步数 | 占 stop_and_probe 比例 | 占总步数比例 |\n"
        )
        md.write(
            "|----------|-------------------|----------------------|--------------|\n"
        )
        for src, cnt in sorted(probe_source_counts.items(), key=lambda x: -x[1]):
            md.write(
                f"| {src} | {cnt} | {_safe_pct(cnt, stop_probe_count):.1f}% | {_safe_pct(cnt, steps):.1f}% |\n"
            )
        md.write("\n")
        md.write("```\n")
        md.write("    title probe 触发来源分布\n")
        for src, cnt in sorted(probe_source_counts.items(), key=lambda x: -x[1]):
            md.write(f'    "{src}" : {cnt}\n')
        md.write("```\n\n")

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

    with open(md_path, "a", encoding="utf-8") as md:
        if map_path:
            md.write("\n## 15. 路径底图与步序标注\n\n")
            md.write(f"- 地图文件: `{os.path.basename(map_path)}`\n\n")
            md.write(f"![step_map]({os.path.basename(map_path)})\n")
        if actr_chart_paths.get("actr_overview"):
            md.write("\n## 16. ACT-R 负荷图\n\n")
            if actr_chart_paths.get("actr_overview"):
                md.write(
                    f"- ACT-R 合并图: `{os.path.basename(actr_chart_paths['actr_overview'])}`\n\n"
                )
                md.write(
                    f"![actr_overview]({os.path.basename(actr_chart_paths['actr_overview'])})\n\n"
                )

    print("\n[报告已生成]:")
    print(f"   Markdown: {md_path}")
    print(f"   CSV 数据: {csv_path}")
    print(f"   单轮模拟数据CSV: {single_simulation_csv_path}")
    print(f"   模块CSV: {module_csv_path}")
    print(f"   JSON 摘要: {json_path}")
    if typical_outputs_md_path:
        print(f"   6场景典型变量MD: {typical_outputs_md_path}")
    if map_path:
        print(f"   地图图片: {map_path}")
    if actr_chart_paths.get("actr_overview"):
        print(f"   ACT-R总览图: {actr_chart_paths['actr_overview']}")
    return md_path, csv_path, json_path
