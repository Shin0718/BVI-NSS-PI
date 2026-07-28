"""Run the BVI non-visual navigation simulation and ACT-R decision loop.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import math
import random
from collections import deque

import networkx as nx
import numpy as np
import pyactr as actr

try:
    from .config import (
        CROSSING_WAIT_STEPS_MIN,
        CROSSING_WAIT_STEPS_MAX,
        CROSSING_TRAVERSE_STEPS_MIN,
        CROSSING_TRAVERSE_STEPS_MAX,
        AVG_STEP_METERS,
        MAX_STEPS,
        BVI_WALKING_SPEED,
        SURFACE_PROBABILITY_DISTRIBUTION,
        SURFACE_PROFILES,
        CANE_OBSTACLE_PROB,
        CANE_CURB_PROB,
        CANE_WALL_PROB,
        CANE_RAILING_PROB,
        SOUND_HORN_PROB,
        SOUND_VEHICLE_APPROACH_PROB,
        SOUND_VEHICLE_APPROACH_CROSSING_PROB,
        SOUND_REVERSE_BEEP_PROB,
        HUMAN_ACTIVITY_PROB,
        VEHICLE_APPROACH_MIN_STEPS,
        VEHICLE_APPROACH_MAX_STEPS,
        VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK,
        CROSSING_HORN_PROB,
        CROSSING_REVERSE_BEEP_PROB,
        CROSSING_HUMAN_ACTIVITY_PROB,
        DBN_CROSSING_VEHICLE_ABSENT_SOFT_EVIDENCE,
        DBN_CROSSING_TRAFFIC_ABSENT_SOFT_EVIDENCE,
        DBN_CROSSING_CANE_HIT_SOFT_EVIDENCE,
        DBN_NEUTRAL_DISTANCE_FEEDBACK,
        SEEV_VALUE_SAFETY_WEIGHT,
        SEEV_VALUE_PROGRESS_WEIGHT,
        SEEV_EXPECTANCY_RISK_WEIGHT,
        ACTR_RISK_CANE_GUIDANCE_RELIEF,
        SEEV_GATE_ADAPTIVE_THRESHOLD_ENABLED,
        SEEV_GATE_THRESHOLD_WINDOW_STEPS,
        SEEV_GATE_THRESHOLD_QUANTILE,
        SEEV_GATE_THRESHOLD_MIN_HISTORY,
        ATTENTION_UNGATED_ENTRY_COEF,
        ATTENTION_GATED_CENTRAL_DANGER_BOOST,
        ATTENTION_GATED_RELIEF_ENABLED,
        LOOMING_RESUME_GATE_ENABLED,
        LOOMING_RESUME_THRESHOLD,
        ACTR_RISK_LANDMARK_RELIEF,
        SURFACE_SEGMENT_MAX_STEPS,
        SURFACE_SEGMENT_MIN_STEPS,
        TACTILE_SEGMENT_MIN_STEPS,
        TACTILE_SEGMENT_MAX_STEPS,
        CANE_GUIDANCE_MIN_STEPS,
        CANE_GUIDANCE_MAX_STEPS,
        ACTR_DYNAMIC_PM_WEIGHTS_ENABLED,
        ACTR_DYNAMIC_PM_TEMP,
        ACTR_DYNAMIC_PM_SMOOTHING,
        ACTR_DYNAMIC_PM_MIN_SHARE,
        ACTR_NAV_ANNOUNCEMENT_ENABLED,
    )
    from .environment import load_environment
    from .inference import build_dbn, infer_risk_posterior, posterior_to_label
    from .actr_setup import (
        ACTION_KEYS,
        KEY_TO_ACTION,
        build_model as _build_model_impl,
        register_productions as _register_productions_impl,
        seed_memory as _seed_memory_impl,
        setup_buffers as _setup_buffers_impl,
    )
    from .profile import get_user_profile_adjustments, normalize_familiarity_level
    from .reporting import generate_report
    from .utils import atom_to_name, clamp, mean_safe
    from .env_schema import (
        get_chunk_activation_modulation,
        get_chunk_activation_terms,
        ENVIRONMENT_TYPES,
    )
except ImportError:
    import sys
    from pathlib import Path

    package_dir = Path(__file__).resolve().parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))

    from config import (
        CROSSING_WAIT_STEPS_MIN,
        CROSSING_WAIT_STEPS_MAX,
        CROSSING_TRAVERSE_STEPS_MIN,
        CROSSING_TRAVERSE_STEPS_MAX,
        AVG_STEP_METERS,
        MAX_STEPS,
        BVI_WALKING_SPEED,
        SURFACE_PROBABILITY_DISTRIBUTION,
        SURFACE_PROFILES,
        CANE_OBSTACLE_PROB,
        CANE_CURB_PROB,
        CANE_WALL_PROB,
        CANE_RAILING_PROB,
        SOUND_HORN_PROB,
        SOUND_VEHICLE_APPROACH_PROB,
        SOUND_VEHICLE_APPROACH_CROSSING_PROB,
        SOUND_REVERSE_BEEP_PROB,
        HUMAN_ACTIVITY_PROB,
        VEHICLE_APPROACH_MIN_STEPS,
        VEHICLE_APPROACH_MAX_STEPS,
        VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK,
        CROSSING_HORN_PROB,
        CROSSING_REVERSE_BEEP_PROB,
        CROSSING_HUMAN_ACTIVITY_PROB,
        DBN_CROSSING_VEHICLE_ABSENT_SOFT_EVIDENCE,
        DBN_CROSSING_TRAFFIC_ABSENT_SOFT_EVIDENCE,
        DBN_CROSSING_CANE_HIT_SOFT_EVIDENCE,
        DBN_NEUTRAL_DISTANCE_FEEDBACK,
        SEEV_VALUE_SAFETY_WEIGHT,
        SEEV_VALUE_PROGRESS_WEIGHT,
        SEEV_EXPECTANCY_RISK_WEIGHT,
        ACTR_RISK_CANE_GUIDANCE_RELIEF,
        SEEV_GATE_ADAPTIVE_THRESHOLD_ENABLED,
        SEEV_GATE_THRESHOLD_WINDOW_STEPS,
        SEEV_GATE_THRESHOLD_QUANTILE,
        SEEV_GATE_THRESHOLD_MIN_HISTORY,
        ATTENTION_UNGATED_ENTRY_COEF,
        ATTENTION_GATED_CENTRAL_DANGER_BOOST,
        ATTENTION_GATED_RELIEF_ENABLED,
        LOOMING_RESUME_GATE_ENABLED,
        LOOMING_RESUME_THRESHOLD,
        ACTR_RISK_LANDMARK_RELIEF,
        SURFACE_SEGMENT_MAX_STEPS,
        SURFACE_SEGMENT_MIN_STEPS,
        TACTILE_SEGMENT_MIN_STEPS,
        TACTILE_SEGMENT_MAX_STEPS,
        CANE_GUIDANCE_MIN_STEPS,
        CANE_GUIDANCE_MAX_STEPS,
        ACTR_DYNAMIC_PM_WEIGHTS_ENABLED,
        ACTR_DYNAMIC_PM_TEMP,
        ACTR_DYNAMIC_PM_SMOOTHING,
        ACTR_DYNAMIC_PM_MIN_SHARE,
        ACTR_NAV_ANNOUNCEMENT_ENABLED,
    )
    from environment import load_environment
    from inference import build_dbn, infer_risk_posterior, posterior_to_label
    from actr_setup import (
        ACTION_KEYS,
        KEY_TO_ACTION,
        build_model as _build_model_impl,
        register_productions as _register_productions_impl,
        seed_memory as _seed_memory_impl,
        setup_buffers as _setup_buffers_impl,
    )
    from profile import get_user_profile_adjustments, normalize_familiarity_level
    from reporting import generate_report
    from utils import atom_to_name, clamp, mean_safe
    from env_schema import (
        get_chunk_activation_modulation,
        get_chunk_activation_terms,
        ENVIRONMENT_TYPES,
    )


SPATIAL_BASE_WM_LOAD = 0.15

ACT_R_PRODUCTION_FIRING_S = 0.050
ACT_R_ATTENTION_SHIFT_S = 0.085
ACT_R_MEMORY_RETRIEVAL_S = 0.150
ACT_R_MOTOR_INITIATION_S = 0.070

EFFORT_WM_MAX = 0.45

LANDMARK_TRIGGER_SCALE = 0.02
LANDMARK_TRIGGER_PROB_MIN = 0.00015
LANDMARK_TRIGGER_PROB_MAX = 0.00100
LANDMARK_EPISODE_STEPS_MIN = 4
LANDMARK_EPISODE_STEPS_MAX = 6
LANDMARK_REFRACTORY_STEPS_MIN = 90
LANDMARK_REFRACTORY_STEPS_MAX = 150

SEEV_PRIORITY_SCALE = 4.0
SEEV_TERM_FLOOR = 0.05
SEEV_EFFORT_WEIGHT = 0.75
SEEV_EFFORT_BASE = 0.0
SEEV_PRIORITY_MAX = 1.5

ACT_R_SPATIAL_PROBE_S = 0.235
ACT_R_WAIT_MONITOR_S = 0.085
NAV_CYCLE_STEPS = 5
NAV_WM_PEAK = 0.25
NAV_WM_BASE = 0.03

GUIDANCE_ABSENT_BASE = 0.01

PROBE_RELIEF_RATIO = 0.30
ANCHOR_SOFT_RETRIEVAL = 0.05

ACTR_PM_WEIGHT = 1.0
ACTR_AUDITORY_SHARE = 0.40
ACTR_TACTILE_SHARE = 0.40
ACTR_MANUAL_SHARE = 0.20
ACTR_CENTRAL_WEIGHT = 2.0
ACTR_MEMORY_WEIGHT = 4.0
ACTR_ERROR_BASE = 1.0
ACTR_ERROR_BOOST = 2.0
RISK_ERROR_BASE_PROB = 0.10
RISK_ERROR_COEF = 0.75
MEMORY_ACTIVE_RETRIEVAL_TH = 0.07
MEMORY_ACTIVE_ABSENT_STEPS_TH = 14
CENTRAL_BASE_INTENSITY = 0.10
CENTRAL_ACTIVE_TH = 0.6
ACTR_IW_HIGH_THRESHOLD = 6.0
ACTR_LOAD_RESUME_THRESHOLD = 5.0

LOOMING_BOOST_PEAK = 0.42
LOOMING_BOOST_DECAY = 0.80

LANDMARK_DECAY_RATE = 0.82
LANDMARK_RESIDUAL_COEF = 0.30
LANDMARK_TRIGGER_PROB_MIN = 0.006
LANDMARK_TRIGGER_PROB_MAX = 0.030
LANDMARK_EPISODE_STEPS_MIN = 5
LANDMARK_EPISODE_STEPS_MAX = 9
LANDMARK_REFRACTORY_STEPS_MIN = 18
LANDMARK_REFRACTORY_STEPS_MAX = 35


def _compute_probe_hold_steps():
    """Handle compute probe hold steps behavior."""
    return 6


def _compute_initial_actr_load():
    """Handle compute initial actr load behavior."""
    return ACTR_PM_WEIGHT * 1 + ACTR_MEMORY_WEIGHT * 1


def _softmax(values, temperature=1.0):
    """Handle softmax behavior."""
    safe_temp = max(1e-6, float(temperature))
    max_v = max(values)
    exps = [math.exp((v - max_v) / safe_temp) for v in values]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in exps]


def _compute_pm_channel_shares(
    prev_shares,
    *,
    vehicle_approach,
    snd_horn,
    crossing_active,
    traffic_sound,
    cane_obstacle,
    cane_guidance_present,
    current_surface_type,
    at_node,
    risk_signal,
):
    """Handle compute pm channel shares behavior."""
    if not ACTR_DYNAMIC_PM_WEIGHTS_ENABLED:
        return ACTR_AUDITORY_SHARE, ACTR_TACTILE_SHARE, ACTR_MANUAL_SHARE

    base_scores = [
        math.log(max(1e-6, ACTR_AUDITORY_SHARE)),
        math.log(max(1e-6, ACTR_TACTILE_SHARE)),
        math.log(max(1e-6, ACTR_MANUAL_SHARE)),
    ]

    if vehicle_approach:
        base_scores[0] += 0.90
    if snd_horn:
        base_scores[0] += 0.35
    if crossing_active and traffic_sound:
        base_scores[0] += 0.25

    if cane_obstacle:
        base_scores[1] += 0.90
    if cane_guidance_present:
        base_scores[1] += 0.25
    if current_surface_type == "tactile_guidance":
        base_scores[1] += 0.30

    if crossing_active:
        base_scores[2] += 0.35
    if not at_node:
        base_scores[2] += 0.15
    if risk_signal >= 0.55:
        base_scores[2] += 0.20

    dynamic_shares = _softmax(base_scores, ACTR_DYNAMIC_PM_TEMP)

    min_share = clamp(float(ACTR_DYNAMIC_PM_MIN_SHARE), low=0.0, high=0.33)
    dynamic_shares = [max(min_share, v) for v in dynamic_shares]
    dynamic_total = sum(dynamic_shares)
    dynamic_shares = [v / dynamic_total for v in dynamic_shares]

    smooth = clamp(float(ACTR_DYNAMIC_PM_SMOOTHING), low=0.0, high=0.98)
    smoothed = [
        smooth * prev + (1.0 - smooth) * cur
        for prev, cur in zip(prev_shares, dynamic_shares)
    ]
    smoothed_total = sum(smoothed)
    if smoothed_total <= 0:
        return ACTR_AUDITORY_SHARE, ACTR_TACTILE_SHARE, ACTR_MANUAL_SHARE
    return tuple(v / smoothed_total for v in smoothed)


def _build_model(profile):
    """Handle build model behavior."""
    return _build_model_impl(profile)


def _compute_route_candidates(graph, current_position, goal_node):
    """Handle compute route candidates behavior."""
    try:
        direct_path = nx.shortest_path(graph, current_position, goal_node)
        direct_next = direct_path[1] if len(direct_path) > 1 else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return direct_next


def _get_edge_length(graph, from_node, to_node):
    """Handle get edge length behavior."""
    edge_data = graph.get_edge_data(from_node, to_node)
    if edge_data is None:
        return 10.0

    if isinstance(edge_data, dict) and "length" in edge_data:
        return max(0.1, float(edge_data.get("length", 10.0)))

    lengths = []
    if isinstance(edge_data, dict):
        for attrs in edge_data.values():
            if isinstance(attrs, dict):
                lengths.append(float(attrs.get("length", 10.0)))

    if lengths:
        return max(0.1, min(lengths))
    return 10.0


def _normalize_surface_probs(surface_probs):
    """Handle normalize surface probs behavior."""
    valid_probs = {
        key: max(0.0, float(value))
        for key, value in surface_probs.items()
        if key in SURFACE_PROFILES
    }
    total = sum(valid_probs.values())
    if total <= 0:
        fallback = 1.0 / max(1, len(SURFACE_PROFILES))
        return {key: fallback for key in SURFACE_PROFILES}
    return {key: value / total for key, value in valid_probs.items()}


def _sample_surface(surface_probs, exclude_surface=None):
    """Handle sample surface behavior."""
    candidates = {
        key: value for key, value in surface_probs.items() if key != exclude_surface
    }
    if not candidates:
        candidates = surface_probs
    names = list(candidates.keys())
    weights = list(candidates.values())
    return random.choices(names, weights=weights, k=1)[0]


def _sample_surface_segment_steps(surface_type):
    """Handle sample surface segment steps behavior."""
    if surface_type == "tactile_guidance":
        return random.randint(TACTILE_SEGMENT_MIN_STEPS, TACTILE_SEGMENT_MAX_STEPS)
    return random.randint(SURFACE_SEGMENT_MIN_STEPS, SURFACE_SEGMENT_MAX_STEPS)


def _dominant_sound_type(
    snd_vehicle_approach, snd_horn, snd_reverse_beep, human_activity_triggered
):
    """Handle dominant sound type behavior."""
    return (
        "vehicle_approach"
        if snd_vehicle_approach
        else (
            "horn"
            if snd_horn
            else (
                "reverse_beep"
                if snd_reverse_beep
                else "human_activity" if human_activity_triggered else "none"
            )
        )
    )


def _dominant_cane_type(
    cane_obstacle,
    cane_tactile,
    cane_curb,
    cane_wall,
    cane_railing,
    surface_change=False,
):
    """Handle dominant cane type behavior."""
    dominant = (
        "obstacle"
        if cane_obstacle
        else (
            "tactile"
            if cane_tactile
            else (
                "curb"
                if cane_curb
                else "wall" if cane_wall else "railing" if cane_railing else "none"
            )
        )
    )
    if dominant != "none":
        return dominant
    return "surface_change" if surface_change else "none"


def _cane_guidance_type(cane_tactile, cane_curb, cane_wall, cane_railing):
    """Handle cane guidance type behavior."""
    return (
        "tactile"
        if cane_tactile
        else (
            "curb"
            if cane_curb
            else "wall" if cane_wall else "railing" if cane_railing else "none"
        )
    )


def _build_dbn_evidence(
    *,
    crossing_active,
    vehicle_approach,
    traffic_sound,
    human_activity,
    sound_salience,
    surface_change,
    current_surface_type,
    at_intersection,
    cane_hit,
    distance_feedback,
):
    """Handle build dbn evidence behavior."""
    if crossing_active:
        vehicle_evidence = (
            vehicle_approach
            if vehicle_approach
            else DBN_CROSSING_VEHICLE_ABSENT_SOFT_EVIDENCE
        )
        traffic_evidence = (
            traffic_sound
            if traffic_sound
            else DBN_CROSSING_TRAFFIC_ABSENT_SOFT_EVIDENCE
        )
        return {
            "traffic_sound": traffic_evidence,
            "vehicle_approach": vehicle_evidence,
            "human_activity": human_activity,
            "sound_salience": sound_salience,
            "cane_hit": DBN_CROSSING_CANE_HIT_SOFT_EVIDENCE,
            "surface_change": surface_change,
            "distance_feedback": DBN_NEUTRAL_DISTANCE_FEEDBACK,
            "on_tactile_guidance": False,
            "at_intersection": True,
            "crossing_active": True,
        }

    if vehicle_approach:
        return {
            "traffic_sound": traffic_sound,
            "vehicle_approach": True,
            "human_activity": human_activity,
            "sound_salience": sound_salience,
            "surface_change": surface_change,
            "distance_feedback": DBN_NEUTRAL_DISTANCE_FEEDBACK,
            "on_tactile_guidance": (current_surface_type == "tactile_guidance"),
            "at_intersection": at_intersection,
            "crossing_active": False,
        }

    return {
        "traffic_sound": traffic_sound,
        "vehicle_approach": vehicle_approach,
        "human_activity": human_activity,
        "sound_salience": sound_salience,
        "cane_hit": cane_hit,
        "surface_change": surface_change,
        "distance_feedback": distance_feedback,
        "on_tactile_guidance": (current_surface_type == "tactile_guidance"),
        "at_intersection": at_intersection,
        "crossing_active": False,
    }


def _classify_actr_risk(actr_risk_signal):
    """Handle classify actr risk behavior."""
    if actr_risk_signal >= 0.60:
        return "high"
    if actr_risk_signal >= 0.30:
        return "medium"
    return "low"


def _compute_actr_risk_signal(
    *,
    imaginal_load_state,
    imaginal_overload_phase,
    imaginal_reference_phase,
    prev_iw_total,
    cane_guidance_present,
    matched_landmark_name,
    attention_gated=True,
):
    """Handle compute actr risk signal behavior."""
    iw_norm = clamp(
        float(prev_iw_total) / max(1e-6, ACTR_IW_HIGH_THRESHOLD), low=0.0, high=1.0
    )
    return clamp(
        0.18 * float(imaginal_load_state == "overloaded")
        + 0.10 * float(imaginal_overload_phase == "sustained")
        + 0.10 * float(imaginal_reference_phase == "absent_long")
        + 0.08 * iw_norm
        - ACTR_RISK_CANE_GUIDANCE_RELIEF
        * float(cane_guidance_present)
        * (1.0 if (attention_gated or not ATTENTION_GATED_RELIEF_ENABLED) else 0.0)
        - ACTR_RISK_LANDMARK_RELIEF
        * float(matched_landmark_name != "none")
        * (1.0 if (attention_gated or not ATTENTION_GATED_RELIEF_ENABLED) else 0.0),
        low=0.0,
        high=1.0,
    )


def _extract_actr_selected_production(sim):
    """Handle extract actr selected production behavior."""
    event = getattr(sim, "current_event", None)
    if event is None:
        return None
    action = str(getattr(event, "action", "") or "")
    for marker in ("RULE SELECTED:", "RULE FIRED:"):
        if marker in action:
            name = action.split(marker, 1)[1].strip()
            return name or None
    return None


def _get_actr_last_rule(sim):
    """Handle get actr last rule behavior."""
    procedural = getattr(sim, "_Simulation__pr", None)
    if procedural is None:
        return None
    name = getattr(procedural, "last_rule", None)
    if name is None:
        return None
    name = str(name).strip()
    return name or None


def _classify_action_source(actr_selected_production, next_action):
    """Handle classify action source behavior."""
    if not actr_selected_production:
        return "actr_unknown"
    name = str(actr_selected_production)
    if name.startswith("commit_"):
        return f"actr_commit_{next_action}"
    if name.startswith("cue_"):
        return "actr_context_cue"
    if name.startswith("bk_"):
        return "actr_bookkeeping"
    if name.startswith("crossing_"):
        return "actr_crossing"
    if name.startswith("react_") or name.startswith("attend_"):
        return "actr_perception"
    if name.startswith("predict_") or name in {
        "probe_when_spatial_lost",
        "probe_when_spatial_drifting",
    }:
        return "actr_production_competition"
    if name.startswith("request_landmark_") or name.startswith("confirm_landmark_"):
        return "actr_landmark_retrieval"
    return f"actr_other:{name}"


def _classify_risk_band(actr_risk_signal):
    """Handle classify risk band behavior."""
    if actr_risk_signal >= 0.55:
        return "high"
    if actr_risk_signal >= 0.28:
        return "medium"
    return "low"


def _short_action_label(action):
    """Handle short action label behavior."""
    if action == "move_direct":
        return "move"
    if action == "stop_and_probe":
        return "probe"
    if action == "wait_at_red":
        return "wait"
    return "none"


def _parse_key_pressed(action_str):
    """Handle parse key pressed behavior."""
    if not action_str or "KEY PRESSED" not in action_str:
        return None
    payload = action_str.split("KEY PRESSED:", 1)[1]
    payload_upper = payload.upper()
    for key_name in ("SPACE", "S", "F"):
        if key_name in payload_upper:
            return key_name
    return None


def _run_until_key_pressed(sim, *, max_wait_s=2.5):
    """Handle run until key pressed behavior."""
    import simpy as _simpy

    start_time = float(sim.show_time())
    deadline = start_time + max_wait_s
    rules_fired = []
    last_decision_rule = None
    last_commit_rule = None
    while True:
        try:
            sim.step()
        except _simpy.core.EmptySchedule:
            return {
                "pressed_key": None,
                "key_time": None,
                "decision_rule": last_decision_rule,
                "commit_rule": last_commit_rule,
                "rules_fired": rules_fired,
                "timeout": False,
            }
        ev = getattr(sim, "current_event", None)
        if ev is None:
            if float(sim.show_time()) >= deadline:
                return {
                    "pressed_key": None,
                    "key_time": None,
                    "decision_rule": last_decision_rule,
                    "commit_rule": last_commit_rule,
                    "rules_fired": rules_fired,
                    "timeout": True,
                }
            continue
        action = str(getattr(ev, "action", "") or "")
        if "RULE FIRED:" in action:
            rule_name = action.split("RULE FIRED:", 1)[1].strip()
            if rule_name:
                rules_fired.append(rule_name)
                if rule_name.startswith("commit_"):
                    last_commit_rule = rule_name
                elif not rule_name.startswith("bk_") and not (
                    rule_name.startswith("reward_") or rule_name.startswith("penalty_")
                ):
                    last_decision_rule = rule_name
        if "KEY PRESSED" in action:
            pressed_key = _parse_key_pressed(action)
            return {
                "pressed_key": pressed_key,
                "key_time": float(getattr(ev, "time", sim.show_time())),
                "decision_rule": last_decision_rule,
                "commit_rule": last_commit_rule,
                "rules_fired": rules_fired,
                "timeout": False,
            }
        if float(sim.show_time()) >= deadline:
            return {
                "pressed_key": None,
                "key_time": None,
                "decision_rule": last_decision_rule,
                "commit_rule": last_commit_rule,
                "rules_fired": rules_fired,
                "timeout": True,
            }


def _seed_memory(model, initial_load, familiarity_level=1):
    """Handle seed memory behavior."""
    return _seed_memory_impl(
        model,
        initial_load,
        surface_profiles=SURFACE_PROFILES,
        surface_cpt_probs=SURFACE_PROBABILITY_DISTRIBUTION,
        familiarity_level=normalize_familiarity_level(familiarity_level),
    )


def _setup_buffers(model, initial_load):
    """Handle setup buffers behavior."""
    return _setup_buffers_impl(model, initial_load, SPATIAL_BASE_WM_LOAD)


def _register_productions(model):
    """Handle register productions behavior."""
    _register_productions_impl(model)


def _compute_landmark_bll_activation(model, sim_time, decay):
    """Handle compute landmark bll activation behavior."""
    dm = model.decmem
    best = -9.0
    for chunk, timestamps in dm._data.items():
        if getattr(chunk, "_typename", None) != "landmark":
            continue
        if len(timestamps) == 0:
            continue
        diffs = np.maximum(sim_time - np.asarray(timestamps, dtype=float), 1e-6)
        bll = float(np.log(np.sum(diffs ** (-decay))))
        if bll > best:
            best = bll
    return best


def _read_imaginal_phase(imaginal):
    """Handle read imaginal phase behavior."""
    chunks = list(imaginal)
    if not chunks:
        return {
            "overload_phase": "none",
            "reference_phase": "present",
            "safety_phase": "none",
            "load_state": "normal",
            "risk": "low",
        }
    chunk = chunks[-1]
    return {
        "overload_phase": atom_to_name(getattr(chunk, "overload_phase", "none")),
        "reference_phase": atom_to_name(getattr(chunk, "reference_phase", "present")),
        "safety_phase": atom_to_name(getattr(chunk, "safety_phase", "none")),
        "load_state": atom_to_name(getattr(chunk, "load_state", "normal")),
        "risk": atom_to_name(getattr(chunk, "risk", "low")),
    }


def _write_tick_signal(
    tick_signal_buf,
    *,
    risk_band,
    iw_high,
    prev_action_label,
    vehicle_or_obstacle,
    reference_now,
    crossing_active_flag,
    crossing_subphase,
    light_state,
    just_entered_crossing,
):
    """Handle write tick signal behavior."""
    tick_signal_buf.add(
        actr.makechunk(
            typename="tick_signal",
            risk_band=risk_band,
            iw_high="yes" if iw_high else "no",
            prev_action=prev_action_label,
            vehicle_or_obstacle="yes" if vehicle_or_obstacle else "no",
            reference_now="yes" if reference_now else "no",
            crossing_active="yes" if crossing_active_flag else "no",
            crossing_subphase=str(crossing_subphase),
            light_state=str(light_state),
            just_entered_crossing="yes" if just_entered_crossing else "no",
        )
    )


def _modify_imaginal_observation(
    imaginal,
    *,
    position,
    actr_iw,
    actr_wave,
    attention_gated,
    attention_source,
    salience_band,
):
    """Handle modify imaginal observation behavior."""
    if not list(imaginal):
        return
    imaginal.modify(
        actr.makechunk(
            typename="current_state",
            position=str(position),
            actr_iw=round(float(actr_iw), 3),
            actr_wave=round(float(actr_wave), 3),
            attention_gated=str(attention_gated),
            attention_source=str(attention_source),
            salience_band=str(salience_band),
        )
    )


def _force_imaginal_reset(imaginal, *, position, actr_iw, actr_wave):
    """Handle force imaginal reset behavior."""
    imaginal.add(
        actr.makechunk(
            typename="current_state",
            position=str(position),
            actr_iw=round(float(actr_iw), 3),
            actr_wave=round(float(actr_wave), 3),
            overload_phase="none",
            reference_phase="present",
            safety_phase="none",
            load_state="normal",
            risk="low",
            attention_gated="no",
            attention_source="none",
            salience_band="low",
        )
    )


def _snapshot_production_utilities(model):
    """Handle snapshot production utilities behavior."""
    utilities = {}
    for prod_name, production in getattr(model, "productions", {}).items():
        if production is None:
            continue
        try:
            utility = production["utility"]
        except (KeyError, TypeError):
            utility = getattr(production, "utility", None)
        if utility is None:
            continue
        utilities[prod_name] = float(utility)
    return utilities


def _update_goal_buffer_if_changed(buffer_obj, chunk_type, prev_state, new_state):
    """Handle update goal buffer if changed behavior."""
    if prev_state == new_state:
        return prev_state
    buffer_obj.add(actr.makechunk(typename=chunk_type, **new_state))
    return dict(new_state)


def run_simulation(familiarity_level=1):
    """Handle run simulation behavior."""
    familiarity_level = normalize_familiarity_level(familiarity_level)
    graph, start_node, goal_node, route_phases = load_environment()

    dynamic_max_steps = MAX_STEPS
    try:
        estimated_route_length_m = float(
            nx.shortest_path_length(graph, start_node, goal_node, weight="length")
        )
        estimated_min_steps = int(math.ceil(estimated_route_length_m / AVG_STEP_METERS))
        dynamic_max_steps = max(MAX_STEPS, int(estimated_min_steps * 5))
        print(
            f"动态步数上限: {dynamic_max_steps} "
            f"(route≈{estimated_route_length_m:.1f}m, step_len={AVG_STEP_METERS:.2f}m)"
        )
    except Exception:
        pass

    profile = get_user_profile_adjustments(
        familiarity_level=familiarity_level,
        user_id="default",
    )
    print("User profile adjustments:", profile)

    familiarity = profile.get(
        "familiarity_level", profile.get("FAMILIARITY_LEVEL", familiarity_level)
    )
    expertise = profile.get("expertise_proxy", profile.get("EXPERTISE_PROXY", 0.8))
    _act_r_decay_d = profile.get("d", profile.get("D", 0.5))

    seg_prob = 0.60 + 0.35 * familiarity
    effective_landmark_prob = clamp(
        1.0 - (1.0 - seg_prob) ** (1.0 / 10.0),
        low=0.00,
        high=0.20,
    )

    initial_load = _compute_initial_actr_load()
    print(f"初始ACT-R负荷 (IW): {initial_load:.2f}")

    model = _build_model(profile)
    _seed_memory(model, initial_load, familiarity_level=familiarity)
    goal, imaginal, _retrieval, perception_buf, spatial_loc, tick_signal = (
        _setup_buffers(model, initial_load)
    )
    _register_productions(model)
    initial_utility_snapshot = _snapshot_production_utilities(model)

    dbn_model = build_dbn()
    prev_risk_posterior = {"low": 0.60, "medium": 0.30, "high": 0.10}
    net_priority_history = deque(maxlen=SEEV_GATE_THRESHOLD_WINDOW_STEPS)

    sim_log = []
    event_log = []

    print("\n=== BVI 模拟开始 ===\n")

    current_position = start_node
    steps = 0
    crossing_nodes_on_route = {
        p["node"] for p in route_phases if p["type"] == "crossing"
    }
    print(f"路径路口数: {len(crossing_nodes_on_route)} 个（来自 route_phases 预处理）")

    priority_threshold = profile["sound_source_threshold"]

    intensity_history = deque(maxlen=6)
    cane_imu_history = deque(maxlen=5)
    looming_boost = 0.0

    prev_actr_iw_total = initial_load
    spatial_anchor_strength = 0.0

    last_anchor_time = -ACT_R_ATTENTION_SHIFT_S * 999
    prev_action = "move_direct"
    pm_channel_shares = (ACTR_AUDITORY_SHARE, ACTR_TACTILE_SHARE, ACTR_MANUAL_SHARE)
    print(
        f"PM权重模式: {'动态' if ACTR_DYNAMIC_PM_WEIGHTS_ENABLED else '静态'} "
        f"(aud={ACTR_AUDITORY_SHARE:.2f}, tac={ACTR_TACTILE_SHARE:.2f}, man={ACTR_MANUAL_SHARE:.2f})"
    )
    surface_probs = _normalize_surface_probs(SURFACE_PROBABILITY_DISTRIBUTION)
    current_surface_type = _sample_surface(surface_probs)
    current_surface_total_steps = _sample_surface_segment_steps(current_surface_type)
    current_surface_remaining = current_surface_total_steps
    landmark_episode_remaining = 0
    landmark_refractory_remaining = 0
    landmark_trigger_step = 0
    landmark_episode_id = 0
    _cane_guide_remaining = 0
    _cane_guide_type = "none"
    _veh_approach_remaining = 0
    crossing_active = False
    crossing_node = None
    crossing_subphase = "wait"
    crossing_wait_remaining = 0
    crossing_traverse_remaining = 0
    light_state = "green"
    guidance_absent_steps = 0
    probe_hold_remaining = 0
    probe_hold_reason = "none"
    prev_planned_direct_next = None
    just_entered_intersection = False
    edge_from_node = None
    edge_to_node = None
    edge_length_m = 0.0
    edge_progress_m = 0.0

    sim = model.simulation(realtime=False, trace=False, gui=False)
    actr_aw_total = 0.0
    actr_total_time = 0.0
    actr_wave = 0.0
    perception_state_prev = None
    current_plan_mode = "direct"
    last_crossing_entry_node = None
    actr_selected_production = None
    actr_commit_rule = None
    actr_rules_fired_this_step = []
    actr_step_pressed_key = None

    for step in range(dynamic_max_steps):
        sim_time = float(sim.show_time())
        step_start_sim_time = sim_time
        at_node = edge_to_node is None
        if not at_node:
            last_crossing_entry_node = None
        if at_node:
            print(f"\n--- Step {step + 1} --- Position(node): {current_position}")
        else:
            print(
                f"\n--- Step {step + 1} --- Position(edge): {edge_from_node}->{edge_to_node}, "
                f"progress={edge_progress_m:.1f}/{edge_length_m:.1f}m"
            )

        at_intersection = at_node and current_position in crossing_nodes_on_route
        just_entered_intersection = (
            at_intersection
            and not crossing_active
            and current_position != last_crossing_entry_node
        )
        if just_entered_intersection:
            crossing_active = True
            crossing_node = current_position
            last_crossing_entry_node = current_position
            crossing_wait_remaining = random.randint(
                CROSSING_WAIT_STEPS_MIN, CROSSING_WAIT_STEPS_MAX
            )
            crossing_traverse_remaining = random.randint(
                CROSSING_TRAVERSE_STEPS_MIN, CROSSING_TRAVERSE_STEPS_MAX
            )
            crossing_subphase = "wait" if crossing_wait_remaining > 0 else "traverse"
            light_state = "red" if crossing_wait_remaining > 0 else "green"
            print(
                f"[路口进入] node={crossing_node}, "
                f"wait={crossing_wait_remaining}步, traverse={crossing_traverse_remaining}步, "
                f"初始灯态={light_state}"
            )

        snd_horn = random.random() < SOUND_HORN_PROB
        snd_reverse_beep = random.random() < SOUND_REVERSE_BEEP_PROB
        human_activity_triggered = random.random() < HUMAN_ACTIVITY_PROB

        if _veh_approach_remaining > 0:
            _veh_approach_remaining -= 1
            snd_vehicle_approach = True
        else:
            _veh_trigger_prob = (
                SOUND_VEHICLE_APPROACH_CROSSING_PROB
                if crossing_active
                else SOUND_VEHICLE_APPROACH_PROB
            )
            if random.random() < _veh_trigger_prob:
                _veh_approach_remaining = (
                    random.randint(
                        VEHICLE_APPROACH_MIN_STEPS,
                        VEHICLE_APPROACH_MAX_STEPS,
                    )
                    - 1
                )
                snd_vehicle_approach = True
            else:
                snd_vehicle_approach = False
        traffic_sound = snd_horn or snd_vehicle_approach or snd_reverse_beep
        vehicle_approach = snd_vehicle_approach
        human_activity = human_activity_triggered
        sound_level = 1 if (traffic_sound or human_activity) else 0
        dominant_sound_type = _dominant_sound_type(
            snd_vehicle_approach,
            snd_horn,
            snd_reverse_beep,
            human_activity_triggered,
        )

        _on_crossing = crossing_active
        cane_obstacle = random.random() < CANE_OBSTACLE_PROB
        if _on_crossing:
            _cane_guide_remaining = 0
            _cane_guide_type = "none"
        elif _cane_guide_remaining > 0:
            _cane_guide_remaining -= 1
        else:
            if random.random() < CANE_CURB_PROB:
                _cane_guide_type = "curb"
                _cane_guide_remaining = (
                    random.randint(CANE_GUIDANCE_MIN_STEPS, CANE_GUIDANCE_MAX_STEPS) - 1
                )
            elif random.random() < CANE_WALL_PROB:
                _cane_guide_type = "wall"
                _cane_guide_remaining = (
                    random.randint(CANE_GUIDANCE_MIN_STEPS, CANE_GUIDANCE_MAX_STEPS) - 1
                )
            elif random.random() < CANE_RAILING_PROB:
                _cane_guide_type = "railing"
                _cane_guide_remaining = (
                    random.randint(CANE_GUIDANCE_MIN_STEPS, CANE_GUIDANCE_MAX_STEPS) - 1
                )
            else:
                _cane_guide_type = "none"
                _cane_guide_remaining = 0
        cane_curb = (not _on_crossing) and (_cane_guide_type == "curb")
        cane_wall = (not _on_crossing) and (_cane_guide_type == "wall")
        cane_railing = (not _on_crossing) and (_cane_guide_type == "railing")
        cane_tactile = (not _on_crossing) and (
            current_surface_type == "tactile_guidance"
        )
        cane_guidance_present = cane_curb or cane_wall or cane_railing or cane_tactile
        cane_hit = cane_obstacle or cane_guidance_present
        _dominant_cane_pre = _dominant_cane_type(
            cane_obstacle,
            cane_tactile,
            cane_curb,
            cane_wall,
            cane_railing,
        )
        previous_surface_type = current_surface_type
        current_surface_remaining -= 1
        if current_surface_remaining <= 0:
            current_surface_type = _sample_surface(
                surface_probs, exclude_surface=current_surface_type
            )
            current_surface_total_steps = _sample_surface_segment_steps(
                current_surface_type
            )
            current_surface_remaining = current_surface_total_steps
        surface_change = current_surface_type != previous_surface_type
        dominant_cane_type = (
            _dominant_cane_pre
            if _dominant_cane_pre != "none"
            else ("surface_change" if surface_change else "none")
        )
        distance_feedback = clamp(
            random.gauss(0.30 if cane_hit else 0.75, 0.15), low=0.0, high=1.0
        )

        if prev_action == "stop_and_probe" and not crossing_active:
            snd_reverse_beep = False
            human_activity_triggered = False
            human_activity = False
            traffic_sound = snd_horn or snd_vehicle_approach
            sound_level = 1 if traffic_sound else 0
            dominant_sound_type = _dominant_sound_type(
                snd_vehicle_approach,
                snd_horn,
                False,
                False,
            )
            cane_guidance_present = False
            cane_hit = cane_obstacle
            dominant_cane_type = "obstacle" if cane_obstacle else "none"
            print(f"[Probe 感知抑制] 非安全声音/引导物已静音（路段probe）")

        if crossing_active:
            previous_surface_type = current_surface_type
            current_surface_type = "carriageway"
            surface_change = current_surface_type != previous_surface_type
            current_surface_remaining = max(1, crossing_traverse_remaining)
            cane_guidance_present = False
            cane_hit = cane_obstacle
            dominant_cane_type = "obstacle" if cane_obstacle else "none"
            snd_horn = random.random() < CROSSING_HORN_PROB
            snd_vehicle_approach = (
                random.random() < SOUND_VEHICLE_APPROACH_CROSSING_PROB
            )
            snd_reverse_beep = random.random() < CROSSING_REVERSE_BEEP_PROB
            human_activity_triggered = random.random() < CROSSING_HUMAN_ACTIVITY_PROB
            traffic_sound = snd_horn or snd_vehicle_approach or snd_reverse_beep
            vehicle_approach = snd_vehicle_approach
            human_activity = human_activity_triggered
            sound_level = 1 if (traffic_sound or human_activity) else 0
            dominant_sound_type = _dominant_sound_type(
                snd_vehicle_approach,
                snd_horn,
                snd_reverse_beep,
                human_activity_triggered,
            )
            print(
                f"[路口感知覆盖] subphase={crossing_subphase}, "
                f"wait_rem={crossing_wait_remaining}, traverse_rem={crossing_traverse_remaining}, "
                f"light={light_state}"
            )

        _surface_imu_base = {
            "flat_road": 0.18,
            "tactile_guidance": 0.18,
            "carriageway": 0.22,
            "slope_surface": 0.28,
            "uneven_natural": 0.34,
            "height_drop": 0.40,
        }.get(current_surface_type, 0.20)
        cane_imu_value = clamp(
            random.gauss(0.75 if cane_hit else _surface_imu_base, 0.12)
        )
        cane_imu_history.append(cane_imu_value)
        if len(cane_imu_history) >= 2:
            _imu_mean = sum(cane_imu_history) / len(cane_imu_history)
            cane_variability = math.sqrt(
                sum((x - _imu_mean) ** 2 for x in cane_imu_history)
                / len(cane_imu_history)
            )
        else:
            cane_variability = 0.0

        current_intensity = (
            random.uniform(0.55, 1.00)
            if sound_level == 1
            else random.uniform(0.05, 0.55)
        )

        baseline = (
            mean_safe(list(intensity_history))
            if intensity_history
            else current_intensity
        )
        change_rate = current_intensity - baseline
        intensity_history.append(current_intensity)

        burst_bonus = 0.18 if (baseline < 0.30 and current_intensity > 0.75) else 0.0
        positive_change = max(0.0, change_rate)
        noise = random.uniform(-0.04, 0.04)
        if vehicle_approach:
            looming_boost = max(looming_boost, LOOMING_BOOST_PEAK)
        else:
            looming_boost = round(looming_boost * LOOMING_BOOST_DECAY, 4)
        sound_salience = clamp(
            0.52 * positive_change + burst_bonus + looming_boost + noise
        )
        vehicle_approach_raw = vehicle_approach
        if crossing_active:
            _crossing_attn_boost = 0.20 if traffic_sound else 0.08
            sound_salience = clamp(sound_salience + _crossing_attn_boost)
        else:
            if (
                vehicle_approach
                and sound_salience < VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK
            ):
                snd_vehicle_approach = False
                vehicle_approach = False
                traffic_sound = snd_horn or snd_vehicle_approach or snd_reverse_beep
                sound_level = 1 if (traffic_sound or human_activity) else 0
                dominant_sound_type = _dominant_sound_type(
                    snd_vehicle_approach,
                    snd_horn,
                    snd_reverse_beep,
                    human_activity_triggered,
                )
                looming_boost = round(looming_boost * LOOMING_BOOST_DECAY, 4)

        evidence = _build_dbn_evidence(
            crossing_active=crossing_active,
            vehicle_approach=vehicle_approach,
            traffic_sound=traffic_sound,
            human_activity=human_activity,
            sound_salience=sound_salience,
            surface_change=surface_change,
            current_surface_type=current_surface_type,
            at_intersection=at_intersection,
            cane_hit=cane_hit,
            distance_feedback=distance_feedback,
        )
        _dbn_distance_feedback = evidence.get("distance_feedback", distance_feedback)
        _dbn_at_intersection = evidence.get("at_intersection", at_intersection)
        _dbn_cane_hit = evidence.get("cane_hit", cane_hit)
        risk_posterior = infer_risk_posterior(prev_risk_posterior, evidence, dbn_model)
        prev_risk_posterior = risk_posterior
        risk_prob = float(risk_posterior["high"])

        _sal_w_aud, _sal_w_tac, _sal_w_man = pm_channel_shares

        sal_intensity_aud = clamp((current_intensity - 0.05) / 0.95)
        sal_novelty_aud = clamp(max(0.0, change_rate) + burst_bonus + looming_boost)
        sal_discriminability_aud = clamp(
            0.65 * float(dominant_sound_type != "none")
            + 0.20 * float(traffic_sound or human_activity)
            + 0.15 * float(sound_level == 1)
        )
        sal_tactile_diff_aud = 0.0
        salience_auditory = clamp(
            0.35 * sal_intensity_aud
            + 0.35 * sal_novelty_aud
            + 0.30 * sal_discriminability_aud
        )

        sal_intensity_tac = clamp(cane_imu_value)
        sal_novelty_tac = clamp(
            0.65 * float(surface_change) + 0.35 * float(cane_obstacle)
        )
        sal_discriminability_tac = clamp(
            0.60 * float(dominant_cane_type != "none")
            + 0.25 * float(cane_guidance_present)
            + 0.15 * float(current_surface_type == "tactile_guidance")
        )
        sal_tactile_diff_tac = clamp(
            0.70 * cane_variability + 0.30 * float(surface_change)
        )
        salience_tactile = clamp(
            0.20 * sal_intensity_tac
            + 0.20 * sal_novelty_tac
            + 0.20 * sal_discriminability_tac
            + 0.40 * sal_tactile_diff_tac
        )

        sal_intensity_man = clamp(
            0.55 * float(crossing_active) + 0.45 * float(not at_node)
        )
        sal_novelty_man = clamp(
            0.70 * float(surface_change) + 0.30 * float(just_entered_intersection)
        )
        sal_discriminability_man = clamp(
            0.60 * float(at_node)
            + 0.40
            * float(cane_guidance_present or current_surface_type == "tactile_guidance")
        )
        sal_tactile_diff_man = 0.0
        salience_manual = clamp(
            0.45 * sal_intensity_man
            + 0.30 * sal_novelty_man
            + 0.25 * sal_discriminability_man
        )

        risk_boost = 0.0
        salience = clamp(
            _sal_w_aud * salience_auditory
            + _sal_w_tac * salience_tactile
            + _sal_w_man * salience_manual
        )

        value_safety = clamp(
            0.35 * float(cane_obstacle or dominant_cane_type == "obstacle")
            + 0.35
            * float(dominant_sound_type in {"vehicle_approach", "horn", "reverse_beep"})
            + 0.30 * float(crossing_active)
        )
        value_progress = clamp(
            0.45
            * float(cane_guidance_present or current_surface_type == "tactile_guidance")
            + 0.30 * float(not crossing_active)
            + 0.25 * float(not at_intersection)
        )
        base_value = clamp(
            SEEV_VALUE_SAFETY_WEIGHT * value_safety
            + SEEV_VALUE_PROGRESS_WEIGHT * value_progress,
            low=0.05,
            high=1.0,
        )
        _surface_expectancy_penalty = {
            "flat_road": 0.00,
            "tactile_guidance": 0.00,
            "carriageway": 0.05,
            "slope_surface": 0.10,
            "uneven_natural": 0.13,
            "height_drop": 0.18,
        }.get(current_surface_type, 0.0)
        base_expectancy = clamp(
            SEEV_EXPECTANCY_RISK_WEIGHT * risk_prob + _surface_expectancy_penalty
        )

        current_env_type = current_surface_type
        if crossing_active:
            current_env_type = "intersection"

        landmark_schema_prob = ENVIRONMENT_TYPES.get(current_env_type, {}).get(
            "landmark_presence", 0.5
        )
        chunk_context = {
            "crossing_active": crossing_active,
            "vehicle_approach": vehicle_approach,
            "cane_guidance_present": cane_guidance_present,
            "no_reference": (
                not cane_guidance_present and current_surface_type != "tactile_guidance"
            ),
        }
        landmark_activation_terms = get_chunk_activation_terms(
            environment_type=current_env_type,
            chunk_type="landmark",
            familiarity_level=familiarity,
            experience_boost=1.0,
            context=chunk_context,
        )
        chunk_activation_mod = get_chunk_activation_modulation(
            environment_type=current_env_type,
            chunk_type="landmark",
            familiarity_level=familiarity,
            experience_boost=1.0,
            context=chunk_context,
        )

        _landmark_trigger_prob = clamp(
            effective_landmark_prob
            * landmark_schema_prob
            * chunk_activation_mod
            * LANDMARK_TRIGGER_SCALE,
            low=LANDMARK_TRIGGER_PROB_MIN,
            high=LANDMARK_TRIGGER_PROB_MAX,
        )
        landmark_triggered = False
        if landmark_episode_remaining > 0:
            landmark_episode_remaining -= 1
            landmark_matched = True
        else:
            landmark_matched = False
            if landmark_refractory_remaining > 0:
                landmark_refractory_remaining -= 1
            elif random.random() < _landmark_trigger_prob:
                landmark_triggered = True
                landmark_matched = True
                landmark_episode_id += 1
                landmark_trigger_step = step + 1
                landmark_episode_remaining = (
                    random.randint(
                        LANDMARK_EPISODE_STEPS_MIN,
                        LANDMARK_EPISODE_STEPS_MAX,
                    )
                    - 1
                )
                landmark_refractory_remaining = random.randint(
                    LANDMARK_REFRACTORY_STEPS_MIN,
                    LANDMARK_REFRACTORY_STEPS_MAX,
                )
        landmark_episode_active = landmark_matched
        if landmark_matched:
            matched_landmark_name = "generic"
            landmark_bonus = profile["landmark_expectancy_bonus"]
            _lm_phase = "新触发" if landmark_triggered else "持续"
            print(
                f"地标{_lm_phase}: trigger_count={landmark_episode_id}, "
                f"active_step_in_episode={step + 1 - landmark_trigger_step + 1}, "
                f"episode_rem={landmark_episode_remaining}, refractory={landmark_refractory_remaining}, "
                f"env={current_env_type}, trigger_p={_landmark_trigger_prob:.4f}, "
                f"A={landmark_activation_terms['activation']:.3f}, bonus={landmark_bonus:.2f}"
            )
        else:
            matched_landmark_name = "none"
            landmark_bonus = 0.0
            print(
                f"地标未触发: trigger_count={landmark_episode_id}, env={current_env_type}, "
                f"trigger_p={_landmark_trigger_prob:.4f}, refractory={landmark_refractory_remaining}, "
                f"A={landmark_activation_terms['activation']:.3f}"
            )

        spatial_anchored = (
            matched_landmark_name != "none"
            or cane_guidance_present
            or current_surface_type == "tactile_guidance"
        )
        any_reference_present = spatial_anchored
        if any_reference_present:
            guidance_absent_steps = 0
        elif prev_action == "stop_and_probe":
            pass
        else:
            guidance_absent_steps += 1

        guidance_context = {
            "crossing_active": crossing_active,
            "vehicle_approach": vehicle_approach,
            "cane_guidance_present": cane_guidance_present,
            "no_reference": (not any_reference_present),
        }
        guidance_absent_activation_terms = get_chunk_activation_terms(
            environment_type=current_env_type,
            chunk_type="guidance_absent",
            familiarity_level=familiarity,
            experience_boost=1.0,
            context=guidance_context,
        )
        guidance_absent_mod = get_chunk_activation_modulation(
            environment_type=current_env_type,
            chunk_type="guidance_absent",
            familiarity_level=familiarity,
            experience_boost=1.0,
            context=guidance_context,
        )

        _dep_steps_norm = clamp(
            guidance_absent_steps / max(1, MEMORY_ACTIVE_ABSENT_STEPS_TH),
            low=0.0,
            high=1.0,
        )
        _dep_act_norm = clamp((guidance_absent_mod - 0.3) / 2.2, low=0.0, high=1.0)
        deprivation_index = clamp(
            0.45 * _dep_steps_norm + 0.55 * _dep_act_norm, low=0.0, high=1.0
        )

        effort_wm = EFFORT_WM_MAX * clamp(prev_actr_iw_total / 10.0, low=0.0, high=1.0)

        if spatial_anchored:
            spatial_anchor_strength = 1.0
        else:
            spatial_anchor_strength = round(
                spatial_anchor_strength * LANDMARK_DECAY_RATE, 4
            )

        if landmark_matched:
            effective_landmark_bonus = landmark_bonus
        else:
            effective_landmark_bonus = (
                profile["landmark_expectancy_bonus"]
                * spatial_anchor_strength
                * LANDMARK_RESIDUAL_COEF
            )

        expectancy = clamp(base_expectancy + 0.35 * effective_landmark_bonus)
        value = clamp(base_value + 0.65 * effective_landmark_bonus, low=0.05, high=1.0)

        print(
            f"【Effort】 effort_wm={effort_wm:.2f}(prev_iw={prev_actr_iw_total:.2f}), "
            f"sp_anchor={spatial_anchor_strength:.3f}"
            f"(lm={int(landmark_matched)}/cane={int(cane_guidance_present)}/tact={int(current_surface_type=='tactile_guidance')}), "
            f"salience={salience:.2f}, eff_lm_bonus={effective_landmark_bonus:.2f}, "
            f"exp={expectancy:.2f}, val={value:.2f}"
        )

        effort_for_priority = clamp(SEEV_EFFORT_BASE + effort_wm, low=0.0, high=1.0)
        salience_term = salience + SEEV_TERM_FLOOR
        expectancy_term = expectancy + SEEV_TERM_FLOOR
        value_term = value + SEEV_TERM_FLOOR
        effort_denominator = 1.0 + SEEV_EFFORT_WEIGHT * effort_for_priority
        net_priority = clamp(
            SEEV_PRIORITY_SCALE
            * salience_term
            * expectancy_term
            * value_term
            / effort_denominator,
            low=0.0,
            high=SEEV_PRIORITY_MAX,
        )
        if (
            SEEV_GATE_ADAPTIVE_THRESHOLD_ENABLED
            and len(net_priority_history) >= SEEV_GATE_THRESHOLD_MIN_HISTORY
        ):
            _sorted_np = sorted(net_priority_history)
            _q_idx = min(
                len(_sorted_np) - 1,
                max(
                    0,
                    int(math.ceil(SEEV_GATE_THRESHOLD_QUANTILE * len(_sorted_np))) - 1,
                ),
            )
            priority_threshold_eff = _sorted_np[_q_idx]
        else:
            priority_threshold_eff = priority_threshold
        net_priority_history.append(net_priority)
        seev_gate_passed = net_priority > priority_threshold_eff
        if net_priority > 0.80:
            salience_band = "high"
        elif seev_gate_passed:
            salience_band = "medium"
        else:
            salience_band = "low"
        if matched_landmark_name != "none":
            attention_source = "landmark"
        elif dominant_sound_type != "none" and seev_gate_passed:
            attention_source = "sound"
        elif (cane_guidance_present or surface_change) and seev_gate_passed:
            attention_source = "tactile"
        else:
            attention_source = "none"
        attention_gated = "yes" if seev_gate_passed else "no"

        sound_id = f"snd_{step + 1}"
        if seev_gate_passed:
            print(
                f"注意门控通过: {sound_id}, net_priority={net_priority:.2f}, 阈值={priority_threshold_eff:.3f}, intensity={current_intensity:.3f}"
            )
        else:
            print(
                f"注意门控未通过: net_priority={net_priority:.2f}, 阈值={priority_threshold_eff:.3f}, intensity={current_intensity:.3f}"
            )

        risk_label = posterior_to_label(risk_posterior)
        _img_phase = _read_imaginal_phase(imaginal)
        _img_risk = _img_phase["risk"]
        _img_load = _img_phase["load_state"]
        _img_safety = _img_phase["safety_phase"]
        _img_overload = _img_phase["overload_phase"]
        _img_reference = _img_phase["reference_phase"]

        actr_risk_signal = _compute_actr_risk_signal(
            imaginal_load_state=_img_load,
            imaginal_overload_phase=_img_overload,
            imaginal_reference_phase=_img_reference,
            prev_iw_total=prev_actr_iw_total,
            cane_guidance_present=cane_guidance_present,
            matched_landmark_name=matched_landmark_name,
            attention_gated=seev_gate_passed,
        )
        actr_risk_label = _classify_actr_risk(actr_risk_signal)

        aural_event_peak = 0.08
        if traffic_sound and not vehicle_approach:
            aural_event_peak = max(aural_event_peak, 0.22)
        if human_activity:
            aural_event_peak = max(aural_event_peak, 0.33)
        aural_event_peak = clamp(
            aural_event_peak + 0.18 * sound_salience, low=0.05, high=0.75
        )

        spatial_risk_load = 0.0
        surface_meta = SURFACE_PROFILES[current_surface_type]
        surface_wm_modifier = float(surface_meta.get("wm_modifier", 1.0))
        surface_drift_rate = float(surface_meta.get("drift_rate", 0.02))
        segment_progress = max(
            0, current_surface_total_steps - current_surface_remaining
        )
        spatial_drift_activation_terms = get_chunk_activation_terms(
            environment_type=current_env_type,
            chunk_type="spatial_drift",
            familiarity_level=familiarity,
            experience_boost=1.0,
            context={
                "crossing_active": crossing_active,
                "vehicle_approach": vehicle_approach,
                "cane_guidance_present": cane_guidance_present,
                "no_reference": (not any_reference_present),
            },
        )
        spatial_drift_mod = get_chunk_activation_modulation(
            environment_type=current_env_type,
            chunk_type="spatial_drift",
            familiarity_level=familiarity,
            experience_boost=1.0,
            context={
                "crossing_active": crossing_active,
                "vehicle_approach": vehicle_approach,
                "cane_guidance_present": cane_guidance_present,
                "no_reference": (not any_reference_present),
            },
        )
        _base_surface_drift_load = clamp(
            segment_progress * surface_drift_rate, low=0.0, high=0.30
        )
        _drift_gain = clamp(
            0.65 + 0.90 * ((spatial_drift_mod - 0.3) / 2.2), low=0.35, high=1.65
        )
        surface_drift_load = clamp(
            _base_surface_drift_load * _drift_gain, low=0.0, high=0.35
        )
        _anchor_str = spatial_anchor_strength
        if _anchor_str > 0.50:
            _anchor_state = "anchored"
        elif _anchor_str > 0.20:
            _anchor_state = "drifting"
        else:
            _anchor_state = "lost"
        landmark_memory_relief = clamp(0.20 + 0.30 * familiarity, low=0.20, high=0.50)
        if landmark_matched:
            retrieval_wm_load = 0.10 * (1.0 - landmark_memory_relief)
        elif spatial_anchor_strength > 0.50:
            retrieval_wm_load = ANCHOR_SOFT_RETRIEVAL * (
                1.0 - 0.50 * landmark_memory_relief
            )
        elif current_intensity > 0.65:
            retrieval_wm_load = 0.10
        else:
            retrieval_wm_load = 0.0

        spatial_loc.add(
            actr.makechunk(
                typename="spatial_state",
                anchored="yes" if spatial_anchored else "no",
                anchor_state=_anchor_state,
                unanchored_steps=guidance_absent_steps,
                load=round(retrieval_wm_load, 3),
                risk_weight=round(spatial_risk_load, 3),
                surface_type=current_surface_type,
                surface_cn=surface_meta.get("cn", current_surface_type),
                wm_modifier=round(surface_wm_modifier, 3),
                drift_rate=round(surface_drift_rate, 3),
                segment_remaining=current_surface_remaining,
            )
        )
        if ACTR_NAV_ANNOUNCEMENT_ENABLED:
            nav_phase = step % NAV_CYCLE_STEPS
            nav_announcement = nav_phase == 0
            nav_wm_load = NAV_WM_BASE + NAV_WM_PEAK * max(
                0.0, 1.0 - nav_phase / max(1, NAV_CYCLE_STEPS - 1)
            )
        else:
            nav_phase = -1
            nav_announcement = False
            nav_wm_load = 0.0

        _threat_active = vehicle_approach or (at_intersection and crossing_active)
        load_anchor_released = spatial_anchored and not _threat_active
        if load_anchor_released:
            last_anchor_time = sim_time
            _anchor_src = (
                f"landmark={matched_landmark_name}"
                if matched_landmark_name != "none"
                else (
                    f"tactile_surface"
                    if current_surface_type == "tactile_guidance"
                    else f"cane_guidance={dominant_cane_type}"
                )
            )
            print(f"[Load anchor] {_anchor_src}, BLL decay starts")

        _t_since_anchor = max(ACT_R_ATTENTION_SHIFT_S, sim_time - last_anchor_time)
        anchor_suppression = clamp(
            (_t_since_anchor / ACT_R_ATTENTION_SHIFT_S) ** (-_act_r_decay_d),
            low=0.0,
            high=1.0,
        )

        retrieval_wm_load = ANCHOR_SOFT_RETRIEVAL + (
            retrieval_wm_load - ANCHOR_SOFT_RETRIEVAL
        ) * (1.0 - anchor_suppression)

        if load_anchor_released:
            deprivation_index = 0.0

        print(
            f"【感觉剥夺】absent_steps={guidance_absent_steps}, dep_idx={deprivation_index:.2f}, "
            f"A_guidance={guidance_absent_activation_terms['activation']:.2f}, "
            f"any_ref={'Y' if any_reference_present else 'N'}(cane_guide={'Y' if cane_guidance_present else 'N'}, lm={matched_landmark_name})"
        )

        _attn_entry = 1.0 if seev_gate_passed else ATTENTION_UNGATED_ENTRY_COEF
        actr_auditory_active = (
            int(dominant_sound_type != "none" or nav_announcement or crossing_active)
            * _attn_entry
        )
        actr_tactile_active = (
            int(dominant_cane_type != "none" or surface_change or cane_guidance_present)
            * _attn_entry
        )
        memory_active_retrieval_th = MEMORY_ACTIVE_RETRIEVAL_TH
        memory_active_absent_steps_th = MEMORY_ACTIVE_ABSENT_STEPS_TH
        actr_central_intensity = clamp(
            CENTRAL_BASE_INTENSITY
            + 0.30 * float(_img_risk in {"medium", "high"})
            + 0.18 * float(_img_load == "overloaded")
            + 0.10 * float(_img_overload == "sustained")
            + 0.10 * float(_img_reference == "absent_long")
            + 0.20 * float(crossing_active)
            + 0.03 * float(prev_action == "stop_and_probe")
            + ATTENTION_GATED_CENTRAL_DANGER_BOOST
            * float(
                seev_gate_passed
                and (
                    vehicle_approach
                    or cane_obstacle
                    or dominant_sound_type in {"horn", "reverse_beep"}
                )
            ),
            low=0.0,
            high=1.0,
        )
        actr_central_active = int(actr_central_intensity >= CENTRAL_ACTIVE_TH)
        memory_retrieval_needed = (
            retrieval_wm_load >= memory_active_retrieval_th
            or guidance_absent_steps >= memory_active_absent_steps_th
            or (
                landmark_matched
                and seev_gate_passed
                and spatial_anchor_strength <= 0.50
            )
        )
        actr_memory_active = int(memory_retrieval_needed)

        pm_channel_shares = _compute_pm_channel_shares(
            pm_channel_shares,
            vehicle_approach=vehicle_approach,
            snd_horn=snd_horn,
            crossing_active=crossing_active,
            traffic_sound=traffic_sound,
            cane_obstacle=cane_obstacle,
            cane_guidance_present=cane_guidance_present,
            current_surface_type=current_surface_type,
            at_node=at_node,
            risk_signal=actr_risk_signal,
        )
        actr_auditory_share, actr_tactile_share, actr_manual_share = pm_channel_shares

        actr_iw_gate = prev_actr_iw_total
        iw_high_now = actr_iw_gate >= ACTR_IW_HIGH_THRESHOLD

        probe_safe_now = (
            prev_action == "stop_and_probe"
            and (not vehicle_approach)
            and (not cane_obstacle)
            and (actr_risk_signal < 0.28)
        )

        _modify_imaginal_observation(
            imaginal,
            position=current_position,
            actr_iw=round(actr_iw_gate, 2),
            actr_wave=round(actr_wave, 2),
            attention_gated=attention_gated,
            attention_source=attention_source,
            salience_band=salience_band,
        )
        _img_phase_after_obs = _read_imaginal_phase(imaginal)
        load_state = _img_phase_after_obs["load_state"]
        actr_risk_label_in_imaginal = _img_phase_after_obs["risk"]

        if not list(goal):
            goal.add(
                actr.makechunk(
                    typename="goal", task="navigating", next_action="continue_forward"
                )
            )

        if at_node:
            planned_direct_next = _compute_route_candidates(
                graph,
                current_position,
                goal_node,
            )
        else:
            planned_direct_next = edge_to_node
        perception_state = {
            "sound_type": dominant_sound_type,
            "dominant_cane_type": dominant_cane_type,
            "surface_change": "yes" if surface_change else "no",
            "sound_intensity": round(current_intensity, 3),
            "sound_salience": round(salience, 3),
            "sound_identified": "yes" if matched_landmark_name != "none" else "no",
            "cane_imu": round(cane_imu_value, 3),
            "cane_variability": round(cane_variability, 3),
            "cane_obstacle": "yes" if cane_obstacle else "no",
            "cane_guidance": "yes" if cane_guidance_present else "no",
            "cane_guidance_type": _cane_guidance_type(
                cane_tactile, cane_curb, cane_wall, cane_railing
            ),
            "at_node": "yes" if at_node else "no",
            "at_intersection": "yes" if at_intersection else "no",
            "crossing_active": "yes" if crossing_active else "no",
            "light_state": light_state,
            "crossing_node": (
                str(crossing_node) if crossing_node is not None else "none"
            ),
        }
        perception_state_prev = _update_goal_buffer_if_changed(
            perception_buf,
            "perception_input",
            perception_state_prev,
            perception_state,
        )

        if actr_memory_active:
            memory_intensity = clamp(
                0.10 + 0.55 * retrieval_wm_load + 0.15 * float(seev_gate_passed),
                low=0.0,
                high=1.0,
            )
            if spatial_anchored and not landmark_matched:
                memory_intensity = max(memory_intensity, 0.15)
            actr_memory_retrieval_dt = ACT_R_MEMORY_RETRIEVAL_S * (
                0.40 + 0.60 * memory_intensity
            )
        else:
            memory_intensity = 0.0
            actr_memory_retrieval_dt = 0.0

        risk_band_now = _classify_risk_band(actr_risk_signal)
        prev_action_label = _short_action_label(prev_action)
        looming_alert_active = bool(
            LOOMING_RESUME_GATE_ENABLED and looming_boost >= LOOMING_RESUME_THRESHOLD
        )
        vehicle_or_obstacle_now = bool(
            vehicle_approach or cane_obstacle or looming_alert_active
        )
        _write_tick_signal(
            tick_signal,
            risk_band=risk_band_now,
            iw_high=iw_high_now,
            prev_action_label=prev_action_label,
            vehicle_or_obstacle=vehicle_or_obstacle_now,
            reference_now=any_reference_present,
            crossing_active_flag=crossing_active,
            crossing_subphase=(crossing_subphase if crossing_active else "none"),
            light_state=(light_state if crossing_active else "none"),
            just_entered_crossing=just_entered_intersection,
        )

        run_result = _run_until_key_pressed(sim, max_wait_s=2.5)
        actr_step_pressed_key = run_result["pressed_key"]
        actr_selected_production = (
            run_result["decision_rule"] or run_result["commit_rule"]
        )
        actr_commit_rule = run_result["commit_rule"]
        actr_rules_fired_this_step = run_result["rules_fired"]

        if actr_step_pressed_key and actr_step_pressed_key in KEY_TO_ACTION:
            next_action = KEY_TO_ACTION[actr_step_pressed_key]
        else:
            goal_chunks = list(goal)
            next_action = "move_direct"
            if goal_chunks:
                goal_intent = atom_to_name(goal_chunks[-1].next_action)
                if goal_intent in {"move_direct", "stop_and_probe", "wait_at_red"}:
                    next_action = goal_intent
            if run_result["timeout"]:
                event_log.append(
                    {
                        "step": step + 1,
                        "type": "actr_run_timeout",
                        "detail": (
                            f"no KEY PRESSED in 2.5s; fallback={next_action}; "
                            f"rules_fired={len(actr_rules_fired_this_step)}"
                        ),
                    }
                )

        action_source = _classify_action_source(actr_selected_production, next_action)
        current_plan_mode = "probe" if next_action == "stop_and_probe" else "direct"

        probe_retrieval_success = False
        probe_gate_released = False
        if prev_action == "stop_and_probe" and not (
            crossing_active and crossing_subphase == "wait"
        ):
            retrieval_gate_prob = clamp(
                0.20 + 0.60 * familiarity + 0.16, low=0.20, high=0.95
            )
            probe_retrieval_success = (matched_landmark_name != "none") and (
                random.random() < retrieval_gate_prob
            )
            probe_guidance_support = cane_guidance_present or (
                spatial_anchor_strength > 0.50 and actr_risk_signal < 0.45
            )
            probe_load_recovered = actr_iw_gate < (
                ACTR_LOAD_RESUME_THRESHOLD + 0.45 * familiarity
            )
            probe_alert_calmed = (
                looming_boost < LOOMING_RESUME_THRESHOLD
                or not LOOMING_RESUME_GATE_ENABLED
            )
            if (
                probe_retrieval_success
                or probe_guidance_support
                or probe_load_recovered
            ) and probe_alert_calmed:
                probe_gate_released = True
                probe_hold_reason = "released"
                event_log.append(
                    {
                        "step": step + 1,
                        "type": "probe_safe_evidence",
                        "detail": (
                            f"retrieval={int(probe_retrieval_success)}, "
                            f"guidance={int(probe_guidance_support)}, "
                            f"load_recovered={int(probe_load_recovered)}, "
                            f"alert_calmed={int(looming_boost < LOOMING_RESUME_THRESHOLD)}"
                        ),
                    }
                )

        central_decision_error = bool(
            next_action == "move_direct"
            and (
                (crossing_active and crossing_subphase == "wait")
                or (actr_risk_signal >= 0.60 and (vehicle_approach or cane_obstacle))
            )
        )
        if central_decision_error:
            event_log.append(
                {
                    "step": step + 1,
                    "type": "central_decision_error",
                    "detail": (
                        f"risk={actr_risk_signal:.3f}, vehicle={int(vehicle_approach)}, "
                        f"obstacle={int(cane_obstacle)}, "
                        f"crossing={'wait' if crossing_active and crossing_subphase == 'wait' else 'no'}"
                    ),
                }
            )

        risk_error_prob = clamp(
            RISK_ERROR_BASE_PROB + RISK_ERROR_COEF * actr_risk_signal,
            low=0.05,
            high=0.98,
        )
        auditory_error_flag = random.random() < risk_error_prob
        tactile_error_flag = random.random() < risk_error_prob
        actr_auditory_error = (
            ACTR_ERROR_BOOST if auditory_error_flag else ACTR_ERROR_BASE
        )
        actr_tactile_error = ACTR_ERROR_BOOST if tactile_error_flag else ACTR_ERROR_BASE

        actr_manual_active = int(next_action in {"move_direct", "stop_and_probe"})
        manual_error_flag = bool(central_decision_error)
        actr_manual_error = ACTR_ERROR_BOOST if manual_error_flag else ACTR_ERROR_BASE

        actr_central_error = (
            ACTR_ERROR_BOOST if central_decision_error else ACTR_ERROR_BASE
        )

        _lm_bll = _compute_landmark_bll_activation(model, sim_time, _act_r_decay_d)
        _bll_adjusted = _lm_bll
        _bll_rt = float(profile.get("rt", profile.get("RT", -2.0)))
        _mem_error_norm = clamp(
            (_bll_rt + 1.5 - _bll_adjusted) / 3.0,
            low=0.0,
            high=1.0,
        )
        actr_memory_error = ACTR_ERROR_BASE + _mem_error_norm * (
            ACTR_ERROR_BOOST - ACTR_ERROR_BASE
        )
        memory_error_flag = _mem_error_norm >= 0.65
        actr_pm_error = max(actr_auditory_error, actr_tactile_error, actr_manual_error)

        actr_iw_auditory = (
            ACTR_PM_WEIGHT
            * actr_auditory_share
            * actr_auditory_error
            * actr_auditory_active
        )
        actr_iw_tactile = (
            ACTR_PM_WEIGHT
            * actr_tactile_share
            * actr_tactile_error
            * actr_tactile_active
        )
        actr_iw_manual = (
            ACTR_PM_WEIGHT * actr_manual_share * actr_manual_error * actr_manual_active
        )
        actr_iw_pm = actr_iw_auditory + actr_iw_tactile + actr_iw_manual
        actr_iw_central = (
            ACTR_CENTRAL_WEIGHT * actr_central_error * actr_central_intensity
        )
        _spatial_load_norm = clamp(retrieval_wm_load / 0.35, low=0.0, high=1.0)
        _memory_intensity = max(float(actr_memory_active), _spatial_load_norm)
        actr_iw_memory = ACTR_MEMORY_WEIGHT * actr_memory_error * _memory_intensity

        probe_safe_now = (
            next_action == "stop_and_probe"
            and not vehicle_approach
            and not cane_obstacle
        )
        if probe_safe_now:
            reference_relief_bonus = (
                0.25 * familiarity
                if (spatial_anchored or spatial_anchor_strength > 0.50)
                else 0.0
            )
            probe_relief = clamp(
                PROBE_RELIEF_RATIO * (0.5 if crossing_active else 1.0)
                + reference_relief_bonus,
                low=0.0,
                high=0.75,
            )
            actr_iw_central *= 1.0 - probe_relief
            actr_iw_memory *= 1.0 - probe_relief

        actr_iw_total = actr_iw_pm + actr_iw_central + actr_iw_memory

        prev_action = next_action

        actr_dt_auditory = ACT_R_ATTENTION_SHIFT_S if actr_auditory_active else 0.0
        actr_dt_tactile = ACT_R_ATTENTION_SHIFT_S if actr_tactile_active else 0.0
        if next_action == "move_direct":
            actr_dt_manual = ACT_R_MOTOR_INITIATION_S
        elif next_action == "stop_and_probe":
            actr_dt_manual = ACT_R_SPATIAL_PROBE_S
        elif next_action == "wait_at_red":
            actr_dt_manual = 0.0
        else:
            actr_dt_manual = 0.0
        actr_dt_central = ACT_R_PRODUCTION_FIRING_S * actr_central_intensity
        actr_dt_memory = actr_memory_retrieval_dt

        route_mode = current_plan_mode

        step_travel_m = 0.0
        if next_action == "wait_at_red":
            print("动作: wait_at_red -> 红灯等待，本步原地")
        elif next_action == "stop_and_probe":
            print("动作: stop_and_probe -> 本步原地探测")
        else:
            target_next = None
            if planned_direct_next is not None:
                target_next = planned_direct_next
            else:
                print("路径阻塞！")
                break

            if not at_node and edge_to_node is not None:
                target_next = edge_to_node

            if at_node:
                edge_from_node = current_position
                edge_to_node = target_next
                edge_length_m = _get_edge_length(graph, edge_from_node, edge_to_node)
                edge_progress_m = 0.0

            edge_remaining_m = max(0.0, edge_length_m - edge_progress_m)
            step_travel_m = min(AVG_STEP_METERS, edge_remaining_m)
            edge_progress_m += step_travel_m

            if edge_progress_m + 1e-6 >= edge_length_m:
                current_position = edge_to_node
                edge_from_node = None
                edge_to_node = None
                edge_length_m = 0.0
                edge_progress_m = 0.0
                print(f"位置更新: 抵达下一节点, step={step_travel_m:.1f}m")
                if crossing_active and crossing_subphase == "traverse":
                    crossing_active = False
                    crossing_node = None
                    light_state = "green"
                    crossing_subphase = "wait"
                    crossing_traverse_remaining = 0
                    print(f"[路口穿越完成-边同步] 抵达下一节点，路口感知覆盖同步退出")
            else:
                print(
                    f"位置更新: 边内推进 step={step_travel_m:.1f}m, "
                    f"remaining={edge_length_m - edge_progress_m:.1f}m"
                )

        if crossing_active:
            if crossing_subphase == "wait":
                if crossing_wait_remaining > 0:
                    crossing_wait_remaining -= 1
                if crossing_wait_remaining == 0:
                    crossing_subphase = "traverse"
                    light_state = "green"
                    print(
                        f"[路口灯变绿] node={crossing_node}, 开始穿越, traverse_rem={crossing_traverse_remaining}步"
                    )
            else:
                if crossing_traverse_remaining > 0:
                    crossing_traverse_remaining -= 1
                if crossing_traverse_remaining == 0:
                    crossing_active = False
                    crossing_node = None
                    light_state = "green"
                    crossing_subphase = "wait"
                    print(f"[路口穿越完成] 退出路口阶段, 恢复路段状态")
        if next_action == "move_direct" and step_travel_m > 0:
            walk_dt = step_travel_m / BVI_WALKING_SPEED
            target_time = float(sim.show_time()) + walk_dt
            try:
                sim.run(target_time)
            except Exception:
                pass

        sim_time = float(sim.show_time())
        actr_step_dt = max(1e-6, sim_time - step_start_sim_time)
        actr_aw_total += actr_iw_total * actr_step_dt
        actr_total_time += actr_step_dt
        actr_wave = actr_aw_total / actr_total_time if actr_total_time > 0 else 0.0

        reached_goal_step = current_position == goal_node

        if reached_goal_step:
            step_outcome_label = "reached_goal"
        elif (
            central_decision_error
            and (vehicle_approach or cane_obstacle)
            and actr_risk_signal >= 0.75
        ):
            step_outcome_label = "collision"
        elif central_decision_error:
            step_outcome_label = "central_error"
        elif probe_retrieval_success:
            step_outcome_label = "probe_success"
        else:
            step_outcome_label = "safe"

        steps += 1
        prev_actr_iw_total = actr_iw_total
        print(
            f"诊断: risk_dbn={risk_prob:.2f}, risk_actr={actr_risk_signal:.2f}, intensity={current_intensity:.2f}, change_rate={change_rate:.2f}, "
            f"salience={salience:.2f}, value={value:.2f}, expectancy={expectancy:.2f}, effort={effort_for_priority:.2f}, "
            f"SEEV_mult=({salience_term:.2f}*{expectancy_term:.2f}*{value_term:.2f})/{effort_denominator:.2f}, "
            f"lm_bonus={landmark_bonus:.2f}→eff={effective_landmark_bonus:.2f}, net_priority={net_priority:.2f}"
        )
        print(
            f"匹配结果: landmark={matched_landmark_name}, threshold={priority_threshold:.2f}, "
            f"action={next_action}, IW={actr_iw_total:.2f}, W_ave={actr_wave:.2f}"
        )
        print(
            f"ACT-R 负荷: IW={actr_iw_total:.2f} "
            f"(aud={actr_iw_auditory:.2f}, tac={actr_iw_tactile:.2f}, man={actr_iw_manual:.2f}, "
            f"c={actr_iw_central:.2f}, m={actr_iw_memory:.2f}), "
            f"W=({actr_auditory_share:.2f},{actr_tactile_share:.2f},{actr_manual_share:.2f}), "
            f"E=({actr_auditory_error:.0f},{actr_tactile_error:.0f},{actr_manual_error:.0f},{actr_central_error:.0f},{actr_memory_error:.0f}), "
            f"A=({actr_auditory_active},{actr_tactile_active},{actr_manual_active},{actr_central_active},{actr_memory_active}), "
            f"dt(ms)=({actr_dt_auditory*1000:.0f},{actr_dt_tactile*1000:.0f},{actr_dt_manual*1000:.0f},{actr_dt_central*1000:.0f},{actr_dt_memory*1000:.0f}), "
            f"W_ave={actr_wave:.2f}"
        )
        print(
            f"ACT-R 内层执行: key={actr_step_pressed_key or 'none'}, "
            f"action={next_action}, decision_rule={actr_selected_production or 'none'}, "
            f"commit={actr_commit_rule or 'none'}, fired={len(actr_rules_fired_this_step)} 条"
        )
        print(f"step outcome (日志): {step_outcome_label}")

        if actr_iw_total >= ACTR_IW_HIGH_THRESHOLD and next_action == "stop_and_probe":
            print("【反向传播触发】ACT-R高负荷风险上升，执行保护性停探测")

        step_record = {
            "step": step + 1,
            "position": str(current_position),
            "sound_level": sound_level,
            "dominant_sound_type": dominant_sound_type,
            "dominant_cane_type": dominant_cane_type,
            "snd_horn": snd_horn,
            "snd_vehicle_approach": snd_vehicle_approach,
            "snd_reverse_beep": snd_reverse_beep,
            "human_activity_triggered": human_activity_triggered,
            "traffic_sound": traffic_sound,
            "vehicle_approach": vehicle_approach,
            "vehicle_approach_raw": vehicle_approach_raw,
            "sound_salience": round(sound_salience, 4),
            "retrieval_wm_load": round(retrieval_wm_load, 4),
            "human_activity": human_activity,
            "cane_hit": _dbn_cane_hit,
            "cane_hit_raw": cane_hit,
            "surface_change": surface_change,
            "surface_type": current_surface_type,
            "surface_cn": surface_meta.get("cn", current_surface_type),
            "surface_segment_remaining": current_surface_remaining,
            "at_intersection": _dbn_at_intersection,
            "at_intersection_raw": at_intersection,
            "crossing_active": crossing_active,
            "light_state": light_state,
            "distance_feedback": round(float(_dbn_distance_feedback), 4),
            "distance_feedback_raw": round(distance_feedback, 4),
            "cane_imu_value": round(cane_imu_value, 4),
            "cane_variability": round(cane_variability, 4),
            "risk_prob": round(risk_prob, 4),
            "risk_post_low": round(risk_posterior["low"], 4),
            "risk_post_medium": round(risk_posterior["medium"], 4),
            "risk_post_high": round(risk_posterior["high"], 4),
            "risk_label": risk_label,
            "actr_risk_signal": round(actr_risk_signal, 4),
            "actr_risk_label": actr_risk_label,
            "intensity": round(current_intensity, 4),
            "baseline": round(baseline, 4),
            "change_rate": round(change_rate, 4),
            "salience": round(salience, 4),
            "salience_auditory": round(salience_auditory, 4),
            "salience_tactile": round(salience_tactile, 4),
            "salience_manual": round(salience_manual, 4),
            "salience_risk_boost": round(risk_boost, 4),
            "sal_intensity_aud": round(sal_intensity_aud, 4),
            "sal_novelty_aud": round(sal_novelty_aud, 4),
            "sal_discriminability_aud": round(sal_discriminability_aud, 4),
            "sal_tactile_diff_aud": round(sal_tactile_diff_aud, 4),
            "sal_intensity_tac": round(sal_intensity_tac, 4),
            "sal_novelty_tac": round(sal_novelty_tac, 4),
            "sal_discriminability_tac": round(sal_discriminability_tac, 4),
            "sal_tactile_diff_tac": round(sal_tactile_diff_tac, 4),
            "sal_intensity_man": round(sal_intensity_man, 4),
            "sal_novelty_man": round(sal_novelty_man, 4),
            "sal_discriminability_man": round(sal_discriminability_man, 4),
            "sal_tactile_diff_man": round(sal_tactile_diff_man, 4),
            "value": round(value, 4),
            "value_safety": round(value_safety, 4),
            "value_progress": round(value_progress, 4),
            "expectancy": round(expectancy, 4),
            "landmark_bonus": round(landmark_bonus, 4),
            "effective_landmark_bonus": round(effective_landmark_bonus, 4),
            "landmark_actr_base": round(landmark_activation_terms["base_level"], 4),
            "landmark_actr_spreading": round(landmark_activation_terms["spreading"], 4),
            "landmark_actr_activation": round(
                landmark_activation_terms["activation"], 4
            ),
            "seev_effort": round(effort_for_priority, 4),
            "seev_salience_term": round(salience_term, 4),
            "seev_expectancy_term": round(expectancy_term, 4),
            "seev_value_term": round(value_term, 4),
            "seev_effort_denominator": round(effort_denominator, 4),
            "net_priority": round(net_priority, 4),
            "gate_passed": seev_gate_passed,
            "seev_attention_gated": attention_gated,
            "seev_attention_source": attention_source,
            "seev_salience_band": salience_band,
            "matched_landmark": matched_landmark_name,
            "landmark_triggered": landmark_triggered,
            "landmark_episode_active": landmark_episode_active,
            "landmark_episode_id": (
                landmark_episode_id if landmark_episode_active else 0
            ),
            "landmark_episode_remaining": landmark_episode_remaining,
            "landmark_refractory_remaining": landmark_refractory_remaining,
            "landmark_trigger_probability": round(_landmark_trigger_prob, 4),
            "landmark_trigger_step": (
                landmark_trigger_step if landmark_episode_active else 0
            ),
            "actr_iw_total": round(actr_iw_total, 4),
            "actr_iw_pm": round(actr_iw_pm, 4),
            "actr_pm_share_auditory": round(actr_auditory_share, 4),
            "actr_pm_share_tactile": round(actr_tactile_share, 4),
            "actr_pm_share_manual": round(actr_manual_share, 4),
            "actr_iw_auditory": round(actr_iw_auditory, 4),
            "actr_iw_tactile": round(actr_iw_tactile, 4),
            "actr_iw_manual": round(actr_iw_manual, 4),
            "actr_iw_central": round(actr_iw_central, 4),
            "actr_iw_memory": round(actr_iw_memory, 4),
            "actr_wave": round(actr_wave, 4),
            "actr_aw_total": round(actr_aw_total, 4),
            "actr_step_dt": round(actr_step_dt, 4),
            "actr_pm_active": int(
                actr_auditory_active or actr_tactile_active or actr_manual_active
            ),
            "actr_auditory_active": actr_auditory_active,
            "actr_tactile_active": actr_tactile_active,
            "actr_manual_active": actr_manual_active,
            "actr_central_active": actr_central_active,
            "actr_central_intensity": round(actr_central_intensity, 4),
            "actr_memory_active": actr_memory_active,
            "actr_memory_intensity": round(memory_intensity, 4),
            "actr_pm_error": round(actr_pm_error, 2),
            "actr_risk_error_prob": round(risk_error_prob, 4),
            "actr_auditory_error": round(actr_auditory_error, 2),
            "actr_tactile_error": round(actr_tactile_error, 2),
            "actr_manual_error": round(actr_manual_error, 2),
            "actr_central_error": round(actr_central_error, 2),
            "actr_memory_error": round(actr_memory_error, 2),
            "actr_central_decision_error": central_decision_error,
            "memory_bll_activation": round(_lm_bll, 4),
            "memory_error_flag": memory_error_flag,
            "actr_dt_auditory": round(actr_dt_auditory, 4),
            "actr_dt_tactile": round(actr_dt_tactile, 4),
            "actr_dt_manual": round(actr_dt_manual, 4),
            "actr_dt_central": round(actr_dt_central, 4),
            "actr_dt_memory": round(actr_dt_memory, 4),
            "imaginal_overload_phase": _img_overload,
            "imaginal_reference_phase": _img_reference,
            "imaginal_safety_phase": _img_safety,
            "imaginal_load_state": load_state,
            "imaginal_risk": actr_risk_label_in_imaginal,
            "spatial_unanchored_steps": guidance_absent_steps,
            "spatial_risk_load": round(spatial_risk_load, 4),
            "surface_wm_modifier": round(surface_wm_modifier, 4),
            "surface_drift_rate": round(surface_drift_rate, 4),
            "surface_drift_load": round(surface_drift_load, 4),
            "spatial_drift_actr_base": round(
                spatial_drift_activation_terms["base_level"], 4
            ),
            "spatial_drift_actr_spreading": round(
                spatial_drift_activation_terms["spreading"], 4
            ),
            "spatial_drift_actr_activation": round(
                spatial_drift_activation_terms["activation"], 4
            ),
            "effort_wm": round(effort_wm, 4),
            "spatial_anchor_strength": round(spatial_anchor_strength, 4),
            "spatial_anchored": spatial_anchored,
            "nav_phase": nav_phase,
            "nav_announcement": nav_announcement,
            "nav_wm_load": round(nav_wm_load, 4),
            "load_anchor_released": load_anchor_released,
            "cane_obstacle": cane_obstacle,
            "cane_guidance_present": cane_guidance_present,
            "cane_guidance_type": _cane_guidance_type(
                cane_tactile, cane_curb, cane_wall, cane_railing
            ),
            "guidance_absent_steps": guidance_absent_steps,
            "deprivation_index": round(deprivation_index, 4),
            "guidance_absent_actr_base": round(
                guidance_absent_activation_terms["base_level"], 4
            ),
            "guidance_absent_actr_spreading": round(
                guidance_absent_activation_terms["spreading"], 4
            ),
            "guidance_absent_actr_activation": round(
                guidance_absent_activation_terms["activation"], 4
            ),
            "probe_hold_remaining": probe_hold_remaining,
            "probe_hold_reason": probe_hold_reason,
            "probe_turn_event": bool(
                at_node
                and planned_direct_next is not None
                and prev_planned_direct_next is not None
                and str(planned_direct_next) != prev_planned_direct_next
            ),
            "central_decision_error": central_decision_error,
            "probe_retrieval_success": probe_retrieval_success,
            "probe_gate_released": probe_gate_released,
            "just_entered_intersection": just_entered_intersection,
            "crossing_subphase": crossing_subphase if crossing_active else "none",
            "crossing_wait_remaining": crossing_wait_remaining,
            "crossing_traverse_remaining": crossing_traverse_remaining,
            "sim_time": round(sim_time, 4),
            "next_action": next_action,
            "action_source": action_source,
            "actr_pressed_key": actr_step_pressed_key or "none",
            "actr_commit_rule": actr_commit_rule or "none",
            "actr_rules_fired_count": len(actr_rules_fired_this_step),
            "actr_rules_fired": "|".join(actr_rules_fired_this_step),
            "actr_selected_production": actr_selected_production or "none",
            "step_outcome_label": step_outcome_label,
            "tick_signal_risk_band": risk_band_now,
            "tick_signal_iw_high": "yes" if iw_high_now else "no",
            "tick_signal_prev_action": prev_action_label,
            "tick_signal_vehicle_or_obstacle": (
                "yes" if vehicle_or_obstacle_now else "no"
            ),
            "tick_signal_reference_now": "yes" if any_reference_present else "no",
            "route_mode": route_mode,
            "looming_boost": round(looming_boost, 4),
            "step_travel_m": round(step_travel_m, 3),
            "step_len_m": round(AVG_STEP_METERS, 3),
            "edge_from": str(edge_from_node) if edge_from_node is not None else "none",
            "edge_to": str(edge_to_node) if edge_to_node is not None else "none",
            "edge_remaining_m": round(max(0.0, edge_length_m - edge_progress_m), 3),
            "at_node": edge_to_node is None,
        }
        sim_log.append(step_record)

        if seev_gate_passed:
            event_log.append(
                {
                    "step": step + 1,
                    "type": "gate_passed",
                    "detail": f"net_priority={net_priority:.3f}, threshold={priority_threshold_eff:.3f}",
                }
            )
        if landmark_triggered:
            event_log.append(
                {
                    "step": step + 1,
                    "type": "landmark_trigger",
                    "detail": matched_landmark_name,
                }
            )
        if matched_landmark_name != "none":
            event_log.append(
                {
                    "step": step + 1,
                    "type": "landmark_match",
                    "detail": matched_landmark_name,
                }
            )
        if actr_iw_total >= ACTR_IW_HIGH_THRESHOLD:
            event_log.append(
                {
                    "step": step + 1,
                    "type": "actr_iw_high",
                    "detail": f"IW={actr_iw_total:.3f}, Wave={actr_wave:.3f}",
                }
            )
        if just_entered_intersection:
            event_log.append(
                {
                    "step": step + 1,
                    "type": "intersection_enter",
                    "detail": f"IW={actr_iw_total:.3f}, Wave={actr_wave:.3f}, dep_idx={deprivation_index:.3f}, absent_steps={guidance_absent_steps}(continues)",
                }
            )
        if load_anchor_released:
            _anchor_detail = f"lm={matched_landmark_name}, surface={current_surface_type}, IW={actr_iw_total:.3f}, Wave={actr_wave:.3f}"
            event_log.append(
                {
                    "step": step + 1,
                    "type": "load_anchor_release",
                    "detail": _anchor_detail,
                }
            )
        if next_action == "stop_and_probe":
            event_log.append(
                {
                    "step": step + 1,
                    "type": "stop_probe",
                    "detail": f"risk={actr_risk_signal:.3f}, IW={actr_iw_total:.3f}, Wave={actr_wave:.3f}",
                }
            )
        prev_planned_direct_next = (
            str(planned_direct_next) if planned_direct_next is not None else None
        )
        if current_position == goal_node:
            print("已到达目标节点！")
            break

    print("\n=== 模拟结束 ===")
    print(f"总步数: {steps}")
    print(f"最终 Goal: {goal.copy()}")
    print(f"最终 Imaginal (Load): {imaginal.copy()}")

    return generate_report(
        sim_log=sim_log,
        event_log=event_log,
        profile=profile,
        steps=steps,
        start_node=start_node,
        goal_node=goal_node,
        current_position=current_position,
        max_steps=dynamic_max_steps,
        graph=graph,
        initial_production_utilities=initial_utility_snapshot,
    )
