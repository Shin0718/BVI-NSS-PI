"""Estimate environmental risk with Bayesian evidence fusion.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

RISK_STATES = ("low", "medium", "high")


EXPERT_CPT = {
    "transition": {
        "low": {"low": 0.48, "medium": 0.42, "high": 0.10},
        "medium": {"low": 0.18, "medium": 0.60, "high": 0.22},
        "high": {"low": 0.10, "medium": 0.45, "high": 0.45},
    },
    "traffic_sound": {"low": 0.48, "medium": 0.54, "high": 0.60},
    "vehicle_approach": {"low": 0.30, "medium": 0.46, "high": 0.62},
    "human_activity": {"low": 0.52, "medium": 0.50, "high": 0.35},
    "cane_hit": {"low": 0.28, "medium": 0.56, "high": 0.44},
    "cane_drop": {"low": 0.26, "medium": 0.32, "high": 0.42},
    "surface_change": {"low": 0.44, "medium": 0.48, "high": 0.52},
    "on_tactile_guidance": {"low": 0.58, "medium": 0.34, "high": 0.20},
    "at_intersection": {"low": 0.25, "medium": 0.45, "high": 0.62},
    "crossing_active": {"low": 0.24, "medium": 0.45, "high": 0.70},
    "sound_salience": {
        "low": {"low": 0.45, "medium": 0.38, "high": 0.17},
        "medium": {"low": 0.26, "medium": 0.48, "high": 0.26},
        "high": {"low": 0.12, "medium": 0.36, "high": 0.52},
    },
    "distance_feedback": {
        "near": {"low": 0.18, "medium": 0.42, "high": 0.40},
        "mid": {"low": 0.24, "medium": 0.50, "high": 0.26},
        "far": {"low": 0.46, "medium": 0.38, "high": 0.16},
    },
}


def build_dbn():
    """Handle build dbn behavior."""
    print("DBN config loaded successfully (PrevRisk + Evidence -> CurrentRisk)")
    return EXPERT_CPT


def _normalize(prob_dict):
    """Handle normalize behavior."""
    total = sum(prob_dict.values())
    if total <= 0:
        uniform = 1.0 / len(RISK_STATES)
        return {state: uniform for state in RISK_STATES}
    return {state: value / total for state, value in prob_dict.items()}


def _bucket_salience(value):
    """Handle bucket salience behavior."""
    if value < 0.33:
        return "low"
    if value < 0.66:
        return "medium"
    return "high"


def _bucket_distance(value):
    """Handle bucket distance behavior."""
    if value < 0.35:
        return "near"
    if value < 0.70:
        return "mid"
    return "far"


def posterior_to_label(posterior):
    """Handle posterior to label behavior."""
    return max(RISK_STATES, key=lambda state: posterior.get(state, 0.0))


def infer_risk_posterior(prev_risk_posterior, evidence, model=None):
    """Handle infer risk posterior behavior."""
    cpt = model or EXPERT_CPT
    prev = _normalize(
        {state: float(prev_risk_posterior.get(state, 0.0)) for state in RISK_STATES}
    )

    salience_bucket = _bucket_salience(float(evidence.get("sound_salience", 0.0)))
    distance_bucket = _bucket_distance(float(evidence.get("distance_feedback", 1.0)))

    unnormalized = {}
    for current_state in RISK_STATES:
        dynamic_prior = 0.0
        for previous_state in RISK_STATES:
            dynamic_prior += (
                prev[previous_state] * cpt["transition"][previous_state][current_state]
            )

        likelihood = 1.0
        for var_name in (
            "traffic_sound",
            "vehicle_approach",
            "human_activity",
            "cane_hit",
            "cane_drop",
            "surface_change",
            "on_tactile_guidance",
            "at_intersection",
            "crossing_active",
        ):
            prob_true = cpt[var_name][current_state]
            obs_raw = evidence.get(var_name, None)
            if obs_raw is None:
                continue
            if isinstance(obs_raw, float) and not isinstance(obs_raw, bool):
                p_obs = obs_raw
                likelihood *= p_obs * prob_true + (1.0 - p_obs) * (1.0 - prob_true)
            else:
                observed = bool(obs_raw)
                likelihood *= prob_true if observed else (1.0 - prob_true)

        likelihood *= cpt["sound_salience"][salience_bucket][current_state]
        likelihood *= cpt["distance_feedback"][distance_bucket][current_state]

        unnormalized[current_state] = max(1e-12, dynamic_prior * likelihood)

    return _normalize(unnormalized)
