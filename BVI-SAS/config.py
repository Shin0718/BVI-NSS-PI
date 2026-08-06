"""Define simulation constants, environment priors, and cognitive workload parameters.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

MAX_STEPS = 200
AVG_STEP_METERS = 0.48
BVI_WALKING_SPEED = 0.558

CROSSING_WAIT_STEPS_MIN = 0
CROSSING_WAIT_STEPS_MAX = 25
CROSSING_TRAVERSE_STEPS_MIN = 5
CROSSING_TRAVERSE_STEPS_MAX = 20

SURFACE_SEGMENT_MIN_STEPS = 3
SURFACE_SEGMENT_MAX_STEPS = 12

TACTILE_SEGMENT_MIN_STEPS = 8
TACTILE_SEGMENT_MAX_STEPS = 20

CANE_GUIDANCE_MIN_STEPS = 2
CANE_GUIDANCE_MAX_STEPS = 5

SURFACE_PROFILES = {
    "carriageway": {
        "cn": "马路/车行道",
        "texture": "smooth_road",
        "wm_modifier": 1.20,
        "drift_rate": 0.018,
    },
    "flat_road": {
        "cn": "平整路面",
        "texture": "smooth",
        "wm_modifier": 1.00,
        "drift_rate": 0.010,
    },
    "uneven_natural": {
        "cn": "不平整自然路面",
        "texture": "irregular",
        "wm_modifier": 1.65,
        "drift_rate": 0.040,
    },
    "slope_surface": {
        "cn": "坡度路面",
        "texture": "ramp",
        "wm_modifier": 1.80,
        "drift_rate": 0.050,
    },
    "height_drop": {
        "cn": "高度落差路面",
        "texture": "steps",
        "wm_modifier": 2.20,
        "drift_rate": 0.080,
    },
    "tactile_guidance": {
        "cn": "提示路面",
        "texture": "tactile",
        "wm_modifier": 1.15,
        "drift_rate": 0.012,
    },
}

SURFACE_PROBABILITY_DISTRIBUTION = {
    "flat_road": 0.6571,
    "uneven_natural": 0.0006,
    "slope_surface": 0.0026,
    "height_drop": 0.0068,
    "tactile_guidance": 0.3329,
}

CANE_OBSTACLE_PROB = 0.0020
CANE_CURB_PROB = 0.0311
CANE_WALL_PROB = 0.0058
CANE_RAILING_PROB = 0.0043

SOUND_HORN_PROB = 0.0047
SOUND_VEHICLE_APPROACH_PROB = 0.0010
SOUND_REVERSE_BEEP_PROB = 0.0003
HUMAN_ACTIVITY_PROB = 0.0108
SOUND_VEHICLE_APPROACH_CROSSING_PROB = 0.0231
CROSSING_HORN_PROB = 0.0031
CROSSING_REVERSE_BEEP_PROB = 0.0036
CROSSING_HUMAN_ACTIVITY_PROB = 0.0715

VEHICLE_APPROACH_MIN_STEPS = 2
VEHICLE_APPROACH_MAX_STEPS = 5
VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK = 0.40


DBN_CROSSING_VEHICLE_ABSENT_SOFT_EVIDENCE = 0.45
DBN_CROSSING_TRAFFIC_ABSENT_SOFT_EVIDENCE = 0.25
DBN_CROSSING_CANE_HIT_SOFT_EVIDENCE = 0.15
DBN_NEUTRAL_DISTANCE_FEEDBACK = 0.52

SEEV_VALUE_SAFETY_WEIGHT = 0.70
SEEV_VALUE_PROGRESS_WEIGHT = 0.30
SEEV_EXPECTANCY_RISK_WEIGHT = 0.55
ACTR_RISK_CANE_GUIDANCE_RELIEF = 0.10
ACTR_RISK_LANDMARK_RELIEF = 0.08

SEEV_GATE_ADAPTIVE_THRESHOLD_ENABLED = True
SEEV_GATE_THRESHOLD_WINDOW_STEPS = 100
SEEV_GATE_THRESHOLD_QUANTILE = 0.80
SEEV_GATE_THRESHOLD_MIN_HISTORY = 20
ATTENTION_UNGATED_ENTRY_COEF = 0.5
ATTENTION_GATED_CENTRAL_DANGER_BOOST = 0.20
ATTENTION_GATED_RELIEF_ENABLED = True

LOOMING_RESUME_GATE_ENABLED = True
LOOMING_RESUME_THRESHOLD = 0.10

ACTR_DYNAMIC_PM_WEIGHTS_ENABLED = True
ACTR_DYNAMIC_PM_TEMP = 1.0
ACTR_DYNAMIC_PM_SMOOTHING = 0.5
ACTR_DYNAMIC_PM_MIN_SHARE = 0.12

ACTR_NAV_ANNOUNCEMENT_ENABLED = True
