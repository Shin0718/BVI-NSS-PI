"""Provide command-line entry points for simulation runs.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import json
import argparse
import csv
import random
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np

try:
    from .simulation import run_simulation
    from .config import REPORT_DIR
    from .profile import normalize_familiarity_level
except ImportError:
    import sys
    from pathlib import Path

    package_dir = Path(__file__).resolve().parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))

    from simulation import run_simulation
    from config import REPORT_DIR
    from profile import normalize_familiarity_level


def load_profile_config(config_path=None, profile_name=None, familiarity=None):
    """Handle load profile config behavior."""
    result = {"familiarity_level": 1}

    if config_path is None:
        config_path = Path(__file__).parent.parent / "profiles.json"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            if profile_name is None:
                profile_name = config_data.get("default_profile", "intermediate")

            if profile_name in config_data.get("profiles", {}):
                profile = config_data["profiles"][profile_name]
                result["familiarity_level"] = profile.get(
                    "familiarity_level", result["familiarity_level"]
                )
                print(f"[OK] 已从配置文件加载 profile: {profile_name}")
                if "description" in profile:
                    print(f"  描述: {profile['description']}")
        except Exception as e:
            print(f"[WARN] 配置文件加载失败，使用默认值: {e}")

    if familiarity is not None:
        result["familiarity_level"] = familiarity
        print(f"[OK] 用命令行参数覆盖 familiarity_level = {familiarity}")

    result["familiarity_level"] = normalize_familiarity_level(
        result["familiarity_level"]
    )
    print(
        f"[OK] 二分类熟悉度: {result['familiarity_level']} ({'熟悉' if result['familiarity_level'] == 1 else '不熟悉'})"
    )
    return result


def run(config_path=None, profile_name=None, familiarity=None):
    """Handle run behavior."""
    user_profile = load_profile_config(config_path, profile_name, familiarity)
    return run_simulation(
        familiarity_level=user_profile["familiarity_level"],
    )


def _safe_stdev(values):
    """Handle safe stdev behavior."""
    return stdev(values) if len(values) >= 2 else 0.0


def _summarize_run_row(run_index, seed, summary_json, sim_csv, summary):
    """Handle summarize run row behavior."""
    return {
        "run": int(run_index),
        "seed": int(seed),
        "summary_json": str(summary_json),
        "sim_csv": str(sim_csv),
        "total_steps": int(summary.get("result", {}).get("total_steps", 0)),
        "reached_goal": bool(summary.get("result", {}).get("reached_goal", False)),
        "stop_probe_count": int(summary.get("result", {}).get("stop_probe_count", 0)),
        "gate_passed_count": int(summary.get("result", {}).get("gate_passed_count", 0)),
        "landmark_trigger_count": int(
            summary.get("result", {}).get("landmark_trigger_count", 0)
        ),
        "landmark_active_step_count": int(
            summary.get("result", {}).get("landmark_active_step_count", 0)
        ),
        "actr_iw_mean": float(
            summary.get("statistics", {}).get("actr_iw", {}).get("mean", 0.0)
        ),
        "actr_wave_mean": float(
            summary.get("statistics", {}).get("actr_wave", {}).get("mean", 0.0)
        ),
        "risk_mean": float(
            summary.get("statistics", {}).get("risk", {}).get("mean", 0.0)
        ),
    }


def run_monte_carlo(
    config_path=None,
    profile_name=None,
    familiarity=None,
    mc_runs=50,
    seed_start=1000,
):
    """Handle run monte carlo behavior."""
    user_profile = load_profile_config(config_path, profile_name, familiarity)
    runs = max(1, int(mc_runs))

    print(f"[MC] 开始蒙特卡洛: runs={runs}, seed_start={seed_start}")
    print(f"[MC] profile={user_profile}")

    rows = []
    for run_idx in range(runs):
        seed = int(seed_start) + run_idx
        random.seed(seed)
        np.random.seed(seed)

        _, csv_path, json_path = run_simulation(
            familiarity_level=user_profile["familiarity_level"],
        )

        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        row = _summarize_run_row(
            run_index=run_idx + 1,
            seed=seed,
            summary_json=json_path,
            sim_csv=csv_path,
            summary=summary,
        )
        rows.append(row)

        if (run_idx + 1) % 10 == 0 or (run_idx + 1) == runs:
            print(f"[MC] 已完成 {run_idx + 1}/{runs}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(REPORT_DIR) / f"mc_runs_{ts}.csv"
    out_json = Path(REPORT_DIR) / f"mc_summary_{ts}.json"

    fieldnames = list(rows[0].keys()) if rows else ["run", "seed"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reached_values = [1.0 if r["reached_goal"] else 0.0 for r in rows] or [0.0]
    total_steps_vals = [r["total_steps"] for r in rows] or [0.0]
    stop_probe_vals = [r["stop_probe_count"] for r in rows] or [0.0]
    gate_vals = [r["gate_passed_count"] for r in rows] or [0.0]
    landmark_trigger_vals = [r["landmark_trigger_count"] for r in rows] or [0.0]
    landmark_active_step_vals = [r["landmark_active_step_count"] for r in rows] or [0.0]
    iw_vals = [r["actr_iw_mean"] for r in rows] or [0.0]
    wave_vals = [r["actr_wave_mean"] for r in rows] or [0.0]
    risk_vals = [r["risk_mean"] for r in rows] or [0.0]

    mc_summary = {
        "timestamp": ts,
        "runs": runs,
        "seed_start": seed_start,
        "profile": user_profile,
        "aggregate": {
            "goal_reach_rate": round(mean(reached_values), 4),
            "total_steps_mean": round(mean(total_steps_vals), 4),
            "total_steps_std": round(_safe_stdev(total_steps_vals), 4),
            "stop_probe_mean": round(mean(stop_probe_vals), 4),
            "stop_probe_std": round(_safe_stdev(stop_probe_vals), 4),
            "gate_passed_mean": round(mean(gate_vals), 4),
            "gate_passed_std": round(_safe_stdev(gate_vals), 4),
            "landmark_trigger_mean": round(mean(landmark_trigger_vals), 4),
            "landmark_trigger_std": round(_safe_stdev(landmark_trigger_vals), 4),
            "landmark_active_step_mean": round(mean(landmark_active_step_vals), 4),
            "landmark_active_step_std": round(
                _safe_stdev(landmark_active_step_vals), 4
            ),
            "actr_iw_mean": round(mean(iw_vals), 4),
            "actr_iw_std": round(_safe_stdev(iw_vals), 4),
            "actr_wave_mean": round(mean(wave_vals), 4),
            "actr_wave_std": round(_safe_stdev(wave_vals), 4),
            "risk_mean": round(mean(risk_vals), 4),
            "risk_std": round(_safe_stdev(risk_vals), 4),
        },
        "run_csv": str(out_csv),
        "run_details": rows,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mc_summary, f, ensure_ascii=False, indent=2)

    print("[MC] 蒙特卡洛完成")
    print(f"[MC] 逐轮明细: {out_csv}")
    print(f"[MC] 汇总结果: {out_json}")
    return str(out_csv), str(out_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BVI 态势感知仿真 - 支持命令行参数和配置文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py
  
  python main.py --profile 0    # 不熟悉
  python main.py --profile 1    # 熟悉
  
  python main.py --familiarity 0    # 不熟悉
  python main.py --familiarity 1    # 熟悉
  
  python main.py --config my_profiles.json --profile 1

  python main.py --profile 1 --mc-runs 100 --seed-start 20260424
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（默认: profiles.json）",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="预设 profile（0=不熟悉, 1=熟悉）",
    )
    parser.add_argument(
        "--familiarity",
        type=float,
        default=None,
        help="熟悉度二分类（0=不熟悉, 1=熟悉；优先级高于配置文件）",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=1,
        help="蒙特卡洛轮数（>1 时将执行多轮并生成汇总）",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000,
        help="蒙特卡洛起始随机种子（每轮 +1）",
    )

    args = parser.parse_args()

    try:
        if args.mc_runs and args.mc_runs > 1:
            run_monte_carlo(
                config_path=args.config,
                profile_name=args.profile,
                familiarity=args.familiarity,
                mc_runs=args.mc_runs,
                seed_start=args.seed_start,
            )
        else:
            run(
                config_path=args.config,
                profile_name=args.profile,
                familiarity=args.familiarity,
            )
    except KeyboardInterrupt:
        print("\n检测到中断信号（KeyboardInterrupt），仿真已安全停止。")
