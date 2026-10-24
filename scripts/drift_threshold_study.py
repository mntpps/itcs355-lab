"""Lab 4 — derive the drift threshold from this problem's own noise, not from a tutorial.

    python scripts/drift_threshold_study.py            # writes reports/lab4-threshold.json

The conventional PSI cut-offs (0.10 / 0.25) come from credit scoring: stable features,
huge volumes. Whether they fit here depends on two things this script measures:

  1. The noise floor. Draw many production-like windows with NO drift — readings from
     machines the model never trained on, against the training reference — and record
     the PSI each one produces. The alert threshold has to sit above that, by enough
     that a check every 15 minutes does not page anyone for noise.
  2. What the threshold buys. For each injected fault, how often does a window cross it,
     and how much does that fault actually change what the model tells the business
     (the share of readings flagged as about to fail)?

Production windows are modelled as rows sampled across the fleet (machines report
interleaved). A window drawn from only a handful of machines is far noisier, because
machines have persistent traits; that case is reported too, because it is what a
low-traffic hour looks like.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src import config, data, seeds
from src.drift_stats import ks_statistic, psi_from_profile, reference_profile

CHECKS_PER_MONTH = 4 * 24 * 30   # the scheduled detector runs every 15 minutes


def windows(pool: pd.DataFrame, n: int, count: int, rng, clustered: bool = False):
    machines = pool[data.GROUP].unique()
    by_machine = {m: g for m, g in pool.groupby(data.GROUP)} if clustered else None
    for _ in range(count):
        if clustered:
            k = max(1, n // data_readings_per_machine(pool))
            chosen = rng.choice(machines, size=k, replace=False)
            yield pd.concat([by_machine[m] for m in chosen]).head(n)
        else:
            yield pool.iloc[rng.integers(0, len(pool), size=n)]


def data_readings_per_machine(pool: pd.DataFrame) -> int:
    return int(pool.groupby(data.GROUP).size().median())


def max_psi(profiles: dict, window: pd.DataFrame) -> tuple[float, dict]:
    per = {f: psi_from_profile(profiles[f], window[f].to_numpy()) for f in data.FEATURES}
    return max(per.values()), per


def inject(window: pd.DataFrame, mode: str, magnitude: float, rng) -> pd.DataFrame:
    w = window.copy()
    if mode == "shift":
        w["temp_c"] = w["temp_c"] + magnitude
    elif mode == "scale":
        mean = w["temp_c"].mean()
        w["temp_c"] = mean + (w["temp_c"] - mean) * magnitude
    elif mode == "mix":
        machines = w[data.GROUP].unique()
        favoured = rng.choice(machines, size=max(1, len(machines) // 5), replace=False)
        weights = np.where(w[data.GROUP].isin(favoured), magnitude, 1.0)
        w = w.sample(n=len(w), replace=True, weights=weights, random_state=int(rng.integers(1 << 31)))
    return w


SENSORS = [f for f in data.FEATURES if f != "load_pct"]
FLEET_MIX = ["load_pct"]   # a per-machine trait: its PSI measures which machines report
FALSE_PAGES_PER_MONTH = 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, nargs="+", default=[250, 500, 1000])
    ap.add_argument("--chosen-window", type=int, default=500)
    ap.add_argument("--null-windows", type=int, default=30000)
    ap.add_argument("--fault-windows", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("reports/lab4-threshold.json"))
    args = ap.parse_args()

    cfg = config.load(strict=False)
    seed = seeds.set_all()
    rng = np.random.default_rng(seed)
    df = data.load_raw(cfg.raw_path)
    train_df, val_df, test_df = data.split(df, seed=seed)
    unseen = pd.concat([val_df, test_df], ignore_index=True)
    profiles = {f: reference_profile(train_df[f].to_numpy()) for f in data.FEATURES}
    # One false page a month, spread over every scheduled check.
    q = 1 - FALSE_PAGES_PER_MONTH / CHECKS_PER_MONTH

    report: dict = {"reference_rows": len(train_df), "unseen_rows": len(unseen),
                    "checks_per_month": CHECKS_PER_MONTH, "target_quantile": round(q, 6),
                    "noise_floor": {}}

    # 1. Noise floor per feature: PSI of windows of unseen machines, with NO drift.
    null_by_n = {}
    for n in args.window:
        sample = [w for w in windows(unseen, n, args.null_windows, rng)]
        per = {f: np.array([psi_from_profile(profiles[f], w[f].to_numpy()) for w in sample])
               for f in data.FEATURES}
        null_by_n[n] = per
        sensor_max = np.max([per[f] for f in SENSORS], axis=0)
        report["noise_floor"][f"n={n}"] = {
            **{f: {"median": round(float(np.median(v)), 4),
                   "p99.9": round(float(np.quantile(v, 0.999)), 4)} for f, v in per.items()},
            "sensor_max_at_target": round(float(np.quantile(sensor_max, q)), 4),
            "load_at_target": round(float(np.quantile(per["load_pct"], q)), 4),
        }
        print(f"noise n={n:<5} sensor-max@target {np.quantile(sensor_max, q):.4f}   "
              f"load_pct@target {np.quantile(per['load_pct'], q):.4f}   "
              + "  ".join(f"{f[:6]} p99.9 {np.quantile(v, .999):.3f}" for f, v in per.items()))
    for n in (250, 500):
        clustered = np.array([max(psi_from_profile(profiles[f], w[f].to_numpy()) for f in SENSORS)
                              for w in windows(unseen, n, 2000, rng, clustered=True)])
        report["noise_floor"][f"n={n} clustered sensor-max median"] = round(float(np.median(clustered)), 4)
        print(f"noise n={n} clustered (n/25 machines) sensor-max median {np.median(clustered):.3f}")

    n = args.chosen_window
    per = null_by_n[n]
    sensor_max = np.max([per[f] for f in SENSORS], axis=0)
    sensor_t = float(np.ceil(np.quantile(sensor_max, q) * 100) / 100)
    load_t = float(np.ceil(np.quantile(per["load_pct"], q) * 100) / 100)
    report["chosen"] = {"window": n, "sensor_threshold": sensor_t, "load_threshold": load_t}
    report["false_pages_per_month"] = {
        "sensor_threshold": round(float((sensor_max >= sensor_t).mean()) * CHECKS_PER_MONTH, 2),
        "load_threshold": round(float((per["load_pct"] >= load_t).mean()) * CHECKS_PER_MONTH, 2),
        "single_0.10_all_features": round(float((np.max([per[f] for f in data.FEATURES], axis=0)
                                                  >= 0.10).mean()) * CHECKS_PER_MONTH, 1),
        "single_0.25_all_features": round(float((np.max([per[f] for f in data.FEATURES], axis=0)
                                                  >= 0.25).mean()) * CHECKS_PER_MONTH, 1),
    }
    print(f"chosen: window {n}, sensors {sensor_t}, load_pct {load_t}; "
          f"false pages/month {report['false_pages_per_month']}")

    # 2. What the chosen rule catches, and what each fault does to the model's output.
    model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   random_state=seed, n_jobs=-1)
    model.fit(train_df[data.FEATURES], train_df[data.TARGET])
    base_proba = model.predict_proba(unseen[data.FEATURES])[:, 1]
    report["baseline"] = {"flagged_rate": round(float((base_proba >= 0.5).mean()), 4),
                          "mean_risk": round(float(base_proba.mean()), 4)}

    faults = [("shift", 2.0), ("shift", 3.0), ("shift", 4.0), ("shift", 6.0),
              ("scale", 1.3), ("scale", 1.5), ("mix", 5.0)]
    report["faults"] = []
    for mode, mag in faults:
        caught = caught_025 = 0
        temp_psi, temp_ks = [], []
        for w in windows(unseen, n, args.fault_windows, rng):
            w = inject(w, mode, mag, rng)
            scores = {f: psi_from_profile(profiles[f], w[f].to_numpy()) for f in data.FEATURES}
            temp_psi.append(scores["temp_c"])
            temp_ks.append(ks_statistic(train_df["temp_c"].to_numpy(), w["temp_c"].to_numpy()))
            caught += max(scores[f] for f in SENSORS) >= sensor_t or scores["load_pct"] >= load_t
            caught_025 += max(scores.values()) >= 0.25
        whole = inject(unseen, mode, mag, rng)
        proba = model.predict_proba(whole[data.FEATURES])[:, 1]
        row = {
            "mode": mode, "magnitude": mag,
            "temp_mean": round(float(whole["temp_c"].mean()), 2),
            "temp_psi_median": round(float(np.median(temp_psi)), 4),
            "temp_ks_median": round(float(np.median(temp_ks)), 4),
            "detect_rate_chosen": round(caught / args.fault_windows, 3),
            "detect_rate_0.25": round(caught_025 / args.fault_windows, 3),
            "flagged_rate": round(float((proba >= 0.5).mean()), 4),
            "mean_risk": round(float(proba.mean()), 4),
        }
        report["faults"].append(row)
        print(f"fault {mode:<5} {mag:<4} temp_mean {row['temp_mean']:.2f}  psi {row['temp_psi_median']:.3f}"
              f"  ks {row['temp_ks_median']:.3f}  caught {row['detect_rate_chosen']:.2f}"
              f" (vs 0.25: {row['detect_rate_0.25']:.2f})  flagged {row['flagged_rate']:.3f}"
              f"  mean risk {row['mean_risk']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
