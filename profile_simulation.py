"""
Profiling script for the ABM simulation.

Runs three measurements:
  1. Correctness baseline — fixed seed, saves scalar outputs to JSON
  2. cProfile of a single full simulation — identifies per-step hotspots
  3. Parallelism scaling — times N_RUNS runs at n_jobs = 1, 2, 4, 8, 16
     to quantify joblib overhead vs. compute

Usage (from repo root):
    python profile_simulation.py
"""

import cProfile, pstats, io, json, time
import numpy as np
from joblib import Parallel, delayed
from package.resources.run import generate_data
from copy import deepcopy

# ---------------------------------------------------------------------------
# Load params — surrogate constants give a clean, self-contained param set
# ---------------------------------------------------------------------------
with open("package/surrogate/constants/base_params_surrogate.json") as f:
    base_params = json.load(f)

# Shorten duration — enough steps for representative hotspot data
# Real durations: burn_in=180, calibration=276, future=144 (total 600 steps)
base_params["duration_burn_in"]     = 60   # 5 years
base_params["duration_calibration"] = 96   # 8 years
base_params["duration_future"]      = 44   # ~4 years  (total = 200 steps)

# No timeseries saving — isolates the core simulation compute
base_params["save_timeseries_data_state"] = 0
base_params["compression_factor_state"] = 1

def make_params(seed):
    p = deepcopy(base_params)
    p["parameters_social_network"]["random_seed"] = seed
    p["parameters_social_network"]["seed_inputs"]  = seed
    return p

# ============================================================
# STEP 1: Correctness baseline (fixed seed)
# ============================================================
print("=" * 60)
print("STEP 1: Correctness baseline (fixed seed=42)")
print("=" * 60)

t0 = time.perf_counter()
ctrl = generate_data(make_params(42))
baseline_time = time.perf_counter() - t0

baseline = {
    "ev_uptake":  float(ctrl.calc_EV_prop()),
    "utility":    float(ctrl.social_network.utility_cumulative),
    "emissions":  float(ctrl.social_network.emissions_cumulative),
    "net_cost":   float(ctrl.calc_net_policy_distortion()),
    "wall_time_s": round(baseline_time, 3),
}
for k, v in baseline.items():
    print(f"  {k:<12}: {v}")

with open("scratchpad_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)
print(f"\nBaseline saved → scratchpad_baseline.json")


# ============================================================
# STEP 2: cProfile (single run, no timeseries)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: cProfile — single simulation")
print("=" * 60)

pr = cProfile.Profile()
pr.enable()
generate_data(make_params(0))
pr.disable()

for sort_key, label in [("cumulative", "CUMULATIVE TIME"), ("tottime", "SELF TIME (hottest lines)")]:
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats(sort_key).print_stats(35)
    text = buf.getvalue()
    print(f"\n--- TOP 35 BY {label} ---")
    print(text)

# Save full profile
with open("scratchpad_profile.txt", "w") as f:
    for sort_key, label in [("cumulative", "CUMULATIVE"), ("tottime", "SELF TIME")]:
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats(sort_key).print_stats(60)
        f.write(f"\n\n=== TOP 60 BY {label} ===\n")
        f.write(buf.getvalue())
print("Full profile saved → scratchpad_profile.txt")


# ============================================================
# STEP 3: Parallelism scaling (16 seeds, loky vs multiprocessing)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Parallelism scaling (16 seeds)")
print("=" * 60)

N_RUNS = 2
params_list = [make_params(s) for s in range(N_RUNS)]

print(f"  Sequential per-run time: {baseline_time:.1f}s  "
      f"→ ideal {N_RUNS}-run total: {baseline_time * N_RUNS:.0f}s")
print()

scaling = {}
for backend in ["loky"]:
    scaling[backend] = {}
    for n_jobs in [1, 4]:
        t0 = time.perf_counter()
        Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
            delayed(generate_data)(p) for p in params_list
        )
        elapsed = time.perf_counter() - t0
        speedup    = (baseline_time * N_RUNS) / elapsed
        efficiency = speedup / n_jobs * 100
        scaling[backend][n_jobs] = {"wall_s": round(elapsed, 2), "speedup": round(speedup, 2), "efficiency_pct": round(efficiency, 1)}
        print(f"  {backend:<16} n_jobs={n_jobs:2d}: {elapsed:5.1f}s  "
              f"speedup={speedup:.2f}x  efficiency={efficiency:.0f}%")
    print()

with open("scratchpad_scaling.json", "w") as f:
    json.dump(scaling, f, indent=2)
print("Scaling results saved → scratchpad_scaling.json")
