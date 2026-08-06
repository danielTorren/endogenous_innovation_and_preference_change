"""
What the homophily parameters actually DO to the network.

Report realised assortativity, never the input parameters alone. The two are not
the same number, for a reason that is structural rather than incidental: the
Watts-Strogatz rewiring probability (SW_prob_rewire, 0.1) is applied AFTER
placement, so roughly 10% of every agent's edges stay uniformly random long-range
bridges however high homophily_strength goes. Homophily therefore SATURATES, and
the ceiling is set by prob_rewire, not by the parameter bounds.

Cheap: builds the population and the ring but does not simulate, so it runs in
seconds and can be swept over a grid.

Run:
    python -m package.validation.homophily_diagnostics
"""

import argparse
import json
import numpy as np

from package.model.synthetic_population import SyntheticPopulation

BASE_PARAMS = "package/constants/base_params_NN_zip.json"


def sweep(base_params_path=BASE_PARAMS, seed_population=22,
          h_values=(0.0, 0.25, 0.5, 0.75, 1.0),
          w_values=(0.0, 0.5, 1.0),
          verbose=True):
    with open(base_params_path) as f:
        bp = json.load(f)
    sp_params = bp["parameters_synthetic_population"]
    psn = bp["parameters_social_network"]
    n = int(psn["num_individuals"])

    # Neighbours on ONE side of the ring, matching socialNetworkUsers:
    # SW_K = round((n-1) * SW_network_density), half of which lie either side.
    k_full = int(round((n - 1) * psn["SW_network_density"]))
    k_ring = max(1, k_full // 2)

    if verbose:
        print(f"agents {n},  SW_K {k_full} (so {k_ring} ring neighbours each side),  "
              f"prob_rewire {psn['SW_prob_rewire']}")
        print("assortativity of the RING, i.e. BEFORE rewiring. Rewiring pulls all of")
        print(f"these toward 0 by roughly the rewired fraction ({psn['SW_prob_rewire']:.0%}).\n")
        print(f"  {'h':>5s} {'w':>5s} | {'income':>8s} {'spatial':>8s} "
              f"{'mean dist km':>13s} {'same-zip %':>11s}")

    rows = []
    for h in h_values:
        for w in (w_values if h > 0 else (w_values[0],)):
            s = SyntheticPopulation(sp_params, n, seed_population,
                                    homophily_strength=h, homophily_spatial_weight=w)
            r = s.realised_homophily(k_ring=k_ring)
            r.update({"h": h, "w": w})
            rows.append(r)
            if verbose:
                print(f"  {h:5.2f} {w:5.2f} | {r['income_assortativity']:8.3f} "
                      f"{r['spatial_assortativity']:8.3f} "
                      f"{r.get('mean_pair_distance_km', np.nan):13.1f} "
                      f"{100 * r.get('same_zip_share', np.nan):11.1f}")

    if verbose:
        print("\n  Reading it:")
        print("   - h = 0 must give ~0 in both columns. That is the original random placement.")
        print("   - h is by construction the rank correlation between an agent's ring position")
        print("     and its similarity index, so it is comparable across w.")
        print("   - w = 0 moves the income column only; w = 1 moves the spatial column only.")
        print("     The income column stays near 0 at w = 1 even though rich zips cluster,")
        print("     because within-zip income spread (log-sd ~0.81) dwarfs the between-zip")
        print("     component (~0.45). That separation is what makes w identifiable.")
        print("   - if the two columns move together, the population's income and geography")
        print("     are too collinear to tell the channels apart, and w will not identify.")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-population", type=int, default=22)
    a = ap.parse_args()
    sweep(seed_population=a.seed_population)
