"""
Sweep parameters_social_network.a_chi and record HHI and fleet car age.

a_chi is the first shape parameter of the Beta distribution that chi_vec, agent
innovativeness, is drawn from (Controller.gen_chi). chi gates whether an agent
will consider an EV at all (consider_ev_vec), so raising a_chi shifts the whole
innovativeness distribution towards 1 and pulls EV consideration up with it. Its
route to the two targets reported here is indirect: more EV consideration moves
demand between the two technologies, which changes which firms sell and therefore
HHI, and changes replacement timing and therefore fleet age. Neither is a
mechanical consequence of a_chi the way fleet age is of delta, which is exactly
why it is worth measuring rather than reasoning about.

READ THIS BEFORE READING THE OUTPUT. This sweep is NOT clean, and it is not clean
in a way the delta and kappa sweeps next to it are.

gen_chi currently draws with RandomState.beta, a rejection sampler, so the number
of uniforms it consumes depends on (a_chi, b_chi). a_chi is the ONLY parameter in
this whole sweep family whose value therefore moves the POSITION of every later
draw on the pinned seed_inputs stream: zero_indices in gen_chi, WTP_E_vec in
gen_gamma, the incomes and the network permutation in gen_beta, then the firm
placement and both the ICE and EV NK landscapes. Each row of this sweep is thus a
different a_chi AND a different population AND a different technology landscape.

That confound does NOT average out over seeds. seed_repetitions varies `seed`,
while the landscape is pinned to seed_inputs, so all 64 seeds of a row share the
same reshuffled landscape.

_precheck below measures the size of it for the grid actually being run and prints
it above the results table. On the default grid at the calibrated b_chi it reports
what the paragraph above predicts: the sorted chi distribution barely moves
(correlation with the calibrated row stays above 0.99 on every value), while the
agent-by-agent correlation collapses from 1 to roughly 0, and all 10 rows leave the
stream at a different position, so all 10 draw a different landscape.

Consequence for interpretation: treat the response curve as the response to
"a_chi and its landscape", not to a_chi. A non-monotonic row-to-row jump is the
expected signature of the landscape moving, not evidence of structure in a_chi. If
you need the isolated a_chi response, switch gen_chi to the stream-stable inverse
CDF form quoted in its Step 1 comment first, and do not compare the result with
runs made on this path -- at matched a_chi/b_chi/delta the 2023 EV stock differs by
roughly an order of magnitude between the two, because seed_inputs = 22 then draws
a different landscape.

The range [0.8, 2] is the a_chi prior used by the SBI calibration
(package/calibration/sbi_seed_av_gen.py, recorded in Data/var_dict of its output),
which brackets the calibrated value in base_params_calibration.json. The wider
[0.8, 5] in NN_multi_round_calibration_multi_gen.py is the older prior; the
tighter one is used here because a 5-fold range of the shape parameter would spend
most of the grid on distributions no calibration would accept, and because a wider
grid buys nothing when each point is already on its own landscape.

Everything else -- what is run, what is saved, what is plotted -- lives in
package/generating_data/sweep_hhi_age_gen.py.

Usage:
    python -m package.generating_data.a_chi_sweep_gen
    python -m package.generating_data.a_chi_sweep_gen --seeds 4      # quick check
"""
import sys

import numpy as np

from package.generating_data.sweep_hhi_age_gen import (
    load_base_params,
    parse_seeds,
    run_and_plot,
)

PARAM_NAME = "a_chi"
SUBDICT = "parameters_social_network"
PARAM_LABEL = r"$a_\chi$"

# 10 values, endpoints included. The SBI calibration prior range.
A_CHI_LIST = np.linspace(0.8, 2.0, 10)

# Overrides seed_repetitions in the base JSON, so editing that file does not
# silently change the sweep. Independent of the worker count: the submit script's
# --cpus-per-task is what sets how many of the 640 runs are resident at once, and
# therefore peak memory. 64 workers in 32G is an OOM kill on this base params
# file, so do not raise the two together without checking `seff` first.
SEEDS = 64


def _precheck(base_params, values):
    """
    Reject an infeasible grid, and measure the stream confound the docstring warns
    about, before 640 runs are launched rather than after reading a puzzling plot.

    The Beta distribution needs both shapes strictly positive, so a grid touching
    zero is rejected outright rather than left to fail inside a worker.

    The rest reproduces gen_chi's draw exactly -- same RandomState(seed_inputs),
    same rejection sampler, same size -- and reports three columns per grid value:

      corr_rank  correlation of that row's chi_vec with the calibrated row's,
                 after sorting BOTH. This measures only how much the innovativeness
                 DISTRIBUTION moved, which over this grid is very little: it stays
                 above 0.99 everywhere. It is here as the contrast for the next
                 column, and it is the number one would naively expect to be the
                 whole story.
      corr_id    the same correlation WITHOUT sorting, so agent i against agent i.
                 This is the reshuffle, and it collapses to roughly zero for every
                 value more than a step away from the calibrated one. The
                 distribution is nearly unchanged while almost no individual keeps
                 their draw, and individuals are what the social network wires
                 together, so the population is effectively re-randomised at every
                 row.
      next_u     the very next uniform the stream yields once gen_chi is done with
                 it. This is the desync itself, in one number: WTP_E_vec, the
                 incomes, the network permutation, the firm placement and both NK
                 landscapes are all drawn after this point. A different next_u means
                 a different landscape. Expect a different value on every row.

    Nothing here can fix the confound. It is printed so the response curve is read
    with the size of it known. On the stream-stable inverse CDF form quoted in
    gen_chi's Step 1 comment, next_u would be identical on every row and corr_id
    would track corr_rank.
    """
    a_lo = float(np.min(values))
    if a_lo <= 0:
        raise ValueError(
            f"a_chi grid reaches {a_lo:.6g}, but the Beta distribution requires "
            f"a_chi > 0. Raise the bottom of the grid."
        )

    seed_inputs = base_params["seed_inputs"]
    psn = base_params["parameters_social_network"]
    b_chi, n = psn["b_chi"], psn["num_individuals"]
    # The calibrated a_chi is read from the JSON, NOT from base_params: run_sweep
    # hands the precheck a dict already built at values[0], so psn["a_chi"] here is
    # the bottom of the grid and using it would make the first row trivially
    # correlate 1.000 with itself and mislabel the grid bottom as calibrated.
    base_a = load_base_params()["parameters_social_network"]["a_chi"]

    def draw(a):
        """gen_chi's Step 1 draw, plus the first uniform left for everything after it."""
        rs = np.random.RandomState(seed_inputs)
        chi = rs.beta(a, b_chi, size=n)
        return chi, float(rs.random_sample())

    chi_base, _ = draw(base_a)

    print(f"a_chi grid {a_lo:.4g} to {float(np.max(values)):.4g} at b_chi = "
          f"{b_chi:.4g}, seed_inputs = {seed_inputs}, calibrated a_chi = {base_a:.4g}")
    print("stream confound, see _precheck docstring. corr_id near 0 with corr_rank "
          "near 1 means the same distribution over re-randomised individuals;")
    print("a different next_u on each row means each row has its own NK landscape.")
    print(f"{'a_chi':>10}{'corr_rank':>11}{'corr_id':>9}{'next_u':>9}")
    next_us = []
    for value in values:
        chi, next_u = draw(float(value))
        next_us.append(next_u)
        print(f"{float(value):>10.4g}"
              f"{np.corrcoef(np.sort(chi), np.sort(chi_base))[0, 1]:>11.3f}"
              f"{np.corrcoef(chi, chi_base)[0, 1]:>9.3f}{next_u:>9.4f}")
    print(f"{len(set(next_us))} of {len(next_us)} rows have a distinct next_u; a "
          f"stream-stable sampler would give 1\n")


if __name__ == "__main__":
    run_and_plot(PARAM_NAME, SUBDICT, A_CHI_LIST, PARAM_LABEL,
                 parse_seeds(sys.argv[1:], SEEDS), precheck=_precheck)
