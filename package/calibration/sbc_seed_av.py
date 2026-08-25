"""
Simulation-based calibration (SBC) for an sbi_seed_av NPE run.

WHAT SBC CHECKS
---------------
If the trained posterior q(theta | x) equals the true posterior p(theta | x),
then for a pair drawn from the joint,

    theta* ~ prior,   x ~ simulator(theta*),   theta_1..theta_L ~ q(. | x),

the rank of theta* among the L posterior draws is UNIFORM on {0, ..., L}. That is
a self-consistency identity of the joint, not an approximation, so any departure
from uniformity is evidence that q is wrong. This is Talts et al. (2018); sbi
implements the machinery in sbi.diagnostics.

The SHAPE of the rank histogram says WHICH way it is wrong, which is the question
actually being asked here:

    flat            q is consistent with the true posterior (no evidence against)
    U-shaped        q is TOO NARROW. theta* keeps landing in the tails of q, so
                    ranks pile up at both ends. Overconfident: stated credible
                    intervals under-cover.
    inverted-U      q is TOO WIDE. theta* keeps landing near the middle of q.
                    Conservative: intervals over-cover. Less dangerous, still wrong.
    tilted / skewed q is BIASED. Ranks piled HIGH mean theta* usually sits above
                    the posterior mass, i.e. q underestimates the parameter; piled
                    LOW means q overestimates it.
    spikes at 0 / L a hard failure: theta* outside q's support.

Two summaries capture the first three, and both are reported with their sampling
error, so "looks a bit sloped" becomes a number:

    mean of normalised ranks   0.5   under H0  ->  tilt / bias
    var  of normalised ranks   1/12  under H0  ->  U vs inverted-U

WHAT SBC DOES NOT CHECK
-----------------------
1. SBC is a GLOBAL check: it averages over the prior predictive. A posterior can
   pass SBC and still be wrong at the one x that matters, x_o. LC2ST (--lc2st) is
   the local counterpart: it asks whether q(. | x_o) itself is calibrated. Run
   both.
2. SBC checks the posterior against THE SIMULATOR IT WAS TRAINED ON. Here that is
   the seed-AVERAGED, logit-scaled simulator: K = num_seeds_per_theta ABM runs
   collapsed by build_x(). SBC cannot say whether averaging K seeds was the right
   thing to do, only whether NPE inverted that averaged simulator correctly. It
   also cannot see model misspecification: x_o coming from the real world rather
   than from the ABM is outside its scope entirely.
3. SBC says nothing about parameters that were PINNED rather than fitted (kappa,
   and anything else absent from var_dict). The whole diagnostic is conditional
   on those being right.

IN-SAMPLE VS HELD-OUT PAIRS
---------------------------
SBC needs pairs from the joint: theta ~ prior, x ~ simulator(theta). A
num_rounds=1 run has exactly those sitting in inference.pkl already, because they
are its training set, so source="training" runs SBC for free.

That is IN-SAMPLE. The posterior was fitted to those pairs, so the check is
optimistic and a pass is weaker evidence than a held-out pass. It is still
informative: a flexible density estimator trained on a few hundred points can and
does fail SBC in-sample when the fit is genuinely bad. Read a FAIL here as
conclusive and a PASS as "no evidence of miscalibration, from the easy version of
the test".

source="heldout" uses fresh pairs written by
package.calibration.sbc_seed_av_gen, which costs num_thetas x num_seeds ABM runs
(a cluster job: submit_sbc_seed_av.slurm). Prefer it when the budget exists.

NOTE ON ROUNDS. SBC as implemented here is valid only for an AMORTISED posterior,
i.e. num_rounds == 1, because it evaluates q at many different x. A multi-round
run's final posterior is trained mostly on thetas drawn near x_o and is not
claimed to be correct anywhere else, so uniform ranks are not expected and
non-uniform ranks would not be a defect. main() refuses such a run.

USAGE
-----
    python -m package.calibration.sbc_seed_av                    # in-sample
    python -m package.calibration.sbc_seed_av --lc2st            # + local check at x_o
    python -m package.calibration.sbc_seed_av --source heldout   # after the slurm job

Writes <run>/Data/sbc_<source>.pkl and <run>/Plots/sbc_*.png.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

from sbi.diagnostics import run_sbc, check_sbc, run_tarp, check_tarp
from sbi.diagnostics.lc2st import LC2ST
from sbi.analysis.plot import sbc_rank_plot

from package.resources.utility import load_object, save_object, createFolder


DEFAULT_RUN = "results/sbi_seed_av_15_31_53__18_08_2026"

# Posterior draws per SBC dataset. L sets the RESOLUTION of the rank grid,
# {0, ..., L}, not the power of the test: power comes from the NUMBER of pairs.
# 1000 is sbi's default and is far finer than a few hundred pairs can resolve; it
# is kept only so the normalised ranks are effectively continuous and the KS test
# needs no tie correction.
DEFAULT_NUM_POSTERIOR_SAMPLES = 1000

# Bins for the chi-square test of the rank histogram. Talts et al. suggest of
# order N/20 expected counts per bin; at N = 512 pairs, 20 bins gives 25.6
# expected per bin, which keeps the chi-square approximation honest.
DEFAULT_NUM_BINS = 20

# Two-sided z beyond which a summary is called a departure rather than noise.
Z_ALERT = 2.0

# Draws per x used to measure prior leakage. Cheap: this is one unrejected pass
# through the flow, not the SBC sampling itself.
LEAKAGE_PROBE_SAMPLES = 2000


def load_run(fileName):
    """Load everything this diagnostic needs from a finished sbi_seed_av run."""
    D = fileName + "/Data"
    run = {
        "posterior": load_object(D, "posterior"),
        "prior": load_object(D, "prior"),
        "var_dict": load_object(D, "var_dict"),
        "x_o": load_object(D, "x_o"),
        "run_config": load_object(D, "run_config"),
    }
    run["param_names"] = [p["name"] for p in run["var_dict"]]
    return run


def training_pairs(fileName):
    """
    The round-0 (theta, x) pairs out of inference.pkl.

    Round 0 and only round 0: those thetas came from the prior, which is what SBC
    requires. Later rounds draw from the posterior at x_o and are not joint
    samples, so including them would break the rank identity outright. Hence the
    round-index check rather than a torch.cat over all rounds.
    """
    inference = load_object(fileName + "/Data", "inference")

    round_index = list(inference._data_round_index)
    if round_index[0] != 0:
        raise RuntimeError(
            f"first stored round has index {round_index[0]}, not 0; its thetas did "
            "not come from the prior and cannot be used for SBC"
        )
    if len(round_index) > 1:
        print(f"  NOTE: run has {len(round_index)} rounds; using round 0 only "
              "(the prior-drawn pairs).")

    return inference._theta_roundwise[0], inference._x_roundwise[0]


def heldout_pairs(fileName):
    """Fresh joint samples written by package.calibration.sbc_seed_av_gen."""
    pairs = load_object(fileName + "/Data", "sbc_pairs")
    print(f"  held-out pairs: {pairs['theta'].shape[0]}, "
          f"K={pairs['num_seeds_per_theta']}, master_seed={pairs['master_seed']}")
    return pairs["theta"], pairs["x"]


def leakage_report(posterior, prior, x, x_o, num_probe=LEAKAGE_PROBE_SAMPLES):
    """
    How much of the flow's mass falls OUTSIDE the prior box, per x.

    NPE fits an unconstrained normalising flow, so nothing stops it putting mass
    beyond the BoxUniform bounds. DirectPosterior.sample() removes that by
    rejection, which is why the posterior actually used for inference is correct
    -- but the acceptance rate is worth knowing for two separate reasons.

    DIAGNOSTIC. A low acceptance at some x means the flow is badly shaped there:
    it is spending most of its density on parameter values the prior rules out.
    Read alongside the SBC ranks, not instead of them. Acceptance AT x_o is the
    one that matters for the published posterior.

    PRACTICAL. It is the reason this script does not use batched sampling; see
    the note at the run_sbc call.
    """
    low, high = prior.base_dist.low, prior.base_dist.high
    with torch.no_grad():
        s = posterior.sample_batched(
            (num_probe,), x=x, reject_outside_prior=False, show_progress_bars=False
        )
        s_o = posterior.posterior_estimator.sample(
            (num_probe,), condition=x_o.reshape(1, -1)
        ).reshape(-1, low.numel())

    inside = ((s >= low) & (s <= high)).all(-1).float().mean(0).numpy()
    inside_o = float(((s_o >= low) & (s_o <= high)).all(-1).float().mean())

    print("\n  prior-support acceptance of the flow (1.0 = no leakage)")
    print(f"    over the SBC x: min {inside.min():.4f}  median "
          f"{np.median(inside):.4f}  mean {inside.mean():.4f}")
    print(f"    x with acceptance < 0.10: {int((inside < 0.10).sum())} of {inside.size}")
    print(f"    AT x_o: {inside_o:.4f}   <- the one the published posterior uses")
    return {"per_x": inside, "at_x_o": inside_o, "num_probe": num_probe}


def rank_shape_stats(ranks, num_posterior_samples, param_names,
                     num_bins=DEFAULT_NUM_BINS):
    """
    Turn one rank column per parameter into a verdict on that histogram's SHAPE.

    Everything here is a statement about normalised ranks r = rank / L, which are
    Uniform(0, 1) under H0 up to the discreteness of the {0, ..., L} grid. The
    discrete correction is carried explicitly (var0 below) rather than assumed
    negligible, so the numbers stay right if L is dropped to 100.

    Returned per parameter:
        mean, z_mean    tilt. z_mean > 0 means ranks skew HIGH, so theta_true
                        tends to sit above the posterior mass: q is biased LOW.
        var, z_var      spread. z_var > 0 is U-shaped, q TOO NARROW.
                        z_var < 0 is inverted-U, q TOO WIDE.
        ks_pval         Kolmogorov-Smirnov against Uniform(0, 1). Sensitive to
                        smooth departures such as a tilt, weak against symmetric
                        ones.
        chi2_pval       chi-square on the binned histogram. The complement: it
                        sees a U-shape, which KS can miss almost entirely because
                        a symmetric U has nearly the uniform CDF.
        coverage        empirical coverage of central credible intervals, which is
                        the tilt/spread story in the units a reader cares about.
        shape           a label, built from z_mean and z_var only.
    """
    L = num_posterior_samples
    ranks = np.asarray(ranks, dtype=np.float64)
    N = ranks.shape[0]

    # Uniform on {0..L} mapped into [0,1] by /L: variance ((L+1)^2 - 1)/12 / L^2.
    var0 = ((L + 1.0) ** 2 - 1.0) / 12.0 / L ** 2
    se_mean = np.sqrt(var0 / N)
    # SE of the sample variance of a Uniform(0,1) sample: sqrt((mu4 - sigma^4)/N)
    # with mu4 = 1/80 and sigma^4 = 1/144, i.e. sqrt(1 / (180 N)).
    se_var = np.sqrt(1.0 / (180.0 * N))

    out = []
    for j, name in enumerate(param_names):
        r = ranks[:, j] / L

        mean, var = r.mean(), r.var(ddof=1)
        z_mean = (mean - 0.5) / se_mean
        z_var = (var - var0) / se_var

        ks_pval = float(stats.kstest(r, "uniform").pvalue)

        counts, _ = np.histogram(r, bins=num_bins, range=(0.0, 1.0))
        chi2_pval = float(stats.chisquare(counts).pvalue)

        cov = {}
        for level in (0.50, 0.90, 0.95):
            lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
            cov[level] = float(((r >= lo) & (r <= hi)).mean())

        bits = []
        if abs(z_mean) > Z_ALERT:
            bits.append("biased LOW (ranks skew high)" if z_mean > 0
                        else "biased HIGH (ranks skew low)")
        if z_var > Z_ALERT:
            bits.append("TOO NARROW (U-shaped, overconfident)")
        elif z_var < -Z_ALERT:
            bits.append("TOO WIDE (inverted-U, conservative)")
        shape = " + ".join(bits) if bits else "flat (consistent with uniform)"

        out.append({
            "name": name, "mean": float(mean), "z_mean": float(z_mean),
            "var": float(var), "z_var": float(z_var), "var_h0": float(var0),
            "ks_pval": ks_pval, "chi2_pval": chi2_pval,
            "coverage": cov, "shape": shape, "counts": counts,
        })
    return out


def print_shape_table(stats_list):
    print("\n  rank-histogram shape (normalised ranks r = rank / L)")
    print("    param        mean    z      var      z      KS p   chi2 p  verdict")
    for s in stats_list:
        print(f"    {s['name']:<10} {s['mean']:.3f}  {s['z_mean']:+5.1f}  "
              f"{s['var']:.4f}  {s['z_var']:+5.1f}  {s['ks_pval']:.3f}  "
              f"{s['chi2_pval']:.3f}  {s['shape']}")
    print(f"    (H0: mean 0.500, var {stats_list[0]['var_h0']:.4f}; "
          f"|z| > {Z_ALERT:.0f} is a departure)")

    print("\n  empirical coverage of central credible intervals (nominal -> actual)")
    for s in stats_list:
        c = s["coverage"]
        print(f"    {s['name']:<10} 50% -> {c[0.50]:.3f}   90% -> {c[0.90]:.3f}   "
              f"95% -> {c[0.95]:.3f}")


def plot_ranks(ranks, num_posterior_samples, param_names, fileName, tag, num_bins):
    """
    Both standard views, because they fail in different directions.

    The CDF view carries SIMULTANEOUS confidence bands: a curve leaving the band
    anywhere is a rejection at the stated level, with no multiple-comparison fudge
    over bins. That is the one for the paper. The histogram is the one to read the
    SHAPE off, since U vs inverted-U vs tilt is obvious there and subtle in a CDF.
    """
    createFolder(fileName)
    paths = []
    for plot_type in ("cdf", "hist"):
        fig, _ = sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type=plot_type,
            num_bins=num_bins if plot_type == "hist" else None,
            parameter_labels=param_names,
        )
        fig.suptitle(f"SBC rank {plot_type} -- {tag}")
        path = f"{fileName}/Plots/sbc_rank_{plot_type}_{tag}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def tarp_check(theta, x, posterior, num_posterior_samples, fileName, tag):
    """
    TARP: the JOINT counterpart to SBC's per-parameter marginal ranks.

    Marginal SBC is blind to a posterior that gets every marginal right and the
    correlation between them wrong. With a_chi and b_chi pushing the stock level
    in opposite directions along one line through the prior box (see the gen
    script's docstring), a mis-estimated correlation is exactly the failure mode
    to worry about here. TARP (Lemos et al. 2023) measures expected coverage of
    distance-to-a-random-reference regions in the full theta space, so it sees it.

    Reads as: ecp against alpha should sit on the diagonal. Above the diagonal is
    conservative (too wide), below is overconfident (too narrow), the same
    directions as SBC's variance but now for the joint.
    """
    ecp, alpha = run_tarp(
        theta, x, posterior,
        num_posterior_samples=num_posterior_samples,
        use_batched_sampling=False,
        show_progress_bar=False,
    )
    atc, ks_pval = check_tarp(ecp, alpha)
    print(f"\n  TARP (joint coverage): atc = {atc:+.3f}, KS p = {ks_pval:.3f}")
    print("    atc near 0 with p > 0.05 is a pass; atc < 0 overconfident, "
          "atc > 0 conservative")

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", c="k", lw=1, label="ideal")
    ax.plot(np.asarray(alpha), np.asarray(ecp), lw=1.8, label="TARP")
    ax.set_xlabel("credibility level")
    ax.set_ylabel("expected coverage probability")
    ax.set_title(f"TARP -- {tag}\natc={atc:+.3f}, KS p={ks_pval:.3f}")
    ax.legend()
    path = f"{fileName}/Plots/sbc_tarp_{tag}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"ecp": ecp, "alpha": alpha, "atc": float(atc),
            "ks_pval": float(ks_pval), "path": path}


def lc2st_check(theta, x, posterior, x_o, num_trials_null, seed):
    """
    LOCAL calibration at x_o, which is the thing the paper actually leans on.

    SBC's verdict is an average over the whole prior predictive. It can be flat
    while q(. | x_o) is badly wrong, if x_o sits somewhere the estimator was
    poorly constrained. x_o is a single real-world observation, not a draw from
    the prior predictive, so there is no reason to assume it is typical.

    LC2ST (Linhart et al. 2023) trains a classifier to separate (theta ~ q(. | x),
    x) from joint samples (theta, x), then reads its output AT x_o. Chance-level
    discrimination there, statistic near 0 and p large, means q is locally
    calibrated. The null distribution comes from permutation, hence
    num_trials_null classifier fits: that is the slow part.

    One posterior draw per pair is not a typo. LC2ST needs exactly one draw per
    (theta, x) so that each class is a sample from a joint distribution.
    """
    with torch.no_grad():
        post_samples = posterior.sample_batched(
            (1,), x=x, show_progress_bars=False
        ).squeeze(0)

    lc2st = LC2ST(
        thetas=theta, xs=x, posterior_samples=post_samples,
        num_trials_null=num_trials_null, seed=seed,
    )
    lc2st.train_on_observed_data(verbosity=0)
    lc2st.train_under_null_hypothesis(verbosity=0)

    theta_o = posterior.sample((1000,), x=x_o, show_progress_bars=False)
    # x_o goes in as a SINGLE row, shape (1, dim). eval_lc2st() repeats it to
    # match theta_o itself; pre-repeating it here makes it repeat the repeat.
    x_o_row = x_o.reshape(1, -1)

    stat = lc2st.get_statistic_on_observed_data(theta_o=theta_o, x_o=x_o_row)
    pval = lc2st.p_value(theta_o=theta_o, x_o=x_o_row)

    print(f"\n  LC2ST (local calibration AT x_o): statistic = {stat:.4f}, "
          f"p = {pval:.3f}")
    print("    statistic near 0 with p > 0.05 means q(theta | x_o) is calibrated")
    return {"statistic": float(stat), "p_value": float(pval),
            "num_trials_null": num_trials_null}


def main(
        fileName=DEFAULT_RUN,
        source="training",
        num_posterior_samples=DEFAULT_NUM_POSTERIOR_SAMPLES,
        num_bins=DEFAULT_NUM_BINS,
        do_tarp=True,
        do_lc2st=False,
        num_trials_null=100,
        seed=1,
    ):
    torch.manual_seed(seed)
    np.random.seed(seed)

    run = load_run(fileName)
    rc = run["run_config"]
    print(f"run: {fileName}")
    print(f"  {rc['num_rounds']} round(s), {rc['num_thetas_per_round']} thetas x "
          f"{rc['num_seeds_per_theta']} seeds, central={rc['central']}")
    print(f"  parameters: {', '.join(run['param_names'])}")

    if rc["num_rounds"] != 1:
        raise RuntimeError(
            f"this run has num_rounds={rc['num_rounds']}. Its final posterior is "
            "sequential, not amortised: it is only claimed to be correct near x_o, "
            "so uniform SBC ranks are not expected and non-uniform ones are not a "
            "defect. SBC applies to num_rounds=1 runs. Use --lc2st alone, which "
            "asks the local question a sequential posterior actually answers."
        )

    if source == "training":
        theta, x = training_pairs(fileName)
        print(f"  IN-SAMPLE SBC on {theta.shape[0]} training pairs: optimistic. A "
              "fail is conclusive, a pass is weak evidence. See the docstring.")
    else:
        theta, x = heldout_pairs(fileName)

    leakage = leakage_report(run["posterior"], run["prior"], x, run["x_o"])

    print(f"\n  drawing {num_posterior_samples} posterior samples for each of "
          f"{theta.shape[0]} pairs...")
    # use_batched_sampling=False ON PURPOSE, and it is a large speed difference
    # here rather than a stylistic choice. DirectPosterior fills its sample_shape
    # by rejecting draws that land outside the prior box, and the BATCHED path
    # redraws the WHOLE block of x every rejection round, so it runs until the
    # single worst-conditioned x is satisfied and every other x pays for it. This
    # run has one x with 0.45% acceptance against a median of 0.89 (see
    # leakage_report above), which made the batched path ~200x more work for the
    # same result. Per-x sampling costs that one x a few extra thousand draws and
    # nobody else anything.
    ranks, dap_samples = run_sbc(
        theta, x, run["posterior"],
        num_posterior_samples=num_posterior_samples,
        reduce_fns="marginals",
        use_batched_sampling=False,
        show_progress_bar=True,
    )

    # sbi's own summary. c2st_ranks is a classifier two-sample test of the ranks
    # against uniform: it catches shapes no single moment does. c2st_dap tests the
    # DATA-AVERAGED POSTERIOR against the prior, since averaging q(.|x) over the
    # prior predictive must return the prior; a failure there is a global
    # miscalibration the marginal ranks can average away.
    sbi_stats = check_sbc(
        ranks, theta, dap_samples, num_posterior_samples=num_posterior_samples
    )
    print("\n  sbi check_sbc:")
    for k, v in sbi_stats.items():
        print(f"    {k}: {np.round(np.asarray(v), 4)}")
    print("    ks_pvals > 0.05 pass; c2st_* near 0.5 pass (0.5 is "
          "indistinguishable, 1.0 is perfectly separable)")

    shape = rank_shape_stats(ranks, num_posterior_samples, run["param_names"],
                             num_bins)
    print_shape_table(shape)

    tag = source
    plot_paths = plot_ranks(ranks, num_posterior_samples, run["param_names"],
                            fileName, tag, num_bins)

    tarp = tarp_check(theta, x, run["posterior"], num_posterior_samples,
                      fileName, tag) if do_tarp else None
    lc2st = lc2st_check(theta, x, run["posterior"], run["x_o"],
                        num_trials_null, seed) if do_lc2st else None

    result = {
        "source": source,
        "num_pairs": int(theta.shape[0]),
        "num_posterior_samples": num_posterior_samples,
        "num_bins": num_bins,
        "param_names": run["param_names"],
        "ranks": ranks,
        "dap_samples": dap_samples,
        "leakage": leakage,
        "check_sbc": sbi_stats,
        "shape_stats": shape,
        "tarp": tarp,
        "lc2st": lc2st,
        "run_config": rc,
    }
    save_object(result, fileName + "/Data", f"sbc_{source}")
    print(f"\n  saved: {fileName}/Data/sbc_{source}.pkl")
    for p in plot_paths:
        print(f"  saved: {p}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SBC check on an sbi_seed_av run")
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--source", default="training",
                    choices=["training", "heldout"])
    ap.add_argument("--num-posterior-samples", type=int,
                    default=DEFAULT_NUM_POSTERIOR_SAMPLES)
    ap.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    ap.add_argument("--no-tarp", action="store_true")
    ap.add_argument("--lc2st", action="store_true",
                    help="also run the LOCAL calibration test at x_o (slow: it "
                         "trains num-trials-null classifiers for the null)")
    ap.add_argument("--num-trials-null", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    main(fileName=a.run, source=a.source,
         num_posterior_samples=a.num_posterior_samples, num_bins=a.num_bins,
         do_tarp=not a.no_tarp, do_lc2st=a.lc2st,
         num_trials_null=a.num_trials_null, seed=a.seed)
