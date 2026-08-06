"""
The four validation splits.

STRUCTURAL CONSTRAINT THAT SHAPES ALL OF THEM. The firms, the two NK
technology landscapes, the second-hand merchant and the social network are all
state-wide and coupled: every agent competes for the same cars. So you CANNOT
drop zips or years from the simulation. Every split therefore simulates the
full population always, and holds out only in the LOSS:

  - a "dimension" split hides some summary-vector dims from the calibration
    (temporal, moment-type)
  - a "zip" split restricts which zips both the model side and the observed
    target are computed over (spatial, extrapolation)

Both are legitimate and neither costs extra simulations. A split is described
by a `Split` carrying a fit side and a test side, and run_validation.py does the
same thing to all four.

THE FOUR, IN INCREASING SEVERITY

1. TEMPORAL. Fit 2016-2021, predict 2022-2023. The strongest test for a
   diffusion model, because the thing being extrapolated is exactly the
   mechanism (adoption dynamics), and the test period is where the curve bends.
   Note that stock_2010-2015 exists in the real aggregate data
   (calibration_data_output.pkl "EV Prop" has 14 years, the calibration uses
   only the last 8) and is a free back-cast test if the zip data reaches back
   that far.

2. MOMENT TYPE. Fit on stock, predict sales. Nearly free, and a genuinely
   different functional of the same trajectory: stock is a slow accumulation,
   sales are the flow, and a model can match one while getting the turnover
   rate badly wrong.

3. RANDOM ZIP. Fit on 70% of zips, predict the other 30%. BE HONEST ABOUT THIS
   ONE: it is weak. The model maps zip covariates to preferences through a
   smooth parametric function, so a randomly held-out zip is an INTERPOLATION
   between trained zips and will pass almost automatically. It is reported
   because a failure here would be alarming, not because a pass is evidence.

4. COVARIATE EXTRAPOLATION. Fit on urban and suburban zips, predict rural. Or
   fit on two political terciles, predict the third. This is the only split
   that tests whether the covariate-to-preference map GENERALISES, which is the
   assumption every policy counterfactual rests on: a charging-infrastructure
   or rebate counterfactual is precisely a claim about parts of the covariate
   space you did not fit. This is the one to report.
"""

import numpy as np

from package.calibration.summary_stats import dim_index, dim_names


class Split:
    """
    A validation design.

    fit_mask / test_mask     boolean over summary dims
    fit_zips / test_zips     boolean over zips, or None for all zips
    """

    def __init__(self, name, description, severity,
                 fit_mask, test_mask, fit_zips=None, test_zips=None, caveat=None):
        self.name = name
        self.description = description
        self.severity = severity
        self.fit_mask = np.asarray(fit_mask, dtype=bool)
        self.test_mask = np.asarray(test_mask, dtype=bool)
        self.fit_zips = None if fit_zips is None else np.asarray(fit_zips, dtype=bool)
        self.test_zips = None if test_zips is None else np.asarray(test_zips, dtype=bool)
        self.caveat = caveat

        overlap = self.fit_mask & self.test_mask
        if self.fit_zips is None and self.test_zips is None and overlap.any():
            raise ValueError(
                f"split {name}: fit and test dimensions overlap and the zip sets are "
                f"identical, so the test set is inside the training set")

    def __repr__(self):
        nz = "" if self.fit_zips is None else f", zips {self.fit_zips.sum()}/{self.test_zips.sum()}"
        return (f"<Split {self.name}: fit {self.fit_mask.sum()} dims, "
                f"test {self.test_mask.sum()} dims{nz}>")

    def dim_report(self, spec):
        names = dim_names(spec)
        return {
            "fit": [n for n, m in zip(names, self.fit_mask) if m],
            "test": [n for n, m in zip(names, self.test_mask) if m],
        }


# ===========================================================================
# 1. Temporal
# ===========================================================================

def temporal_split(spec, fit_through=2021):
    """
    Fit every dimension whose year is <= fit_through, test on the rest.

    Applies to gradient and dispersion dims as well as the two level series, so
    the held-out period tests the cross-sectional pattern too, not just the
    aggregate level. That matters: a model can be pushed onto the right
    aggregate path by the wrong mechanism, and the late-period gradients are
    where that shows.
    """
    idx = dim_index(spec)
    n = len(idx)
    fit = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    for name, i in idx.items():
        year = int(name.rsplit("_", 1)[1])
        if year <= fit_through:
            fit[i] = True
        else:
            test[i] = True
    if not test.any():
        raise ValueError(f"fit_through={fit_through} leaves no test dimensions")
    return Split(
        "temporal",
        f"fit years <= {fit_through}, predict {fit_through+1} onward",
        severity="high",
        fit_mask=fit, test_mask=test,
    )


# ===========================================================================
# 2. Moment type
# ===========================================================================

def moment_type_split(spec):
    """
    Fit stock (levels, gradients, dispersion), predict the sales flow.

    The gradients and dispersion are computed from the stock snapshot, so they
    go on the fit side; the test side is purely the sales series.
    """
    names = dim_names(spec)
    fit = np.array([not n.startswith("sales_") for n in names])
    test = ~fit
    if not test.any():
        raise ValueError("no sales dimensions in the spec, nothing to hold out")
    return Split(
        "moment_type",
        "fit EV stock (levels + cross-section), predict the EV sales flow",
        severity="medium",
        fit_mask=fit, test_mask=test,
    )


# ===========================================================================
# 3. Random zip
# ===========================================================================

def random_zip_split(spec, zip_table, frac_fit=0.7, seed=0):
    """
    Fit on a random `frac_fit` of zips, predict the rest, on ALL dimensions.

    Sampled with probability proportional to nothing in particular (uniform over
    zips), but the resulting fit and test zip sets are reported with their
    household shares by run_validation, because a uniform zip draw in California
    puts most of the POPULATION on one side or the other only by luck.
    """
    nz = len(zip_table["zip"])
    rs = np.random.RandomState(seed)
    order = rs.permutation(nz)
    k = int(round(frac_fit * nz))
    fit_zips = np.zeros(nz, dtype=bool)
    fit_zips[order[:k]] = True
    all_dims = np.ones(len(dim_index(spec)), dtype=bool)
    return Split(
        "random_zip",
        f"fit a random {frac_fit:.0%} of zips, predict the held-out {1-frac_fit:.0%}",
        severity="low",
        fit_mask=all_dims, test_mask=all_dims,
        fit_zips=fit_zips, test_zips=~fit_zips,
        caveat=("weak by construction: the covariate-to-preference map is smooth, so a "
                "randomly held-out zip is an interpolation between trained zips. A pass "
                "here is close to automatic; only a failure is informative."),
    )


# ===========================================================================
# 4. Covariate extrapolation
# ===========================================================================

def covariate_extrapolation_split(spec, zip_table, covariate="ruralness", holdout_quantile=0.75):
    """
    Fit on the low end of a covariate, predict the held-out high end.

    Default: fit urban and suburban zips (bottom 75% of ruralness), predict the
    most rural quartile. That directly interrogates `a_rural`, and it is the
    split whose result a charging-infrastructure counterfactual actually depends
    on, because such a counterfactual is a claim about the part of the ruralness
    range you did not fit.

    Use covariate="political_leaning" for the political channel (`rho_pol`) or
    "median_income" for the income channel (`eps_beta`).
    """
    if covariate not in zip_table:
        raise KeyError(f"{covariate!r} not in the zip table (have {sorted(zip_table)})")
    v = zip_table[covariate]
    w = zip_table["households"]

    from package.model.synthetic_population import weighted_quantile
    cut = weighted_quantile(v, holdout_quantile, w)[0]
    fit_zips = v <= cut
    test_zips = ~fit_zips
    if test_zips.sum() < 5:
        raise ValueError(
            f"holdout_quantile={holdout_quantile} leaves only {test_zips.sum()} test zips; "
            f"the gradient regression on the test side will be meaningless")

    all_dims = np.ones(len(dim_index(spec)), dtype=bool)
    return Split(
        f"extrapolate_{covariate}",
        f"fit zips with {covariate} <= {cut:.4g} (bottom {holdout_quantile:.0%}), "
        f"predict the top {1-holdout_quantile:.0%}",
        severity="highest",
        fit_mask=all_dims, test_mask=all_dims,
        fit_zips=fit_zips, test_zips=test_zips,
    )


# ===========================================================================
# The default battery
# ===========================================================================

def all_splits(spec, zip_table):
    """The four designs, in the order they should be read."""
    return [
        temporal_split(spec, fit_through=2021),
        moment_type_split(spec),
        random_zip_split(spec, zip_table, frac_fit=0.7, seed=0),
        covariate_extrapolation_split(spec, zip_table, covariate="ruralness",
                                      holdout_quantile=0.75),
    ]
