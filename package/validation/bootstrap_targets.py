"""
How much of the model-data gap is just noise in the DATA?

Without this number every validation result is uninterpretable: you cannot say
a model misses a target until you know how well the target is known. Three
noise sources matter and they are separate things:

  (a) OBSERVATION noise in the zip counts -- this module
  (b) model Monte Carlo across seeds -- run_validation.py measures it
  (c) population-sample noise, which agents were drawn -- also (b), because
      seed_population varies inside the seed triples

Two different bootstraps are implemented, because the right notion of
uncertainty differs by dimension and using one for everything is wrong in one
direction or the other:

  COUNT bootstrap (binomial within zip). Treats the zips as the whole
  population and the only error as counting: redraw ev_stock ~ Binomial(
  total_stock, ev_stock/total_stock). This is the correct band for the STATE
  AGGREGATE dims, where there is no zip sampling error at all because the data
  covers every zip in California.

  ZIP-CLUSTER bootstrap (resample zips with replacement). Treats the observed
  zips as a sample from a population of possible zips, so it carries the
  residual zip-level heterogeneity -- the 0.35-log-odds unexplained spread in
  the fake data, and whatever its real counterpart is. This is the correct band
  for the GRADIENT and DISPERSION dims, which are cross-zip regressions whose
  standard error is driven by that residual, not by counting.

Reported side by side. run_validation.py takes the LARGER of the two per
dimension, which is the conservative choice: it will never call a gap
significant on the strength of having picked the smaller band.
"""

import numpy as np

from package.calibration.summary_stats import data_summary, dim_names


def count_bootstrap(bundle, n_boot=400, seed=0):
    """Binomial resample of every (zip, year) count. Captures counting noise only."""
    rs = np.random.RandomState(seed)
    spec, zip_table = bundle["spec"], bundle["zip_table"]
    draws = []
    for _ in range(n_boot):
        stock = _resample_counts(bundle["stock_arrays"], rs)
        sales = _resample_counts(bundle["sales_arrays"], rs)
        draws.append(data_summary(stock, sales, spec, zip_table))
    return np.asarray(draws)


def _resample_counts(arrays, rs):
    ev, tot = {}, {}
    for y in arrays["ev"]:
        t = arrays["total"][y]
        p = np.divide(arrays["ev"][y], t, out=np.zeros_like(t), where=t > 0)
        ev[y] = rs.binomial(t.astype(np.int64), np.clip(p, 0, 1)).astype(float)
        tot[y] = t
    return {"ev": ev, "total": tot}


def zip_cluster_bootstrap(bundle, n_boot=400, seed=0):
    """
    Resample zips with replacement. Captures residual zip heterogeneity.

    Implemented as an integer MULTIPLICITY on each zip rather than by building a
    resampled table, because every statistic in summary_stats is already
    count-weighted: multiplying a zip's ev and total counts by its bootstrap
    multiplicity is exactly equivalent to including it that many times, and it
    keeps the canonical zip ordering (and hence the z-scored covariates) intact.
    """
    rs = np.random.RandomState(seed)
    spec, zip_table = bundle["spec"], bundle["zip_table"]
    nz = len(zip_table["zip"])
    draws = []
    for _ in range(n_boot):
        mult = rs.multinomial(nz, np.full(nz, 1.0 / nz)).astype(float)
        stock = _scale_counts(bundle["stock_arrays"], mult)
        sales = _scale_counts(bundle["sales_arrays"], mult)
        draws.append(data_summary(stock, sales, spec, zip_table))
    return np.asarray(draws)


def _scale_counts(arrays, mult):
    return {
        "ev": {y: v * mult for y, v in arrays["ev"].items()},
        "total": {y: v * mult for y, v in arrays["total"].items()},
    }


def target_uncertainty(bundle, n_boot=400, seed=0, verbose=True):
    """
    Returns dict with per-dimension SDs from both bootstraps and the
    conservative `sd` (elementwise max) that validation should use.
    """
    spec = bundle["spec"]
    names = dim_names(spec)

    cnt = count_bootstrap(bundle, n_boot, seed)
    clu = zip_cluster_bootstrap(bundle, n_boot, seed + 1)

    sd_count = np.nanstd(cnt, axis=0)
    sd_cluster = np.nanstd(clu, axis=0)
    sd = np.maximum(sd_count, sd_cluster)

    if verbose:
        x_o = np.asarray(bundle["x_o"])
        print(f"observation-noise SD on the target ({n_boot} bootstrap draws)\n")
        print(f"  {'dim':<24s} {'value':>11s} {'sd_count':>11s} {'sd_cluster':>11s} {'used':>11s} {'rel':>8s}")
        for i, n in enumerate(names):
            rel = abs(sd[i] / x_o[i]) if x_o[i] != 0 else np.inf
            print(f"  {n:<24s} {x_o[i]:11.6f} {sd_count[i]:11.6f} "
                  f"{sd_cluster[i]:11.6f} {sd[i]:11.6f} {rel:7.1%}")
        print("\n  sd_count   = binomial within zip; the right band for the state aggregate dims")
        print("  sd_cluster = resample zips; the right band for the gradient/dispersion dims")
        print("  used       = elementwise max, the conservative choice")

    return {
        "sd": sd,
        "sd_count": sd_count,
        "sd_cluster": sd_cluster,
        "dim_names": names,
        "n_boot": n_boot,
    }


if __name__ == "__main__":
    from package.resources.utility import load_object
    b = load_object("package/calibration_data", "calibration_data_output_zip")
    if b.get("IS_FAKE_DATA"):
        print("NOTE: FAKE placeholder data\n")
    target_uncertainty(b)
