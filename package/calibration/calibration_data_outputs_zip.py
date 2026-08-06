"""
Build the zip-level calibration target.

Companion to the existing calibration_data_outputs.py, which produces the
8-dim state-level target. This produces the 28-dim target described in
summary_stats.py, plus everything the validation splits and the observation
bootstrap need, and saves it as one pickle bundle.

Run:
    python -m package.calibration.calibration_data_outputs_zip

Swapping FAKE for real data: pass the real paths (or edit the defaults). The
bundle records which files it came from and whether they were the fake ones,
and every consumer prints that back, so a real result can never be quietly
confused with a placeholder result.
"""

import argparse
import numpy as np

from package.model.synthetic_population import build_zip_table, load_population, DEFAULT_STRATA_SPEC
from package.calibration.summary_stats import (
    DEFAULT_SPEC, dim_names, data_summary, load_zip_observations,
)
from package.resources.utility import save_object

FAKE_DIR = "package/calibration_data/FAKE_zip_data"
DEFAULT_POPULATION = f"{FAKE_DIR}/FAKE_synthetic_population.csv"
DEFAULT_STOCK = f"{FAKE_DIR}/FAKE_zip_ev_stock.csv"
DEFAULT_SALES = f"{FAKE_DIR}/FAKE_zip_ev_sales.csv"

OUT_ROOT = "package/calibration_data"
OUT_NAME = "calibration_data_output_zip"


def build(population_path=DEFAULT_POPULATION,
          stock_path=DEFAULT_STOCK,
          sales_path=DEFAULT_SALES,
          spec=None,
          strata_spec=None,
          verbose=True):
    spec = dict(DEFAULT_SPEC if spec is None else spec)
    strata_spec = dict(DEFAULT_STRATA_SPEC if strata_spec is None else strata_spec)

    is_fake = any("FAKE" in p for p in (population_path, stock_path, sales_path))
    if verbose and is_fake:
        print("=" * 74)
        print(" USING FAKE PLACEHOLDER DATA -- results are pipeline tests, not findings")
        print("=" * 74)

    pop = load_population(population_path)
    zip_table = build_zip_table(pop, strata_spec)
    stock_arrays, sales_arrays = load_zip_observations(stock_path, sales_path, zip_table)

    x_o = data_summary(stock_arrays, sales_arrays, spec, zip_table, zip_subset=None)
    names = dim_names(spec)

    if np.any(~np.isfinite(x_o)):
        bad = [names[i] for i in np.where(~np.isfinite(x_o))[0]]
        raise ValueError(f"target vector has non-finite entries at {bad}")

    bundle = {
        "x_o": x_o,
        "dim_names": names,
        "spec": spec,
        "strata_spec": strata_spec,
        "zip_table": zip_table,
        # kept so validation can rebuild the target over any zip subset, and so
        # bootstrap_targets.py can resample the raw counts
        "stock_arrays": stock_arrays,
        "sales_arrays": sales_arrays,
        "population_path": population_path,
        "stock_path": stock_path,
        "sales_path": sales_path,
        "IS_FAKE_DATA": is_fake,
    }

    if verbose:
        print(f"population : {population_path}")
        print(f"stock      : {stock_path}")
        print(f"sales      : {sales_path}")
        print(f"zips       : {len(zip_table['zip'])}")
        print(f"strata     : {len(zip_table['strata_labels'])}  {zip_table['strata_labels']}")
        print(f"target dims: {len(x_o)}")
        print()
        for n, v in zip(names, x_o):
            print(f"  {n:<24s} {v: .6f}")

    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default=DEFAULT_POPULATION)
    ap.add_argument("--stock", default=DEFAULT_STOCK)
    ap.add_argument("--sales", default=DEFAULT_SALES)
    args = ap.parse_args()

    bundle = build(args.population, args.stock, args.sales)
    save_object(bundle, OUT_ROOT, OUT_NAME)
    print(f"\nsaved {OUT_ROOT}/{OUT_NAME}.pkl")
