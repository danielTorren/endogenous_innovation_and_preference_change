"""
surrogate.py — Gaussian Process surrogate fitting and validation.

WHY A GP?
---------
The ABM has stochastic outputs (multiple seeds) and an unknown, non-linear
response surface over the 5D policy space. A GP with a Matern 5/2 kernel:
  - Provides uncertainty estimates (critical for knowing where to sample next
    and for honest reporting of optimisation confidence)
  - Works well with 50–200 training points (typical for expensive simulators)
  - Handles the ABM's stochastic noise via the WhiteKernel nugget term

HOW TO ASSESS SURROGATE QUALITY
---------------------------------
Run validate() or loo_cv() and check these metrics:

  R²  (coefficient of determination, 0–1, higher is better)
  ├── > 0.95 : excellent — trust the BO recommendations
  ├── 0.85–0.95 : good — add ~20 more LHS points in high-error regions
  └── < 0.85 : poor — do not run BO yet; collect more data first

  RMSE / MAE  (in original output units)
  └── Cross-reference with the output range; RMSE < 5% of range is fine

  95% prediction interval coverage  (should be ≈ 0.95)
  ├── >> 0.95 : GP is over-confident — it will mislead BO about uncertainty
  └── << 0.95 : GP is under-confident — safe but makes BO less efficient
                (try decreasing the WhiteKernel noise_level_bounds lower limit)

HOW TO IMPROVE THE SURROGATE
------------------------------
1. Most reliable: add more LHS points (especially where parity plots show
   large errors — this reveals where the response is most complex)
2. If the parity plot shows systematic bias in one region, that region needs
   targeted samples (concentrate new LHS points there)
3. If coverage is much < 0.95, the ABM has high run-to-run variance relative
   to your n_seeds — either increase n_seeds or accept wider CIs
4. If R² is fine for some outputs but not others, the poorly-fitted output
   has a more complex (possibly non-monotonic) relationship with policy inputs
   — consider a separate, denser LHS for that output
5. The Matern 5/2 kernel is a strong default. Only change it if you have
   a physical reason (e.g., perfectly smooth response → RBF; highly local
   variation → Matern 3/2)
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from typing import Optional

from .sampling import OUTPUT_NAMES


# ---------------------------------------------------------------------------
# GP surrogate
# ---------------------------------------------------------------------------

def _make_kernel():
    """
    Matern 5/2 is the standard choice for physical/economic simulators:
    it assumes the response is twice differentiable — smoother than reality
    is unlikely, rougher than RBF is realistic.

    ConstantKernel handles output scale.
    WhiteKernel absorbs ABM seed-to-seed noise (the 'nugget').
    """
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(length_scale=np.ones(5), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e0))
    )


class SurrogateGP:
    """
    One GP per output, fitted independently.

    Inputs  are normalised to [0, 1] internally (required for length-scale
    optimisation to work well across policies with very different units).
    Outputs are standardised to zero mean, unit variance internally.
    Predictions are returned in original (un-normalised) units.
    """

    def __init__(self, bounds_lower: np.ndarray, bounds_upper: np.ndarray,
                 n_restarts: int = 5):
        self.bounds_lower = bounds_lower.copy()
        self.bounds_upper = bounds_upper.copy()
        self.n_restarts = n_restarts
        self.gps: list = []
        self.y_mean = np.zeros(4)
        self.y_std = np.ones(4)
        self._input_range = np.where(
            bounds_upper - bounds_lower == 0, 1.0, bounds_upper - bounds_lower
        )

    def _norm_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.bounds_lower) / self._input_range

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit one GP per output column. X: (n, p), Y: (n, 4)."""
        X_n = self._norm_X(X)
        self.y_mean = Y.mean(axis=0)
        self.y_std = np.where(Y.std(axis=0) < 1e-8, 1.0, Y.std(axis=0))
        Y_n = (Y - self.y_mean) / self.y_std

        self.gps = []
        for j in range(Y.shape[1]):
            gp = GaussianProcessRegressor(
                kernel=_make_kernel(),
                n_restarts_optimizer=self.n_restarts,
                normalize_y=False,
                alpha=0.0,
            )
            gp.fit(X_n, Y_n[:, j])
            self.gps.append(gp)

    def predict(self, X: np.ndarray) -> tuple:
        """
        Returns (mean, std) in original units.
        Both arrays have shape (n, 4).
        """
        X_n = self._norm_X(X)
        means, stds = [], []
        for j, gp in enumerate(self.gps):
            mu_n, sigma_n = gp.predict(X_n, return_std=True)
            means.append(mu_n * self.y_std[j] + self.y_mean[j])
            stds.append(np.abs(sigma_n) * self.y_std[j])
        return np.stack(means, axis=1), np.stack(stds, axis=1)

    def predict_ev(self, X: np.ndarray) -> tuple:
        """Convenience method: predict only the EV uptake GP (index 0)."""
        X_n = self._norm_X(X)
        mu_n, sigma_n = self.gps[0].predict(X_n, return_std=True)
        return (mu_n * self.y_std[0] + self.y_mean[0],
                np.abs(sigma_n) * self.y_std[0])


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_std: np.ndarray, name: str, label: str) -> dict:
    z = 1.96
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    lo = y_pred - z * y_std
    hi = y_pred + z * y_std
    coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
    print(f"  [{label}] {name:<20}  R²={r2:+.3f}  RMSE={rmse:.4g}  "
          f"MAE={mae:.4g}  95%-coverage={coverage:.1%}")
    return dict(R2=r2, RMSE=rmse, MAE=mae, Coverage_95=coverage)


def _flag_metrics(metrics: dict):
    """Print actionable warnings based on validation metrics."""
    print()
    for name, m in metrics.items():
        r2, cov = m["R2"], m["Coverage_95"]
        if r2 < 0.85:
            print(f"  WARNING [{name}]: R²={r2:.3f} < 0.85 — surrogate is unreliable."
                  " Collect more LHS samples before optimising.")
        elif r2 < 0.95:
            print(f"  NOTE    [{name}]: R²={r2:.3f} — acceptable but consider adding "
                  "~20 more samples in high-error regions of the parity plot.")
        if cov < 0.85:
            print(f"  WARNING [{name}]: 95%-coverage={cov:.1%} — GP is over-confident."
                  " BO uncertainty estimates will be misleadingly small.")
        elif cov > 0.99:
            print(f"  NOTE    [{name}]: 95%-coverage={cov:.1%} — GP is under-confident."
                  " Safe but slightly inefficient for BO.")


def _parity_plots(mu: np.ndarray, sigma: np.ndarray, Y_true: np.ndarray,
                  label: str = "", save_dir: Optional[str] = None):
    """
    Predicted vs actual scatter plots with 95% CI error bars.

    Points that fall outside the ±45° diagonal indicate poor predictions
    in that region — those are the areas where you need more training data.
    """
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    z = 1.96
    for j, (ax, name) in enumerate(zip(axs, OUTPUT_NAMES)):
        y_true = Y_true[:, j]
        y_pred = mu[:, j]
        y_err = z * sigma[:, j]
        lo_lim = min(y_true.min(), (y_pred - y_err).min())
        hi_lim = max(y_true.max(), (y_pred + y_err).max())

        ax.errorbar(y_true, y_pred, yerr=y_err, fmt='o', alpha=0.6,
                    capsize=3, markersize=4, color='steelblue',
                    label='GP pred ± 95% CI')
        ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], 'k--', lw=1, label='perfect')
        ax.set_title(f"{name}\nR²={r2_score(y_true, y_pred):.3f}", fontsize=11)
        ax.set_xlabel("ABM (true)")
        ax.set_ylabel("GP (predicted)")
        ax.legend(fontsize=7)

    fig.suptitle(f"Surrogate parity plots — {label}", fontsize=12)
    fig.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/parity_{label.replace(' ', '_')}.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Validation: train/test split
# ---------------------------------------------------------------------------

def validate(
    surrogate: SurrogateGP,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    label: str = "test set",
    plot: bool = True,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Evaluate surrogate quality on a held-out test set.

    Use this when n_train >= 60 so you can afford to withhold 20% of points.
    The test set should be drawn from the same LHS design but withheld before
    fitting — never fit and then test on the same data.

    Returns dict of {output_name: {R2, RMSE, MAE, Coverage_95}}.
    """
    print(f"\nValidation ({label}, n={len(X_test)}):")
    mu, sigma = surrogate.predict(X_test)
    metrics = {}
    for j, name in enumerate(OUTPUT_NAMES):
        metrics[name] = _compute_metrics(Y_test[:, j], mu[:, j], sigma[:, j], name, label)
    _flag_metrics(metrics)
    if plot:
        _parity_plots(mu, sigma, Y_test, label=label, save_dir=save_dir)
    return metrics


# ---------------------------------------------------------------------------
# Validation: leave-one-out cross-validation
# ---------------------------------------------------------------------------

def loo_cv(
    X: np.ndarray,
    Y: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    plot: bool = True,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Leave-one-out cross-validation (LOO-CV).

    Use this when n < 60 and a held-out test set would waste too much data.
    Each point is removed in turn, the surrogate is re-fit on the remaining
    n-1 points, and the excluded point is predicted.

    This is n times more expensive than a simple train/test split, but it
    gives an unbiased estimate of generalisation error for small datasets.

    Returns dict of {output_name: {R2, RMSE, MAE, Coverage_95}}.
    """
    print(f"\nLOO-CV (n={len(X)}):")
    loo = LeaveOneOut()
    true_list, pred_list, std_list = [], [], []

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        print(f"  fold {i+1}/{len(X)}", end="\r")
        s = SurrogateGP(bounds_lower, bounds_upper)
        s.fit(X[train_idx], Y[train_idx])
        mu, sigma = s.predict(X[test_idx])
        true_list.append(Y[test_idx])
        pred_list.append(mu)
        std_list.append(sigma)

    print()
    Y_true = np.vstack(true_list)
    Y_pred = np.vstack(pred_list)
    Y_std = np.vstack(std_list)

    metrics = {}
    for j, name in enumerate(OUTPUT_NAMES):
        metrics[name] = _compute_metrics(Y_true[:, j], Y_pred[:, j],
                                         Y_std[:, j], name, "LOO-CV")
    _flag_metrics(metrics)
    if plot:
        _parity_plots(Y_pred, Y_std, Y_true, label="LOO-CV", save_dir=save_dir)
    return metrics


# ---------------------------------------------------------------------------
# Persistence — save and reload a fitted surrogate
# ---------------------------------------------------------------------------

def save_surrogate(surrogate: SurrogateGP, path: str):
    """
    Save a fitted SurrogateGP to disk using pickle.

    The saved file includes the fitted GP objects, kernel hyperparameters,
    input normalisation bounds, and output standardisation stats — everything
    needed to make predictions without retraining.

    Usage:
        save_surrogate(surrogate, "results/surrogate_optimisation/Data/surrogate.pkl")

    Typical file size: ~1–5 MB for a 100-point, 5-input, 4-output GP.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(surrogate, f)
    print(f"Surrogate saved to {path}")


def load_surrogate(path: str) -> SurrogateGP:
    """
    Load a previously saved SurrogateGP from disk.

    The loaded surrogate is immediately ready to call .predict() on — no
    retraining needed. The original training data is NOT stored inside it
    (only the fitted kernel parameters), so keep your lhs_data.npz / bo_data.npz
    files if you want to add more training points later.

    Usage:
        surrogate = load_surrogate("results/surrogate_optimisation/Data/surrogate.pkl")
        mu, sigma = surrogate.predict(new_policy_combinations)
    """
    with open(path, "rb") as f:
        surrogate = pickle.load(f)
    print(f"Surrogate loaded from {path}  "
          f"({len(surrogate.gps)} GPs, "
          f"input dim={len(surrogate.bounds_lower)})")
    return surrogate
