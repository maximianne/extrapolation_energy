# methods/calibration.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression

try:
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError(
        "scipy is required for calibration weighting. "
        "Please install it via `pip install scipy`."
    ) from e


def _to_2d_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Ensure X is a 2D numpy array with shape (n_samples, n_features).
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    return X_arr


@dataclass
class CalibrationWeightingResult:
    """
    Result container for calibration-weighting generalizability estimator.

    Attributes
    ----------
    ate : float
        Estimated average treatment effect in the chosen target population.
    ate_treated : float
        Estimated mean potential outcome under treatment in the target population.
    ate_control : float
        Estimated mean potential outcome under control in the target population.
    weights : ndarray, shape (n_samples,)
        Calibration weights for all units. Non-trial units (S=0) have weight 0.
    target_means : ndarray, shape (p,)
        Covariate means in the target population (S=0 or combined).
    trial_weighted_means : ndarray, shape (p,)
        Covariate means among S=1 using calibration weights.
    converged : bool
        Whether the optimization for calibration weights converged.
    optimization_message : str
        Message from the optimizer (helpful for debugging).
    """

    ate: float
    ate_treated: float
    ate_control: float
    weights: np.ndarray
    target_means: np.ndarray
    trial_weighted_means: np.ndarray
    converged: bool
    optimization_message: str


class CalibrationGeneralizabilityEstimator:
    """
    Calibration weighting estimator for extrapolating an ATE to a target population.

    Variables
    ---------
    - S: sample indicator (S=1 trial / experimental sample, S=0 target-only units)
    - T: treatment indicator (T=1 treated, T=0 control) in the trial
    - Y: outcome

    Idea
    ----
    Choose weights w_i for trial units (S=1) such that:
        sum_{i: S=1} w_i X_i  ≈  sum_{i in target} X_i / N_target
    i.e., weighted covariate means in the trial match the (unweighted)
    means in the target population.

    Implementation
    --------------
    We use an entropy balancing / exponential tilting formulation:
        w_i(γ) ∝ exp(γ' X_i)   for S=1 units
    and choose γ to minimize the dual convex objective:
        f(γ) = log( sum_i exp(γ' X_i) ) - γ' μ_target,
    whose gradient equals weighted means minus μ_target.

    Once weights are obtained, we estimate:
        μ1 = weighted mean of Y among T=1, S=1
        μ0 = weighted mean of Y among T=0, S=1
        ATE = μ1 - μ0
    """

    def __init__(
        self,
        *,
        target: Literal["nontrial", "combined"] = "nontrial",
        max_iter: int = 500,
        tol: float = 1e-8,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        target : {"nontrial", "combined"}, default "nontrial"
            Target population definition:
            - "nontrial": target is units with S=0.
            - "combined": target is all units (S=0 and S=1).
        max_iter : int, default 500
            Maximum number of iterations for the optimizer.
        tol : float, default 1e-8
            Convergence tolerance for the optimizer.
        optimizer_kwargs : dict, optional
            Additional keyword arguments for scipy.optimize.minimize.
        """
        self.target = target
        self.max_iter = max_iter
        self.tol = tol
        self.optimizer_kwargs = optimizer_kwargs or {}

        # Fitted attributes
        self.weights_: Optional[np.ndarray] = None
        self.target_means_: Optional[np.ndarray] = None
        self.trial_weighted_means_: Optional[np.ndarray] = None
        self.S_: Optional[np.ndarray] = None
        self.X_: Optional[np.ndarray] = None
        self._opt_result: Optional[Any] = None

        self._fitted = False

    @staticmethod
    def _entropy_dual_objective(
        gamma: np.ndarray,
        X_trial: np.ndarray,
        target_means: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Dual objective for calibration via entropy balancing.

        Parameters
        ----------
        gamma : ndarray, shape (p,)
            Current parameter vector.
        X_trial : ndarray, shape (n_trial, p)
            Covariates for trial units (S=1).
        target_means : ndarray, shape (p,)
            Covariate means in the target population.

        Returns
        -------
        f_val : float
            Objective value.
        grad : ndarray, shape (p,)
            Gradient with respect to gamma.
        """
        Xg = X_trial @ gamma  # shape (n_trial,)

        # Numerical stability with log-sum-exp trick
        max_Xg = np.max(Xg)
        exp_shifted = np.exp(Xg - max_Xg)
        Z = np.sum(exp_shifted)
        w = exp_shifted / Z  # these are normalized to sum to 1

        # f(γ) = log( sum exp(γ' x_i) ) - γ' μ_target
        f_val = np.log(Z) + max_Xg - np.dot(gamma, target_means)

        # gradient = sum_i w_i x_i - μ_target
        weighted_means = X_trial.T @ w  # shape (p,)
        grad = weighted_means - target_means

        return f_val, grad

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        S: np.ndarray,
    ) -> "CalibrationGeneralizabilityEstimator":
        """
        Fit calibration weights using covariates X and sample indicator S.

        Parameters
        ----------
        X : DataFrame or 2D ndarray
            Covariates for all units (trial and non-trial).
        S : 1D ndarray
            Sample indicator (1 = trial, 0 = non-trial).

        Returns
        -------
        self : CalibrationGeneralizabilityEstimator
        """
        X_mat = _to_2d_array(X)
        S = np.asarray(S).ravel()

        if X_mat.shape[0] != S.shape[0]:
            raise ValueError("X and S must have the same number of rows.")

        self.X_ = X_mat
        self.S_ = S

        # Define trial and target sets
        trial_mask = (S == 1)
        if not np.any(trial_mask):
            raise ValueError("No trial units (S=1) in the data.")

        if self.target == "nontrial":
            target_mask = (S == 0)
        elif self.target == "combined":
            target_mask = np.ones_like(S, dtype=bool)
        else:
            raise ValueError("target must be 'nontrial' or 'combined'")

        if not np.any(target_mask):
            raise ValueError("No units in chosen target population; check `target` and `S`.")

        X_trial = X_mat[trial_mask, :]
        X_target = X_mat[target_mask, :]

        # Target covariate means
        target_means = X_target.mean(axis=0)  # shape (p,)

        # Optimize dual objective to find gamma
        p = X_trial.shape[1]
        gamma0 = np.zeros(p, dtype=float)

        def obj(gamma):
            f_val, grad = self._entropy_dual_objective(gamma, X_trial, target_means)
            return f_val, grad

        opt_result = minimize(
            fun=lambda g: obj(g)[0],
            x0=gamma0,
            jac=lambda g: obj(g)[1],
            method="BFGS",
            options={"maxiter": self.max_iter, "gtol": self.tol, **self.optimizer_kwargs},
        )

        if not opt_result.success:
            # We don't fail hard; we still produce weights, but mark converged=False
            converged = False
        else:
            converged = True

        gamma_hat = opt_result.x
        # Compute normalized weights for trial units
        Xg = X_trial @ gamma_hat
        max_Xg = np.max(Xg)
        exp_shifted = np.exp(Xg - max_Xg)
        w_trial = exp_shifted / np.sum(exp_shifted)  # sums to 1 over trial units

        # Expand to all units: S=1 gets w_trial, S=0 gets 0
        weights = np.zeros_like(S, dtype=float)
        weights[trial_mask] = w_trial

        # Trial weighted means (for diagnostics)
        trial_weighted_means = (X_trial.T @ w_trial)  # shape (p,)

        self.weights_ = weights
        self.target_means_ = target_means
        self.trial_weighted_means_ = trial_weighted_means
        self._opt_result = opt_result
        self._converged = converged
        self._fitted = True

        return self

    def estimate_ate(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        S: np.ndarray,
    ) -> CalibrationWeightingResult:
        """
        Estimate ATE using the fitted calibration weights.

        Parameters
        ----------
        Y : 1D ndarray
            Outcome; assumed observed for S=1. For S=0, values can be np.nan.
        T : 1D ndarray
            Treatment indicator (1 = treated, 0 = control).
        S : 1D ndarray
            Sample indicator (1 = trial, 0 = non-trial).

        Returns
        -------
        result : CalibrationWeightingResult
        """
        if not self._fitted:
            raise RuntimeError("Estimator must be fit before calling estimate_ate().")

        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T).ravel()
        S = np.asarray(S).ravel()

        if any(arr.shape[0] != Y.shape[0] for arr in [T, S, self.weights_]):
            raise ValueError("Y, T, S, and weights must all have the same length.")

        weights = self.weights_
        target_means = self.target_means_
        trial_weighted_means = self.trial_weighted_means_

        # Restrict to trial units with observed outcomes
        mask_trial_obs = (S == 1) & ~np.isnan(Y)
        Y_trial = Y[mask_trial_obs]
        T_trial = T[mask_trial_obs]
        w_trial = weights[mask_trial_obs]

        # Weighted means in treated and control groups
        mask_treated = (T_trial == 1)
        mask_control = (T_trial == 0)

        if not np.any(mask_treated) or not np.any(mask_control):
            raise RuntimeError("Need both treated and control units in the trial to estimate ATE.")

        mu1_hat = np.average(Y_trial[mask_treated], weights=w_trial[mask_treated])
        mu0_hat = np.average(Y_trial[mask_control], weights=w_trial[mask_control])
        ate_hat = mu1_hat - mu0_hat

        return CalibrationWeightingResult(
            ate=float(ate_hat),
            ate_treated=float(mu1_hat),
            ate_control=float(mu0_hat),
            weights=weights,
            target_means=target_means,
            trial_weighted_means=trial_weighted_means,
            converged=self._converged,
            optimization_message=self._opt_result.message if self._opt_result is not None else "",
        )


def estimate_ate_calibration(
    Y: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    *,
    target: Literal["nontrial", "combined"] = "nontrial",
    max_iter: int = 500,
    tol: float = 1e-8,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> CalibrationWeightingResult:
    """
    Convenience function: one-shot calibration-weighting ATE estimation.

    Parameters
    ----------
    Y : 1D ndarray
        Outcome.
    T : 1D ndarray
        Treatment indicator (1 = treated, 0 = control).
    S : 1D ndarray
        Sample indicator (1 = trial, 0 = non-trial).
    X : DataFrame or 2D ndarray
        Covariates used for calibration constraints.

    target : {"nontrial", "combined"}, default "nontrial"
        Target population definition.
    max_iter : int, default 500
        Maximum iterations for optimizer.
    tol : float, default 1e-8
        Convergence tolerance.
    optimizer_kwargs : dict, optional
        Additional kwargs for scipy.optimize.minimize.

    Returns
    -------
    result : CalibrationWeightingResult
    """
    est = CalibrationGeneralizabilityEstimator(
        target=target,
        max_iter=max_iter,
        tol=tol,
        optimizer_kwargs=optimizer_kwargs or {},
    )
    est.fit(X=X, S=S)
    return est.estimate_ate(Y=Y, T=T, S=S)
