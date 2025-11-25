# methods/calibration.py  (or augmented_calibration.py)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression

try:
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError(
        "scipy is required for calibration weighting. "
        "Please install it via `pip install scipy`."
    ) from e


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# Pure calibration weighting
# ---------------------------------------------------------------------


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
        sum_{i: S=1} w_i X_i  ≈  mean of X in target population.

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
        self._converged: bool = False

        self._fitted: bool = False

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
        w = exp_shifted / Z  # normalized to sum to 1 over trial units

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

        def obj(gamma: np.ndarray) -> Tuple[float, np.ndarray]:
            return self._entropy_dual_objective(gamma, X_trial, target_means)

        opt_result = minimize(
            fun=lambda g: obj(g)[0],
            x0=gamma0,
            jac=lambda g: obj(g)[1],
            method="BFGS",
            options={"maxiter": self.max_iter, "gtol": self.tol, **self.optimizer_kwargs},
        )

        if not opt_result.success:
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


# ---------------------------------------------------------------------
# Augmented (doubly-robust) calibration weighting
# ---------------------------------------------------------------------


@dataclass
class AugmentedCalibrationResult:
    """
    Result container for Augmented Calibration Weighting (ACW) estimator.

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
    m1 : ndarray, shape (n_samples,)
        Predicted potential outcome under treatment for all units.
    m0 : ndarray, shape (n_samples,)
        Predicted potential outcome under control for all units.
    e_hat : float
        Estimated (or provided) treatment probability within the trial.
    converged : bool
        Whether the calibration optimization converged.
    optimization_message : str
        Message from the optimizer (helpful for debugging).
    """

    ate: float
    ate_treated: float
    ate_control: float
    weights: np.ndarray
    target_means: np.ndarray
    trial_weighted_means: np.ndarray
    m1: np.ndarray
    m0: np.ndarray
    e_hat: float
    converged: bool
    optimization_message: str


class AugmentedCalibrationGeneralizabilityEstimator(CalibrationGeneralizabilityEstimator):
    """
    Augmented Calibration Weighting estimator (doubly-robust generalizability).

    Combines:
      (i) calibration weights w_i for trial units (S=1), and
      (ii) outcome models m_1(X), m_0(X) fit on the trial.

    DR estimator:
        ATE^G = E_G[m_1(X) - m_0(X)]
                + E_w[ (T/e)*(Y - m_1(X)) - ((1-T)/(1-e))*(Y - m_0(X)) ]

    where:
      - G is the target population ("nontrial" or "combined"),
      - w are calibration weights on S=1 units normalized to sum to 1,
      - e is treatment probability in the trial (scalar).
    """

    def __init__(
        self,
        *,
        target: Literal["nontrial", "combined"] = "nontrial",
        max_iter: int = 500,
        tol: float = 1e-8,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        outcome_model: Optional[RegressorMixin] = None,
        outcome_model_kwargs: Optional[Dict[str, Any]] = None,
        treatment_prob: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        target : {"nontrial", "combined"}, default "nontrial"
            Target population definition.
        max_iter : int, default 500
            Maximum iterations for calibration optimizer.
        tol : float, default 1e-8
            Convergence tolerance for calibration optimizer.
        optimizer_kwargs : dict, optional
            Additional kwargs for scipy.optimize.minimize.
        outcome_model : sklearn-like regressor, optional
            Base outcome model (must implement .fit(X, y), .predict(X)).
            If None, default is LinearRegression(**outcome_model_kwargs).
            This base model is cloned separately for T=1 and T=0.
        outcome_model_kwargs : dict, optional
            Extra kwargs passed to LinearRegression if outcome_model is None.
        treatment_prob : float, optional
            Known treatment probability e = P(T=1 | S=1). If None, it is
            estimated from the trial data as the empirical share of T=1.
        """
        super().__init__(
            target=target,
            max_iter=max_iter,
            tol=tol,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.outcome_model = outcome_model
        self.outcome_model_kwargs = outcome_model_kwargs or {}
        self.treatment_prob = treatment_prob

        # Extra fitted attributes
        self.m1_: Optional[np.ndarray] = None
        self.m0_: Optional[np.ndarray] = None
        self.Y_: Optional[np.ndarray] = None
        self.T_: Optional[np.ndarray] = None
        self.model1_: Optional[RegressorMixin] = None
        self.model0_: Optional[RegressorMixin] = None
        self.e_hat_: Optional[float] = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        S: np.ndarray,
    ) -> "AugmentedCalibrationGeneralizabilityEstimator":
        """
        Fit calibration weights and outcome models using (X, Y, T, S).

        Parameters
        ----------
        X : DataFrame or 2D ndarray
            Covariates for all units (trial and non-trial).
        Y : 1D ndarray
            Outcome; assumed only observed (or valid) for S=1.
            For S=0, values can be np.nan or ignored.
        T : 1D ndarray
            Treatment indicator (1 = treated, 0 = control).
        S : 1D ndarray
            Sample indicator (1 = trial, 0 = non-trial).

        Returns
        -------
        self : AugmentedCalibrationGeneralizabilityEstimator
        """
        X_mat = _to_2d_array(X)
        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T).ravel()
        S = np.asarray(S).ravel()

        if not (X_mat.shape[0] == Y.shape[0] == T.shape[0] == S.shape[0]):
            raise ValueError("X, Y, T, and S must all have the same number of rows.")

        # 1) Fit calibration weights (on X, S) via parent class
        super().fit(X=X_mat, S=S)

        # 2) Fit outcome models m_1(X), m_0(X) on trial units
        if self.outcome_model is None:
            base_model: RegressorMixin = LinearRegression(**self.outcome_model_kwargs)
        else:
            base_model = self.outcome_model

        mask_trial_obs = (S == 1) & ~np.isnan(Y)
        X_trial = X_mat[mask_trial_obs]
        Y_trial = Y[mask_trial_obs]
        T_trial = T[mask_trial_obs]

        # Treated model
        mask_treated = (T_trial == 1)
        if not np.any(mask_treated):
            raise RuntimeError("No treated trial units (T=1, S=1) to fit m_1(X).")

        model1 = clone(base_model)
        model1.fit(X_trial[mask_treated], Y_trial[mask_treated])
        m1 = model1.predict(X_mat)

        # Control model
        mask_control = (T_trial == 0)
        if not np.any(mask_control):
            raise RuntimeError("No control trial units (T=0, S=1) to fit m_0(X).")

        model0 = clone(base_model)
        model0.fit(X_trial[mask_control], Y_trial[mask_control])
        m0 = model0.predict(X_mat)

        # 3) Treatment probability e
        if self.treatment_prob is None:
            e_hat = float(np.mean(T_trial))
        else:
            e_hat = float(self.treatment_prob)

        self.m1_ = m1
        self.m0_ = m0
        self.Y_ = Y
        self.T_ = T
        self.e_hat_ = e_hat

        return self

    def estimate_ate(self) -> AugmentedCalibrationResult:
        """
        Compute the DR ATE, μ1, and μ0 in the target population using
        calibration weights and outcome models.

        Returns
        -------
        result : AugmentedCalibrationResult
        """
        if not self._fitted:
            raise RuntimeError("Estimator must be fit before calling estimate_ate().")

        if self.m1_ is None or self.m0_ is None or self.Y_ is None or self.T_ is None:
            raise RuntimeError("Outcome models or stored Y/T are missing; call fit() first.")

        S = self.S_
        w = self.weights_
        m1 = self.m1_
        m0 = self.m0_
        Y = self.Y_
        T = self.T_
        e_hat = self.e_hat_

        if S is None or w is None:
            raise RuntimeError("Calibration components missing; ensure fit() ran successfully.")

        # Target population mask
        if self.target == "nontrial":
            target_mask = (S == 0)
        elif self.target == "combined":
            target_mask = np.ones_like(S, dtype=bool)
        else:
            raise ValueError("target must be 'nontrial' or 'combined'")

        if not np.any(target_mask):
            raise ValueError("No units in chosen target population; check `target` and `S`.")

        # Plug-in term: average of predicted effects in target
        mu1_tar = float(np.mean(m1[target_mask]))
        mu0_tar = float(np.mean(m0[target_mask]))
        plug_in = mu1_tar - mu0_tar

        # Residual correction term using trial units with outcomes
        mask_trial_obs = (S == 1) & ~np.isnan(Y)
        Y_trial = Y[mask_trial_obs]
        T_trial = T[mask_trial_obs]
        m1_trial = m1[mask_trial_obs]
        m0_trial = m0[mask_trial_obs]
        w_trial = w[mask_trial_obs]

        if not np.any(T_trial == 1) or not np.any(T_trial == 0):
            raise RuntimeError("Need both treated and control units in the trial for DR correction.")

        correction = np.sum(
            w_trial
            * (
                (T_trial / e_hat) * (Y_trial - m1_trial)
                - ((1.0 - T_trial) / (1.0 - e_hat)) * (Y_trial - m0_trial)
            )
        )

        ate_hat = plug_in + correction

        # DR estimates of μ1 and μ0 in target (optionally reported)
        mu1_hat = mu1_tar + np.sum(w_trial * (T_trial / e_hat) * (Y_trial - m1_trial))
        mu0_hat = mu0_tar + np.sum(
            w_trial * ((1.0 - T_trial) / (1.0 - e_hat)) * (Y_trial - m0_trial)
        )

        return AugmentedCalibrationResult(
            ate=float(ate_hat),
            ate_treated=float(mu1_hat),
            ate_control=float(mu0_hat),
            weights=w,
            target_means=self.target_means_,
            trial_weighted_means=self.trial_weighted_means_,
            m1=m1,
            m0=m0,
            e_hat=e_hat,
            converged=self._converged,
            optimization_message=self._opt_result.message if self._opt_result is not None else "",
        )


def estimate_ate_calibration_augmented(
    Y: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    *,
    target: Literal["nontrial", "combined"] = "nontrial",
    max_iter: int = 500,
    tol: float = 1e-8,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    outcome_model: Optional[RegressorMixin] = None,
    outcome_model_kwargs: Optional[Dict[str, Any]] = None,
    treatment_prob: Optional[float] = None,
) -> AugmentedCalibrationResult:
    """
    Convenience function: one-shot Augmented Calibration Weighting ATE estimation.

    Parameters
    ----------
    Y : 1D ndarray
        Outcome.
    T : 1D ndarray
        Treatment indicator (1 = treated, 0 = control).
    S : 1D ndarray
        Sample indicator (1 = trial, 0 = non-trial).
    X : DataFrame or 2D ndarray
        Covariates used for calibration constraints and outcome models.
    target : {"nontrial", "combined"}, default "nontrial"
        Target population definition.
    max_iter : int, default 500
        Maximum iterations for calibration optimizer.
    tol : float, default 1e-8
        Convergence tolerance for calibration optimizer.
    optimizer_kwargs : dict, optional
        Additional kwargs for scipy.optimize.minimize.
    outcome_model : sklearn-like regressor, optional
        Base outcome model.
    outcome_model_kwargs : dict, optional
        Extra kwargs passed if outcome_model is None (LinearRegression).
    treatment_prob : float, optional
        Known treatment probability e = P(T=1 | S=1). If None, estimated from trial.

    Returns
    -------
    result : AugmentedCalibrationResult
    """
    est = AugmentedCalibrationGeneralizabilityEstimator(
        target=target,
        max_iter=max_iter,
        tol=tol,
        optimizer_kwargs=optimizer_kwargs or {},
        outcome_model=outcome_model,
        outcome_model_kwargs=outcome_model_kwargs or {},
        treatment_prob=treatment_prob,
    )
    est.fit(X=X, Y=Y, T=T, S=S)
    return est.estimate_ate()
