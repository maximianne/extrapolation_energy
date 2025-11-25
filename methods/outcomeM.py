# methods/outcome_modeling.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression


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
class OutcomeModelResult:
    """
    Result container for pure outcome-modeling generalizability estimator.

    Attributes
    ----------
    ate : float
        Estimated average treatment effect in the chosen target population.
    ate_treated : float
        Estimated mean potential outcome under treatment in the target population.
    ate_control : float
        Estimated mean potential outcome under control in the target population.
    m1 : ndarray, shape (n_samples,)
        Predicted potential outcome under treatment for all units.
    m0 : ndarray, shape (n_samples,)
        Predicted potential outcome under control for all units.
    """

    ate: float
    ate_treated: float
    ate_control: float
    m1: np.ndarray
    m0: np.ndarray


class OutcomeModelGeneralizabilityEstimator:
    """
    Pure outcome modeling estimator for extrapolating an ATE to a target population.

    Variables
    ---------
    - S: sample indicator (S=1 trial / experimental sample, S=0 target-only units)
    - T: treatment indicator (T=1 treated, T=0 control) in the trial
    - Y: outcome

    Idea
    ----
    1. Fit separate models using trial units (S=1):
        m_1(x) = E[Y | T=1, S=1, X=x]
        m_0(x) = E[Y | T=0, S=1, X=x]

    2. Predict both potential outcomes for all units (trial and non-trial):
        m_1(X_i), m_0(X_i)

    3. For a chosen target population G (e.g., S=0 or S in {0,1}):
        μ1^G = average of m_1(X_i) over i in G
        μ0^G = average of m_0(X_i) over i in G
        ATE^G = μ1^G - μ0^G

    Notes
    -----
    - No weighting or sampling-score modeling is used here.
    - This is the "pure" outcome model extrapolation approach.
    """

    def __init__(
        self,
        *,
        target: Literal["nontrial", "combined"] = "nontrial",
        outcome_model: Optional[RegressorMixin] = None,
        outcome_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        target : {"nontrial", "combined"}, default "nontrial"
            Target population definition:
            - "nontrial": target is units with S=0.
            - "combined": target is all units (S=0 and S=1).
        outcome_model : sklearn-like regressor, optional
            Base outcome model (must implement .fit(X, y) and .predict(X)).
            If None, default is LinearRegression(**outcome_model_kwargs).
            This base model is cloned separately for T=1 and T=0.
        outcome_model_kwargs : dict, optional
            Extra keyword arguments passed to LinearRegression if
            outcome_model is None.
        """
        self.target = target
        self.outcome_model = outcome_model
        self.outcome_model_kwargs = outcome_model_kwargs or {}

        # Fitted attributes
        self.m1_: Optional[np.ndarray] = None
        self.m0_: Optional[np.ndarray] = None
        self.S_: Optional[np.ndarray] = None
        self.model1_: Optional[RegressorMixin] = None
        self.model0_: Optional[RegressorMixin] = None

        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        S: np.ndarray,
    ) -> "OutcomeModelGeneralizabilityEstimator":
        """
        Fit the outcome models m_1(X) and m_0(X) using trial data (S=1).

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
        self : OutcomeModelGeneralizabilityEstimator
        """
        X_mat = _to_2d_array(X)
        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T).ravel()
        S = np.asarray(S).ravel()

        if not (X_mat.shape[0] == Y.shape[0] == T.shape[0] == S.shape[0]):
            raise ValueError("X, Y, T, and S must all have the same number of rows.")

        if self.outcome_model is None:
            base_model: RegressorMixin = LinearRegression(**self.outcome_model_kwargs)
        else:
            base_model = self.outcome_model

        # Use only trial units with observed outcomes
        mask_trial_obs = (S == 1) & ~np.isnan(Y)
        X_trial = X_mat[mask_trial_obs]
        Y_trial = Y[mask_trial_obs]
        T_trial = T[mask_trial_obs]

        # Fit model for treated (T=1)
        mask_treated = (T_trial == 1)
        if not np.any(mask_treated):
            raise RuntimeError("No treated trial units (T=1, S=1) to fit m_1(X).")

        model1 = clone(base_model)
        model1.fit(X_trial[mask_treated], Y_trial[mask_treated])
        m1 = model1.predict(X_mat)

        # Fit model for control (T=0)
        mask_control = (T_trial == 0)
        if not np.any(mask_control):
            raise RuntimeError("No control trial units (T=0, S=1) to fit m_0(X).")

        model0 = clone(base_model)
        model0.fit(X_trial[mask_control], Y_trial[mask_control])
        m0 = model0.predict(X_mat)

        # Store predictions and S for later aggregation
        self.m1_ = m1
        self.m0_ = m0
        self.S_ = S
        self.model1_ = model1
        self.model0_ = model0
        self._fitted = True

        return self

    def estimate_ate(self) -> OutcomeModelResult:
        """
        Compute ATE, μ1, and μ0 in the chosen target population
        using the fitted outcome models.

        Returns
        -------
        result : OutcomeModelResult
        """
        if not self._fitted:
            raise RuntimeError("Estimator must be fit before calling estimate_ate().")

        S = self.S_
        m1 = self.m1_
        m0 = self.m0_

        if S is None or m1 is None or m0 is None:
            raise RuntimeError("Internal state incomplete; call fit() first.")

        # Define target population
        if self.target == "nontrial":
            target_mask = (S == 0)
        elif self.target == "combined":
            target_mask = np.ones_like(S, dtype=bool)
        else:
            raise ValueError("target must be 'nontrial' or 'combined'")

        if not np.any(target_mask):
            raise ValueError("No units in chosen target population; check `target` and `S`.")

        N_tar = int(np.sum(target_mask))

        mu1_hat = float(np.sum(m1[target_mask]) / N_tar)
        mu0_hat = float(np.sum(m0[target_mask]) / N_tar)
        ate_hat = mu1_hat - mu0_hat

        return OutcomeModelResult(
            ate=ate_hat,
            ate_treated=mu1_hat,
            ate_control=mu0_hat,
            m1=m1,
            m0=m0,
        )


def estimate_ate_outcome(
    Y: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    *,
    target: Literal["nontrial", "combined"] = "nontrial",
    outcome_model: Optional[RegressorMixin] = None,
    outcome_model_kwargs: Optional[Dict[str, Any]] = None,
) -> OutcomeModelResult:
    """
    Convenience function: one-shot outcome-modeling ATE estimation.

    Parameters
    ----------
    Y : 1D ndarray
        Outcome.
    T : 1D ndarray
        Treatment indicator (1 = treated, 0 = control).
    S : 1D ndarray
        Sample indicator (1 = trial, 0 = non-trial).
    X : DataFrame or 2D ndarray
        Covariates for all units.

    target : {"nontrial", "combined"}, default "nontrial"
        Target population definition.
    outcome_model : sklearn-like regressor, optional
        Base regressor for m_1(X) and m_0(X).
    outcome_model_kwargs : dict, optional
        Extra kwargs if using default LinearRegression.

    Returns
    -------
    result : OutcomeModelResult
    """
    est = OutcomeModelGeneralizabilityEstimator(
        target=target,
        outcome_model=outcome_model,
        outcome_model_kwargs=outcome_model_kwargs or {},
    )
    est.fit(X=X, Y=Y, T=T, S=S)
    return est.estimate_ate()
