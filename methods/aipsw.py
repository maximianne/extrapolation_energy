# extrapolation/aipsw.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression

from .ipsw import estimate_sampling_scores, compute_ipsw_weights


@dataclass
class AugmentedIPSWResult:
    """
    Container for Augmented IPSW (AIPSW) estimates.

    Attributes
    ----------
    ate : float
        Estimated average treatment effect in the chosen target population.
    ate_treated : float
        Estimated mean potential outcome under treatment in the target population.
    ate_control : float
        Estimated mean potential outcome under control in the target population.
    weights : ndarray
        Sampling weights applied to trial units (S=1). Non-trial units have weight 0.
    sampling_scores : ndarray
        Estimated sampling scores P(S=1 | X).
    m1 : ndarray
        Predicted potential outcome under treatment for all units.
    m0 : ndarray
        Predicted potential outcome under control for all units.
    e_hat : float
        Estimated (or provided) treatment probability within the trial.
    """

    ate: float
    ate_treated: float
    ate_control: float
    weights: np.ndarray
    sampling_scores: np.ndarray
    m1: np.ndarray
    m0: np.ndarray
    e_hat: float


def _to_2d_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Ensure X is a 2D numpy array (n_samples, n_features).
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    return X_arr


def fit_outcome_models(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    s: np.ndarray,
    *,
    outcome_model: Optional[RegressorMixin] = None,
    outcome_model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, RegressorMixin, RegressorMixin]:
    """
    Fit separate outcome models m_1(x) and m_0(x) on the trial sample (S=1).

    Parameters
    ----------
    X : DataFrame or 2D ndarray
        Covariates for all units (trial and non-trial).
    y : 1D ndarray
        Outcome; assumed only observed (or valid) for S=1.
        For S=0 you can pass np.nan or any placeholder; these observations
        are not used in fitting.
    a : 1D ndarray
        Treatment indicator (1 = treated, 0 = control).
    s : 1D ndarray
        Sample indicator (1 = trial, 0 = non-trial).
    outcome_model : sklearn-like regressor, optional
        Any object with .fit(X, y) and .predict(X). If None, LinearRegression is used.
    outcome_model_kwargs : dict, optional
        Extra keyword arguments for the outcome_model if it is None.

    Returns
    -------
    m0 : 1D ndarray
        Predicted potential outcomes under control, for all units.
    m1 : 1D ndarray
        Predicted potential outcomes under treatment, for all units.
    model0 : RegressorMixin
        Fitted model for control outcomes.
    model1 : RegressorMixin
        Fitted model for treated outcomes.
    """
    X_mat = _to_2d_array(X)
    y = np.asarray(y, dtype=float).ravel()
    a = np.asarray(a).ravel()
    s = np.asarray(s).ravel()

    if outcome_model_kwargs is None:
        outcome_model_kwargs = {}

    if outcome_model is None:
        base_model: RegressorMixin = LinearRegression(**outcome_model_kwargs)
    else:
        base_model = outcome_model

    # Use only trial units with non-missing outcomes
    mask_trial_obs = (s == 1) & ~np.isnan(y)
    X_trial = X_mat[mask_trial_obs]
    y_trial = y[mask_trial_obs]
    a_trial = a[mask_trial_obs]

    # Fit model for treated units
    model1 = clone(base_model)
    mask_treated = a_trial == 1
    model1.fit(X_trial[mask_treated], y_trial[mask_treated])
    m1 = model1.predict(X_mat)

    # Fit model for control units
    model0 = clone(base_model)
    mask_control = a_trial == 0
    model0.fit(X_trial[mask_control], y_trial[mask_control])
    m0 = model0.predict(X_mat)

    return m0, m1, model0, model1


def estimate_ate_aipsw(
    y: np.ndarray,
    a: np.ndarray,
    s: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    *,
    # Sampling model / scores
    ps: Optional[np.ndarray] = None,
    sampling_model: Optional[LogisticRegression] = None,
    sampling_model_kwargs: Optional[Dict[str, Any]] = None,
    # Outcome model
    outcome_model: Optional[RegressorMixin] = None,
    outcome_model_kwargs: Optional[Dict[str, Any]] = None,
    # IPSW weight options
    weight_type: Literal["ipsw", "inverse_odds"] = "inverse_odds",
    stabilized: bool = True,
    clip: Optional[Tuple[float, float]] = None,
    # Treatment assignment probability in the trial
    treatment_prob: Optional[float] = None,
    # Target population definition
    target: Literal["nontrial", "combined"] = "nontrial",
) -> AugmentedIPSWResult:
    """
    Estimate an ATE in a target population using Augmented IPSW (AIPSW).

    This is a doubly-robust extrapolation estimator that combines:
      (i) reweighting of the trial using sampling scores P(S=1 | X), and
      (ii) an outcome model m_a(X) for each treatment arm.

    Assumptions
    -----------
    - Treatment is randomized within the trial (S=1) with probability e = P(A=1 | S=1),
      which can be known (e.g., from design) or estimated from data.
    - Sampling scores P(S=1 | X) are either provided (ps) or estimated via logistic
      regression using `estimate_sampling_scores`.
    - Outcome model is fitted only on trial units (S=1).

    Parameters
    ----------
    y : 1D ndarray
        Outcome. For non-trial units (S=0), values can be np.nan or ignored.
    a : 1D ndarray
        Treatment indicator (1 = treated, 0 = control).
    s : 1D ndarray
        Sample indicator (1 = trial, 0 = non-trial).
    X : DataFrame or 2D ndarray
        Covariates for both trial and non-trial units.

    ps : 1D ndarray, optional
        Pre-estimated sampling scores P(S=1 | X). If None, they are estimated
        with `estimate_sampling_scores`.
    sampling_model : LogisticRegression, optional
        If provided and ps is None, this model is fit to estimate sampling scores.
    sampling_model_kwargs : dict, optional
        Extra keyword arguments passed to LogisticRegression if ps is None and
        sampling_model is None.

    outcome_model : sklearn-like regressor, optional
        Outcome model for m_a(X). If None, LinearRegression is used.
    outcome_model_kwargs : dict, optional
        Extra keyword arguments passed to the outcome model if outcome_model is None.

    weight_type : {"ipsw", "inverse_odds"}, default "inverse_odds"
        Type of sampling weights:
        - "ipsw": w_i ∝ 1 / ps_i
        - "inverse_odds": w_i ∝ (1 - ps_i) / ps_i
        Typically "inverse_odds" is used to map trial to the non-trial population.
    stabilized : bool, default True
        Whether to use stabilized sampling weights.
    clip : (low, high), optional
        If provided, clip weights to [low, high].

    treatment_prob : float, optional
        Known treatment probability e = P(A=1 | S=1). If None, it is estimated
        as the empirical mean of A among trial units with observed outcomes.

    target : {"nontrial", "combined"}, default "nontrial"
        Defines the target population:
        - "nontrial": target is units with S=0.
        - "combined": target is all units (S=0 and S=1).

    Returns
    -------
    result : AugmentedIPSWResult
        Dataclass with ATE and related quantities.
    """
    y = np.asarray(y, dtype=float).ravel()
    a = np.asarray(a).ravel()
    s = np.asarray(s).ravel()
    X_mat = _to_2d_array(X)

    if sampling_model_kwargs is None:
        sampling_model_kwargs = {}
    if outcome_model_kwargs is None:
        outcome_model_kwargs = {}

    # 1) Sampling scores P(S=1 | X)
    if ps is None:
        ps, _sampling_model = estimate_sampling_scores(
            X_mat,
            s,
            model=sampling_model,
            **sampling_model_kwargs,
        )
    else:
        ps = np.asarray(ps).ravel()

    # 2) Sampling weights (IPS weights for the trial)
    w = compute_ipsw_weights(
        s=s,
        ps=ps,
        weight_type=weight_type,
        stabilized=stabilized,
        clip=clip,
    )

    # 3) Outcome models m_0(X), m_1(X)
    m0, m1, _model0, _model1 = fit_outcome_models(
        X=X_mat,
        y=y,
        a=a,
        s=s,
        outcome_model=outcome_model,
        outcome_model_kwargs=outcome_model_kwargs,
    )

    # 4) Treatment probability within the trial
    mask_trial_obs = (s == 1) & ~np.isnan(y)
    if treatment_prob is None:
        e_hat = float(np.mean(a[mask_trial_obs]))
    else:
        e_hat = float(treatment_prob)

    # 5) Define target population index set
    if target == "nontrial":
        target_mask = (s == 0)
    elif target == "combined":
        target_mask = np.ones_like(s, dtype=bool)
    else:
        raise ValueError("target must be 'nontrial' or 'combined'")

    N_tar = int(np.sum(target_mask))
    if N_tar == 0:
        raise ValueError("No units in chosen target population; check `target` and `s`.")

    # 6) Plug-in term: average of predicted effects in the target population
    plug_in = np.sum(m1[target_mask] - m0[target_mask]) / N_tar

    # 7) Residual correction term using trial units and sampling weights
    w_trial = w[mask_trial_obs]
    y_trial = y[mask_trial_obs]
    a_trial = a[mask_trial_obs]
    m1_trial = m1[mask_trial_obs]
    m0_trial = m0[mask_trial_obs]

    # AIPSW correction term:
    # (1 / N_tar) * sum_{i in trial} w_i * [
    #   (A_i / e_hat) * (Y_i - m1_i) - ((1 - A_i) / (1 - e_hat)) * (Y_i - m0_i)
    # ]
    correction = np.sum(
        w_trial
        * (
            (a_trial / e_hat) * (y_trial - m1_trial)
            - ((1.0 - a_trial) / (1.0 - e_hat)) * (y_trial - m0_trial)
        )
    ) / N_tar

    ate_hat = plug_in + correction

    # 8) DR estimates of μ1 and μ0 in the target population (optional but helpful)
    mu1_hat = (
        np.sum(m1[target_mask]) / N_tar
        + np.sum(w_trial * (a_trial / e_hat) * (y_trial - m1_trial)) / N_tar
    )

    mu0_hat = (
        np.sum(m0[target_mask]) / N_tar
        + np.sum(
            w_trial
            * ((1.0 - a_trial) / (1.0 - e_hat))
            * (y_trial - m0_trial)
        )
        / N_tar
    )

    return AugmentedIPSWResult(
        ate=ate_hat,
        ate_treated=mu1_hat,
        ate_control=mu0_hat,
        weights=w,
        sampling_scores=ps,
        m1=m1,
        m0=m0,
        e_hat=e_hat,
    )
