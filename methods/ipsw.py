# extrapolation/ipsw.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class IPSW:
    """
    Container for IPSW results.

    Attributes
    ----------
    ate : float
        Estimated average treatment effect.
    ate_treated : float
        Estimated mean outcome under treatment.
    ate_control : float
        Estimated mean outcome under control.
    weights : np.ndarray
        Final IPSW (or inverse-odds) weights for all observations.
    sampling_scores : np.ndarray
        Estimated sampling scores P(S=1 | X).
    weight_type : {"ipsw", "inverse_odds"}
        Type of weights used.
    stabilized : bool
        Whether stabilized weights were used.
    clip : Optional[(float, float)]
        Clipping bounds used for weights, if any.
    """
    ate: float
    ate_treated: float
    ate_control: float
    weights: np.ndarray
    sampling_scores: np.ndarray
    weight_type: Literal["ipsw", "inverse_odds"] = "ipsw"
    stabilized: bool = True
    clip: Optional[Tuple[float, float]] = None

def estimate_sampling_scores(
    X: pd.DataFrame | np.ndarray,
    s: np.ndarray,
    *,
    model=None,
    proba_method: str = "predict_proba",
    **model_kwargs,
) -> Tuple[np.ndarray, object]:
    """
    Estimate sampling scores P(S=1 | X) using any classification model.

    Parameters
    ----------
    X : DataFrame or 2D ndarray
        Covariates used to predict selection into the trial.
    s : 1D ndarray
        Sample indicator, 1 if in the trial / experimental sample, 0 otherwise.
    model : sklearn-like classifier, optional
        Any object with .fit(X, s) and either .predict_proba(X) or .predict(X),
        e.g. LogisticRegression, XGBClassifier, RandomForestClassifier, etc.
        If None, use LogisticRegression(**model_kwargs).
    proba_method : {"predict_proba", "predict"}, default "predict_proba"
        Method used to extract probabilities.
    model_kwargs :
        Extra keyword arguments passed to the model if model is None.

    Returns
    -------
    ps : 1D ndarray
        Estimated sampling scores P(S=1 | X).
    model : fitted model
    """
    # Ensure arrays
    X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    s = np.asarray(s).ravel()

    # Default model
    if model is None:
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            **model_kwargs,
        )

    # Fit
    model.fit(X_mat, s)

    # Get probabilities
    if proba_method == "predict_proba":
        if not hasattr(model, "predict_proba"):
            raise AttributeError("model has no predict_proba method")
        ps = model.predict_proba(X_mat)[:, 1]
    elif proba_method == "predict":
        if not hasattr(model, "predict"):
            raise AttributeError("model has no predict method")
        ps = model.predict(X_mat)
    else:
        raise ValueError("proba_method must be 'predict_proba' or 'predict'")

    return ps, model



def compute_ipsw_weights(
    s: np.ndarray,
    ps: np.ndarray,
    *,
    weight_type: Literal["ipsw", "inverse_odds"] = "ipsw",
    stabilized: bool = True,
    clip: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Compute inverse-probability-of-sampling weights for trial units.

    Parameters
    ----------
    s : 1D ndarray
        Sample indicator, 1 if in the trial, 0 otherwise.
    ps : 1D ndarray
        Estimated sampling scores P(S=1 | X).
    weight_type : {"ipsw", "inverse_odds"}, default "ipsw"
        - "ipsw": w_i ∝ 1 / ps_i  for S=1 units.
        - "inverse_odds": w_i ∝ (1 - ps_i) / ps_i for S=1 units
          (often used for generalizability to the S=0 population).
    stabilized : bool, default True
        If True, multiply by a stabilizing constant so that average weight is ~1.
        For ipsw: c = E[ps]; for inverse_odds: c = E[1 - ps].
    clip : (low, high), optional
        If provided, clip weights to [low, high] to reduce the influence of
        extreme weights.

    Returns
    -------
    w : 1D ndarray
        Weights for all observations. Units with S=0 have weight 0
        (since they don't contribute trial outcomes).
    """
    s = np.asarray(s).ravel()
    ps = np.asarray(ps).ravel()

    if ps.shape[0] != s.shape[0]:
        raise ValueError("ps and s must have the same length")

    w = np.zeros_like(ps, dtype=float)

    if weight_type == "ipsw":
        base = 1.0 / ps
        if stabilized:
            c = np.mean(ps)
            base = c * base
    elif weight_type == "inverse_odds":
        odds = (1.0 - ps) / ps
        if stabilized:
            c = np.mean(1.0 - ps)
            base = c * odds
        else:
            base = odds
    else:
        raise ValueError("weight_type must be 'ipsw' or 'inverse_odds'")

    # assign weights only to trial units
    w[s == 1] = base[s == 1]

    if clip is not None:
        low, high = clip
        w = np.clip(w, low, high)

    return w


def estimate_ate_ipsw(
    y: np.ndarray,
    a: np.ndarray,
    s: np.ndarray,
    ps: np.ndarray,
    *,
    weight_type: Literal["ipsw", "inverse_odds"] = "ipsw",
    stabilized: bool = True,
    clip: Optional[Tuple[float, float]] = None,
) -> IPSW:
    """
    Estimate an ATE using inverse-probability-of-sampling weights.

    This assumes:
      - You already have a randomized treatment within the trial (S=1),
      - You estimated P(S=1 | X) = ps,
      - You want to reweight the trial to a target population (e.g. S=0).

    Parameters
    ----------
    y : 1D ndarray
        Outcome.
    a : 1D ndarray
        Treatment indicator (1 = treated, 0 = control).
    s : 1D ndarray
        Sample indicator (1 = trial, 0 = non-trial).
    ps : 1D ndarray
        Estimated sampling scores P(S=1 | X).
    weight_type : {"ipsw", "inverse_odds"}, default "ipsw"
        See `compute_ipsw_weights`.
    stabilized : bool, default True
        See `compute_ipsw_weights`.
    clip : (low, high), optional
        See `compute_ipsw_weights`.

    Returns
    -------
    result : IPSW
        Dataclass with ATE, group means, weights, and sampling scores.
    """
    y = np.asarray(y).ravel()
    a = np.asarray(a).ravel()
    s = np.asarray(s).ravel()
    ps = np.asarray(ps).ravel()

    w = compute_ipsw_weights(
        s=s,
        ps=ps,
        weight_type=weight_type,
        stabilized=stabilized,
        clip=clip,
    )

    # Only use trial units (S=1) for outcomes
    mask_trial = (s == 1)
    w_trial = w[mask_trial]
    y_trial = y[mask_trial]
    a_trial = a[mask_trial]

    # Weighted means in treated and control groups
    w_treated = w_trial[a_trial == 1]
    y_treated = y_trial[a_trial == 1]

    w_control = w_trial[a_trial == 0]
    y_control = y_trial[a_trial == 0]

    mu1 = np.average(y_treated, weights=w_treated)
    mu0 = np.average(y_control, weights=w_control)
    ate_hat = mu1 - mu0

    return IPSW(
        ate=ate_hat,
        ate_treated=mu1,
        ate_control=mu0,
        weights=w,
        sampling_scores=ps,
        weight_type=weight_type,
        stabilized=stabilized,
        clip=clip,
    )
# ---------------------------------------------------------------------
# Estimator wrapper: lets you plug in logit / XGBoost / RF / trees
# ---------------------------------------------------------------------
class IPSWEstimator:
    """
    Estimator wrapper for IPSW generalizability.

    Example
    -------
    from xgboost import XGBClassifier

    est = IPSWEstimator(
        model=XGBClassifier(...),
        weight_type="inverse_odds",   # generalize to S=0
        stabilized=True,
        clip=(0.01, 50),
    )
    est.fit(X, s)
    result = est.estimate(y, a)

    result.ate
    """

    def __init__(
        self,
        model=None,
        *,
        proba_method: str = "predict_proba",
        weight_type: Literal["ipsw", "inverse_odds"] = "inverse_odds",
        stabilized: bool = True,
        clip: Optional[Tuple[float, float]] = None,
        **model_kwargs,
    ):
        self.model = model
        self.model_kwargs = model_kwargs
        self.proba_method = proba_method

        self.weight_type = weight_type
        self.stabilized = stabilized
        self.clip = clip

        # Fitted attributes
        self.ps_: Optional[np.ndarray] = None
        self.fitted_model_ = None
        self.s_: Optional[np.ndarray] = None
        self.result_: Optional[IPSW] = None

    def fit(self, X: pd.DataFrame | np.ndarray, s: np.ndarray):
        """
        Fit the sampling model P(S=1 | X).
        """
        ps, fitted_model = estimate_sampling_scores(
            X=X,
            s=s,
            model=self.model,
            proba_method=self.proba_method,
            **self.model_kwargs,
        )
        self.ps_ = ps
        self.fitted_model_ = fitted_model
        self.s_ = np.asarray(s).ravel()
        return self

    def estimate(self, y: np.ndarray, a: np.ndarray, s: Optional[np.ndarray] = None) -> IPSW:
        """
        Estimate ATE in the target population using IPSW.
        """
        if self.ps_ is None:
            raise RuntimeError("Call .fit(X, s) before .estimate(y, a, s).")

        s_used = np.asarray(s).ravel() if s is not None else self.s_

        result = estimate_ate_ipsw(
            y=y,
            a=a,
            s=s_used,
            ps=self.ps_,
            weight_type=self.weight_type,
            stabilized=self.stabilized,
            clip=self.clip,
        )
        self.result_ = result
        return result