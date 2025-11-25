# extrapolation/stratification.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .ipsw import estimate_sampling_scores


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


@dataclass
class StratificationResult:
    """
    Result container for stratification-based ATE estimation.

    Attributes
    ----------
    ate : float
        Overall stratified ATE in the chosen target population.
    ate_treated : float
        Estimated mean potential outcome under treatment in the target population.
    ate_control : float
        Estimated mean potential outcome under control in the target population.
    stratum_ates : ndarray
        Stratum-specific ATEs (may contain np.nan if a stratum has no treated or control).
    stratum_mu1 : ndarray
        Stratum-specific treated means E[Y | T=1, S=1, stratum].
    stratum_mu0 : ndarray
        Stratum-specific control means E[Y | T=0, S=1, stratum].
    stratum_weights : ndarray
        Weights for each stratum based on its share in the target population.
    stratum_sizes_trial : ndarray
        Number of trial units (S=1) in each stratum.
    stratum_sizes_target : ndarray
        Number of target-population units in each stratum.
    ps : ndarray
        Sampling scores P(S=1 | X) used to define strata.
    strata : ndarray
        Integer stratum assignment for each observation (0, 1, ..., n_strata-1).
    cutpoints : ndarray
        Cutpoints used to define the strata (quantiles of ps).
    """

    ate: float
    ate_treated: float
    ate_control: float
    stratum_ates: np.ndarray
    stratum_mu1: np.ndarray
    stratum_mu0: np.ndarray
    stratum_weights: np.ndarray
    stratum_sizes_trial: np.ndarray
    stratum_sizes_target: np.ndarray
    ps: np.ndarray
    strata: np.ndarray
    cutpoints: np.ndarray


class StratifiedGeneralizabilityEstimator:
    """
    Stratification-based estimator for extrapolating an ATE to a target population.

    Variables
    ---------
    - S: sample indicator (S=1 trial / experimental sample, S=0 target-only units)
    - T: treatment indicator (T=1 treated, T=0 control) in the trial
    - Y: outcome

    Workflow
    --------
    1. Fit:
        - Estimate sampling scores P(S=1 | X) (unless ps is provided).
        - Form K strata using quantiles of P(S=1 | X).
        - Compute stratum weights based on target population choice.

    2. Estimate:
        - Using trial units within each stratum, compute:
            tau_k = E[Y | T=1, S=1, stratum=k] - E[Y | T=0, S=1, stratum=k].
        - Aggregate:
            ATE = sum_k w_k * tau_k,
          where w_k are the stratum weights from the target population.

    Assumptions
    -----------
    - Treatment T is randomized within the trial (S=1).
    - No outcome model and no IPS weights are used here; this is
      a "pure" stratification / subclassification estimator.
    """

    def __init__(
        self,
        n_strata: int = 5,
        *,
        target: Literal["nontrial", "combined"] = "nontrial",
        sampling_model: Optional[LogisticRegression] = None,
        sampling_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_strata : int, default 5
            Number of strata (subclasses) on P(S=1 | X).
        target : {"nontrial", "combined"}, default "nontrial"
            Target population definition:
            - "nontrial": target is units with S=0.
            - "combined": target is all units (S=0 and S=1).
        sampling_model : LogisticRegression or compatible classifier, optional
            Model used to estimate P(S=1 | X) if ps is not provided.
        sampling_model_kwargs : dict, optional
            Extra keyword arguments passed to LogisticRegression if both
            ps is None and sampling_model is None.
        """
        if n_strata < 1:
            raise ValueError("n_strata must be >= 1")

        self.n_strata = n_strata
        self.target = target
        self.sampling_model = sampling_model
        self.sampling_model_kwargs = sampling_model_kwargs or {}

        # Fitted attributes
        self.ps_: Optional[np.ndarray] = None
        self.strata_: Optional[np.ndarray] = None
        self.cutpoints_: Optional[np.ndarray] = None
        self.stratum_weights_: Optional[np.ndarray] = None
        self.stratum_sizes_target_: Optional[np.ndarray] = None

        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        S: np.ndarray,
        *,
        ps: Optional[np.ndarray] = None,
    ) -> "StratifiedGeneralizabilityEstimator":
        """
        Fit the stratification structure: sampling scores, strata, and stratum weights.

        Parameters
        ----------
        X : DataFrame or 2D ndarray
            Covariates for all units.
        S : 1D ndarray
            Sample indicator (1 = trial, 0 = non-trial).
        ps : 1D ndarray, optional
            Precomputed sampling scores P(S=1 | X). If not provided, they are
            estimated via logistic regression using `estimate_sampling_scores`.

        Returns
        -------
        self : StratifiedGeneralizabilityEstimator
        """
        X_mat = _to_2d_array(X)
        S = np.asarray(S).ravel()

        # 1) Sampling scores
        if ps is None:
            ps, _ = estimate_sampling_scores(
                X_mat,
                S,
                model=self.sampling_model,
                **self.sampling_model_kwargs,
            )
        else:
            ps = np.asarray(ps).ravel()

        if ps.shape[0] != S.shape[0]:
            raise ValueError("ps and S must have the same length")

        self.ps_ = ps

        # 2) Define strata based on quantiles of ps
        quantiles = np.linspace(0.0, 1.0, self.n_strata + 1)
        cutpoints = np.quantile(ps, quantiles)
        self.cutpoints_ = cutpoints

        # Assign each observation to a stratum 0, ..., n_strata-1
        strata = np.searchsorted(cutpoints[1:-1], ps, side="right")
        strata = np.clip(strata, 0, self.n_strata - 1)
        self.strata_ = strata

        # 3) Compute stratum weights in the chosen target population
        if self.target == "nontrial":
            target_mask = (S == 0)
        elif self.target == "combined":
            target_mask = np.ones_like(S, dtype=bool)
        else:
            raise ValueError("target must be 'nontrial' or 'combined'")

        if not np.any(target_mask):
            raise ValueError("No units in chosen target population; check `target` and `S`.")

        N_tar = int(np.sum(target_mask))

        stratum_weights = np.zeros(self.n_strata, dtype=float)
        stratum_sizes_target = np.zeros(self.n_strata, dtype=int)

        for k in range(self.n_strata):
            mask_k_tar = target_mask & (strata == k)
            stratum_sizes_target[k] = int(np.sum(mask_k_tar))
            stratum_weights[k] = stratum_sizes_target[k] / N_tar

        self.stratum_weights_ = stratum_weights
        self.stratum_sizes_target_ = stratum_sizes_target

        self._fitted = True
        return self

    def estimate_ate(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        S: np.ndarray,
    ) -> StratificationResult:
        """
        Estimate a stratified ATE using the fitted strata and weights.

        Parameters
        ----------
        Y : 1D ndarray
            Outcome; assumed observed for S=1. For S=0 values can be np.nan.
        T : 1D ndarray
            Treatment indicator (1 = treated, 0 = control).
        S : 1D ndarray
            Sample indicator (1 = trial, 0 = non-trial).

        Returns
        -------
        result : StratificationResult
        """
        if not self._fitted:
            raise RuntimeError("Estimator must be fit before calling estimate_ate().")

        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T).ravel()
        S = np.asarray(S).ravel()

        if any(arr.shape[0] != Y.shape[0] for arr in [T, S, self.ps_, self.strata_]):
            raise ValueError("All input arrays must have the same length as Y.")

        ps = self.ps_
        strata = self.strata_
        cutpoints = self.cutpoints_
        w_strata = self.stratum_weights_
        n_strata = self.n_strata

        # Arrays to store per-stratum stats
        stratum_mu1 = np.full(n_strata, np.nan, dtype=float)
        stratum_mu0 = np.full(n_strata, np.nan, dtype=float)
        stratum_ates = np.full(n_strata, np.nan, dtype=float)
        stratum_sizes_trial = np.zeros(n_strata, dtype=int)

        # Use only trial units with observed outcomes
        mask_trial = (S == 1) & ~np.isnan(Y)

        for k in range(n_strata):
            mask_k = mask_trial & (strata == k)
            stratum_sizes_trial[k] = int(np.sum(mask_k))
            if stratum_sizes_trial[k] == 0:
                # No trial units in this stratum
                continue

            Y_k = Y[mask_k]
            T_k = T[mask_k]

            mask_treated = (T_k == 1)
            mask_control = (T_k == 0)

            if not np.any(mask_treated) or not np.any(mask_control):
                # Can't estimate a within-stratum ATE if no treated or no control
                continue

            mu1_k = np.mean(Y_k[mask_treated])
            mu0_k = np.mean(Y_k[mask_control])

            stratum_mu1[k] = mu1_k
            stratum_mu0[k] = mu0_k
            stratum_ates[k] = mu1_k - mu0_k

        # Handle strata where ATE is undefined (lack of treated or control)
        valid = ~np.isnan(stratum_ates)
        if not np.any(valid):
            raise RuntimeError("No stratum has both treated and control trial units; cannot estimate ATE.")

        # Renormalize weights over strata with valid ATEs
        w_valid = w_strata[valid]
        if w_valid.sum() == 0:
            raise RuntimeError("Stratum weights sum to zero over valid strata; check target population or strata.")

        w_valid = w_valid / w_valid.sum()

        ate_hat = np.sum(w_valid * stratum_ates[valid])
        mu1_hat = np.sum(w_valid * stratum_mu1[valid])
        mu0_hat = np.sum(w_valid * stratum_mu0[valid])

        return StratificationResult(
            ate=ate_hat,
            ate_treated=mu1_hat,
            ate_control=mu0_hat,
            stratum_ates=stratum_ates,
            stratum_mu1=stratum_mu1,
            stratum_mu0=stratum_mu0,
            stratum_weights=w_strata,
            stratum_sizes_trial=stratum_sizes_trial,
            stratum_sizes_target=self.stratum_sizes_target_,
            ps=ps,
            strata=strata,
            cutpoints=cutpoints,
        )


def estimate_ate_stratified(
    Y: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    *,
    n_strata: int = 5,
    target: Literal["nontrial", "combined"] = "nontrial",
    ps: Optional[np.ndarray] = None,
    sampling_model: Optional[LogisticRegression] = None,
    sampling_model_kwargs: Optional[Dict[str, Any]] = None,
) -> StratificationResult:
    """
    Convenience function: one-shot stratified ATE estimation with (S, T, Y).

    Parameters
    ----------
    Y : 1D ndarray
        Outcome.
    T : 1D ndarray
        Treatment indicator (1 = treated, 0 = control).
    S : 1D ndarray
        Sample indicator (1 = trial, 0 = non-trial).
    X : DataFrame or 2D ndarray
        Covariates used for P(S=1 | X) and to define strata.

    n_strata : int, default 5
        Number of strata.
    target : {"nontrial", "combined"}, default "nontrial"
        Target population definition.
    ps : 1D ndarray, optional
        Precomputed P(S=1 | X).
    sampling_model : LogisticRegression or compatible classifier, optional
        Model used to estimate P(S=1 | X) if ps is None.
    sampling_model_kwargs : dict, optional
        Extra kwargs for sampling_model (or default logistic).

    Returns
    -------
    result : StratificationResult
    """
    est = StratifiedGeneralizabilityEstimator(
        n_strata=n_strata,
        target=target,
        sampling_model=sampling_model,
        sampling_model_kwargs=sampling_model_kwargs or {},
    )
    est.fit(X=X, S=S, ps=ps)
    return est.estimate_ate(Y=Y, T=T, S=S)
