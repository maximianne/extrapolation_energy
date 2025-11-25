
from __future__ import annotations

from typing import Callable, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.utils import resample

def run_ate_with_bootstrap(
    estimator: Callable[..., Any],
    Y,
    T,
    S,
    X,
    *,
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: Optional[int] = 42,
    **estimator_kwargs,
) -> Tuple[float, float, float, float]:
    """
    Compute ATE, bootstrap SE, and CI for a single estimator.

    Parameters
    ----------
    estimator : callable
        Function implementing your estimator.
        Must have signature: estimator(Y, T, S, X, **kwargs)
        and return either:
          - a scalar ATE, or
          - an object with attribute `.ate`.
    Y, T, S, X :
        Outcome, treatment, sample indicator, covariates.
        Y, T, S can be array-like; X can be pandas DataFrame or ndarray.
    n_boot : int, default 500
        Number of bootstrap replications.
    alpha : float, default 0.05
        Significance level for CI (0.05 => 95% CI).
    random_state : int or None, default 42
        Seed for reproducibility.
    estimator_kwargs :
        Extra keyword arguments passed directly to `estimator`.

    Returns
    -------
    ate_hat : float
        Point estimate of the ATE from the full sample.
    se : float
        Bootstrap standard error.
    ci_lower : float
        Lower bound of (1 - alpha) CI.
    ci_upper : float
        Upper bound of (1 - alpha) CI.
    """
    rng = np.random.default_rng(random_state)

    # Convert inputs
    Y = np.asarray(Y)
    T = np.asarray(T)
    S = np.asarray(S)

    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        X_is_df = True
    else:
        X_arr = np.asarray(X)
        X_is_df = False

    n = Y.shape[0]

    # 1) Full-sample estimate
    full_res = estimator(Y, T, S, X, **estimator_kwargs)
    ate_hat = float(full_res.ate) if hasattr(full_res, "ate") else float(full_res)

    # 2) Bootstrap
    boot_ates = []

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)

        Y_b = Y[idx]
        T_b = T[idx]
        S_b = S[idx]

        if X_is_df:
            X_b = X.iloc[idx].reset_index(drop=True)
        else:
            X_b = X_arr[idx, :]

        try:
            res_b = estimator(Y_b, T_b, S_b, X_b, **estimator_kwargs)
            ate_b = float(res_b.ate) if hasattr(res_b, "ate") else float(res_b)
            if np.isfinite(ate_b):
                boot_ates.append(ate_b)
        except Exception:
            # Some resamples may fail (e.g., no treated/control units)
            continue

    if len(boot_ates) < 5:
        raise RuntimeError(
            f"Too few successful bootstrap draws ({len(boot_ates)}). "
            "Try increasing sample size or reducing n_boot."
        )

    boot_ates = np.asarray(boot_ates)
    se = float(boot_ates.std(ddof=1))
    ci_lower = float(np.percentile(boot_ates, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_ates, 100 * (1 - alpha / 2)))

    return ate_hat, se, ci_lower, ci_upper


def run_clustered_bootstrap(
    estimator_fn,
    Y: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    X: pd.DataFrame,
    cluster_ids: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
):
    """
    Clustered bootstrap for ATE estimators.
    
    Parameters
    ----------
    estimator_fn : function
        Function that returns ATE when passed (Y, T, S, X)
    Y, T, S : np.ndarray
        Outcome, treatment, and sample indicator arrays
    X : pd.DataFrame
        Covariates
    cluster_ids : np.ndarray
        Cluster/group IDs (e.g., experiment number) of same length as Y
    n_boot : int
        Number of bootstrap reps
    alpha : float
        Confidence level (e.g., 0.05 for 95%)
    
    Returns
    -------
    ate_mean, se, ci_lower, ci_upper
    """
    rng = np.random.default_rng(random_state)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    boot_ates = []

    for _ in range(n_boot):
        # Sample cluster IDs with replacement
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)

        # Collect the indices of all rows from sampled clusters
        sampled_idx = np.concatenate([
            np.where(cluster_ids == cid)[0]
            for cid in sampled_clusters
        ])

        # Bootstrap sample
        Y_b = Y[sampled_idx]
        T_b = T[sampled_idx]
        S_b = S[sampled_idx]
        X_b = X.iloc[sampled_idx]

        # Estimate ATE
        try:
            ate_b = estimator_fn(Y_b, T_b, S_b, X_b)
            boot_ates.append(ate_b)
        except Exception:
            continue  # skip failed bootstrap samples

    boot_ates = np.array(boot_ates)
    ate_mean = np.mean(boot_ates)
    se = np.std(boot_ates, ddof=1)

    lo = np.percentile(boot_ates, 100 * (alpha / 2))
    hi = np.percentile(boot_ates, 100 * (1 - alpha / 2))

    return ate_mean, se, lo, hi

def run_clustered_bootstrap(
    estimator_fn: Callable,
    Y: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    X: pd.DataFrame,
    cluster_ids: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Run clustered bootstrap to estimate ATE, SE, and confidence interval.
    """
    np.random.seed(random_state)

    df = pd.DataFrame(X.copy())
    df["Y"] = Y
    df["T"] = T
    df["S"] = S
    df["cluster_id"] = cluster_ids

    clusters = df["cluster_id"].unique()
    n_clusters = len(clusters)
    boot_ates = []

    for _ in range(n_boot):
        sampled_clusters = resample(clusters, replace=True, n_samples=n_clusters)
        boot_df = pd.concat([df[df["cluster_id"] == cid] for cid in sampled_clusters], ignore_index=True)

        try:
            ate_result = estimator_fn(
                Y=boot_df["Y"].values,
                T=boot_df["T"].values,
                S=boot_df["S"].values,
                X=boot_df[X.columns]
            )
            # === FIX ===
            if hasattr(ate_result, "ate"):
                ate = ate_result.ate
            elif isinstance(ate_result, tuple):
                ate = ate_result[0]
            else:
                ate = ate_result

            boot_ates.append(ate)
        except Exception:
            continue

    boot_ates = np.array(boot_ates)

    # Final estimate on full sample
    final_result = estimator_fn(Y, T, S, X)
    if hasattr(final_result, "ate"):
        ate_hat = final_result.ate
    elif isinstance(final_result, tuple):
        ate_hat = final_result[0]
    else:
        ate_hat = final_result

    se = np.std(boot_ates, ddof=1)
    ci_low = np.percentile(boot_ates, 100 * alpha / 2)
    ci_high = np.percentile(boot_ates, 100 * (1 - alpha / 2))

    return float(ate_hat), float(se), float(ci_low), float(ci_high)