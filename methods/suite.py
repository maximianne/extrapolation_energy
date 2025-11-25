# methods/suite.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable, Tuple

import numpy as np
import pandas as pd

from .ipsw import estimate_ate_ipsw
from .aipsw import estimate_ate_aipsw
from .stratification import estimate_ate_stratified
from .outcomeM import estimate_ate_outcome
from .acalibration import (
    estimate_ate_calibration,
    estimate_ate_calibration_augmented,
)


@dataclass
class SuiteEstimatorResult:
    """
    One row of results for a given estimator.
    """
    name: str
    ate: float
    se: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]


class GeneralizabilitySuite:
    """
    Run a suite of generalizability estimators on (Y, T, S, X).

    Estimators included
    -------------------
    - IPSW
    - Augmented IPSW
    - Stratification
    - Outcome modeling
    - Calibration weighting
    - Augmented calibration weighting

    Notation
    --------
    - Y : outcome
    - T : treatment (1/0)
    - S : sample indicator (1 = trial, 0 = non-trial)
    - X : covariates
    """

    def __init__(
        self,
        *,
        target: str = "nontrial",
        n_strata: int = 5,
        ipsw_kwargs: Optional[Dict[str, Any]] = None,
        aipsw_kwargs: Optional[Dict[str, Any]] = None,
        strat_kwargs: Optional[Dict[str, Any]] = None,
        outcome_kwargs: Optional[Dict[str, Any]] = None,
        calib_kwargs: Optional[Dict[str, Any]] = None,
        aug_calib_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        target : {"nontrial", "combined"}, default "nontrial"
            Target population definition for methods that need it.
        n_strata : int, default 5
            Number of strata for stratification estimator.
        *_kwargs : dict, optional
            Extra keyword arguments passed into each estimator's convenience
            function (e.g., outcome_model=..., sampling_model=..., etc.).
        """
        self.target = target
        self.n_strata = n_strata
        self.ipsw_kwargs = ipsw_kwargs or {}
        self.aipsw_kwargs = aipsw_kwargs or {}
        self.strat_kwargs = strat_kwargs or {}
        self.outcome_kwargs = outcome_kwargs or {}
        self.calib_kwargs = calib_kwargs or {}
        self.aug_calib_kwargs = aug_calib_kwargs or {}

    # ------------------------------------------------------------------
    # Internal helper: one estimator
    # ------------------------------------------------------------------
    def _run_one(
        self,
        name: str,
        Y: np.ndarray,
        T: np.ndarray,
        S: np.ndarray,
        X: pd.DataFrame | np.ndarray,
    ) -> float:
        """
        Run a single estimator and return its ATE.
        """
        if name == "IPSW":
            res = estimate_ate_ipsw(
                y=Y,
                a=T,
                s=S,
                **self.ipsw_kwargs,
            )
            return float(res.ate)

        elif name == "Augmented IPSW":
            res = estimate_ate_aipsw(
                y=Y,
                a=T,
                s=S,
                X=X,
                target=self.target,
                **self.aipsw_kwargs,
            )
            return float(res.ate)

        elif name == "Stratification":
            res = estimate_ate_stratified(
                Y=Y,
                T=T,
                S=S,
                X=X,
                n_strata=self.n_strata,
                target=self.target,
                **self.strat_kwargs,
            )
            return float(res.ate)

        elif name == "Outcome modeling":
            res = estimate_ate_outcome(
                Y=Y,
                T=T,
                S=S,
                X=X,
                target=self.target,
                **self.outcome_kwargs,
            )
            return float(res.ate)

        elif name == "Calibration weighting":
            res = estimate_ate_calibration(
                Y=Y,
                T=T,
                S=S,
                X=X,
                target=self.target,
                **self.calib_kwargs,
            )
            return float(res.ate)

        elif name == "Augmented calibration":
            res = estimate_ate_calibration_augmented(
                Y=Y,
                T=T,
                S=S,
                X=X,
                target=self.target,
                **self.aug_calib_kwargs,
            )
            return float(res.ate)

        else:
            raise ValueError(f"Unknown estimator name: {name}")

    # ------------------------------------------------------------------
    # Bootstrap helper
    # ------------------------------------------------------------------
    def _bootstrap_estimator(
        self,
        name: str,
        Y: np.ndarray,
        T: np.ndarray,
        S: np.ndarray,
        X: pd.DataFrame | np.ndarray,
        *,
        n_boot: int,
        alpha: float,
        random_state: Optional[int] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Bootstrap SE and CI for a single estimator.

        Returns
        -------
        (se, ci_lower, ci_upper)
        """
        n = Y.shape[0]
        rng = np.random.default_rng(random_state)

        boot_ates: List[float] = []

        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)

            Y_b = Y[idx]
            T_b = T[idx]
            S_b = S[idx]

            if isinstance(X, pd.DataFrame):
                X_b = X.iloc[idx].reset_index(drop=True)
            else:
                X_b = X[idx, :]

            try:
                ate_b = self._run_one(name, Y_b, T_b, S_b, X_b)
                if np.isfinite(ate_b):
                    boot_ates.append(ate_b)
            except Exception:
                # Some bootstrap samples may fail (e.g., no treated in a stratum).
                # We just skip them.
                continue

        if len(boot_ates) < 5:
            # Not enough successful bootstrap draws
            return None, None, None

        boot_arr = np.array(boot_ates)
        se = float(boot_arr.std(ddof=1))
        lower = float(np.percentile(boot_arr, 100 * (alpha / 2)))
        upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))

        return se, lower, upper

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def run(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        S: np.ndarray,
        X: pd.DataFrame | np.ndarray,
        *,
        estimators: Optional[List[str]] = None,
        n_boot: int = 0,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run the suite of estimators and (optionally) bootstrap CIs.

        Parameters
        ----------
        Y, T, S, X :
            Data arrays / DataFrame (consistent with your other modules).
        estimators : list of str, optional
            Subset of estimators to run. Default is all.
        n_boot : int, default 0
            Number of bootstrap replications. If 0, no bootstrap is run.
        alpha : float, default 0.05
            Significance level for confidence intervals (e.g. 0.05 -> 95% CI).
        random_state : int, optional
            Seed for reproducibility of bootstrap.

        Returns
        -------
        df_results : DataFrame
            One row per estimator with columns:
            ["estimator", "ate", "se", "ci_lower", "ci_upper"]
        """
        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T).ravel()
        S = np.asarray(S).ravel()

        if estimators is None:
            estimators = [
                "IPSW",
                "Augmented IPSW",
                "Stratification",
                "Outcome modeling",
                "Calibration weighting",
                "Augmented calibration",
            ]

        results: List[SuiteEstimatorResult] = []

        for name in estimators:
            ate = self._run_one(name, Y, T, S, X)

            if n_boot > 0:
                se, ci_l, ci_u = self._bootstrap_estimator(
                    name,
                    Y,
                    T,
                    S,
                    X,
                    n_boot=n_boot,
                    alpha=alpha,
                    random_state=None if random_state is None else random_state + hash(name) % 10000,
                )
            else:
                se, ci_l, ci_u = None, None, None

            results.append(
                SuiteEstimatorResult(
                    name=name,
                    ate=ate,
                    se=se,
                    ci_lower=ci_l,
                    ci_upper=ci_u,
                )
            )

        df = pd.DataFrame(
            {
                "estimator": [r.name for r in results],
                "ate": [r.ate for r in results],
                "se": [r.se for r in results],
                "ci_lower": [r.ci_lower for r in results],
                "ci_upper": [r.ci_upper for r in results],
            }
        )
        return df

    # ------------------------------------------------------------------
    # LaTeX helper
    # ------------------------------------------------------------------
    @staticmethod
    def to_latex_table(
        df_results: pd.DataFrame,
        *,
        caption: str = "Generalizability estimators",
        label: str = "tab:generalizability",
        float_format: str = "{:.3f}",
    ) -> str:
        """
        Convert suite results into a LaTeX tabular environment.

        Columns: Estimator, ATE, (SE), [CI_lower, CI_upper]

        Returns
        -------
        latex_str : str
        """
        lines: List[str] = []
        lines.append("\\begin{table}[htbp]")
        lines.append("  \\centering")
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{{label}}}")
        lines.append("  \\begin{tabular}{lccc}")
        lines.append("    \\toprule")
        lines.append("    Estimator & ATE & SE & 95\\% CI \\\\")
        lines.append("    \\midrule")

        for _, row in df_results.iterrows():
            est = row["estimator"]
            ate = float_format.format(row["ate"])
            if pd.isna(row["se"]):
                se = "--"
                ci_str = "--"
            else:
                se = float_format.format(row["se"])
                ci_l = float_format.format(row["ci_lower"])
                ci_u = float_format.format(row["ci_upper"])
                ci_str = f"[{ci_l}, {ci_u}]"

            lines.append(f"    {est} & {ate} & {se} & {ci_str} \\\\")

        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)
