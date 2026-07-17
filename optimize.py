"""
optimize.py — Minimum-variance portfolio optimization.

This is the piece that wasn't in the original project: the actual solver.
The Excel workbook had the *outputs* of this optimization (weights, variance,
Sharpe) but the solve itself was done "outside Excel" and never saved.

Objective (long-only minimum-variance QP):

    minimize    w' Σ w
    subject to  sum(w) == 1
                0 <= w_i <= max_weight   for all i

Note there is deliberately no expected-return term in the objective — that's
what makes this "minimum-variance" rather than full mean-variance (Markowitz)
optimization. It avoids relying on noisy expected-return forecasts, at the
cost of not directly targeting the best risk-adjusted return.

Also included:
    - efficient_frontier(): sweep a target-return constraint to trace the
      frontier (min-variance is just one point on it, at the frontier's
      minimum-vol end).
    - max_sharpe(): the tangency portfolio, for comparison.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_MONTHS_PER_YEAR = 12


@dataclass
class OptimizationResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    effective_n: float

    def summary(self) -> str:
        lines = [
            "Weights:",
            self.weights.round(4).to_string(),
            "",
            f"Expected Annual Return : {self.expected_return:.4%}",
            f"Annual Volatility      : {self.volatility:.4%}",
            f"Sharpe Ratio            : {self.sharpe:.4f}",
            f"Effective # of Assets   : {self.effective_n:.2f}",
        ]
        return "\n".join(lines)


def annualize_stats(monthly_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualized mean return vector and covariance matrix from monthly returns."""
    mu = monthly_returns.mean() * TRADING_MONTHS_PER_YEAR
    cov = monthly_returns.cov() * TRADING_MONTHS_PER_YEAR
    return mu, cov


def _effective_n(weights: np.ndarray) -> float:
    """1 / Herfindahl-Hirschman Index — diversification measure."""
    hhi = np.sum(weights**2)
    return 1.0 / hhi if hhi > 0 else 0.0


def minimize_variance(
    cov: pd.DataFrame,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
) -> np.ndarray:
    """Solve the long-only, box-constrained minimum-variance QP."""
    n = len(cov)
    cov_matrix = cov.values

    def objective(w):
        return w @ cov_matrix @ w

    def objective_grad(w):
        return 2 * cov_matrix @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n
    w0 = np.full(n, 1.0 / n)  # equal-weight start

    result = minimize(
        objective,
        w0,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    # Clean up floating point noise (e.g. -1e-17 instead of 0)
    weights = np.clip(result.x, min_weight, max_weight)
    weights = weights / weights.sum()
    return weights


def build_result(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> OptimizationResult:
    w = pd.Series(weights, index=cov.columns)
    exp_return = float(w.values @ mu.values)
    variance = float(w.values @ cov.values @ w.values)
    vol = float(np.sqrt(variance))
    sharpe = (exp_return - risk_free_rate) / vol if vol > 0 else np.nan
    eff_n = _effective_n(w.values)
    return OptimizationResult(w, exp_return, vol, sharpe, eff_n)


def equal_weight_result(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float = 0.04
) -> OptimizationResult:
    n = len(mu)
    weights = np.full(n, 1.0 / n)
    return build_result(weights, mu, cov, risk_free_rate)


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
    risk_free_rate: float = 0.04,
) -> np.ndarray:
    """
    Tangency portfolio (maximum Sharpe ratio). Included for comparison —
    note this one DOES use the (noisy) expected-return estimates, which is
    exactly the sensitivity minimum-variance is designed to avoid.
    """
    n = len(mu)
    cov_matrix = cov.values
    mu_vec = mu.values

    def neg_sharpe(w):
        ret = w @ mu_vec
        vol = np.sqrt(w @ cov_matrix @ w)
        return -(ret - risk_free_rate) / vol if vol > 0 else 1e6

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n
    w0 = np.full(n, 1.0 / n)

    result = minimize(
        neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")
    weights = np.clip(result.x, min_weight, max_weight)
    return weights / weights.sum()


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 30,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Trace the efficient frontier by minimizing variance for a sweep of
    target returns between the min-variance portfolio's return and the
    highest achievable return.
    """
    n = len(mu)
    cov_matrix = cov.values
    mu_vec = mu.values

    min_var_w = minimize_variance(cov, max_weight, min_weight)
    ret_low = min_var_w @ mu_vec
    ret_high = mu.max()  # best case: 100% in the single best-return asset (if unconstrained by cap)

    targets = np.linspace(ret_low, ret_high, n_points)
    frontier = []

    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu_vec - t},
        ]
        bounds = [(min_weight, max_weight)] * n
        w0 = np.full(n, 1.0 / n)
        result = minimize(
            lambda w: w @ cov_matrix @ w,
            w0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if result.success:
            w = np.clip(result.x, min_weight, max_weight)
            w = w / w.sum()
            vol = np.sqrt(w @ cov_matrix @ w)
            frontier.append({"target_return": target, "volatility": vol, "weights": w})

    return pd.DataFrame(frontier)


if __name__ == "__main__":
    # Quick smoke test using the annualized covariance matrix reported in
    # the workbook (Section 2 of the "Portfolio Optimization" sheet), so
    # this can be sanity-checked without a network connection.
    tickers = ["AGG", "EFA", "GLD", "QQQ", "SPY", "TIP", "TLT", "VNQ"]
    mu = pd.Series(
        [0.018413, 0.087882, 0.120091, 0.199714, 0.151634, 0.029113, -0.000373, 0.075852],
        index=tickers,
    )
    cov_data = [
        [0.002649, 0.003582, 0.003057, 0.003937, 0.003014, 0.002233, 0.006289, 0.005197],
        [0.003582, 0.022606, 0.004314, 0.020641, 0.019456, 0.003973, 0.003653, 0.019530],
        [0.003057, 0.004314, 0.019619, 0.001949, 0.001843, 0.003342, 0.007482, 0.004715],
        [0.003937, 0.020641, 0.001949, 0.034791, 0.026202, 0.004874, 0.005694, 0.021391],
        [0.003014, 0.019456, 0.001843, 0.026202, 0.023481, 0.004018, 0.003070, 0.021449],
        [0.002233, 0.003973, 0.003342, 0.004874, 0.004018, 0.002589, 0.005039, 0.005891],
        [0.006289, 0.003653, 0.007482, 0.005694, 0.003070, 0.005039, 0.018617, 0.009756],
        [0.005197, 0.019530, 0.004715, 0.021391, 0.021449, 0.005891, 0.009756, 0.031713],
    ]
    cov = pd.DataFrame(cov_data, index=tickers, columns=tickers)

    w = minimize_variance(cov, max_weight=0.25)
    result = build_result(w, mu, cov)
    print("=== Minimum-Variance Portfolio ===")
    print(result.summary())
    print()
    eq = equal_weight_result(mu, cov)
    print("=== Equal-Weight Benchmark ===")
    print(eq.summary())
