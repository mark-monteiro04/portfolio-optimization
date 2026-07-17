"""
rolling.py — Rolling-window re-optimization ("live" portfolio updates).

This is the practical version of "real-time" optimization discussed in the
README: minimum-variance portfolios are driven by a covariance matrix that
only changes meaningfully over weeks/months, so re-solving on every tick
adds noise and transaction costs without adding accuracy. Institutional
risk systems typically refresh EXPOSURE/RISK METRICS continuously but only
REBALANCE on a daily/weekly/monthly cadence — this module implements that
pattern.

For each rebalance date, it:
    1. Takes a trailing window of monthly returns (e.g. 36 months)
    2. Re-estimates the covariance matrix
    3. Re-solves the minimum-variance QP
    4. Records the new weights and the "weight drift" vs the prior period

This produces a weight-drift-over-time chart, which is a much stronger
demo artifact than a single static allocation.
"""

import pandas as pd

from optimize import annualize_stats, build_result, minimize_variance


def rolling_reoptimization(
    monthly_returns: pd.DataFrame,
    window: int = 36,
    step: int = 1,
    max_weight: float = 0.25,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """
    Walk forward through the return series, re-optimizing every `step`
    months on a trailing `window`-month lookback.

    Returns a DataFrame indexed by rebalance date, one column per ticker
    weight, plus expected_return / volatility / sharpe columns.
    """
    dates = monthly_returns.index
    records = []

    for end_idx in range(window, len(dates), step):
        window_data = monthly_returns.iloc[end_idx - window : end_idx]
        rebalance_date = dates[end_idx]

        mu, cov = annualize_stats(window_data)
        weights = minimize_variance(cov, max_weight=max_weight)
        result = build_result(weights, mu, cov, risk_free_rate)

        row = result.weights.to_dict()
        row.update(
            {
                "date": rebalance_date,
                "expected_return": result.expected_return,
                "volatility": result.volatility,
                "sharpe": result.sharpe,
                "effective_n": result.effective_n,
            }
        )
        records.append(row)

    out = pd.DataFrame(records).set_index("date")
    return out


def turnover(weight_history: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    One-way turnover between consecutive rebalances: sum(|w_t - w_{t-1}|) / 2.
    A cheap proxy for the trading cost this rebalancing schedule would incur
    — worth showing alongside the weight-drift chart to make the "why not
    rebalance every tick" argument concrete rather than hand-wavy.
    """
    w = weight_history[tickers]
    diffs = w.diff().abs().sum(axis=1) / 2
    return diffs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rolling minimum-variance re-optimization.")
    parser.add_argument("--returns-csv", required=True, help="Monthly returns CSV (date-indexed).")
    parser.add_argument("--window", type=int, default=36, help="Lookback window in months.")
    parser.add_argument("--step", type=int, default=1, help="Months between rebalances.")
    parser.add_argument("--out", default="outputs/rolling_weights.csv")
    args = parser.parse_args()

    returns = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)
    history = rolling_reoptimization(returns, window=args.window, step=args.step)
    history.to_csv(args.out)

    tickers = [c for c in returns.columns]
    to = turnover(history, tickers)
    print(f"Saved {len(history)} rebalance periods -> {args.out}")
    print(f"Average one-way turnover per rebalance: {to.mean():.2%}")
    print(f"Max one-way turnover: {to.max():.2%}")
