"""
data.py — Market data ingestion for the portfolio optimizer.

Downloads adjusted daily prices for the ETF universe and converts them to
monthly simple returns. This replaces the original risk.py, which had the
monthly-resampling logic commented out and was actually emitting *yearly*
returns despite the rest of the project (covariance matrix, weights, report)
being built on 119 months of monthly data. That mismatch meant the script
in the repo couldn't reproduce the numbers in the report — this version
fixes that so the pipeline is reproducible end to end.

Usage:
    python src/data.py --start 2015-10-01 --end 2025-08-31 --out outputs/asset_returns.csv
"""

import argparse
import sys

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

TICKERS = [
    "SPY",  # US Equities
    "QQQ",  # Growth / Tech
    "EFA",  # International Developed Equities
    "AGG",  # Core Bonds
    "TLT",  # Long-Term Treasuries
    "VNQ",  # Real Estate (REITs)
    "GLD",  # Gold
    "TIP",  # Inflation-Protected Bonds (TIPS)
]


def fetch_monthly_returns(
    tickers: list[str] = TICKERS,
    start: str = "2015-10-01",
    end: str = "2025-08-31",
) -> pd.DataFrame:
    """
    Download daily adjusted close prices and return a DataFrame of
    month-end simple returns, indexed by date, one column per ticker.
    """
    if yf is None:
        raise ImportError(
            "yfinance is required for live data pulls. Install it with:\n"
            "    pip install yfinance"
        )

    prices = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Month-end prices -> simple monthly returns.
    # (This is the step that was commented out in the original script.)
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="all")

    return monthly_returns


def load_returns_from_csv(path: str) -> pd.DataFrame:
    """Load a previously saved monthly-returns CSV (date-indexed)."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch monthly ETF returns.")
    parser.add_argument("--start", default="2015-10-01")
    parser.add_argument("--end", default="2025-08-31")
    parser.add_argument("--out", default="outputs/asset_returns.csv")
    args = parser.parse_args()

    returns = fetch_monthly_returns(start=args.start, end=args.end)
    returns.to_csv(args.out)
    print(f"Saved {returns.shape[0]} months x {returns.shape[1]} tickers -> {args.out}")
    print(returns.describe())


if __name__ == "__main__":
    sys.exit(main())
