"""CLI para backtesting."""

from __future__ import annotations

import argparse

from ..rl.backtest import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Runner de backtest")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    report = run_backtest(args.model_path, days=args.days)
    print(report)


if __name__ == "__main__":
    main()
