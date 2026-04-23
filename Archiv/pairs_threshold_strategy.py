from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


@dataclass(slots=True)
class PairModel:
    left: str
    right: str
    hedge_ratio: float
    intercept: float


@dataclass(slots=True)
class Position:
    left_shares: float = 0.0
    right_shares: float = 0.0


@dataclass(slots=True)
class ExecutionResult:
    cash: float
    cost: float
    turnover: float
    traded: bool


def load_price_table(path: str | Path) -> pd.DataFrame:
    resolved_path = resolve_path(path)
    prices = pd.read_csv(resolved_path, index_col="date", parse_dates=True).sort_index()
    prices.index.name = "date"
    return prices


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


def estimate_pair_model(prices: pd.DataFrame, left: str, right: str) -> PairModel:
    pair_prices = prices[[left, right]].dropna()
    x = np.column_stack([np.ones(len(pair_prices)), pair_prices[right].to_numpy()])
    y = pair_prices[left].to_numpy()
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    intercept, hedge_ratio = coefficients
    return PairModel(
        left=left,
        right=right,
        hedge_ratio=float(hedge_ratio),
        intercept=float(intercept),
    )


def compute_spread(prices: pd.DataFrame, model: PairModel) -> pd.Series:
    pair_prices = prices[[model.left, model.right]].dropna()
    spread = pair_prices[model.left] - model.intercept - model.hedge_ratio * pair_prices[model.right]
    spread.name = f"{model.left}_{model.right}_spread"
    return spread


def compute_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std(ddof=0).replace(0.0, np.nan)
    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = "zscore"
    return zscore


def generate_signal(
    zscore_value: float,
    current_side: int,
    entry_threshold: float,
    exit_threshold: float,
) -> int:
    if np.isnan(zscore_value):
        return current_side
    if current_side == 0:
        if zscore_value >= entry_threshold:
            return -1
        if zscore_value <= -entry_threshold:
            return 1
        return 0
    if current_side == 1 and zscore_value >= -exit_threshold:
        return 0
    if current_side == -1 and zscore_value <= exit_threshold:
        return 0
    return current_side


def build_target_position(
    signal: int,
    left_price: float,
    right_price: float,
    model: PairModel,
    gross_notional: float,
) -> Position:
    if signal == 0:
        return Position()

    denominator = left_price + abs(model.hedge_ratio) * right_price
    if denominator <= 0:
        return Position()

    scale = gross_notional / denominator
    left_shares = signal * scale
    right_shares = -signal * model.hedge_ratio * scale
    return Position(left_shares=float(left_shares), right_shares=float(right_shares))


def execute_trade(
    current_position: Position,
    target_position: Position,
    left_price: float,
    right_price: float,
    cash: float,
    transaction_cost_bps: float,
) -> ExecutionResult:
    delta_left = target_position.left_shares - current_position.left_shares
    delta_right = target_position.right_shares - current_position.right_shares
    turnover = abs(delta_left) * left_price + abs(delta_right) * right_price
    cost = turnover * transaction_cost_bps / 10_000.0
    trade_cash = -(delta_left * left_price + delta_right * right_price)
    traded = bool(abs(delta_left) > 0 or abs(delta_right) > 0)
    return ExecutionResult(
        cash=float(cash + trade_cash - cost),
        cost=float(cost),
        turnover=float(turnover),
        traded=traded,
    )


def mark_to_market(cash: float, position: Position, left_price: float, right_price: float) -> float:
    return float(cash + position.left_shares * left_price + position.right_shares * right_price)


def run_threshold_backtest(
    prices: pd.DataFrame,
    model: PairModel,
    lookback: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    gross_notional: float = 100.0,
    transaction_cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    spread = compute_spread(prices, model)
    zscore = compute_zscore(spread, lookback=lookback)
    pair_prices = prices.loc[spread.index, [model.left, model.right]]

    current_position = Position()
    current_side = 0
    cash = 0.0
    previous_equity = 0.0
    rows: list[dict[str, float | int | pd.Timestamp]] = []

    for timestamp in pair_prices.index:
        left_price = float(pair_prices.at[timestamp, model.left])
        right_price = float(pair_prices.at[timestamp, model.right])
        zscore_value = float(zscore.loc[timestamp]) if not pd.isna(zscore.loc[timestamp]) else np.nan

        target_side = generate_signal(
            zscore_value=zscore_value,
            current_side=current_side,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
        target_position = build_target_position(
            signal=target_side,
            left_price=left_price,
            right_price=right_price,
            model=model,
            gross_notional=gross_notional,
        )
        execution = execute_trade(
            current_position=current_position,
            target_position=target_position,
            left_price=left_price,
            right_price=right_price,
            cash=cash,
            transaction_cost_bps=transaction_cost_bps,
        )
        cash = execution.cash
        current_position = target_position
        current_side = target_side

        equity = mark_to_market(cash, current_position, left_price, right_price)
        daily_pnl = equity - previous_equity
        previous_equity = equity

        rows.append(
            {
                "date": timestamp,
                "left_price": left_price,
                "right_price": right_price,
                "spread": float(spread.loc[timestamp]),
                "zscore": zscore_value,
                "signal": float(target_side),
                "left_shares": current_position.left_shares,
                "right_shares": current_position.right_shares,
                "cash": cash,
                "turnover": execution.turnover,
                "cost": execution.cost,
                "equity": equity,
                "daily_pnl": daily_pnl,
                "traded": int(execution.traded),
            }
        )

    results = pd.DataFrame(rows).set_index("date")
    metrics = summarize_backtest(results)
    return results, metrics


def summarize_backtest(results: pd.DataFrame, annualization_factor: int = 252) -> dict[str, float]:
    if results.empty:
        return {
            "final_equity": 0.0,
            "total_pnl": 0.0,
            "annualized_pnl_volatility": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
            "trades": 0.0,
        }

    daily_pnl = results["daily_pnl"]
    cumulative_pnl = results["equity"]
    pnl_vol = float(daily_pnl.std(ddof=0) * np.sqrt(annualization_factor))
    sharpe_like = float(daily_pnl.mean() / daily_pnl.std(ddof=0) * np.sqrt(annualization_factor)) if daily_pnl.std(ddof=0) else 0.0
    running_peak = cumulative_pnl.cummax()
    max_drawdown = float((cumulative_pnl - running_peak).min())
    return {
        "final_equity": float(cumulative_pnl.iloc[-1]),
        "total_pnl": float(cumulative_pnl.iloc[-1]),
        "annualized_pnl_volatility": pnl_vol,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
        "trades": float(results["traded"].sum()),
    }


def run_train_test_strategy(
    train_prices: pd.DataFrame,
    test_prices: pd.DataFrame,
    left: str,
    right: str,
    lookback: int,
    entry_threshold: float,
    exit_threshold: float,
    gross_notional: float,
    transaction_cost_bps: float,
) -> tuple[PairModel, pd.DataFrame, dict[str, float]]:

    model = estimate_pair_model(train_prices, left=left, right=right)

    results, metrics = run_threshold_backtest(
        prices=test_prices,
        model=model,
        lookback=lookback,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        gross_notional=gross_notional,
        transaction_cost_bps=transaction_cost_bps,
    )
    return model, results, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured threshold pairs trading backtest.")
    parser.add_argument("--train-file", default="semiconductor_close_analysis.csv")
    parser.add_argument("--test-file", default="semiconductor_close_trade.csv")
    parser.add_argument("--left", default="LRCX")
    parser.add_argument("--right", default="AMAT")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--entry-threshold", type=float, default=2.0)
    parser.add_argument("--exit-threshold", type=float, default=0.5)
    parser.add_argument("--gross-notional", type=float, default=100.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--output-file", default="threshold_backtest_results.csv")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    train_prices = load_price_table(args.train_file)
    test_prices = load_price_table(args.test_file)
    model, results, metrics = run_train_test_strategy(
        train_prices=train_prices,
        test_prices=test_prices,
        left=args.left,
        right=args.right,
        lookback=args.lookback,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        gross_notional=args.gross_notional,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    output_path = resolve_path(args.output_file)
    results.to_csv(output_path, index_label="date")

    print(f"Pair: {model.left}-{model.right}")
    print(f"Estimated intercept: {model.intercept:.6f}")
    print(f"Estimated hedge ratio: {model.hedge_ratio:.6f}")
    print(f"Results saved to: {output_path.resolve()}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
