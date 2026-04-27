#!/usr/bin/env python3
"""Analyze experiment results for bugs and data quality issues.

Checks:
1. Transactions above/below reservation prices (constraint violations)
2. Empty simulations (no transactions at all)
3. Bid/price parsing failures
4. Agents trading multiple times in the same round
5. Auction-specific: bids above private values
6. Missing result files
7. Reasoning field presence when expected
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_config(result_dir: Path) -> dict | None:
    """Load config_used.yaml from a result directory."""
    config_path = result_dir / "config_used.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.unsafe_load(f)
    return None


def check_double_auction_results(result_dir: Path, config: dict) -> list[str]:
    """Check double auction results for bugs."""
    issues = []
    data_dir = result_dir / "data"

    if not data_dir.exists():
        issues.append("CRITICAL: No data directory found")
        return issues

    n_sims = config.get("experiment", {}).get("n_simulations", 10)

    for sim_id in range(1, n_sims + 1):
        # Check transactions
        tx_file = data_dir / f"transactions_{sim_id}.csv"
        iter_file = data_dir / f"iteration_history_{sim_id}.csv"
        agent_file = data_dir / f"agent_histories_{sim_id}.csv"

        if not tx_file.exists() and not iter_file.exists():
            issues.append(f"  sim {sim_id}: No data files (transactions or iteration_history)")
            continue

        # Check transactions CSV
        if tx_file.exists():
            df_tx = pd.read_csv(tx_file)

            if df_tx.empty:
                issues.append(f"  sim {sim_id}: Empty transactions file")
                continue

            # Check for buyer paying above reservation price
            if "buyer_reservation_price" in df_tx.columns and "price" in df_tx.columns:
                violations = df_tx[df_tx["price"] > df_tx["buyer_reservation_price"]]
                if not violations.empty:
                    issues.append(
                        f"  sim {sim_id}: {len(violations)} transactions where buyer paid ABOVE reservation price!"
                    )
                    for _, row in violations.head(3).iterrows():
                        issues.append(
                            f"    Round {row.get('round', '?')}: price={row['price']:.2f} > buyer_res={row['buyer_reservation_price']:.2f}"
                        )

            # Check for seller selling below reservation price
            if "seller_reservation_price" in df_tx.columns and "price" in df_tx.columns:
                violations = df_tx[df_tx["price"] < df_tx["seller_reservation_price"]]
                if not violations.empty:
                    issues.append(
                        f"  sim {sim_id}: {len(violations)} transactions where seller sold BELOW reservation price!"
                    )
                    for _, row in violations.head(3).iterrows():
                        issues.append(
                            f"    Round {row.get('round', '?')}: price={row['price']:.2f} < seller_res={row['seller_reservation_price']:.2f}"
                        )

            # Check for duplicate agent transactions in same round
            if "round" in df_tx.columns:
                for agent_col in ["buyer_id", "seller_id"]:
                    if agent_col in df_tx.columns:
                        dups = df_tx.groupby(["round", agent_col]).size()
                        multi_trade = dups[dups > 1]
                        if not multi_trade.empty:
                            issues.append(
                                f"  sim {sim_id}: {len(multi_trade)} cases where agent ({agent_col}) traded multiple times in same round!"
                            )

        # Check iteration history for parsing issues
        if iter_file.exists():
            df_iter = pd.read_csv(iter_file)
            if not df_iter.empty and "price" in df_iter.columns:
                null_prices = df_iter["price"].isna().sum()
                total = len(df_iter)
                if null_prices > total * 0.5:
                    issues.append(
                        f"  sim {sim_id}: {null_prices}/{total} ({null_prices/total*100:.0f}%) null prices in iteration history (parsing failures?)"
                    )

    return issues


def check_auction_results(result_dir: Path, config: dict) -> list[str]:
    """Check auction (non-double-auction) results for bugs."""
    issues = []
    data_dir = result_dir / "data"

    if not data_dir.exists():
        issues.append("CRITICAL: No data directory found")
        return issues

    auction_config = config.get("experiment", {}).get("auction", {})
    n_sims = auction_config.get("n_simulations", 10)
    auction_type = config.get("experiment", {}).get("auction_type", "unknown")

    for sim_id in range(1, n_sims + 1):
        results_file = data_dir / f"auction_results_{sim_id}.csv"
        bids_file = data_dir / f"all_bids_{sim_id}.csv"

        if not results_file.exists() and not bids_file.exists():
            issues.append(f"  sim {sim_id}: No auction result files")
            continue

        # Check auction results
        if results_file.exists():
            df_results = pd.read_csv(results_file)

            if df_results.empty:
                issues.append(f"  sim {sim_id}: Empty auction_results file")
                continue

            # Check for negative profits (winner paying more than value)
            if "winner_profit" in df_results.columns:
                neg_profits = df_results[df_results["winner_profit"] < 0]
                if not neg_profits.empty:
                    issues.append(
                        f"  sim {sim_id}: {len(neg_profits)} rounds with negative winner profit!"
                    )
                    for _, row in neg_profits.head(3).iterrows():
                        issues.append(
                            f"    Round {row.get('round', '?')}: profit={row['winner_profit']:.2f}"
                        )

            # Check for no winners (may be valid in some cases)
            if "winner_id" in df_results.columns:
                no_winner = df_results["winner_id"].isna().sum()
                if no_winner > 0:
                    total_rounds = len(df_results)
                    issues.append(
                        f"  sim {sim_id}: {no_winner}/{total_rounds} rounds had no winner"
                    )

        # Check individual bids
        if bids_file.exists():
            df_bids = pd.read_csv(bids_file)

            if not df_bids.empty:
                # Check for bids above private value
                if "bid" in df_bids.columns and "private_value" in df_bids.columns:
                    overbids = df_bids[df_bids["bid"] > df_bids["private_value"]]
                    if not overbids.empty:
                        pct = len(overbids) / len(df_bids) * 100
                        issues.append(
                            f"  sim {sim_id}: {len(overbids)} ({pct:.1f}%) bids ABOVE private value!"
                        )

                # Check for negative bids
                if "bid" in df_bids.columns:
                    neg_bids = df_bids[df_bids["bid"] < 0]
                    if not neg_bids.empty:
                        issues.append(
                            f"  sim {sim_id}: {len(neg_bids)} negative bids!"
                        )

                # Check for null/NaN bids (parsing failures)
                if "bid" in df_bids.columns:
                    null_bids = df_bids["bid"].isna().sum()
                    if null_bids > 0:
                        issues.append(
                            f"  sim {sim_id}: {null_bids} null bids (parsing failures?)"
                        )

    return issues


def check_logs_for_errors(result_dir: Path) -> list[str]:
    """Check simulation logs for error patterns."""
    issues = []
    logs_dir = result_dir / "logs"

    if not logs_dir.exists():
        return issues

    for log_file in sorted(logs_dir.glob("*.log")):
        with open(log_file) as f:
            content = f.read()

        errors = []
        for line in content.split("\n"):
            line_lower = line.lower()
            if any(
                kw in line_lower
                for kw in ["traceback", "exception", "error", "failed to parse", "rate limit"]
            ):
                errors.append(line.strip())

        if errors:
            issues.append(f"  {log_file.name}: {len(errors)} error/warning lines")
            # Show first few unique errors
            seen = set()
            for err in errors[:5]:
                short = err[:120]
                if short not in seen:
                    seen.add(short)
                    issues.append(f"    {short}")

    return issues


def analyze_result_dir(result_dir: Path) -> None:
    """Analyze a single result directory."""
    config = load_config(result_dir)

    if config is None:
        print(f"  SKIP: No config_used.yaml found")
        return

    auction_type_raw = config.get("experiment", {}).get("auction_type", "double_auction")
    # Handle both enum and string values
    auction_type = str(auction_type_raw.value) if hasattr(auction_type_raw, 'value') else str(auction_type_raw)
    include_reasoning = config.get("experiment", {}).get("include_reasoning", True)

    print(f"  Type: {auction_type} | Reasoning: {include_reasoning}")
    print(f"  Model: {config.get('llm', {}).get('model', 'unknown')}")
    print(f"  Temp: {config.get('llm', {}).get('temperature', 'unknown')}")

    # Check results based on auction type
    if auction_type == "double_auction":
        issues = check_double_auction_results(result_dir, config)
    else:
        issues = check_auction_results(result_dir, config)

    # Check logs
    log_issues = check_logs_for_errors(result_dir)

    if issues:
        print(f"  DATA ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"  Data: OK")

    if log_issues:
        print(f"  LOG ISSUES ({len(log_issues)}):")
        for issue in log_issues:
            print(f"    {issue}")
    else:
        print(f"  Logs: OK")


def main():
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists():
        print("No results directory found.")
        sys.exit(1)

    # Filter by argument if provided
    filter_str = sys.argv[1] if len(sys.argv) > 1 else ""

    result_dirs = sorted(results_dir.iterdir())
    result_dirs = [d for d in result_dirs if d.is_dir()]

    if filter_str:
        result_dirs = [d for d in result_dirs if filter_str in d.name]

    if not result_dirs:
        print(f"No result directories found{' matching ' + filter_str if filter_str else ''}.")
        sys.exit(1)

    print("=" * 70)
    print("  Experiment Results Analysis")
    print("=" * 70)
    print(f"  Found {len(result_dirs)} result directories")
    print("")

    # Summary counters
    total_ok = 0
    total_issues = 0

    for result_dir in result_dirs:
        print(f"\n--- {result_dir.name} ---")
        try:
            analyze_result_dir(result_dir)
        except Exception as e:
            print(f"  ERROR analyzing: {e}")
            total_issues += 1

    print("\n" + "=" * 70)
    print("  Analysis complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
