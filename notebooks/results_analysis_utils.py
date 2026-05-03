"""
Market Experiment Analysis
==========================
Loads three data sources per simulation:
- iteration_history_{sim}.csv: full order flow (all submitted bids/asks, posted/traded/no_announcement)
- transactions_{sim}.csv: completed transactions with execution prices (from order book crossing)
- agent_histories_{sim}.csv: per-agent action log with reservation prices

Computes:
- Allocative efficiency (per round, aggregated across simulations)
- Smith's alpha (coefficient of convergence)
- Marshallian path analysis (within-round price convergence)
- Across-round trajectory (flat for ZI-C, convergent for LLMs)
- Rent-seeking analysis (attempted rent, rent ratio, realized rent, extraction efficiency)
- Concession rate analysis
- Constraint violation tracking
- Order submission frequency and bid/ask dispersion
- Who-trades analysis (inframarginal participation)
- Fill rate analysis (fraction of orders that execute immediately)

Works with both:
- ZI-C experiments: 1 round per simulation, many simulations
- LLM experiments: N rounds per simulation, rounds are NOT independent

IMPORTANT: In order-book experiments, the execution price differs from the
submitted order price. The incoming order crosses the resting order at the
resting order's price (price-time priority). iteration_history.price is the
SUBMITTED limit price; transactions.price is the EXECUTION price. All
transaction-level metrics (efficiency, alpha, realized rent, surplus) use
execution prices from the transactions file.

Usage:
    from results_analysis import run_validation, run_rent_analysis, run_full_analysis
    results = run_full_analysis(results_path, n_sims=500)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from pathlib import Path
from dataclasses import dataclass
import warnings


# ============================================================
# Smith (1962) benchmark data
# ============================================================

SMITH_ALPHA = {
    "smith1": {1: 11.8, 2: 8.1, 3: 5.2, 4: 5.5, 5: 3.5},
    "smith2": {1: 9.9, 2: 5.4, 3: 2.2},
    "smith3": {1: 16.5, 2: 6.6, 3: 3.7, 4: 5.7},
    "smith4a": {1: 19.1, 2: 10.4, 3: 7.8, 4: 7.6},
    "smith4b": {1: 6.9, 2: 7.1, 3: 6.5},
    "smith5a": {1: 2.0, 2: 0.7, 3: 0.7, 4: 0.6},
    "smith5b": {1: 9.4, 2: 4.3},
    "smith6a": {1: 53.8, 2: 38.7, 3: 21.1, 4: 9.4},
    "smith6b": {1: 11.0},
    "smith7": {1: 49.1, 2: 22.2, 3: 7.1, 4: 5.4, 5: 3.0, 6: 2.7},
    "smith8a": {1: 19.0, 2: 2.9, 3: 7.4, 4: 7.0},
    "smith8b": {1: 7.8, 2: 6.1},
    "smith9a": {1: 21.8, 2: 15.4, 3: 13.2},
    "smith9b": {1: 10.3},
    "smith10": {1: 11.0, 2: 3.2, 3: 2.2},
}


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_experiment(results_path: Path, n_sims: int
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load iteration_history, transactions, and agent_histories for all
    available simulations.

    Returns:
        df_iter: full iteration history (all submitted orders)
        df_tx: completed transactions with execution prices
        df_agents: agent histories with reservation prices
    """
    available_sims = [
        sim for sim in range(1, n_sims + 1)
        if (results_path / 'data' / f'iteration_history_{sim}.csv').exists()
        and (results_path / 'data' / f'transactions_{sim}.csv').exists()
        and (results_path / 'data' / f'agent_histories_{sim}.csv').exists()
    ]

    if not available_sims:
        raise FileNotFoundError(f"No matching files found in {results_path / 'data'}")

    ih_list = [
        pd.read_csv(results_path / 'data' / f'iteration_history_{sim}.csv').assign(sim=sim)
        for sim in available_sims
    ]
    tx_list = [
        pd.read_csv(results_path / 'data' / f'transactions_{sim}.csv').assign(sim=sim)
        for sim in available_sims
    ]
    ah_list = [
        pd.read_csv(results_path / 'data' / f'agent_histories_{sim}.csv').assign(sim=sim)
        for sim in available_sims
    ]

    df_iter = pd.concat(ih_list, ignore_index=True)
    df_tx = pd.concat(tx_list, ignore_index=True)
    df_agents = pd.concat(ah_list, ignore_index=True)

    n_rounds = df_tx.groupby('sim')['round'].nunique().iloc[0]

    print(f"Loaded {len(available_sims)} simulations from {results_path.name}")
    print(f"  Iteration history rows: {len(df_iter)}")
    print(f"  Transaction rows:       {len(df_tx)}")
    print(f"  Rounds per sim:         {n_rounds}")
    print(f"  Total round-obs:        {df_tx.groupby(['sim', 'round']).ngroups}")

    return df_iter, df_tx, df_agents


def enrich_transactions(df_tx: pd.DataFrame, eq: 'Equilibrium') -> pd.DataFrame:
    """
    Add reservation prices, surplus columns, and role flags to transactions.
    Uses Equilibrium buyer_map/seller_map to look up reservation prices.
    """
    tx = df_tx.copy()
    tx['buyer_val'] = tx['buyer_id'].map(eq.buyer_map)
    tx['seller_cost'] = tx['seller_id'].map(eq.seller_map)
    tx['buyer_rent'] = tx['buyer_val'] - tx['price']
    tx['seller_rent'] = tx['price'] - tx['seller_cost']
    tx['total_surplus'] = tx['buyer_val'] - tx['seller_cost']

    valid = (tx['buyer_rent'] >= 0) & (tx['seller_rent'] >= 0)
    tx['buyer_share'] = np.where(
        ~valid, np.nan,
        np.where(tx['total_surplus'] > 0, tx['buyer_rent'] / tx['total_surplus'], 0.5),
    )
    tx['seller_share'] = np.where(
        ~valid, np.nan,
        np.where(tx['total_surplus'] > 0, tx['seller_rent'] / tx['total_surplus'], 0.5),
    )

    tx['buyer_inframarginal'] = tx['buyer_val'] >= eq.price
    tx['seller_inframarginal'] = tx['seller_cost'] <= eq.price
    tx['both_inframarginal'] = tx['buyer_inframarginal'] & tx['seller_inframarginal']
    tx['buyer_violation'] = tx['buyer_rent'] < 0
    tx['seller_violation'] = tx['seller_rent'] < 0

    return tx


def extract_announcements(df_iter: pd.DataFrame) -> pd.DataFrame:
    """
    Extract deduplicated submitted orders from iteration history.
    price column = submitted limit price (NOT execution price).
    """
    ann = (
        df_iter[df_iter['announcement_made'] == True]
        .drop_duplicates(
            subset=['sim', 'round', 'iteration', 'announcing_agent_id',
                    'price', 'announcement_type']
        )
        .copy()
    )
    ann['side'] = ann['announcement_type'].map({'buy': 'buyer', 'sell': 'seller'})
    ann['attempted_rent'] = np.where(
        ann['announcement_type'] == 'buy',
        ann['announcing_agent_reservation_price'] - ann['price'],
        ann['price'] - ann['announcing_agent_reservation_price'],
    )
    ann['rent_ratio'] = ann['attempted_rent'] / ann['announcing_agent_reservation_price']
    ann['violation'] = ann['attempted_rent'] < 0

    if 'order_outcome' in ann.columns:
        ann['filled'] = ann['order_outcome'] == 'traded'
    elif 'transaction_made' in ann.columns:
        ann['filled'] = ann['transaction_made'] == True
    else:
        ann['filled'] = False

    return ann


# ============================================================
# 2. EQUILIBRIUM COMPUTATION
# ============================================================

@dataclass
class Equilibrium:
    quantity: int
    price: float
    surplus: float
    buyer_surplus: float
    seller_surplus: float
    demand: np.ndarray
    supply: np.ndarray
    buyer_map: dict
    seller_map: dict


def find_equilibrium(supply, demand, flat_threshold=1e-10):
    """Find equilibrium quantity and price where demand >= supply."""
    supply = np.asarray(supply)
    demand = np.asarray(demand)
    n = min(len(supply), len(demand))
    viable_mask = demand[:n] >= supply[:n]
    if not viable_mask.any():
        return 0, None
    last_idx = np.flatnonzero(viable_mask)[-1]
    next_idx = last_idx + 1
    q_eq = last_idx + 1
    p_supply = supply[last_idx]
    p_demand = demand[last_idx]
    has_next_supply = next_idx < len(supply)
    has_next_demand = next_idx < len(demand)
    if not has_next_supply:
        p_eq = (p_demand + demand[next_idx]) / 2 if has_next_demand else p_demand
    elif not has_next_demand:
        p_eq = (p_supply + supply[next_idx]) / 2 if has_next_supply else p_supply
    elif abs(supply[next_idx] - p_supply) < flat_threshold:
        p_eq = p_supply
    elif abs(demand[next_idx] - p_demand) < flat_threshold:
        p_eq = p_demand
    else:
        p_eq = (p_supply + p_demand) / 2
    return q_eq, p_eq


def plot_equilibrium(supply, demand, q_eq, p_eq, title=None, ax=None):
    """Visualize supply and demand curves with equilibrium point."""
    supply = np.asarray(supply)
    demand = np.asarray(demand)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    quantities_supply = np.arange(len(supply)) + 1
    quantities_demand = np.arange(len(demand)) + 1
    ax.step(quantities_supply, supply, 'r-', where='pre', label='Supply', linewidth=2)
    ax.step(quantities_demand, demand, 'b-', where='pre', label='Demand', linewidth=2)
    y_max = max(supply[-1], demand[0]) * 1.2
    ax.vlines(len(supply), supply[-1], y_max, colors='red', linewidth=2)
    ax.vlines(len(demand), demand[-1], 0, colors='blue', linewidth=2)
    if q_eq > 0 and p_eq is not None:
        ax.plot(q_eq, p_eq, 'go', markersize=12,
                label=f'Equilibrium: Q={q_eq}, P={p_eq:.2f}', zorder=5)
        ax.axhline(p_eq, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.axvline(q_eq, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(title or 'Supply and Demand - Equilibrium')
    ax.set_ylim(0, y_max)
    return ax


def compute_equilibrium(config: dict, df_agents: pd.DataFrame) -> Equilibrium:
    """Compute competitive equilibrium from config schedules."""
    demand = np.round(np.linspace(
        config['experiment']['buyers']['max'],
        config['experiment']['buyers']['min'],
        config['experiment']['buyers']['num'],
    ), 2)
    supply = np.round(np.linspace(
        config['experiment']['sellers']['min'],
        config['experiment']['sellers']['max'],
        config['experiment']['sellers']['num'],
    ), 2)
    q_eq, p_eq = find_equilibrium(supply=supply, demand=demand)
    ce_surplus = float(np.sum(demand[:q_eq] - supply[:q_eq]))
    buyer_surplus = float(np.sum(demand[:q_eq] - p_eq))
    seller_surplus = float(np.sum(p_eq - supply[:q_eq]))
    agents = df_agents[['agent_id', 'agent_type', 'reservation_price']].drop_duplicates()
    buyers = agents[agents['agent_type'] == 'buyer']
    sellers = agents[agents['agent_type'] == 'seller']
    bv_dict = dict(zip(buyers['agent_id'], buyers['reservation_price']))
    sv_dict = dict(zip(sellers['agent_id'], sellers['reservation_price']))
    eq = Equilibrium(
        quantity=q_eq, price=p_eq, surplus=ce_surplus,
        buyer_surplus=buyer_surplus, seller_surplus=seller_surplus,
        demand=demand, supply=supply,
        buyer_map=bv_dict, seller_map=sv_dict,
    )
    print(f"\nCompetitive Equilibrium:")
    print(f"  Quantity:       {eq.quantity}")
    print(f"  Price:          {eq.price:.4f}")
    print(f"  Total surplus:  {eq.surplus:.4f}")
    print(f"  Buyer surplus:  {eq.buyer_surplus:.2f}")
    print(f"  Seller surplus: {eq.seller_surplus:.2f}")
    print(f"  Demand:         {demand}")
    print(f"  Supply:         {supply}")
    plot_equilibrium(supply, demand, q_eq, p_eq)
    plt.show()
    return eq


# ============================================================
# 3. PER-ROUND METRICS (uses execution prices from transactions)
# ============================================================

def compute_round_metrics(df_tx: pd.DataFrame, eq: Equilibrium) -> pd.DataFrame:
    """Compute metrics for each (sim, round) pair using execution prices."""
    tx = df_tx.copy()
    if 'buyer_val' not in tx.columns:
        tx['buyer_val'] = tx['buyer_id'].map(eq.buyer_map)
        tx['seller_cost'] = tx['seller_id'].map(eq.seller_map)
    records = []
    for (sim, rnd), group in tx.groupby(['sim', 'round']):
        bv = group['buyer_val'].values
        sc = group['seller_cost'].values
        prices = group['price'].values
        actual_surplus = float(np.sum(bv - sc))
        efficiency = actual_surplus / eq.surplus if eq.surplus > 0 else np.nan
        rmse = np.sqrt(np.mean((prices - eq.price) ** 2))
        alpha = 100 * rmse / eq.price
        n_trades = len(group)
        mean_price = prices.mean()
        n_extramarginal = int(np.sum((bv < eq.price) | (sc > eq.price)))
        n_negative = int(np.sum((bv - sc) < 0))
        records.append({
            'sim': sim, 'round': rnd,
            'efficiency': efficiency, 'alpha': alpha,
            'n_trades': n_trades, 'mean_price': mean_price,
            'actual_surplus': actual_surplus,
            'n_extramarginal': n_extramarginal,
            'n_negative_surplus': n_negative,
        })
    return pd.DataFrame(records)


# ============================================================
# 4. MARSHALLIAN PATH ANALYSIS (uses execution prices)
# ============================================================

def compute_marshallian_path(df_tx: pd.DataFrame, eq: Equilibrium) -> pd.DataFrame:
    """Analyse within-round price convergence using execution prices."""
    df = df_tx.copy()
    if 'buyer_val' not in df.columns:
        df['buyer_val'] = df['buyer_id'].map(eq.buyer_map)
        df['seller_cost'] = df['seller_id'].map(eq.seller_map)
    df['tx_seq'] = df.groupby(['sim', 'round']).cumcount() + 1
    df['price_dev'] = abs(df['price'] - eq.price)
    df['signed_dev'] = df['price'] - eq.price
    df['pair_surplus'] = df['buyer_val'] - df['seller_cost']
    df['buyer_dist_from_ce'] = abs(df['buyer_val'] - eq.price)
    df['seller_dist_from_ce'] = abs(df['seller_cost'] - eq.price)
    return df


# ============================================================
# 5. RENT-SEEKING ANALYSIS
# ============================================================

def compute_rent_metrics(df_tx: pd.DataFrame, df_ann: pd.DataFrame,
                         eq: Equilibrium) -> dict:
    """
    Compute rent-seeking metrics.
    Transaction-level metrics use EXECUTION prices from df_tx.
    Announcement-level metrics use SUBMITTED prices from df_ann.
    """
    tx = enrich_transactions(df_tx, eq) if 'buyer_val' not in df_tx.columns else df_tx.copy()
    ann = df_ann.copy()

    # Concession rate (submitted order prices)
    ann_sorted = ann.sort_values(['sim', 'announcing_agent_id', 'round', 'iteration']).copy()
    concessions = []
    for (sim, agent, rnd), grp in ann_sorted.groupby(['sim', 'announcing_agent_id', 'round']):
        if len(grp) < 2:
            continue
        grp = grp.sort_values('iteration')
        rents = grp['attempted_rent'].values
        iters = grp['iteration'].values
        side = grp['side'].iloc[0]
        for j in range(1, len(rents)):
            concession = rents[j - 1] - rents[j]
            remaining = rents[j - 1]
            if remaining <= 0:
                continue
            frac_concession = concession / remaining if remaining > 0 else np.nan
            concessions.append({
                'sim': sim, 'agent_id': agent, 'side': side, 'round': rnd,
                'from_iter': iters[j - 1], 'to_iter': iters[j],
                'concession': concession, 'frac_concession': frac_concession,
            })
    conc = pd.DataFrame(concessions)

    # Rent surrendered per round
    ann_by_round = (
        ann.sort_values('iteration')
        .groupby(['sim', 'announcing_agent_id', 'round', 'side'])
        .agg(first_rent=('attempted_rent', 'first'),
             last_rent=('attempted_rent', 'last'),
             n_announcements=('attempted_rent', 'count'))
        .reset_index()
    )
    ann_by_round = ann_by_round[
        (ann_by_round['first_rent'] > 0) & (ann_by_round['n_announcements'] > 1)
    ].copy()
    ann_by_round['rent_surrendered'] = ann_by_round['first_rent'] - ann_by_round['last_rent']
    ann_by_round['frac_surrendered'] = ann_by_round['rent_surrendered'] / ann_by_round['first_rent']

    # Rent extraction efficiency
    ann_first = (
        ann.sort_values('iteration')
        .groupby(['sim', 'announcing_agent_id', 'round'])
        .first().reset_index()
    )
    ann_first = ann_first.rename(columns={
        'attempted_rent': 'first_attempted_rent', 'price': 'first_price',
    })
    # Realized rent from EXECUTION prices
    records = []
    for _, row in tx.iterrows():
        s, rnd = row['sim'], row['round']
        records.append({'sim': s, 'agent_id': row['buyer_id'], 'round': rnd,
                        'realized_rent': row['buyer_rent']})
        records.append({'sim': s, 'agent_id': row['seller_id'], 'round': rnd,
                        'realized_rent': row['seller_rent']})
    realized = pd.DataFrame(records)

    eff = ann_first.merge(
        realized,
        left_on=['sim', 'announcing_agent_id', 'round'],
        right_on=['sim', 'agent_id', 'round'],
        how='inner',
    )
    eff = eff[eff['first_attempted_rent'] > 0].copy()
    eff['side'] = eff['announcement_type'].map({'buy': 'buyer', 'sell': 'seller'})
    eff['efficiency'] = eff['realized_rent'] / eff['first_attempted_rent']

    return {
        'ann': ann, 'tx': tx, 'concessions': conc,
        'rent_surrendered': ann_by_round,
        'ann_first': ann_first, 'extraction_eff': eff,
    }


# ============================================================
# 6. SUMMARY PRINTING
# ============================================================

def print_summary(df_metrics: pd.DataFrame, eq: Equilibrium, independent_rounds: bool):
    """Print comprehensive summary statistics."""
    df = df_metrics
    n_obs = len(df)
    n_sims = df['sim'].nunique()
    n_rounds = df['round'].nunique()
    if independent_rounds:
        obs_label = f"{n_obs} simulations (1 round each)"
    else:
        obs_label = f"{n_sims} simulations \u00d7 {n_rounds} rounds = {n_obs} observations"
    print(f"\n{'=' * 60}")
    print(f"ALLOCATIVE EFFICIENCY ({obs_label})")
    print(f"{'=' * 60}")
    print(f"  Mean:   {df['efficiency'].mean():.4f}")
    print(f"  Median: {df['efficiency'].median():.4f}")
    print(f"  Std:    {df['efficiency'].std():.4f}")
    print(f"  Min:    {df['efficiency'].min():.4f}")
    print(f"  Max:    {df['efficiency'].max():.4f}")
    print(f"  < 80%:  {(df['efficiency'] < 0.80).sum()}/{n_obs}")
    print(f"  > 90%:  {(df['efficiency'] > 0.90).sum()}/{n_obs}")
    print(f"  > 95%:  {(df['efficiency'] > 0.95).sum()}/{n_obs}")
    print(f"  = 100%: {(abs(df['efficiency'] - 1.0) < 0.001).sum()}/{n_obs}")
    if independent_rounds:
        print(f"\nSMITH'S ALPHA (coefficient of convergence)")
        print(f"  Mean:   {df['alpha'].mean():.2f}%")
        print(f"  Median: {df['alpha'].median():.2f}%")
        print(f"  Std:    {df['alpha'].std():.2f}%")
    print(f"\nQUANTITY")
    print(f"  CE quantity:  {eq.quantity}")
    print(f"  Mean trades:  {df['n_trades'].mean():.2f}")
    print(f"  Distribution: {df['n_trades'].value_counts().sort_index().to_dict()}")
    print(f"\nPRICE")
    print(f"  CE price:       {eq.price:.4f}")
    print(f"  Mean price:     {df['mean_price'].mean():.4f}")
    print(f"  Std of means:   {df['mean_price'].std():.4f}")
    print(f"\nEXTRAMARGINAL ACTIVITY")
    print(f"  Rounds with extramarginal: {(df['n_extramarginal'] > 0).sum()}/{n_obs} "
          f"({(df['n_extramarginal'] > 0).mean() * 100:.1f}%)")
    print(f"  Mean extramarginal/round:  {df['n_extramarginal'].mean():.2f}")
    print(f"  Negative surplus trades:   {(df['n_negative_surplus'] > 0).sum()}/{n_obs}")
    if not independent_rounds:
        print(f"\nACROSS-ROUND TRAJECTORY (mean \u00b1 SEM across {n_sims} sims)")
        by_round = df.groupby('round').agg(
            eff_mean=('efficiency', 'mean'), eff_sem=('efficiency', 'sem'),
            alpha_mean=('alpha', 'mean'), alpha_sem=('alpha', 'sem'),
            n_trades_mean=('n_trades', 'mean'),
        ).round(4)
        print(by_round.to_string())


def print_marshallian_summary(df_marsh: pd.DataFrame, eq: Equilibrium):
    """Print Marshallian path analysis results."""
    print(f"\n{'=' * 60}")
    print(f"MARSHALLIAN PATH (within-round convergence)")
    print(f"{'=' * 60}")
    max_seq = int(df_marsh['tx_seq'].quantile(0.95))
    print(f"\nBy transaction position (up to {max_seq}):")
    print(f"{'Pos':>4} {'|p-CE|':>7} {'BuyerVal':>9} {'SellCost':>9} "
          f"{'Surplus':>8} {'B_dist':>7} {'S_dist':>7} {'n':>5}")
    for seq in range(1, max_seq + 1):
        sub = df_marsh[df_marsh['tx_seq'] == seq]
        if len(sub) == 0:
            break
        print(f"{seq:>4} {sub['price_dev'].mean():>7.3f} "
              f"{sub['buyer_val'].mean():>9.2f} {sub['seller_cost'].mean():>9.2f} "
              f"{sub['pair_surplus'].mean():>8.2f} "
              f"{sub['buyer_dist_from_ce'].mean():>7.2f} "
              f"{sub['seller_dist_from_ce'].mean():>7.2f} {len(sub):>5}")
    slopes = []
    for (sim, rnd), group in df_marsh.groupby(['sim', 'round']):
        if len(group) >= 3:
            s = np.polyfit(group['tx_seq'].values.astype(float),
                           group['price_dev'].values, 1)[0]
            slopes.append(s)
    slopes = np.array(slopes)
    if len(slopes) > 0:
        t_stat = slopes.mean() / (slopes.std() / np.sqrt(len(slopes)))
        print(f"\nPer-round slope (|price - CE| ~ transaction_seq):")
        print(f"  Mean slope:  {slopes.mean():.4f}")
        print(f"  Median:      {np.median(slopes):.4f}")
        print(f"  % negative (convergent): {(slopes < 0).sum()}/{len(slopes)} "
              f"({(slopes < 0).mean() * 100:.1f}%)")
        print(f"  t-stat (H0: slope=0):    {t_stat:.3f}")
        print(f"  {'Significant' if abs(t_stat) > 1.96 else 'Not significant'} at 5% level")
    corr_b = np.corrcoef(df_marsh['tx_seq'], df_marsh['buyer_dist_from_ce'])[0, 1]
    corr_s = np.corrcoef(df_marsh['tx_seq'], df_marsh['seller_dist_from_ce'])[0, 1]
    print(f"\n  Corr(seq, buyer_dist_from_CE):  {corr_b:.4f}")
    print(f"  Corr(seq, seller_dist_from_CE): {corr_s:.4f}")
    print(f"  {'Marshallian path confirmed' if corr_b < -0.1 and corr_s < -0.1 else 'Weak/no Marshallian path'}")


def print_rent_summary(rent: dict, eq: Equilibrium):
    """Print rent-seeking analysis summary."""
    ann = rent['ann']
    tx = rent['tx']
    conc = rent['concessions']
    eff = rent['extraction_eff']
    print(f"\n{'=' * 60}")
    print(f"RENT-SEEKING ANALYSIS")
    print(f"{'=' * 60}")
    print(f"\nAttempted Rent (submitted order prices):")
    for side in ['buyer', 'seller']:
        sub = ann[ann['side'] == side]['attempted_rent']
        print(f"  {side.title():>6}: mean={sub.mean():.3f}, median={sub.median():.3f}, std={sub.std():.3f}")
    print(f"\nRealized Rent (execution prices):")
    print(f"  Buyer rent:  mean={tx['buyer_rent'].mean():.3f}")
    print(f"  Seller rent: mean={tx['seller_rent'].mean():.3f}")
    valid_tx = tx[tx['buyer_share'].notna()]
    if len(valid_tx) > 0:
        print(f"  Buyer share:  {valid_tx['buyer_share'].mean():.1%}")
        print(f"  Seller share: {valid_tx['seller_share'].mean():.1%}")
    if len(conc) > 0:
        print(f"\nConcessions (submitted order prices):")
        for side in ['buyer', 'seller']:
            sub = conc[conc['side'] == side]
            if len(sub) > 0:
                print(f"  {side.title():>6}: mean concession={sub['concession'].mean():.3f}, "
                      f"frac={sub['frac_concession'].mean():.1%}")
    print(f"\nExtraction Efficiency:")
    for side in ['buyer', 'seller']:
        sub = eff[eff['side'] == side]['efficiency']
        if len(sub) > 0:
            print(f"  {side.title():>6}: mean={sub.mean():.3f}, median={sub.median():.3f}")
    print(f"\nConstraint Violations:")
    total_ann = len(ann)
    total_viol = ann['violation'].sum()
    print(f"  Announcements: {total_viol}/{total_ann} ({total_viol / total_ann:.1%})")
    total_tx = len(tx)
    print(f"  Buyer tx violations:  {tx['buyer_violation'].sum()}/{total_tx}")
    print(f"  Seller tx violations: {tx['seller_violation'].sum()}/{total_tx}")
    print(f"\nWho Trades:")
    print(f"  Buyer inframarginal:  {tx['buyer_inframarginal'].mean():.1%}")
    print(f"  Seller inframarginal: {tx['seller_inframarginal'].mean():.1%}")
    print(f"  Both inframarginal:   {tx['both_inframarginal'].mean():.1%}")
    if 'filled' in ann.columns:
        print(f"\nFill Rate (fraction of submitted orders that executed immediately):")
        for side in ['buyer', 'seller']:
            sub = ann[ann['side'] == side]
            print(f"  {side.title():>6}: {sub['filled'].mean():.1%}")


def print_market_summary_table(df_metrics: pd.DataFrame, eq: Equilibrium,
                               experiment_id: str = None):
    """Print Smith (1962) style summary table."""
    print(f"\n{'=' * 60}")
    print(f"MARKET SUMMARY TABLE (Smith 1962 style)")
    print(f"{'=' * 60}")
    summary = df_metrics.groupby('round').agg(
        n_trades=('n_trades', 'mean'), mean_price=('mean_price', 'mean'),
        alpha=('alpha', 'mean'), efficiency=('efficiency', 'mean'),
        n_extramarginal=('n_extramarginal', 'mean'),
    ).round(3)
    summary.insert(0, 'eq_quantity', eq.quantity)
    summary.insert(2, 'eq_price', eq.price)
    smith_key = None
    if experiment_id:
        smith_key = experiment_id.split('_')[0]
    if smith_key and smith_key in SMITH_ALPHA:
        summary['smith_alpha'] = summary.index.map(SMITH_ALPHA[smith_key])
    print(summary.to_string())
    return summary


# ============================================================
# 7. PLOTTING -- VALIDATION
# ============================================================

def _plot_trajectory_panel(ax, df_metrics, col, color, ylabel,
                           independent_rounds, n_sims,
                           hline_val=None, hline_color='#e74c3c',
                           hline_label=None):
    """Helper for bottom row of validation plot."""
    if independent_rounds:
        ax.scatter(df_metrics['sim'], df_metrics[col], alpha=0.15, s=10, color=color)
        ax.axhline(df_metrics[col].mean(), color=color, lw=2,
                   label=f"mean = {df_metrics[col].mean():.3f}")
        if hline_val is not None:
            ax.axhline(hline_val, color=hline_color, ls='--', lw=1, alpha=0.5, label=hline_label)
        ax.set_xlabel('Simulation'); ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} across simulations\n(stationarity check)')
        ax.legend(fontsize=9)
    else:
        by_round = df_metrics.groupby('round')[col].agg(['mean', 'sem'])
        for sim_id, sim_data in df_metrics.groupby('sim'):
            ax.plot(sim_data['round'], sim_data[col], alpha=0.15, color=color, lw=0.8)
        ax.errorbar(by_round.index, by_round['mean'], yerr=by_round['sem'],
                    fmt='-o', color=color, capsize=3, markersize=5, lw=2, zorder=5)
        if hline_val is not None:
            ax.axhline(hline_val, color=hline_color, ls='--', lw=1, alpha=0.5, label=hline_label)
        ax.set_xlabel('Round'); ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} by round\n(mean \u00b1 SEM, n={n_sims} sims)')
        ax.set_xticks(sorted(df_metrics['round'].unique()))
        if hline_label: ax.legend(fontsize=9)


def plot_validation(df_metrics, df_marsh, eq, title="Market Experiment",
                    independent_rounds=False):
    """Generate 2x3 validation plots."""
    n_rounds = df_metrics['round'].nunique()
    n_sims = df_metrics['sim'].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    # Row 0
    ax = axes[0, 0]
    ax.hist(df_metrics['efficiency'], bins=30, color='#16a34a', alpha=0.7, edgecolor='white')
    ax.axvline(df_metrics['efficiency'].mean(), color='black', ls='--', lw=1.5,
               label=f"mean = {df_metrics['efficiency'].mean():.3f}")
    ax.axvline(1.0, color='#e74c3c', ls='-', lw=1, label='CE (100%)')
    ax.set_xlabel('Allocative efficiency'); ax.set_ylabel('Count')
    ax.set_title('Efficiency distribution'); ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.hist(df_metrics['alpha'], bins=30, color='#2563eb', alpha=0.7, edgecolor='white')
    ax.axvline(df_metrics['alpha'].mean(), color='black', ls='--', lw=1.5,
               label=f"mean = {df_metrics['alpha'].mean():.1f}%")
    ax.set_xlabel("Smith's \u03b1 (%)"); ax.set_ylabel('Count')
    ax.set_title('\u03b1 distribution'); ax.legend(fontsize=9)

    ax = axes[0, 2]
    trade_counts = df_metrics['n_trades'].value_counts().sort_index()
    ax.bar(trade_counts.index, trade_counts.values, color='#d97706', alpha=0.7, edgecolor='white')
    ax.axvline(eq.quantity, color='#e74c3c', ls='--', lw=1.5, label=f'CE = {eq.quantity}')
    ax.set_xlabel('Trades per round'); ax.set_ylabel('Count')
    ax.set_title('Quantity distribution'); ax.legend(fontsize=9)

    # Row 1
    _plot_trajectory_panel(axes[1, 0], df_metrics, 'efficiency', '#16a34a',
                           'Efficiency', independent_rounds, n_sims, hline_val=1.0)
    _plot_trajectory_panel(axes[1, 1], df_metrics, 'mean_price', '#d97706',
                           'Mean price', independent_rounds, n_sims,
                           hline_val=eq.price, hline_label=f'CE = {eq.price:.2f}')
    ax = axes[1, 2]
    max_seq = int(df_marsh['tx_seq'].quantile(0.95))
    marsh_agg = df_marsh[df_marsh['tx_seq'] <= max_seq].groupby('tx_seq').agg(
        mean_dev=('price_dev', 'mean'), sem_dev=('price_dev', 'sem'))
    ax.errorbar(marsh_agg.index, marsh_agg['mean_dev'], yerr=marsh_agg['sem_dev'],
                fmt='-o', color='#7c3aed', capsize=3, markersize=5)
    ax.set_xlabel('Transaction position within round')
    ax.set_ylabel('Mean |price - CE|'); ax.set_title('Within-round convergence\n(Marshallian path)')
    plt.tight_layout(); plt.show()
    return fig


def plot_price_convergence(df_metrics, eq, title=None, independent_rounds=False):
    """Price convergence chart with alpha annotations."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sim_color, avg_color = '#1f77b4', '#d62728'
    y_min = max(min(eq.supply[0], eq.demand[-1]) - 1, 0)
    y_max = max(eq.supply[-1], eq.demand[0]) + 1
    y_range = y_max - y_min
    alpha_y_pos = y_max - 0.05 * y_range
    avg_alpha = df_metrics.groupby('round')['alpha'].mean()
    if independent_rounds:
        ax.scatter(df_metrics['sim'], df_metrics['mean_price'], alpha=0.3, s=15, color=sim_color)
        ax.axhline(df_metrics['mean_price'].mean(), color=avg_color, lw=2, label='Average Price')
        ax.axhline(eq.price, color='grey', ls='--', lw=1.2, alpha=0.6)
        ax.text(df_metrics['sim'].min(), eq.price - 0.15, 'Competitive Equilibrium Price',
                color='grey', fontsize=10, family='serif', va='bottom')
        ax.text(df_metrics['sim'].median(), alpha_y_pos, f'\u03b1={avg_alpha.iloc[0]:.1f}',
                ha='center', va='top', fontsize=10, family='serif',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('Simulation', fontsize=12, family='serif')
    else:
        df_plot = df_metrics.pivot_table(index='round', columns='sim', values='mean_price')
        for sim_id in df_plot.columns:
            ax.plot(df_plot.index, df_plot[sim_id], color=sim_color, alpha=0.3, linewidth=1)
        ax.plot(df_plot.index, df_plot.mean(axis=1), color=avg_color, linewidth=2, label='Average Price')
        ax.axhline(y=eq.price, color='grey', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.text(df_plot.index[0], eq.price - 0.15, 'Competitive Equilibrium Price',
                color='grey', fontsize=10, family='serif', va='bottom')
        for round_num, alpha_value in avg_alpha.items():
            ax.text(round_num, alpha_y_pos, f'\u03b1={alpha_value:.1f}',
                    ha='center', va='top', fontsize=10, family='serif',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
        ax.set_xticks(sorted(df_metrics['round'].unique()))
        ax.set_xlabel('Round', fontsize=12, family='serif')
    ax.set_ylabel('Average Transaction Price per Round', fontsize=12, family='serif')
    ax.set_title(title or 'Average Transaction Price per Round Across Simulations',
                 fontsize=14, family='serif', pad=10)
    ax.tick_params(axis='both', direction='in', labelsize=10, colors='#333333')
    for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_family('serif')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8); ax.spines['bottom'].set_linewidth(0.8)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='lower left', frameon=False, prop={'family': 'serif', 'size': 10})
    ax.grid(False); plt.tight_layout(); plt.show()
    return fig


def plot_smith_comparison(df_metrics, eq, experiment_id=None, title=None):
    """Plot alpha: LLM vs Smith (1962)."""
    smith_key = experiment_id.split('_')[0] if experiment_id else None
    if smith_key not in SMITH_ALPHA: smith_key = 'smith1'
    fig, ax = plt.subplots(figsize=(7, 5))
    sr = df_metrics.groupby('round')['alpha'].agg(['mean', 'sem'])
    ax.errorbar(sr.index, sr['mean'], yerr=sr['sem'], fmt='-o', color='#1565C0', capsize=3, markersize=6, label='LLM agents')
    smith_data = SMITH_ALPHA[smith_key]
    exp_rounds = set(df_metrics['round'].unique())
    rs = [(r, smith_data[r]) for r in sorted(smith_data.keys()) if r in exp_rounds]
    if rs:
        ax.plot([r for r, v in rs], [v for r, v in rs], '--s', color='#E65100', markersize=6, label=f'Smith (1962) [{smith_key}]')
    ax.set_xlabel('Round'); ax.set_ylabel('Coefficient of convergence (%)')
    ax.set_title(title or 'Coefficient of Convergence\n(LLM agents mean \u00b1 SEM vs Smith 1962)')
    ax.set_xticks(sorted(df_metrics['round'].unique())); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


def plot_single_sim_prices(df_tx, df_metrics, eq, sim=1):
    """Transaction prices for a single simulation (Smith style)."""
    df_plot = df_tx[df_tx['sim'] == sim].copy()
    df_plot = df_plot.sort_values(['round', 'iteration']).reset_index(drop=True)
    df_plot['transaction_number'] = df_plot.groupby('round').cumcount() + 1
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(df_plot) > 0:
        ax.plot(df_plot.index, df_plot['price'], color='black', linewidth=2, label='Transaction Price', zorder=2)
        ax.scatter(df_plot.index, df_plot['price'], marker='s', s=80, color='#1f77b4', zorder=3)
    y_min = max(min(eq.supply[0], eq.demand[-1]) - 1, 0)
    y_max = max(eq.supply[-1], eq.demand[0]) + 1
    alpha_y_pos = y_max - 0.05 * (y_max - y_min)
    last_indices = df_plot.groupby('round').apply(lambda g: g.index.max())
    for x in last_indices.values[:-1]:
        ax.axvline(x=x + 0.5, color='grey', linestyle='--', linewidth=1.5)
    rounds = df_plot['round'].unique()
    first_indices = df_plot.groupby('round').apply(lambda g: g.index.min())
    sim_metrics = df_metrics[df_metrics['sim'] == sim]
    for i, round_num in enumerate(rounds):
        start_pos = first_indices.iloc[i] if i == 0 else last_indices.iloc[i - 1] + 0.5
        end_pos = last_indices.iloc[i] + 0.5 if i < len(last_indices) else df_plot.index.max()
        mid_pos = (start_pos + end_pos) / 2
        alpha_row = sim_metrics[sim_metrics['round'] == round_num]
        if len(alpha_row) > 0:
            ax.text(mid_pos, alpha_y_pos, f'\u03b1={alpha_row["alpha"].values[0]:.1f}',
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    ax.axhline(y=eq.price, color='red', linestyle='--')
    ax.set_xticks(df_plot.index); ax.set_xticklabels(df_plot['transaction_number'], rotation=0)
    ax.set_title(f'Transaction Prices Across Rounds, simulation {sim}')
    ax.set_xlabel('Transaction within each round'); ax.set_ylabel('Price ($)')
    ax.set_ylim(y_min, y_max); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


# ============================================================
# 8. PLOTTING -- ORDER FLOW & RENT ANALYSIS
# ============================================================

def plot_order_flow(df_iter, eq):
    """Plot submitted bid/ask prices per simulation with trades highlighted."""
    ann_rows = df_iter[df_iter['announcement_made'] == True].copy()
    sims = sorted(ann_rows['sim'].unique())
    n_sims = len(sims)
    n_cols = min(2, n_sims)
    n_rows = (n_sims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    y_min = max(min(eq.supply[0], eq.demand[-1]) - 1, 0)
    y_max = max(eq.supply[-1], eq.demand[0]) + 1
    for i, sim in enumerate(sims):
        ax = axes_flat[i]
        df_plot = ann_rows[ann_rows['sim'] == sim]
        buy_mask = df_plot['announcement_type'] == 'buy'
        sell_mask = df_plot['announcement_type'] == 'sell'
        ax.plot(df_plot.loc[buy_mask].index, df_plot.loc[buy_mask, 'price'],
                marker='^', markersize=8, linestyle='-', label='Bid', color='#1f77b4')
        ax.plot(df_plot.loc[sell_mask].index, df_plot.loc[sell_mask, 'price'],
                marker='s', markersize=8, linestyle='-', label='Ask', color='#d62728')
        traded = df_plot[df_plot['transaction_made'] == True]
        for at, mk, c in [('buy', '^', '#1f77b4'), ('sell', 's', '#d62728')]:
            sub = traded[traded['announcement_type'] == at]
            ax.plot(sub.index, sub['price'], marker=mk, markersize=10, linestyle='',
                    markeredgecolor='black', markeredgewidth=1.5, color=c, label=f'{at.title()} \u2192 Trade')
        last_indices = df_plot.groupby('round').apply(lambda g: g.index.max())
        for x in last_indices.values[:-1]:
            ax.axvline(x=x + 0.5, color='grey', linestyle='--', linewidth=1.5)
        ax.axhline(y=eq.price, color='red', linestyle='--')
        ax.set_xticks([]); ax.set_title(f'Order Flow, simulation {sim}')
        ax.set_ylabel('Submitted Price'); ax.set_ylim(y_min, y_max); ax.legend(fontsize=8)
    for j in range(i + 1, len(axes_flat)): axes_flat[j].set_visible(False)
    plt.tight_layout(); plt.show()
    return fig


def plot_bid_ask_dispersion(df_ann, eq):
    """Std of bids/asks within round (mean +/- std across sims)."""
    buy_std = df_ann[df_ann['side'] == 'buyer'].groupby(['sim', 'round'])['price'].std()
    sell_std = df_ann[df_ann['side'] == 'seller'].groupby(['sim', 'round'])['price'].std()
    df_std = pd.concat([buy_std, sell_std], axis=1)
    df_std.columns = ['bid', 'ask']
    mean = df_std.groupby(level=1).mean()
    std = df_std.groupby(level=1).std()
    colors = {'ask': '#d62728', 'bid': '#1f77b4'}
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df_std.columns:
        ax.plot(mean.index, mean[col], label=col.title(), marker="o", color=colors[col])
        ax.fill_between(mean.index, mean[col] - std[col], mean[col] + std[col], alpha=0.2, color=colors[col])
    ax.set_xticks(mean.index)
    ax.legend(loc='upper right', frameon=False, prop={'family': 'serif', 'size': 10})
    ax.set_xlabel("Round", fontsize=12, family='serif')
    ax.set_ylabel("Mean Standard Deviation", fontsize=12, family='serif')
    ax.set_title("Std of Bids/Asks within Round (mean across simulations)", fontsize=14, family='serif', pad=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.show()
    return fig


def plot_order_price_vs_reservation(df_ann, eq):
    """Scatter: submitted order price vs reservation price."""
    import seaborn as sns
    buy_mask = df_ann['side'] == 'buyer'
    sell_mask = df_ann['side'] == 'seller'
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8, 8))
        bc = sns.color_palette("colorblind")[0]
        sc = sns.color_palette("colorblind")[1]
        ax.scatter(df_ann.loc[buy_mask, 'announcing_agent_reservation_price'],
                   df_ann.loc[buy_mask, 'price'], color=bc, label='Buyers',
                   edgecolor='k', s=70, linewidth=0.5, marker='o')
        ax.scatter(df_ann.loc[sell_mask, 'announcing_agent_reservation_price'],
                   df_ann.loc[sell_mask, 'price'], color=sc, label='Sellers',
                   edgecolor='k', s=70, linewidth=0.5, marker='^')
        min_val = max(np.concatenate([eq.supply, eq.demand]).min() - 1, 0)
        max_val = np.concatenate([eq.supply, eq.demand]).max() + 1
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='45\u00b0 Line')
        ax.set_xlabel("Reservation Price", fontsize=14, labelpad=10)
        ax.set_ylabel("Submitted Order Price", fontsize=14, labelpad=10)
        ax.set_title("Submitted Order Price vs Reservation Price", fontsize=16, pad=15)
        ax.legend(frameon=True, fontsize=12); ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal'); ax.set_xlim(min_val, max_val); ax.set_ylim(min_val, max_val)
    plt.tight_layout(); plt.show()
    return fig


def plot_fill_rate(df_ann, eq):
    """Fill rate analysis (replaces old acceptance rate)."""
    if 'filled' not in df_ann.columns:
        print("No fill rate data available."); return None
    agent_fill = (df_ann.groupby(['sim', 'side', 'announcing_agent_id',
                                   'announcing_agent_reservation_price'])['filled']
                  .mean().reset_index(name='fill_rate'))
    agg = agent_fill.groupby(['side', 'announcing_agent_reservation_price']).agg(
        mean_rate=('fill_rate', 'mean'),
        se=('fill_rate', lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
    ).reset_index()
    agg['ci95'] = 1.96 * agg['se']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    colors = {'buyer': '#1f77b4', 'seller': '#d62728'}
    for side in ['buyer', 'seller']:
        df_t = agg[agg['side'] == side].sort_values('announcing_agent_reservation_price')
        ax.plot(df_t['announcing_agent_reservation_price'], df_t['mean_rate'],
                marker='o', color=colors[side], label=side.title())
        ax.fill_between(df_t['announcing_agent_reservation_price'],
                        df_t['mean_rate'] - df_t['ci95'], df_t['mean_rate'] + df_t['ci95'],
                        color=colors[side], alpha=0.3)
    ax.axvline(eq.price, color='grey', ls='--', lw=0.8, label='CE price')
    ax.set_xlabel("Reservation Price", fontsize=12, family='serif')
    ax.set_ylabel("Fill Rate", fontsize=12, family='serif')
    ax.set_title("Fill Rate vs Reservation Price", fontsize=14, family='serif', pad=10)
    ax.legend(frameon=False, prop={'family': 'serif', 'size': 10})
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax = axes[1]
    for side, color, marker in [('buyer', '#1f77b4', 'o'), ('seller', '#d62728', 's')]:
        sr = (df_ann[df_ann['side'] == side].groupby(['sim', 'round'])['filled']
              .mean().reset_index(name='fill_rate'))
        r_agg = sr.groupby('round')['fill_rate'].agg(['mean', 'sem'])
        ax.errorbar(r_agg.index, r_agg['mean'], yerr=r_agg['sem'],
                    fmt=f'-{marker}', color=color, capsize=3, label=side.title())
    ax.set_xlabel('Round'); ax.set_ylabel('Fill Rate')
    ax.set_title('Fill Rate by Round\n(mean \u00b1 SEM across sims)')
    ax.set_xticks(sorted(df_ann['round'].unique()))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.show()
    return fig


def _histogram_by_side(ax, data, value_col, sims, n_bins=20, title='', xlabel=''):
    """Helper: side-by-side histogram with mean +/- SEM per bin."""
    all_vals = data[value_col].dropna()
    if len(all_vals) == 0: return
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]; half = width / 2
    for side, color, offset in [('buyer', '#2196F3', -half/2), ('seller', '#F44336', half/2)]:
        counts = np.zeros((len(sims), len(bin_centers)))
        for i, s in enumerate(sims):
            vals = data[(data['side'] == side) & (data['sim'] == s)][value_col]
            counts[i], _ = np.histogram(vals, bins=bin_edges)
        mean_counts = counts.mean(axis=0)
        sem_counts = counts.std(axis=0, ddof=1) / np.sqrt(len(sims))
        ax.bar(bin_centers + offset, mean_counts, width=half, color=color, edgecolor='white', label=side.title())
        ax.errorbar(bin_centers + offset, mean_counts, yerr=sem_counts, fmt='none', ecolor='black', capsize=1.5, lw=0.8)
    ax.axvline(0, color='grey', ls='--', lw=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel('Mean count per simulation'); ax.set_title(title); ax.legend()


def plot_attempted_rent(rent, eq):
    """Metric 1: Attempted Rent Margin (2x2). Uses submitted order prices."""
    ann = rent['ann']; sims = sorted(ann['sim'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _histogram_by_side(axes[0, 0], ann, 'attempted_rent', sims,
                       title='1a) Distribution (mean \u00b1 SEM)', xlabel='Attempted rent ($)')
    ax = axes[0, 1]
    for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
        sub = ann[ann['side'] == side]
        sim_agent = sub.groupby(['sim', 'announcing_agent_reservation_price'])['attempted_rent'].mean().reset_index()
        agg = sim_agent.groupby('announcing_agent_reservation_price')['attempted_rent'].agg(['mean', 'sem']).reset_index()
        ax.errorbar(agg['announcing_agent_reservation_price'], agg['mean'], yerr=agg['sem'],
                    fmt=f'-{marker}', color=color, capsize=3, label=side.title(), markersize=5)
    ax.axhline(0, color='grey', ls='--', lw=0.8); ax.set_xlabel('Reservation price ($)')
    ax.set_ylabel('Mean attempted rent ($)'); ax.set_title('1b) By reservation price'); ax.legend()
    ax = axes[1, 0]
    for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
        sr = ann[ann['side'] == side].groupby(['sim', 'round'])['attempted_rent'].mean().reset_index()
        agg = sr.groupby('round')['attempted_rent'].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt=f'-{marker}', color=color, capsize=3, label=side.title())
    ax.set_xlabel('Round'); ax.set_ylabel('Mean attempted rent ($)'); ax.set_title('1c) Over rounds')
    ax.set_xticks(sorted(ann['round'].unique())); ax.legend()
    ax = axes[1, 1]
    agent_sim = ann.groupby(['announcing_agent_id', 'side', 'sim'])['attempted_rent'].mean().reset_index()
    agent_agg = agent_sim.groupby(['announcing_agent_id', 'side'])['attempted_rent'].agg(['mean', 'sem']).reset_index()
    buyers = agent_agg[agent_agg['side'] == 'buyer'].sort_values('mean')
    sellers = agent_agg[agent_agg['side'] == 'seller'].sort_values('mean')
    y_pos = np.arange(len(buyers) + len(sellers))
    labels, means, sems, colors = [], [], [], []
    for _, r in buyers.iterrows():
        labels.append(f'B{int(r.announcing_agent_id)}'); means.append(r['mean']); sems.append(r['sem']); colors.append('#2196F3')
    for _, r in sellers.iterrows():
        labels.append(f'S{int(r.announcing_agent_id)}'); means.append(r['mean']); sems.append(r['sem']); colors.append('#F44336')
    ax.barh(y_pos, means, xerr=sems, color=colors, edgecolor='white', height=0.7, capsize=2, error_kw={'lw': 0.8})
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean attempted rent ($)'); ax.set_title('1d) Per-agent')
    fig.suptitle('Metric 1: Attempted Rent Margin (submitted order prices)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()
    return fig


def plot_rent_ratio(rent, eq):
    """Metric 2: Attempted Rent Ratio (2x2)."""
    ann = rent['ann']; sims = sorted(ann['sim'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _histogram_by_side(axes[0, 0], ann, 'rent_ratio', sims, n_bins=25,
                       title='2a) Distribution', xlabel='Rent ratio')
    ax = axes[0, 1]
    for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
        sub = ann[ann['side'] == side]
        sim_agent = sub.groupby(['sim', 'announcing_agent_reservation_price'])['rent_ratio'].mean().reset_index()
        agg = sim_agent.groupby('announcing_agent_reservation_price')['rent_ratio'].agg(['mean', 'sem']).reset_index()
        ax.errorbar(agg['announcing_agent_reservation_price'], agg['mean'], yerr=agg['sem'],
                    fmt=f'-{marker}', color=color, capsize=3, label=side.title(), markersize=5)
    ax.axhline(0, color='grey', ls='--', lw=0.8); ax.set_xlabel('Reservation price ($)')
    ax.set_ylabel('Mean rent ratio'); ax.set_title('2b) By reservation price'); ax.legend()
    ax = axes[1, 0]
    for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
        sr = ann[ann['side'] == side].groupby(['sim', 'round'])['rent_ratio'].mean().reset_index()
        agg = sr.groupby('round')['rent_ratio'].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt=f'-{marker}', color=color, capsize=3, label=side.title())
    ax.set_xlabel('Round'); ax.set_ylabel('Mean rent ratio'); ax.set_title('2c) Over rounds')
    ax.set_xticks(sorted(ann['round'].unique())); ax.legend()
    ax = axes[1, 1]
    agent_sim = ann.groupby(['announcing_agent_id', 'side', 'sim'])['rent_ratio'].mean().reset_index()
    agent_agg = agent_sim.groupby(['announcing_agent_id', 'side'])['rent_ratio'].agg(['mean', 'sem']).reset_index()
    buyers = agent_agg[agent_agg['side'] == 'buyer'].sort_values('mean')
    sellers = agent_agg[agent_agg['side'] == 'seller'].sort_values('mean')
    y_pos = np.arange(len(buyers) + len(sellers))
    labels, means, sems, colors = [], [], [], []
    for _, r in buyers.iterrows():
        labels.append(f'B{int(r.announcing_agent_id)}'); means.append(r['mean']); sems.append(r['sem']); colors.append('#2196F3')
    for _, r in sellers.iterrows():
        labels.append(f'S{int(r.announcing_agent_id)}'); means.append(r['mean']); sems.append(r['sem']); colors.append('#F44336')
    ax.barh(y_pos, means, xerr=sems, color=colors, edgecolor='white', height=0.7, capsize=2, error_kw={'lw': 0.8})
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean rent ratio'); ax.set_title('2d) Per-agent')
    fig.suptitle('Metric 2: Attempted Rent Ratio', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()
    return fig


def plot_realized_rent(rent, eq):
    """Metric 3: Realized Rent (2x2). Uses execution prices."""
    tx = rent['tx']; sims = sorted(tx['sim'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    all_rents = pd.concat([tx['buyer_rent'], tx['seller_rent']])
    bin_edges = np.linspace(all_rents.min(), all_rents.max(), 26)
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2; w = bin_edges[1] - bin_edges[0]; h = w / 2
    for col, color, offset, label in [('buyer_rent', '#2196F3', -h/2, 'Buyer'), ('seller_rent', '#F44336', h/2, 'Seller')]:
        counts = np.zeros((len(sims), len(bc)))
        for i, s in enumerate(sims): counts[i], _ = np.histogram(tx[tx['sim'] == s][col], bins=bin_edges)
        ax.bar(bc + offset, counts.mean(0), width=h, color=color, edgecolor='white', label=label)
        ax.errorbar(bc + offset, counts.mean(0), yerr=counts.std(0, ddof=1)/np.sqrt(len(sims)), fmt='none', ecolor='black', capsize=1.5, lw=0.8)
    ax.set_xlabel('Realized rent ($)'); ax.set_ylabel('Mean count'); ax.set_title('3a) Distribution'); ax.legend()
    ax = axes[0, 1]
    tx_pos = tx[tx['total_surplus'] > 0].copy()
    n_excluded = tx['buyer_share'].isna().sum()
    srs = tx_pos.groupby(['sim', 'round'])[['buyer_share', 'seller_share']].mean().reset_index()
    for col, color, label in [('buyer_share', '#2196F3', 'Buyer'), ('seller_share', '#F44336', 'Seller')]:
        agg = srs.groupby('round')[col].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt='-o', color=color, capsize=3, label=label)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, label='Equal split')
    ax.set_ylabel('Surplus share'); ax.set_xlabel('Round'); ax.set_title('3b) Surplus share by round')
    ax.set_xticks(sorted(tx['round'].unique())); ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend(fontsize=8)
    ax = axes[1, 0]
    srm = tx.groupby(['sim', 'round'])['total_surplus'].mean().reset_index()
    agg = srm.groupby('round')['total_surplus'].agg(['mean', 'sem'])
    ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt='-o', color='#4CAF50', capsize=3)
    ax.set_xlabel('Round'); ax.set_ylabel('Mean surplus/tx ($)'); ax.set_title('3c) Mean surplus per tx')
    ax.set_xticks(sorted(tx['round'].unique()))
    ax = axes[1, 1]
    srs2 = tx.groupby(['sim', 'round'])['total_surplus'].sum().reset_index()
    agg = srs2.groupby('round')['total_surplus'].agg(['mean', 'sem'])
    ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt='-o', color='#4CAF50', capsize=3, label='Realised')
    ax.axhline(eq.surplus, color='grey', ls='--', lw=0.8, label=f'Equilibrium (${eq.surplus:.2f})')
    ax.set_xlabel('Round'); ax.set_ylabel('Total surplus ($)'); ax.set_title('3d) Total surplus vs eq')
    ax.set_xticks(sorted(tx['round'].unique())); ax.legend(fontsize=8)
    fig.suptitle('Metric 3: Realized Rent (execution prices)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()
    return fig


def plot_extraction_efficiency(rent, eq):
    """Metric 4: Rent Extraction Efficiency (1x3)."""
    eff = rent['extraction_eff']; ann = rent['ann']; sims = sorted(eff['sim'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _histogram_by_side(axes[0], eff, 'efficiency', sims, n_bins=25,
                       title='4a) Distribution', xlabel='Extraction efficiency')
    axes[0].axvline(1.0, color='grey', ls='--', lw=0.8, label='100%'); axes[0].legend(fontsize=8)
    ax = axes[1]
    for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
        sr = eff[eff['side'] == side].groupby(['sim', 'round'])['efficiency'].mean().reset_index()
        agg = sr.groupby('round')['efficiency'].agg(['mean', 'sem'])
        if len(agg) > 0: ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt=f'-{marker}', color=color, capsize=3, label=side.title())
    ax.axhline(1.0, color='grey', ls='--', lw=0.8); ax.set_xlabel('Round'); ax.set_ylabel('Mean efficiency')
    ax.set_title('4b) Over rounds'); ax.set_xticks(sorted(ann['round'].unique())); ax.legend()
    ax = axes[2]
    for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
        sub = eff[eff['side'] == side]
        ax.scatter(sub['first_attempted_rent'], sub['realized_rent'], alpha=0.2, c=color, marker=marker, s=25, label=side.title())
    mv = max(eff['first_attempted_rent'].max(), eff['realized_rent'].max()) * 1.05
    ax.plot([0, mv], [0, mv], 'k--', lw=0.8, alpha=0.5, label='100% line')
    ax.set_xlabel('First attempted rent ($)'); ax.set_ylabel('Realised rent ($)')
    ax.set_title('4c) Ambition vs outcome'); ax.legend(fontsize=8)
    fig.suptitle('Metric 4: Rent Extraction Efficiency', fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout(); plt.show()
    return fig


def plot_concession_rate(rent, eq):
    """Metric 5: Concession Rate (1x3)."""
    conc = rent['concessions']; ann = rent['ann']; abr = rent['rent_surrendered']
    sims = sorted(conc['sim'].unique()) if len(conc) > 0 else []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    if len(conc) > 0:
        _histogram_by_side(axes[0], conc, 'concession', sims, n_bins=30,
                           title='5a) Concession distribution', xlabel='Concession ($)')
    ax = axes[1]
    if len(conc) > 0:
        for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
            sr = conc[conc['side'] == side].groupby(['sim', 'round'])['frac_concession'].mean().reset_index()
            agg = sr.groupby('round')['frac_concession'].agg(['mean', 'sem'])
            if len(agg) > 0: ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt=f'-{marker}', color=color, capsize=3, label=side.title())
    ax.axhline(0, color='grey', ls='--', lw=0.8); ax.set_xlabel('Round'); ax.set_ylabel('Frac concession')
    ax.set_title('5b) Over rounds'); ax.set_xticks(sorted(ann['round'].unique()))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend()
    ax = axes[2]
    if len(abr) > 0:
        for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
            sr = abr[abr['side'] == side].groupby(['sim', 'round'])['frac_surrendered'].mean().reset_index()
            agg = sr.groupby('round')['frac_surrendered'].agg(['mean', 'sem'])
            if len(agg) > 0: ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt=f'-{marker}', color=color, capsize=3, label=side.title())
    ax.axhline(0, color='grey', ls='--', lw=0.8); ax.axhline(1, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Round'); ax.set_ylabel('Frac surrendered'); ax.set_title('5c) Cumulative surrender')
    ax.set_xticks(sorted(ann['round'].unique())); ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend()
    fig.suptitle('Metric 5: Concession Rate (submitted prices)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()
    return fig


def plot_zero_profit_orders(rent, eq):
    """Metric 6: % of orders at exactly reservation price."""
    ann = rent['ann']
    fig, ax = plt.subplots(figsize=(6, 4))
    for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
        sr = (ann[ann['side'] == side].groupby(['sim', 'round'])
              .apply(lambda g: (g['attempted_rent'] == 0).mean(), include_groups=False)
              .reset_index(name='pct'))
        agg = sr.groupby('round')['pct'].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt=f'-{marker}', color=color, capsize=3, label=side.title())
    ax.set_xlabel('Round'); ax.set_ylabel('% at reservation price')
    ax.set_title('Metric 6: Orders at Exactly Reservation Price')
    ax.set_xticks(sorted(ann['round'].unique())); ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


def plot_constraint_violations(rent, eq):
    """Constraint violations: order-level and transaction-level."""
    ann = rent['ann']; tx = rent['tx']
    ann_viol = ann.groupby(['sim', 'round', 'side']).agg(n_total=('violation', 'count'), n_viol=('violation', 'sum')).reset_index()
    ann_viol['rate'] = ann_viol['n_viol'] / ann_viol['n_total']
    tx_viol = tx.groupby(['sim', 'round']).agg(n_tx=('buyer_violation', 'count'),
        n_bv=('buyer_violation', 'sum'), n_sv=('seller_violation', 'sum')).reset_index()
    tx_viol['bv_rate'] = tx_viol['n_bv'] / tx_viol['n_tx']; tx_viol['sv_rate'] = tx_viol['n_sv'] / tx_viol['n_tx']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
        sub = ann_viol[ann_viol['side'] == side]
        agg = sub.groupby('round')['rate'].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt=f'-{marker}', color=color, capsize=3, label=side.title())
    ax.set_xlabel('Round'); ax.set_ylabel('Violation rate')
    ax.set_title('Order Violations\n(submitted beyond reservation)'); ax.set_xticks(sorted(ann['round'].unique()))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend()
    ax = axes[1]
    for col, color, marker, label in [('bv_rate', '#2196F3', 'o', 'Buyer'), ('sv_rate', '#F44336', 's', 'Seller')]:
        agg = tx_viol.groupby('round')[col].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt=f'-{marker}', color=color, capsize=3, label=label)
    ax.set_xlabel('Round'); ax.set_ylabel('Violation rate')
    ax.set_title('Transaction Violations\n(execution at a loss)'); ax.set_xticks(sorted(tx['round'].unique()))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


def plot_order_frequency(rent, eq):
    """Order frequency by round."""
    ann = rent['ann']
    ac = ann.groupby(['sim', 'round', 'side']).size().reset_index(name='n')
    fig, ax = plt.subplots(figsize=(6, 4))
    for side, color, marker in [('buyer', '#2196F3', 'o'), ('seller', '#F44336', 's')]:
        sub = ac[ac['side'] == side]
        agg = sub.groupby('round')['n'].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt=f'-{marker}', color=color, capsize=3, label=side.title())
    ax.set_xlabel('Round'); ax.set_ylabel('Orders per round')
    ax.set_title('Order Frequency\n(mean \u00b1 SEM)'); ax.set_xticks(sorted(ann['round'].unique())); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


def plot_who_trades(rent, eq):
    """Who trades: inframarginal participation."""
    tx = rent['tx']; sims = sorted(tx['sim'].unique())
    tw = tx.groupby(['sim', 'round']).agg(n=('both_inframarginal', 'count'),
        nb=('buyer_inframarginal', 'sum'), ns=('seller_inframarginal', 'sum'),
        nboth=('both_inframarginal', 'sum')).reset_index()
    tw['pb'] = tw['nb']/tw['n']; tw['ps'] = tw['ns']/tw['n']; tw['pboth'] = tw['nboth']/tw['n']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for col, color, marker, label in [('pb', '#2196F3', 'o', 'Buyer infra'), ('ps', '#F44336', 's', 'Seller infra'), ('pboth', '#4CAF50', 'D', 'Both infra')]:
        agg = tw.groupby('round')[col].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt=f'-{marker}', color=color, capsize=3, label=label)
    ax.set_xlabel('Round'); ax.set_ylabel('Fraction'); ax.set_title('Inframarginal Participation')
    ax.set_xticks(sorted(tx['round'].unique())); ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend(fontsize=8)
    ax = axes[1]
    all_res = np.sort(np.union1d(eq.supply, eq.demand)); x = np.arange(len(all_res)); w = 0.35
    for col, color, offset, label in [('buyer_val', '#2196F3', -w/2, 'Buyer'), ('seller_cost', '#F44336', w/2, 'Seller')]:
        counts = np.zeros((len(sims), len(all_res)))
        for i, s in enumerate(sims):
            rc = tx[tx['sim'] == s][col].value_counts()
            for j, r in enumerate(all_res): counts[i, j] = rc.get(r, 0)
        ax.bar(x + offset, counts.mean(0), w, color=color, edgecolor='white', label=label)
        ax.errorbar(x + offset, counts.mean(0), yerr=counts.std(0, ddof=1)/np.sqrt(len(sims)), fmt='none', ecolor='black', capsize=1.5, lw=0.8)
    eq_idx = np.argmin(np.abs(all_res - eq.price))
    ax.axvline(x=eq_idx + 0.5, color='grey', ls='--', lw=0.8, label=f'Eq price')
    ax.set_xticks(x); ax.set_xticklabels([f'${r:.2f}' for r in all_res], rotation=45, fontsize=8)
    ax.set_xlabel('Reservation price ($)'); ax.set_ylabel('Mean tx/sim'); ax.set_title('Tx by Reservation Price'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    return fig


def plot_agent_rent_trajectories(rent, eq):
    """Agent rent trajectories grid: sims x rounds."""
    ann = rent['ann']; sims = sorted(ann['sim'].unique()); rounds = sorted(ann['round'].unique())
    fig, axes = plt.subplots(len(sims), len(rounds), figsize=(4*len(rounds), 3.5*len(sims)), squeeze=False)
    for row, s in enumerate(sims):
        for col, rnd in enumerate(rounds):
            ax = axes[row, col]
            ae = ann[(ann['sim'] == s) & (ann['round'] == rnd)]
            for agent, grp in ae.groupby('announcing_agent_id'):
                grp = grp.sort_values('iteration'); side = grp['side'].iloc[0]
                color = '#2196F3' if side == 'buyer' else '#F44336'
                if len(grp) == 1: ax.plot(grp['iteration'].values[0], grp['attempted_rent'].values[0], 'o', color=color, alpha=0.35, markersize=4)
                else: ax.plot(grp['iteration'], grp['attempted_rent'], '-o', color=color, alpha=0.5, markersize=3, lw=1.0)
            ax.axhline(0, color='grey', ls='--', lw=0.8)
            if row == 0: ax.set_title(f'Round {rnd}', fontsize=10)
            ax.set_xlabel('Iteration')
            if col == 0: ax.set_ylabel(f'Sim {s}\nAttempted rent ($)')
    axes[0, 0].legend(handles=[Line2D([0],[0],color='#2196F3',marker='o',label='Buyer'), Line2D([0],[0],color='#F44336',marker='o',label='Seller')], fontsize=8)
    fig.suptitle('Agent Rent Trajectories', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.show()
    return fig


# ============================================================
# 10. MAIN RUNNERS
# ============================================================

def run_validation(results_path: Path, n_sims: int, title: str = None,
                   config: dict = None, independent_rounds: bool = False) -> dict:
    """
    Run the core validation pipeline (efficiency, alpha, Marshallian path).

    Returns dict with keys:
        'eq', 'metrics', 'marshallian', 'tx', 'iter', 'agents', 'config',
        'fig_summary', 'fig_convergence'
    """
    import yaml

    if config is None:
        config_path = results_path / 'config_used.yaml'
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(
                f"No config found at {config_path}. "
                f"Pass config dict explicitly via config= parameter."
            )

    df_iter, df_tx, df_agents = load_experiment(results_path, n_sims)
    eq = compute_equilibrium(config, df_agents)
    df_tx = enrich_transactions(df_tx, eq)
    df_metrics = compute_round_metrics(df_tx, eq)
    df_marsh = compute_marshallian_path(df_tx, eq)

    print_summary(df_metrics, eq, independent_rounds)
    print_marshallian_summary(df_marsh, eq)

    plot_title = title or results_path.name
    fig_summary = plot_validation(df_metrics, df_marsh, eq, title=plot_title,
                                  independent_rounds=independent_rounds)
    fig_convergence = plot_price_convergence(df_metrics, eq, title=plot_title,
                                             independent_rounds=independent_rounds)

    return {
        'eq': eq, 'metrics': df_metrics, 'marshallian': df_marsh,
        'tx': df_tx, 'iter': df_iter, 'agents': df_agents, 'config': config,
        'fig_summary': fig_summary, 'fig_convergence': fig_convergence,
    }


def run_rent_analysis(df_tx: pd.DataFrame, df_ann: pd.DataFrame,
                      df_agents: pd.DataFrame, eq: Equilibrium,
                      experiment_id: str = None) -> dict:
    """Run the full rent-seeking / order flow analysis."""
    rent = compute_rent_metrics(df_tx, df_ann, eq)
    print_rent_summary(rent, eq)

    figs = {}
    figs['fig_attempted_rent'] = plot_attempted_rent(rent, eq)
    figs['fig_rent_ratio'] = plot_rent_ratio(rent, eq)
    figs['fig_realized_rent'] = plot_realized_rent(rent, eq)
    figs['fig_extraction_eff'] = plot_extraction_efficiency(rent, eq)
    figs['fig_concession'] = plot_concession_rate(rent, eq)
    figs['fig_zero_profit'] = plot_zero_profit_orders(rent, eq)
    figs['fig_violations'] = plot_constraint_violations(rent, eq)
    figs['fig_order_frequency'] = plot_order_frequency(rent, eq)
    figs['fig_who_trades'] = plot_who_trades(rent, eq)
    figs['fig_agent_trajectories'] = plot_agent_rent_trajectories(rent, eq)
    figs['fig_fill_rate'] = plot_fill_rate(df_ann, eq)

    return {'rent': rent, **figs}


def run_full_analysis(results_path: Path, n_sims: int, title: str = None,
                      config: dict = None, experiment_id: str = None,
                      independent_rounds: bool = False) -> dict:
    """
    Run both validation and rent-seeking analysis pipelines.

    Usage:
        from results_analysis import run_full_analysis
        results = run_full_analysis(Path('results/my_experiment'), n_sims=10)
    """
    val = run_validation(results_path, n_sims, title=title, config=config, 
                         independent_rounds=independent_rounds)
    df_ann = extract_announcements(val['iter'])

    rent_results = run_rent_analysis(
        val['tx'], df_ann, val['agents'], val['eq'],
        experiment_id=experiment_id,
    )

    extra_figs = {}
    if val['metrics']['round'].nunique() > 1:
        extra_figs['fig_smith'] = plot_smith_comparison(
            val['metrics'], val['eq'], experiment_id=experiment_id, title=title)

    extra_figs['fig_single_sim'] = plot_single_sim_prices(
        val['tx'], val['metrics'], val['eq'], sim=1)
    extra_figs['fig_order_flow'] = plot_order_flow(val['iter'], val['eq'])
    extra_figs['fig_dispersion'] = plot_bid_ask_dispersion(df_ann, val['eq'])
    extra_figs['fig_price_vs_res'] = plot_order_price_vs_reservation(df_ann, val['eq'])

    summary_table = print_market_summary_table(
        val['metrics'], val['eq'], experiment_id=experiment_id)

    return {
        **val, **rent_results, **extra_figs,
        'ann': df_ann, 'summary_table': summary_table,
    }


# ============================================================
# Standalone usage
# ============================================================
if __name__ == '__main__':
    import sys
    import yaml

    if len(sys.argv) < 2:
        print("Usage: python results_analysis.py <results_path> [n_sims]")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    config_path = results_path / 'config_used.yaml'
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"No config found at {config_path}")
        sys.exit(1)

    n_sims = int(sys.argv[2]) if len(sys.argv) >= 3 else config['experiment']['n_simulations']
    experiment_id = results_path.name
    run_full_analysis(results_path, n_sims, config=config, experiment_id=experiment_id)
