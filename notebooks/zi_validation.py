"""
Market Experiment Analysis
==========================
Loads agent_histories and transactions files, computes:
- Allocative efficiency (per round, aggregated across simulations)
- Smith's alpha (coefficient of convergence)
- Marshallian path analysis (within-round price convergence)
- Across-round trajectory (flat for ZI-C, convergent for LLMs)

Works with both:
- ZI-C experiments: 1 round per simulation, many simulations
- LLM experiments: N rounds per simulation, rounds are NOT independent

Usage:
    from zi_validation import run_validation
    results = run_validation(results_path, n_sims=500)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from dataclasses import dataclass


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_experiment(results_path: Path, n_sims: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load transactions and agent_histories for all available simulations.
    
    Returns:
        df_tx: transactions with columns [round, iteration, price, buyer_id, 
               seller_id, announcement_type, sim]
        df_agents: agent histories with columns [agent_id, agent_type, 
                   reservation_price, ..., sim]
    """
    available_sims = [
        sim for sim in range(1, n_sims + 1)
        if (results_path / 'data' / f'transactions_{sim}.csv').exists()
        and (results_path / 'data' / f'agent_histories_{sim}.csv').exists()
    ]
    
    if not available_sims:
        raise FileNotFoundError(f"No matching files found in {results_path / 'data'}")
    
    tx_list = [
        pd.read_csv(results_path / 'data' / f'transactions_{sim}.csv').assign(sim=sim)
        for sim in available_sims
    ]
    ah_list = [
        pd.read_csv(results_path / 'data' / f'agent_histories_{sim}.csv').assign(sim=sim)
        for sim in available_sims
    ]
    
    df_tx = pd.concat(tx_list, ignore_index=True)
    df_agents = pd.concat(ah_list, ignore_index=True)
    
    n_rounds = df_tx.groupby('sim')['round'].nunique().iloc[0]
    
    print(f"Loaded {len(available_sims)} simulations from {results_path.name}")
    print(f"  Transactions: {len(df_tx)} rows")
    print(f"  Rounds per sim: {n_rounds}")
    print(f"  Total round-observations: {df_tx.groupby(['sim', 'round']).ngroups}")
    
    return df_tx, df_agents


# ============================================================
# 2. EQUILIBRIUM COMPUTATION
# ============================================================

@dataclass
class Equilibrium:
    quantity: int
    price: float
    surplus: float
    demand: np.ndarray         # demand schedule (descending values)
    supply: np.ndarray         # supply schedule (ascending costs)
    buyer_map: dict            # agent_id -> reservation_price
    seller_map: dict           # agent_id -> reservation_price


def find_equilibrium(supply, demand, flat_threshold=1e-10):
    """
    Find equilibrium quantity and price where demand >= supply.
    
    Parameters:
        supply: array of supply prices at each quantity
        demand: array of demand prices at each quantity
        flat_threshold: tolerance for considering a curve "flat"
    
    Returns:
        q_eq: equilibrium quantity (number of trades)
        p_eq: equilibrium price
    """
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
    """
    Visualize supply and demand curves with equilibrium point.
    """
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
    """
    Compute competitive equilibrium from config schedules and build
    agent_id -> reservation_price maps from agent_histories.
    """
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
    
    agents = df_agents[['agent_id', 'agent_type', 'reservation_price']].drop_duplicates()
    buyers = agents[agents['agent_type'] == 'buyer']
    sellers = agents[agents['agent_type'] == 'seller']
    bv_dict = dict(zip(buyers['agent_id'], buyers['reservation_price']))
    sv_dict = dict(zip(sellers['agent_id'], sellers['reservation_price']))
    
    eq = Equilibrium(
        quantity=q_eq, price=p_eq, surplus=ce_surplus,
        demand=demand, supply=supply,
        buyer_map=bv_dict, seller_map=sv_dict,
    )
    
    print(f"\nCompetitive Equilibrium:")
    print(f"  Quantity: {eq.quantity}")
    print(f"  Price:    {eq.price:.4f}")
    print(f"  Surplus:  {eq.surplus:.4f}")
    print(f"  Demand:   {demand}")
    print(f"  Supply:   {supply}")
    
    plot_equilibrium(supply, demand, q_eq, p_eq)
    plt.show()
    
    return eq


# ============================================================
# 3. PER-ROUND METRICS
# ============================================================

def compute_round_metrics(df_tx: pd.DataFrame, eq: Equilibrium) -> pd.DataFrame:
    """
    Compute metrics for each (sim, round) pair.
    """
    records = []
    for (sim, rnd), group in df_tx.groupby(['sim', 'round']):
        actual_surplus = sum(
            eq.buyer_map[r['buyer_id']] - eq.seller_map[r['seller_id']]
            for _, r in group.iterrows()
        )
        efficiency = actual_surplus / eq.surplus if eq.surplus > 0 else np.nan
        
        prices = group['price'].values
        rmse = np.sqrt(np.mean((prices - eq.price) ** 2))
        alpha = 100 * rmse / eq.price
        
        n_trades = len(group)
        mean_price = prices.mean()
        
        n_extramarginal = sum(
            1 for _, r in group.iterrows()
            if eq.buyer_map[r['buyer_id']] < eq.price
            or eq.seller_map[r['seller_id']] > eq.price
        )
        
        n_negative = sum(
            1 for _, r in group.iterrows()
            if eq.buyer_map[r['buyer_id']] - eq.seller_map[r['seller_id']] < 0
        )
        
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
# 4. MARSHALLIAN PATH ANALYSIS
# ============================================================

def compute_marshallian_path(df_tx: pd.DataFrame, eq: Equilibrium) -> pd.DataFrame:
    """
    Analyse within-round price convergence (Marshallian path).
    """
    df = df_tx.copy()
    df['tx_seq'] = df.groupby(['sim', 'round']).cumcount() + 1
    df['price_dev'] = abs(df['price'] - eq.price)
    df['signed_dev'] = df['price'] - eq.price
    df['buyer_val'] = df['buyer_id'].map(eq.buyer_map)
    df['seller_cost'] = df['seller_id'].map(eq.seller_map)
    df['pair_surplus'] = df['buyer_val'] - df['seller_cost']
    df['buyer_dist_from_ce'] = abs(df['buyer_val'] - eq.price)
    df['seller_dist_from_ce'] = abs(df['seller_cost'] - eq.price)
    return df


# ============================================================
# 5. SUMMARY PRINTING
# ============================================================

def print_summary(df_metrics: pd.DataFrame, eq: Equilibrium):
    """Print comprehensive summary statistics."""
    df = df_metrics
    n_obs = len(df)
    n_sims = df['sim'].nunique()
    n_rounds = df['round'].nunique()
    is_single_round = n_rounds == 1
    
    if is_single_round:
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
    
    print(f"\nEFFICIENCY PERCENTILES")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:2d}: {df['efficiency'].quantile(p / 100) * 100:.1f}%")
    
    # Across-round trajectory (only meaningful with multiple rounds)
    if not is_single_round:
        print(f"\nACROSS-ROUND TRAJECTORY (mean \u00b1 SEM across {n_sims} sims)")
        by_round = df.groupby('round').agg(
            eff_mean=('efficiency', 'mean'),
            eff_sem=('efficiency', 'sem'),
            alpha_mean=('alpha', 'mean'),
            alpha_sem=('alpha', 'sem'),
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


# ============================================================
# 6. PLOTTING
# ============================================================

def plot_validation(df_metrics: pd.DataFrame, df_marsh: pd.DataFrame, eq: Equilibrium,
                    title: str = "Market Experiment"):
    """
    Generate validation plots. Adapts layout based on experiment type:
    - Single round per sim (ZI-C): distributions across simulations
    - Multi-round per sim (LLM): across-round trajectories (mean +/- SEM)
    """
    n_rounds = df_metrics['round'].nunique()
    n_sims = df_metrics['sim'].nunique()
    is_single_round = n_rounds == 1
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    # ── Row 0, Col 0: Efficiency histogram ──
    ax = axes[0, 0]
    ax.hist(df_metrics['efficiency'], bins=30, color='#16a34a', alpha=0.7, edgecolor='white')
    ax.axvline(df_metrics['efficiency'].mean(), color='black', ls='--', lw=1.5,
               label=f"mean = {df_metrics['efficiency'].mean():.3f}")
    ax.axvline(1.0, color='#e74c3c', ls='-', lw=1, label='CE (100%)')
    ax.set_xlabel('Allocative efficiency')
    ax.set_ylabel('Count')
    ax.set_title('Efficiency distribution')
    ax.legend(fontsize=9)
    
    # ── Row 0, Col 1: Alpha histogram ──
    ax = axes[0, 1]
    ax.hist(df_metrics['alpha'], bins=30, color='#2563eb', alpha=0.7, edgecolor='white')
    ax.axvline(df_metrics['alpha'].mean(), color='black', ls='--', lw=1.5,
               label=f"mean = {df_metrics['alpha'].mean():.1f}%")
    ax.set_xlabel("Smith's \u03b1 (%)")
    ax.set_ylabel('Count')
    ax.set_title('\u03b1 distribution')
    ax.legend(fontsize=9)
    
    # ── Row 0, Col 2: Trades per round ──
    ax = axes[0, 2]
    trade_counts = df_metrics['n_trades'].value_counts().sort_index()
    ax.bar(trade_counts.index, trade_counts.values, color='#d97706', alpha=0.7, edgecolor='white')
    ax.axvline(eq.quantity, color='#e74c3c', ls='--', lw=1.5, label=f'CE = {eq.quantity}')
    ax.set_xlabel('Trades per round')
    ax.set_ylabel('Count')
    ax.set_title('Quantity distribution')
    ax.legend(fontsize=9)
    
    # ── Row 1, Col 0: Efficiency across rounds/simulations ──
    ax = axes[1, 0]
    if is_single_round:
        # ZI-C: scatter by simulation index (should be flat / no trend)
        ax.scatter(df_metrics['sim'], df_metrics['efficiency'],
                   alpha=0.15, s=10, color='#16a34a')
        ax.axhline(df_metrics['efficiency'].mean(), color='#16a34a', lw=2,
                   label=f"mean = {df_metrics['efficiency'].mean():.3f}")
        ax.axhline(1.0, color='#e74c3c', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel('Simulation')
        ax.set_ylabel('Efficiency')
        ax.set_title('Efficiency across simulations\n(stationarity check)')
        ax.legend(fontsize=9)
    else:
        # LLM: trajectory across rounds, mean +/- SEM across simulations
        by_round = df_metrics.groupby('round')['efficiency'].agg(['mean', 'sem'])
        # Individual sim trajectories (faint)
        for sim_id, sim_data in df_metrics.groupby('sim'):
            ax.plot(sim_data['round'], sim_data['efficiency'],
                    alpha=0.15, color='#16a34a', lw=0.8)
        # Mean +/- SEM (bold)
        ax.errorbar(by_round.index, by_round['mean'], yerr=by_round['sem'],
                    fmt='-o', color='#16a34a', capsize=3, markersize=5,
                    lw=2, zorder=5)
        ax.axhline(1.0, color='#e74c3c', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel('Round')
        ax.set_ylabel('Efficiency')
        ax.set_title(f'Efficiency by round\n(mean \u00b1 SEM, n={n_sims} sims)')
        ax.set_xticks(sorted(df_metrics['round'].unique()))
    
    # ── Row 1, Col 1: Price across rounds/simulations ──
    ax = axes[1, 1]
    if is_single_round:
        ax.scatter(df_metrics['sim'], df_metrics['mean_price'],
                   alpha=0.15, s=10, color='#d97706')
        ax.axhline(df_metrics['mean_price'].mean(), color='#d97706', lw=2)
        ax.axhline(eq.price, color='#e74c3c', ls='--', lw=1.5,
                   label=f'CE = {eq.price:.2f}')
        ax.set_xlabel('Simulation')
        ax.set_ylabel('Mean price')
        ax.set_title('Price across simulations')
        ax.legend(fontsize=9)
    else:
        by_round = df_metrics.groupby('round')['mean_price'].agg(['mean', 'sem'])
        for sim_id, sim_data in df_metrics.groupby('sim'):
            ax.plot(sim_data['round'], sim_data['mean_price'],
                    alpha=0.15, color='#d97706', lw=0.8)
        ax.errorbar(by_round.index, by_round['mean'], yerr=by_round['sem'],
                    fmt='-o', color='#d97706', capsize=3, markersize=5,
                    lw=2, zorder=5)
        ax.axhline(eq.price, color='#e74c3c', ls='--', lw=1.5,
                   label=f'CE = {eq.price:.2f}')
        ax.set_xlabel('Round')
        ax.set_ylabel('Mean price')
        ax.set_title(f'Price by round\n(mean \u00b1 SEM, n={n_sims} sims)')
        ax.set_xticks(sorted(df_metrics['round'].unique()))
        ax.legend(fontsize=9)
    
    # ── Row 1, Col 2: Marshallian path ──
    ax = axes[1, 2]
    max_seq = int(df_marsh['tx_seq'].quantile(0.95))
    marsh_agg = df_marsh[df_marsh['tx_seq'] <= max_seq].groupby('tx_seq').agg(
        mean_dev=('price_dev', 'mean'),
        sem_dev=('price_dev', 'sem'),
    )
    ax.errorbar(marsh_agg.index, marsh_agg['mean_dev'], yerr=marsh_agg['sem_dev'],
                fmt='-o', color='#7c3aed', capsize=3, markersize=5)
    ax.set_xlabel('Transaction position within round')
    ax.set_ylabel('Mean |price - CE|')
    ax.set_title('Within-round convergence\n(Marshallian path)')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_price_convergence(df_metrics: pd.DataFrame, eq: Equilibrium,
                           title: str = None):
    """
    Price convergence chart with Smith's alpha annotations per round.
    
    Shows individual simulation price trajectories (faint), the average
    price (bold), the CE price line, and alpha values annotated per round.
    
    For single-round experiments (ZI-C), shows price distribution across
    simulations instead of a trajectory.
    """
    n_rounds = df_metrics['round'].nunique()
    n_sims = df_metrics['sim'].nunique()
    is_single_round = n_rounds == 1
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sim_color = '#1f77b4'
    avg_color = '#d62728'
    
    # Axis limits from supply/demand range
    y_min = max(min(eq.supply[0], eq.demand[-1]) - 1, 0)
    y_max = max(eq.supply[-1], eq.demand[0]) + 1
    y_range = y_max - y_min
    alpha_y_pos = y_max - 0.05 * y_range
    
    # Compute average alpha per round across simulations
    avg_alpha = df_metrics.groupby('round')['alpha'].mean()
    
    if is_single_round:
        # ZI-C: no trajectory, show price distribution as swarm
        ax.scatter(df_metrics['sim'], df_metrics['mean_price'],
                   alpha=0.3, s=15, color=sim_color)
        ax.axhline(df_metrics['mean_price'].mean(), color=avg_color, lw=2,
                   label='Average Price')
        ax.axhline(eq.price, color='grey', ls='--', lw=1.2, alpha=0.6)
        ax.text(df_metrics['sim'].min(), eq.price - 0.15,
                'Competitive Equilibrium Price',
                color='grey', fontsize=10, family='serif', va='bottom')
        
        # Single alpha annotation
        alpha_val = avg_alpha.iloc[0]
        ax.text(df_metrics['sim'].median(), alpha_y_pos,
                f'\u03b1={alpha_val:.1f}',
                ha='center', va='top', fontsize=10, family='serif',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Simulation', fontsize=12, family='serif')
    else:
        # LLM / multi-round: trajectory plot
        # Pivot mean prices: rows=round, cols=sim
        df_plot = df_metrics.pivot_table(
            index='round', columns='sim', values='mean_price')
        
        # Individual sim trajectories
        for sim_id in df_plot.columns:
            ax.plot(df_plot.index, df_plot[sim_id],
                    color=sim_color, alpha=0.3, linewidth=1)
        
        # Average price across simulations
        avg_prices = df_plot.mean(axis=1)
        ax.plot(df_plot.index, avg_prices, color=avg_color, linewidth=2,
                label='Average Price')
        
        # CE price line
        ax.axhline(y=eq.price, color='grey', linestyle='--',
                   linewidth=1.2, alpha=0.6)
        ax.text(df_plot.index[0], eq.price - 0.15,
                'Competitive Equilibrium Price',
                color='grey', fontsize=10, family='serif', va='bottom')
        
        # Alpha annotations per round
        for round_num, alpha_value in avg_alpha.items():
            ax.text(round_num, alpha_y_pos, f'\u03b1={alpha_value:.1f}',
                    ha='center', va='top', fontsize=10, family='serif',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='wheat', alpha=0.5))
        
        ax.set_xticks(sorted(df_metrics['round'].unique()))
        ax.set_xlabel('Round', fontsize=12, family='serif')
    
    ax.set_ylabel('Average Transaction Price per Round',
                  fontsize=12, family='serif')
    ax.set_title(title or 'Average Transaction Price per Round Across Simulations',
                 fontsize=14, family='serif', pad=10)
    
    ax.tick_params(axis='both', direction='in', labelsize=10, colors='#333333')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family('serif')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='lower left', frameon=False,
              prop={'family': 'serif', 'size': 10})
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig


# ============================================================
# 7. MAIN RUNNER
# ============================================================

def run_validation(results_path: Path, n_sims: int, title: str = None,
                   config: dict = None) -> dict:
    """
    Run the full analysis pipeline.
    
    Works for both ZI-C (1 round/sim, many sims) and LLM (N rounds/sim).
    The code is identical; the interpretation differs:
    - ZI-C: each simulation is an independent draw, 1 round each
    - LLM: each simulation is a multi-round experiment with learning
    
    Parameters:
        results_path: Path to the experiment results directory
        n_sims: Number of simulations to load
        title: Plot title (defaults to directory name)
        config: Experiment config dict. If None, loads from config_used.yaml.
    
    Returns:
        dict with keys: 'eq', 'metrics', 'marshallian', 'tx', 'agents', 'config'
    """
    import yaml
    
    # Load config
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
    
    # Load data
    df_tx, df_agents = load_experiment(results_path, n_sims)
    
    # Equilibrium (from config schedules + agent ID mapping)
    eq = compute_equilibrium(config, df_agents)
    
    # Metrics
    df_metrics = compute_round_metrics(df_tx, eq)
    
    # Marshallian
    df_marsh = compute_marshallian_path(df_tx, eq)
    
    # Print
    print_summary(df_metrics, eq)
    print_marshallian_summary(df_marsh, eq)
    
    # Plot
    plot_title = title or results_path.name
    fig_summary = plot_validation(df_metrics, df_marsh, eq, title=plot_title)
    fig_convergence = plot_price_convergence(df_metrics, eq, title=plot_title)
    
    return {
        'eq': eq,
        'metrics': df_metrics,
        'marshallian': df_marsh,
        'tx': df_tx,
        'agents': df_agents,
        'config': config,
        'fig_summary': fig_summary,
        'fig_convergence': fig_convergence,
    }


# ============================================================
# Standalone usage
# ============================================================
if __name__ == '__main__':
    import sys
    import yaml
    
    if len(sys.argv) < 2:
        print("Usage: python zi_validation.py <results_path> [n_sims]")
        print("  results_path: path to experiment results directory")
        print("  n_sims: number of simulations (default: read from config)")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    
    config_path = results_path / 'config_used.yaml'
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"No config found at {config_path}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        n_sims = int(sys.argv[2])
    else:
        n_sims = config['experiment']['n_simulations']
    
    run_validation(results_path, n_sims, config=config)