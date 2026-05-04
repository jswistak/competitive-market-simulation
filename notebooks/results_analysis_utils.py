"""
Market Experiment Analysis
==========================
Loads three data sources per simulation:
- iteration_history_{sim}.csv: full order flow (all submitted bids/asks)
- transactions_{sim}.csv: completed transactions with execution prices
- agent_histories_{sim}.csv: per-agent action log with reservation prices

Computes:
- Allocative efficiency, Smith's alpha, Marshallian path
- Rent-seeking metrics (attempted/realised rent, extraction efficiency,
  concession rate, constraint violations)
- Order flow metrics (frequency, dispersion, fill rate, spread, initiation)

Works with both ZI-C (1 round/sim) and LLM (N rounds/sim) experiments.

IMPORTANT: In order-book experiments, execution price differs from submitted
order price. iteration_history.price is the SUBMITTED limit price;
transactions.price is the EXECUTION price. Transaction-level metrics use
execution prices; announcement-level metrics use submitted prices.

Public API:
    data    = load_experiment_data(path, n_sims, config)
    metrics = compute_all_metrics(data)
    figs    = render_all_plots(data, metrics)
    # or: data, metrics, figs = run_full_analysis(path, n_sims)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ============================================================
# §1. CONSTANTS & CONVENTIONS
# ============================================================

SMITH_ALPHA = {
    "smith1":  {1: 11.8, 2: 8.1, 3: 5.2, 4: 5.5, 5: 3.5},
    "smith2":  {1: 9.9,  2: 5.4, 3: 2.2},
    "smith3":  {1: 16.5, 2: 6.6, 3: 3.7, 4: 5.7},
    "smith4a": {1: 19.1, 2: 10.4, 3: 7.8, 4: 7.6},
    "smith4b": {1: 6.9,  2: 7.1, 3: 6.5},
    "smith5a": {1: 2.0,  2: 0.7, 3: 0.7, 4: 0.6},
    "smith5b": {1: 9.4,  2: 4.3},
    "smith6a": {1: 53.8, 2: 38.7, 3: 21.1, 4: 9.4},
    "smith6b": {1: 11.0},
    "smith7":  {1: 49.1, 2: 22.2, 3: 7.1, 4: 5.4, 5: 3.0, 6: 2.7},
    "smith8a": {1: 19.0, 2: 2.9, 3: 7.4, 4: 7.0},
    "smith8b": {1: 7.8,  2: 6.1},
    "smith9a": {1: 21.8, 2: 15.4, 3: 13.2},
    "smith9b": {1: 10.3},
    "smith10": {1: 11.0, 2: 3.2, 3: 2.2},
}

# Side styling — single source of truth for colours/markers across all plots.
SIDE_STYLES = {
    'buyer':  {'color': '#2196F3', 'marker': 'o', 'label': 'Buyer'},
    'seller': {'color': '#F44336', 'marker': 's', 'label': 'Seller'},
}
# Alternate palette used by validation/order-flow plots (kept for visual parity
# with the original module).
SIDE_STYLES_ALT = {
    'buyer':  {'color': '#1f77b4', 'marker': '^', 'label': 'Bid'},
    'seller': {'color': '#d62728', 'marker': 's', 'label': 'Ask'},
}


# ============================================================
# §2. DATA CLASSES
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


@dataclass
class ExperimentData:
    """Loaded + enriched data for one experiment."""
    experiment_id: str
    eq: Equilibrium
    iter: pd.DataFrame      # raw iteration history (all submitted orders)
    tx: pd.DataFrame        # enriched transactions (buyer_val, surplus, ...)
    agents: pd.DataFrame
    ann: pd.DataFrame       # extracted, deduplicated announcements
    config: dict
    independent_rounds: bool = False  # True for ZI baselines (1 round/sim)


@dataclass
class ExperimentMetrics:
    """Computed metric DataFrames for one experiment."""
    round_metrics: pd.DataFrame
    marshallian: pd.DataFrame
    rent: dict                       # {ann, tx, concessions, rent_surrendered,
                                     #  ann_first, extraction_eff}
    spread_series: pd.DataFrame
    spread_by_round: pd.DataFrame
    initiation: pd.DataFrame
    summary_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# ============================================================
# §3. LOADING & ENRICHMENT
# ============================================================

def _load_csvs(results_path: Path, n_sims: int
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and concat the three CSV families across all available sims."""
    available = [
        s for s in range(1, n_sims + 1)
        if (results_path / 'data' / f'iteration_history_{s}.csv').exists()
        and (results_path / 'data' / f'transactions_{s}.csv').exists()
        and (results_path / 'data' / f'agent_histories_{s}.csv').exists()
    ]
    if not available:
        raise FileNotFoundError(f"No matching files found in {results_path / 'data'}")

    def _concat(prefix):
        return pd.concat([
            pd.read_csv(results_path / 'data' / f'{prefix}_{s}.csv').assign(sim=s)
            for s in available
        ], ignore_index=True)

    df_iter = _concat('iteration_history')
    df_tx = _concat('transactions')
    df_agents = _concat('agent_histories')

    n_rounds = df_tx.groupby('sim')['round'].nunique().iloc[0]
    print(f"Loaded {len(available)} simulations from {results_path.name}")
    print(f"  Iteration history rows: {len(df_iter)}")
    print(f"  Transaction rows:       {len(df_tx)}")
    print(f"  Rounds per sim:         {n_rounds}")
    print(f"  Total round-obs:        {df_tx.groupby(['sim', 'round']).ngroups}")
    return df_iter, df_tx, df_agents


def _enrich_transactions(df_tx: pd.DataFrame, eq: Equilibrium) -> pd.DataFrame:
    """Add reservation prices, surplus, share, role flags. Idempotent-safe but
    only ever called once per dataset (at load time)."""
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


def _extract_announcements(df_iter: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate submitted orders. price is SUBMITTED limit, not execution."""
    ann = (
        df_iter[df_iter['announcement_made'] == True]
        .drop_duplicates(subset=['sim', 'round', 'iteration', 'announcing_agent_id',
                                 'price', 'announcement_type'])
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


# ----- Equilibrium ------------------------------------------------------------

def _find_equilibrium(supply, demand, flat_threshold=1e-10):
    """Find equilibrium quantity and price where demand >= supply."""
    supply, demand = np.asarray(supply), np.asarray(demand)
    n = min(len(supply), len(demand))
    viable = demand[:n] >= supply[:n]
    if not viable.any():
        return 0, None
    last_idx = np.flatnonzero(viable)[-1]
    next_idx = last_idx + 1
    q_eq = last_idx + 1
    p_supply, p_demand = supply[last_idx], demand[last_idx]
    has_next_s, has_next_d = next_idx < len(supply), next_idx < len(demand)
    if not has_next_s:
        p_eq = (p_demand + demand[next_idx]) / 2 if has_next_d else p_demand
    elif not has_next_d:
        p_eq = (p_supply + supply[next_idx]) / 2 if has_next_s else p_supply
    elif abs(supply[next_idx] - p_supply) < flat_threshold:
        p_eq = p_supply
    elif abs(demand[next_idx] - p_demand) < flat_threshold:
        p_eq = p_demand
    else:
        p_eq = (p_supply + p_demand) / 2
    return q_eq, p_eq


def _plot_equilibrium(supply, demand, q_eq, p_eq, title=None, ax=None):
    """Visualise S/D curves with equilibrium point."""
    supply, demand = np.asarray(supply), np.asarray(demand)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    qs = np.arange(len(supply)) + 1
    qd = np.arange(len(demand)) + 1
    ax.step(qs, supply, 'r-', where='pre', label='Supply', linewidth=2)
    ax.step(qd, demand, 'b-', where='pre', label='Demand', linewidth=2)
    y_max = max(supply[-1], demand[0]) * 1.2
    ax.vlines(len(supply), supply[-1], y_max, colors='red', linewidth=2)
    ax.vlines(len(demand), demand[-1], 0, colors='blue', linewidth=2)
    if q_eq > 0 and p_eq is not None:
        ax.plot(q_eq, p_eq, 'go', markersize=12,
                label=f'Equilibrium: Q={q_eq}, P={p_eq:.2f}', zorder=5)
        ax.axhline(p_eq, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.axvline(q_eq, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Quantity'); ax.set_ylabel('Price')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(title or 'Supply and Demand - Equilibrium')
    ax.set_ylim(0, y_max)
    return ax


def _compute_equilibrium(config: dict, df_agents: pd.DataFrame,
                         show_plot: bool = True) -> Equilibrium:
    """Compute competitive equilibrium from config schedules."""
    demand = np.round(np.linspace(
        config['experiment']['buyers']['max'],
        config['experiment']['buyers']['min'],
        config['experiment']['buyers']['num']), 2)
    supply = np.round(np.linspace(
        config['experiment']['sellers']['min'],
        config['experiment']['sellers']['max'],
        config['experiment']['sellers']['num']), 2)
    q_eq, p_eq = _find_equilibrium(supply, demand)
    ce_surplus = float(np.sum(demand[:q_eq] - supply[:q_eq]))
    buyer_surplus = float(np.sum(demand[:q_eq] - p_eq))
    seller_surplus = float(np.sum(p_eq - supply[:q_eq]))

    agents = df_agents[['agent_id', 'agent_type', 'reservation_price']].drop_duplicates()
    bv = dict(zip(agents[agents['agent_type'] == 'buyer']['agent_id'],
                  agents[agents['agent_type'] == 'buyer']['reservation_price']))
    sv = dict(zip(agents[agents['agent_type'] == 'seller']['agent_id'],
                  agents[agents['agent_type'] == 'seller']['reservation_price']))

    eq = Equilibrium(quantity=q_eq, price=p_eq, surplus=ce_surplus,
                     buyer_surplus=buyer_surplus, seller_surplus=seller_surplus,
                     demand=demand, supply=supply,
                     buyer_map=bv, seller_map=sv)
    print(f"\nCompetitive Equilibrium:")
    print(f"  Quantity:       {eq.quantity}")
    print(f"  Price:          {eq.price:.4f}")
    print(f"  Total surplus:  {eq.surplus:.4f}")
    print(f"  Buyer surplus:  {eq.buyer_surplus:.2f}")
    print(f"  Seller surplus: {eq.seller_surplus:.2f}")
    print(f"  Demand:         {demand}")
    print(f"  Supply:         {supply}")
    if show_plot:
        _plot_equilibrium(supply, demand, q_eq, p_eq)
        plt.show()
    return eq


def load_experiment_data(results_path: Path, n_sims: int,
                         config: dict | None = None,
                         experiment_id: str | None = None,
                         independent_rounds: bool = False,
                         show_eq_plot: bool = True) -> ExperimentData:
    """Load + enrich one experiment. Reads config_used.yaml if config not given."""
    import yaml
    if config is None:
        cfg_path = results_path / 'config_used.yaml'
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"No config at {cfg_path}. Pass config dict explicitly.")
        with open(cfg_path) as f:
            config = yaml.safe_load(f)

    df_iter, df_tx, df_agents = _load_csvs(results_path, n_sims)
    eq = _compute_equilibrium(config, df_agents, show_plot=show_eq_plot)
    tx = _enrich_transactions(df_tx, eq)
    ann = _extract_announcements(df_iter)

    return ExperimentData(
        experiment_id=experiment_id or results_path.name,
        eq=eq, iter=df_iter, tx=tx, agents=df_agents, ann=ann,
        config=config, independent_rounds=independent_rounds,
    )


# ============================================================
# §4. METRICS (pure functions returning DataFrames)
# ============================================================

def compute_round_metrics(tx: pd.DataFrame, eq: Equilibrium) -> pd.DataFrame:
    """Per (sim, round): efficiency, alpha, n_trades, mean_price, etc."""
    records = []
    for (sim, rnd), g in tx.groupby(['sim', 'round']):
        bv, sc, prices = g['buyer_val'].values, g['seller_cost'].values, g['price'].values
        actual_surplus = float(np.sum(bv - sc))
        rmse = np.sqrt(np.mean((prices - eq.price) ** 2))
        records.append({
            'sim': sim, 'round': rnd,
            'efficiency': actual_surplus / eq.surplus if eq.surplus > 0 else np.nan,
            'alpha': 100 * rmse / eq.price,
            'n_trades': len(g), 'mean_price': prices.mean(),
            'actual_surplus': actual_surplus,
            'n_extramarginal': int(np.sum((bv < eq.price) | (sc > eq.price))),
            'n_negative_surplus': int(np.sum((bv - sc) < 0)),
        })
    return pd.DataFrame(records)


def compute_marshallian_path(tx: pd.DataFrame, eq: Equilibrium) -> pd.DataFrame:
    """Within-round price convergence (uses execution prices)."""
    df = tx.copy()
    df['tx_seq'] = df.groupby(['sim', 'round']).cumcount() + 1
    df['price_dev'] = abs(df['price'] - eq.price)
    df['signed_dev'] = df['price'] - eq.price
    df['pair_surplus'] = df['buyer_val'] - df['seller_cost']
    df['buyer_dist_from_ce'] = abs(df['buyer_val'] - eq.price)
    df['seller_dist_from_ce'] = abs(df['seller_cost'] - eq.price)
    return df


def _compute_concessions(ann: pd.DataFrame) -> pd.DataFrame:
    """Per-step concession in attempted rent (submitted prices)."""
    out = []
    for (sim, agent, rnd), g in ann.groupby(
            ['sim', 'announcing_agent_id', 'round']):
        if len(g) < 2:
            continue
        g = g.sort_values('iteration')
        rents, iters = g['attempted_rent'].values, g['iteration'].values
        side = g['side'].iloc[0]
        for j in range(1, len(rents)):
            remaining = rents[j - 1]
            if remaining <= 0:
                continue
            out.append({
                'sim': sim, 'agent_id': agent, 'side': side, 'round': rnd,
                'from_iter': iters[j - 1], 'to_iter': iters[j],
                'concession': remaining - rents[j],
                'frac_concession': (remaining - rents[j]) / remaining,
            })
    return pd.DataFrame(out)


def _compute_rent_surrendered(ann: pd.DataFrame) -> pd.DataFrame:
    """Per-(sim, agent, round) total rent surrendered first→last announcement."""
    abr = (ann.sort_values('iteration')
           .groupby(['sim', 'announcing_agent_id', 'round', 'side'])
           .agg(first_rent=('attempted_rent', 'first'),
                last_rent=('attempted_rent', 'last'),
                n_announcements=('attempted_rent', 'count'))
           .reset_index())
    abr = abr[(abr['first_rent'] > 0) & (abr['n_announcements'] > 1)].copy()
    abr['rent_surrendered'] = abr['first_rent'] - abr['last_rent']
    abr['frac_surrendered'] = abr['rent_surrendered'] / abr['first_rent']
    return abr


def _compute_extraction_efficiency(ann: pd.DataFrame, tx: pd.DataFrame
                                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Efficiency = realised_rent / first_attempted_rent."""
    ann_first = (ann.sort_values('iteration')
                 .groupby(['sim', 'announcing_agent_id', 'round'])
                 .first().reset_index()
                 .rename(columns={'attempted_rent': 'first_attempted_rent',
                                  'price': 'first_price'}))

    realised_rows = []
    for _, r in tx.iterrows():
        realised_rows.append({'sim': r['sim'], 'agent_id': r['buyer_id'],
                              'round': r['round'], 'realized_rent': r['buyer_rent']})
        realised_rows.append({'sim': r['sim'], 'agent_id': r['seller_id'],
                              'round': r['round'], 'realized_rent': r['seller_rent']})
    realised = pd.DataFrame(realised_rows)

    eff = ann_first.merge(realised,
                          left_on=['sim', 'announcing_agent_id', 'round'],
                          right_on=['sim', 'agent_id', 'round'], how='inner')
    eff = eff[eff['first_attempted_rent'] > 0].copy()
    eff['side'] = eff['announcement_type'].map({'buy': 'buyer', 'sell': 'seller'})
    eff['efficiency'] = eff['realized_rent'] / eff['first_attempted_rent']
    return ann_first, eff


def compute_rent_metrics(data: ExperimentData) -> dict:
    """Bundle of rent-seeking DataFrames keyed for downstream plots."""
    ann, tx = data.ann, data.tx
    conc = _compute_concessions(ann)
    abr = _compute_rent_surrendered(ann)
    ann_first, eff = _compute_extraction_efficiency(ann, tx)
    return {
        'ann': ann, 'tx': tx, 'concessions': conc,
        'rent_surrendered': abr,
        'ann_first': ann_first, 'extraction_eff': eff,
    }


def compute_spread_series(df_iter: pd.DataFrame) -> pd.DataFrame:
    cols = ['sim', 'round', 'iteration', 'standing_bid', 'standing_ask']
    df = df_iter[cols].copy()
    df['standing_bid'] = pd.to_numeric(df['standing_bid'], errors='coerce')
    df['standing_ask'] = pd.to_numeric(df['standing_ask'], errors='coerce')
    df['spread'] = df['standing_ask'] - df['standing_bid']
    return df


def spread_by_round(spread_series: pd.DataFrame) -> pd.DataFrame:
    g = spread_series.groupby(['sim', 'round'])
    mean_s = g['spread'].mean().rename('mean_spread')
    min_s = g['spread'].min().rename('min_spread')
    final_s = (spread_series.sort_values('iteration')
               .groupby(['sim', 'round']).last()[['spread']]
               .rename(columns={'spread': 'final_spread'}))
    both = (spread_series
            .assign(both=lambda d: d['standing_bid'].notna() & d['standing_ask'].notna())
            .groupby(['sim', 'round'])['both'].mean().rename('pct_both_sides'))
    return pd.concat([mean_s, min_s, final_s, both], axis=1).reset_index()


def compute_initiation(tx: pd.DataFrame) -> pd.DataFrame:
    """Per-round fractions of buyer- vs seller-initiated trades."""
    counts = (tx.groupby(['sim', 'round', 'announcement_type'])
              .size().rename('n').reset_index())
    total = counts.groupby(['sim', 'round'])['n'].sum().rename('total')
    counts = counts.join(total, on=['sim', 'round'])
    counts['fraction'] = counts['n'] / counts['total']
    return (counts.groupby(['round', 'announcement_type'])
            .agg(mean_fraction=('fraction', 'mean'),
                 se_fraction=('fraction', 'sem'),
                 mean_n=('n', 'mean'))
            .reset_index())


def compute_all_metrics(data: ExperimentData) -> ExperimentMetrics:
    """Compute every metric DataFrame. Order does not matter."""
    spread = compute_spread_series(data.iter)
    return ExperimentMetrics(
        round_metrics=compute_round_metrics(data.tx, data.eq),
        marshallian=compute_marshallian_path(data.tx, data.eq),
        rent=compute_rent_metrics(data),
        spread_series=spread,
        spread_by_round=spread_by_round(spread),
        initiation=compute_initiation(data.tx),
    )


# ============================================================
# §5. SUMMARIZE / REPORT (compute → dict, then print)
# ============================================================

def summarize_round_metrics(metrics: pd.DataFrame, eq: Equilibrium,
                            independent_rounds: bool) -> dict:
    df = metrics
    n_obs = len(df)
    out = {
        'n_obs': n_obs, 'n_sims': df['sim'].nunique(),
        'n_rounds': df['round'].nunique(),
        'independent_rounds': independent_rounds,
        'efficiency': {
            'mean': df['efficiency'].mean(), 'median': df['efficiency'].median(),
            'std': df['efficiency'].std(), 'min': df['efficiency'].min(),
            'max': df['efficiency'].max(),
            'n_below_80': int((df['efficiency'] < 0.80).sum()),
            'n_above_90': int((df['efficiency'] > 0.90).sum()),
            'n_above_95': int((df['efficiency'] > 0.95).sum()),
            'n_at_100':   int((abs(df['efficiency'] - 1.0) < 0.001).sum()),
        },
        'alpha': {
            'mean': df['alpha'].mean(), 'median': df['alpha'].median(),
            'std': df['alpha'].std(),
        },
        'quantity': {
            'eq': eq.quantity, 'mean': df['n_trades'].mean(),
            'distribution': df['n_trades'].value_counts().sort_index().to_dict(),
        },
        'price': {
            'eq': eq.price, 'mean_of_means': df['mean_price'].mean(),
            'std_of_means': df['mean_price'].std(),
        },
        'extramarginal': {
            'rounds_with_any': int((df['n_extramarginal'] > 0).sum()),
            'frac_rounds':     (df['n_extramarginal'] > 0).mean(),
            'mean_per_round':  df['n_extramarginal'].mean(),
            'rounds_with_neg_surplus': int((df['n_negative_surplus'] > 0).sum()),
        },
    }
    if not independent_rounds:
        out['by_round'] = (df.groupby('round').agg(
            eff_mean=('efficiency', 'mean'), eff_sem=('efficiency', 'sem'),
            alpha_mean=('alpha', 'mean'), alpha_sem=('alpha', 'sem'),
            n_trades_mean=('n_trades', 'mean')).round(4))
    return out


def report_round_metrics(s: dict) -> None:
    n = s['n_obs']
    label = (f"{n} simulations (1 round each)" if s['independent_rounds']
             else f"{s['n_sims']} simulations × {s['n_rounds']} rounds = {n} observations")
    print(f"\n{'=' * 60}")
    print(f"ALLOCATIVE EFFICIENCY ({label})")
    print(f"{'=' * 60}")
    e = s['efficiency']
    print(f"  Mean:   {e['mean']:.4f}")
    print(f"  Median: {e['median']:.4f}")
    print(f"  Std:    {e['std']:.4f}")
    print(f"  Min:    {e['min']:.4f}")
    print(f"  Max:    {e['max']:.4f}")
    print(f"  < 80%:  {e['n_below_80']}/{n}")
    print(f"  > 90%:  {e['n_above_90']}/{n}")
    print(f"  > 95%:  {e['n_above_95']}/{n}")
    print(f"  = 100%: {e['n_at_100']}/{n}")
    if s['independent_rounds']:
        a = s['alpha']
        print(f"\nSMITH'S ALPHA (coefficient of convergence)")
        print(f"  Mean:   {a['mean']:.2f}%")
        print(f"  Median: {a['median']:.2f}%")
        print(f"  Std:    {a['std']:.2f}%")
    q = s['quantity']
    print(f"\nQUANTITY")
    print(f"  CE quantity:  {q['eq']}")
    print(f"  Mean trades:  {q['mean']:.2f}")
    print(f"  Distribution: {q['distribution']}")
    p = s['price']
    print(f"\nPRICE")
    print(f"  CE price:       {p['eq']:.4f}")
    print(f"  Mean price:     {p['mean_of_means']:.4f}")
    print(f"  Std of means:   {p['std_of_means']:.4f}")
    em = s['extramarginal']
    print(f"\nEXTRAMARGINAL ACTIVITY")
    print(f"  Rounds with extramarginal: {em['rounds_with_any']}/{n} "
          f"({em['frac_rounds'] * 100:.1f}%)")
    print(f"  Mean extramarginal/round:  {em['mean_per_round']:.2f}")
    print(f"  Negative surplus trades:   {em['rounds_with_neg_surplus']}/{n}")
    if 'by_round' in s:
        print(f"\nACROSS-ROUND TRAJECTORY (mean ± SEM across {s['n_sims']} sims)")
        print(s['by_round'].to_string())


def summarize_marshallian(df_marsh: pd.DataFrame, eq: Equilibrium) -> dict:
    max_seq = int(df_marsh['tx_seq'].quantile(0.95))
    rows = []
    for seq in range(1, max_seq + 1):
        sub = df_marsh[df_marsh['tx_seq'] == seq]
        if len(sub) == 0:
            break
        rows.append({
            'seq': seq, 'price_dev': sub['price_dev'].mean(),
            'buyer_val': sub['buyer_val'].mean(), 'seller_cost': sub['seller_cost'].mean(),
            'pair_surplus': sub['pair_surplus'].mean(),
            'buyer_dist': sub['buyer_dist_from_ce'].mean(),
            'seller_dist': sub['seller_dist_from_ce'].mean(),
            'n': len(sub),
        })
    by_seq = pd.DataFrame(rows)

    slopes = []
    for _, g in df_marsh.groupby(['sim', 'round']):
        if len(g) >= 3:
            slopes.append(np.polyfit(g['tx_seq'].values.astype(float),
                                     g['price_dev'].values, 1)[0])
    slopes = np.array(slopes)
    slope_summary = {}
    if len(slopes) > 0:
        t_stat = slopes.mean() / (slopes.std() / np.sqrt(len(slopes)))
        slope_summary = {
            'mean': slopes.mean(), 'median': np.median(slopes),
            'n_negative': int((slopes < 0).sum()), 'n_total': len(slopes),
            'frac_negative': (slopes < 0).mean(),
            't_stat': t_stat, 'significant': abs(t_stat) > 1.96,
        }

    return {
        'max_seq': max_seq, 'by_seq': by_seq, 'slopes': slope_summary,
        'corr_buyer':  np.corrcoef(df_marsh['tx_seq'], df_marsh['buyer_dist_from_ce'])[0, 1],
        'corr_seller': np.corrcoef(df_marsh['tx_seq'], df_marsh['seller_dist_from_ce'])[0, 1],
    }


def report_marshallian(s: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"MARSHALLIAN PATH (within-round convergence)")
    print(f"{'=' * 60}")
    print(f"\nBy transaction position (up to {s['max_seq']}):")
    print(f"{'Pos':>4} {'|p-CE|':>7} {'BuyerVal':>9} {'SellCost':>9} "
          f"{'Surplus':>8} {'B_dist':>7} {'S_dist':>7} {'n':>5}")
    for _, r in s['by_seq'].iterrows():
        print(f"{int(r.seq):>4} {r.price_dev:>7.3f} {r.buyer_val:>9.2f} "
              f"{r.seller_cost:>9.2f} {r.pair_surplus:>8.2f} "
              f"{r.buyer_dist:>7.2f} {r.seller_dist:>7.2f} {int(r.n):>5}")
    if s['slopes']:
        sl = s['slopes']
        print(f"\nPer-round slope (|price - CE| ~ transaction_seq):")
        print(f"  Mean slope:  {sl['mean']:.4f}")
        print(f"  Median:      {sl['median']:.4f}")
        print(f"  % negative (convergent): {sl['n_negative']}/{sl['n_total']} "
              f"({sl['frac_negative'] * 100:.1f}%)")
        print(f"  t-stat (H0: slope=0):    {sl['t_stat']:.3f}")
        print(f"  {'Significant' if sl['significant'] else 'Not significant'} at 5% level")
    print(f"\n  Corr(seq, buyer_dist_from_CE):  {s['corr_buyer']:.4f}")
    print(f"  Corr(seq, seller_dist_from_CE): {s['corr_seller']:.4f}")
    marshallian_ok = s['corr_buyer'] < -0.1 and s['corr_seller'] < -0.1
    print(f"  {'Marshallian path confirmed' if marshallian_ok else 'Weak/no Marshallian path'}")


def summarize_rent(rent: dict) -> dict:
    ann, tx, conc, eff = rent['ann'], rent['tx'], rent['concessions'], rent['extraction_eff']
    out = {
        'attempted_by_side': {side: {
            'mean': ann[ann['side'] == side]['attempted_rent'].mean(),
            'median': ann[ann['side'] == side]['attempted_rent'].median(),
            'std': ann[ann['side'] == side]['attempted_rent'].std(),
        } for side in ['buyer', 'seller']},
        'realized': {
            'buyer_mean': tx['buyer_rent'].mean(),
            'seller_mean': tx['seller_rent'].mean(),
        },
        'shares': {},
        'concessions_by_side': {},
        'extraction_eff_by_side': {},
        'violations': {
            'ann_total': len(ann), 'ann_violations': int(ann['violation'].sum()),
            'tx_total': len(tx),
            'tx_buyer_violations': int(tx['buyer_violation'].sum()),
            'tx_seller_violations': int(tx['seller_violation'].sum()),
        },
        'who_trades': {
            'buyer_inframarginal': tx['buyer_inframarginal'].mean(),
            'seller_inframarginal': tx['seller_inframarginal'].mean(),
            'both_inframarginal': tx['both_inframarginal'].mean(),
        },
    }
    valid = tx[tx['buyer_share'].notna()]
    if len(valid) > 0:
        out['shares'] = {'buyer': valid['buyer_share'].mean(),
                         'seller': valid['seller_share'].mean()}
    if len(conc) > 0:
        for side in ['buyer', 'seller']:
            sub = conc[conc['side'] == side]
            if len(sub) > 0:
                out['concessions_by_side'][side] = {
                    'mean_concession': sub['concession'].mean(),
                    'mean_frac': sub['frac_concession'].mean(),
                }
    for side in ['buyer', 'seller']:
        sub = eff[eff['side'] == side]['efficiency']
        if len(sub) > 0:
            out['extraction_eff_by_side'][side] = {
                'mean': sub.mean(), 'median': sub.median(),
            }
    if 'filled' in ann.columns:
        out['fill_rate_by_side'] = {
            side: ann[ann['side'] == side]['filled'].mean()
            for side in ['buyer', 'seller']
        }
    return out


def report_rent(s: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"RENT-SEEKING ANALYSIS")
    print(f"{'=' * 60}")
    print(f"\nAttempted Rent (submitted order prices):")
    for side, v in s['attempted_by_side'].items():
        print(f"  {side.title():>6}: mean={v['mean']:.3f}, "
              f"median={v['median']:.3f}, std={v['std']:.3f}")
    r = s['realized']
    print(f"\nRealized Rent (execution prices):")
    print(f"  Buyer rent:  mean={r['buyer_mean']:.3f}")
    print(f"  Seller rent: mean={r['seller_mean']:.3f}")
    if s['shares']:
        print(f"  Buyer share:  {s['shares']['buyer']:.1%}")
        print(f"  Seller share: {s['shares']['seller']:.1%}")
    if s['concessions_by_side']:
        print(f"\nConcessions (submitted order prices):")
        for side, v in s['concessions_by_side'].items():
            print(f"  {side.title():>6}: mean concession={v['mean_concession']:.3f}, "
                  f"frac={v['mean_frac']:.1%}")
    print(f"\nExtraction Efficiency:")
    for side, v in s['extraction_eff_by_side'].items():
        print(f"  {side.title():>6}: mean={v['mean']:.3f}, median={v['median']:.3f}")
    v = s['violations']
    print(f"\nConstraint Violations:")
    print(f"  Announcements: {v['ann_violations']}/{v['ann_total']} "
          f"({v['ann_violations'] / v['ann_total']:.1%})")
    print(f"  Buyer tx violations:  {v['tx_buyer_violations']}/{v['tx_total']}")
    print(f"  Seller tx violations: {v['tx_seller_violations']}/{v['tx_total']}")
    w = s['who_trades']
    print(f"\nWho Trades:")
    print(f"  Buyer inframarginal:  {w['buyer_inframarginal']:.1%}")
    print(f"  Seller inframarginal: {w['seller_inframarginal']:.1%}")
    print(f"  Both inframarginal:   {w['both_inframarginal']:.1%}")
    if 'fill_rate_by_side' in s:
        print(f"\nFill Rate (fraction of submitted orders that executed immediately):")
        for side, rate in s['fill_rate_by_side'].items():
            print(f"  {side.title():>6}: {rate:.1%}")


def build_market_summary_table(round_metrics: pd.DataFrame, eq: Equilibrium,
                               experiment_id: str | None = None) -> pd.DataFrame:
    """Smith (1962) style table. Returns DataFrame; caller decides whether to print."""
    summary = round_metrics.groupby('round').agg(
        n_trades=('n_trades', 'mean'), mean_price=('mean_price', 'mean'),
        alpha=('alpha', 'mean'), efficiency=('efficiency', 'mean'),
        n_extramarginal=('n_extramarginal', 'mean'),
    ).round(3)
    summary.insert(0, 'eq_quantity', eq.quantity)
    summary.insert(2, 'eq_price', eq.price)
    smith_key = experiment_id.split('_')[0] if experiment_id else None
    if smith_key and smith_key in SMITH_ALPHA:
        summary['smith_alpha'] = summary.index.map(SMITH_ALPHA[smith_key])
    return summary


def report_market_summary_table(summary: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(f"MARKET SUMMARY TABLE (Smith 1962 style)")
    print(f"{'=' * 60}")
    print(summary.to_string())


# ============================================================
# §6. PLOT HELPERS
# ============================================================

def _apply_paper_style(ax) -> None:
    """Serif fonts, clean spines. Use selectively where the plot is paper-bound."""
    ax.tick_params(axis='both', direction='in', labelsize=10, colors='#333333')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family('serif')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)


def _safe_hist(ax, data, target_bins=30, **kwargs):
    """Histogram robust to constant or near-constant data."""
    vals = data.dropna()
    if len(vals) == 0:
        return
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-10:
        margin = max(abs(vmin) * 0.1, 0.5)
        ax.hist(vals, bins=1, range=(vmin - margin, vmin + margin), **kwargs)
    else:
        n_bins = min(target_bins, max(1, len(vals) // 2))
        ax.hist(vals, bins=n_bins, **kwargs)


def _errorbar_by_round_by_side(ax, df: pd.DataFrame, value_col: str,
                                round_col: str = 'round',
                                aggregate: str = 'mean') -> None:
    """Errorbar per round per side. df must contain `sim`, `side`, round_col, value_col.
    aggregate: how to collapse multiple rows per (sim, round) to one value first."""
    for side, st in SIDE_STYLES.items():
        sub = df[df['side'] == side]
        if len(sub) == 0:
            continue
        sr = sub.groupby(['sim', round_col])[value_col].agg(aggregate).reset_index()
        agg = sr.groupby(round_col)[value_col].agg(['mean', 'sem'])
        if len(agg) == 0:
            continue
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'],
                    fmt=f"-{st['marker']}", color=st['color'],
                    capsize=3, label=st['label'])


def _errorbar_by_reservation(ax, df: pd.DataFrame, value_col: str,
                             res_col: str = 'announcing_agent_reservation_price') -> None:
    """Errorbar across reservation prices, by side."""
    for side, st in SIDE_STYLES.items():
        sub = df[df['side'] == side]
        if len(sub) == 0:
            continue
        sim_agent = sub.groupby(['sim', res_col])[value_col].mean().reset_index()
        agg = sim_agent.groupby(res_col)[value_col].agg(['mean', 'sem']).reset_index()
        ax.errorbar(agg[res_col], agg['mean'], yerr=agg['sem'],
                    fmt=f"-{st['marker']}", color=st['color'],
                    capsize=3, label=st['label'], markersize=5)


def _per_agent_barh(ax, df: pd.DataFrame, value_col: str,
                    agent_col: str = 'announcing_agent_id') -> None:
    """Per-agent horizontal bar chart with ±SEM."""
    sim_agent = df.groupby([agent_col, 'side', 'sim'])[value_col].mean().reset_index()
    agg = sim_agent.groupby([agent_col, 'side'])[value_col].agg(['mean', 'sem']).reset_index()
    buyers = agg[agg['side'] == 'buyer'].sort_values('mean')
    sellers = agg[agg['side'] == 'seller'].sort_values('mean')
    labels, means, sems, colors = [], [], [], []
    for _, r in buyers.iterrows():
        labels.append(f"B{int(r[agent_col])}")
        means.append(r['mean']); sems.append(r['sem'])
        colors.append(SIDE_STYLES['buyer']['color'])
    for _, r in sellers.iterrows():
        labels.append(f"S{int(r[agent_col])}")
        means.append(r['mean']); sems.append(r['sem'])
        colors.append(SIDE_STYLES['seller']['color'])
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, xerr=sems, color=colors, edgecolor='white',
            height=0.7, capsize=2, error_kw={'lw': 0.8})
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8)


def _histogram_by_side(ax, data, value_col, sims, n_bins=20,
                       title='', xlabel=''):
    """Side-by-side histogram with mean ±SEM per bin."""
    vals = data[value_col].dropna()
    if len(vals) == 0:
        return
    bin_edges = np.linspace(vals.min(), vals.max(), n_bins + 1)
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]; half = width / 2
    for side, st, offset in [
        ('buyer', SIDE_STYLES['buyer'], -half / 2),
        ('seller', SIDE_STYLES['seller'], half / 2),
    ]:
        counts = np.zeros((len(sims), len(bc)))
        for i, s in enumerate(sims):
            v = data[(data['side'] == side) & (data['sim'] == s)][value_col]
            counts[i], _ = np.histogram(v, bins=bin_edges)
        m = counts.mean(0)
        sem = counts.std(0, ddof=1) / np.sqrt(len(sims)) if len(sims) > 1 else np.zeros_like(m)
        ax.bar(bc + offset, m, width=half, color=st['color'],
               edgecolor='white', label=st['label'])
        ax.errorbar(bc + offset, m, yerr=sem, fmt='none', ecolor='black',
                    capsize=1.5, lw=0.8)
    ax.axvline(0, color='grey', ls='--', lw=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel('Mean count per simulation')
    ax.set_title(title); ax.legend()


def _yax_pct(ax) -> None:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def _price_y_bounds(eq: Equilibrium) -> tuple[float, float]:
    y_min = max(min(eq.supply[0], eq.demand[-1]) - 1, 0)
    y_max = max(eq.supply[-1], eq.demand[0]) + 1
    return y_min, y_max


# ============================================================
# §7. PLOTS — VALIDATION
# ============================================================

def _plot_trajectory_panel(ax, df_metrics, col, color, ylabel,
                            independent_rounds, n_sims,
                            hline_val=None, hline_color='#e74c3c',
                            hline_label=None):
    """Bottom-row panel of validation plot: trajectory or scatter."""
    if independent_rounds:
        ax.scatter(df_metrics['sim'], df_metrics[col], alpha=0.15, s=10, color=color)
        ax.axhline(df_metrics[col].mean(), color=color, lw=2,
                   label=f"mean = {df_metrics[col].mean():.3f}")
        if hline_val is not None:
            ax.axhline(hline_val, color=hline_color, ls='--', lw=1, alpha=0.5,
                       label=hline_label)
        ax.set_xlabel('Simulation'); ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} across simulations\n(stationarity check)')
        ax.legend(fontsize=9)
    else:
        by_round = df_metrics.groupby('round')[col].agg(['mean', 'sem'])
        for _, sim_data in df_metrics.groupby('sim'):
            ax.plot(sim_data['round'], sim_data[col], alpha=0.15,
                    color=color, lw=0.8)
        ax.errorbar(by_round.index, by_round['mean'], yerr=by_round['sem'],
                    fmt='-o', color=color, capsize=3, markersize=5, lw=2, zorder=5)
        if hline_val is not None:
            ax.axhline(hline_val, color=hline_color, ls='--', lw=1, alpha=0.5,
                       label=hline_label)
        ax.set_xlabel('Round'); ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} by round\n(mean ± SEM, n={n_sims} sims)')
        ax.set_xticks(sorted(df_metrics['round'].unique()))
        if hline_label:
            ax.legend(fontsize=9)


def plot_validation(data: ExperimentData, metrics: ExperimentMetrics,
                    title: str | None = None) -> plt.Figure:
    """2x3 validation panel: efficiency / alpha / quantity histograms +
    trajectory panels + Marshallian path."""
    df_metrics, df_marsh, eq = metrics.round_metrics, metrics.marshallian, data.eq
    n_sims = df_metrics['sim'].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title or data.experiment_id, fontsize=14, fontweight='bold', y=1.02)

    ax = axes[0, 0]
    _safe_hist(ax, df_metrics['efficiency'], color='#16a34a', alpha=0.7, edgecolor='white')
    ax.axvline(df_metrics['efficiency'].mean(), color='black', ls='--', lw=1.5,
               label=f"mean = {df_metrics['efficiency'].mean():.3f}")
    ax.axvline(1.0, color='#e74c3c', ls='-', lw=1, label='CE (100%)')
    ax.set_xlabel('Allocative efficiency'); ax.set_ylabel('Count')
    ax.set_title('Efficiency distribution'); ax.legend(fontsize=9)

    ax = axes[0, 1]
    _safe_hist(ax, df_metrics['alpha'], color='#2563eb', alpha=0.7, edgecolor='white')
    ax.axvline(df_metrics['alpha'].mean(), color='black', ls='--', lw=1.5,
               label=f"mean = {df_metrics['alpha'].mean():.1f}%")
    ax.set_xlabel("Smith's α (%)"); ax.set_ylabel('Count')
    ax.set_title('α distribution'); ax.legend(fontsize=9)

    ax = axes[0, 2]
    tc = df_metrics['n_trades'].value_counts().sort_index()
    ax.bar(tc.index, tc.values, color='#d97706', alpha=0.7, edgecolor='white')
    ax.axvline(eq.quantity, color='#e74c3c', ls='--', lw=1.5, label=f'CE = {eq.quantity}')
    ax.set_xlabel('Trades per round'); ax.set_ylabel('Count')
    ax.set_title('Quantity distribution'); ax.legend(fontsize=9)

    _plot_trajectory_panel(axes[1, 0], df_metrics, 'efficiency', '#16a34a',
                            'Efficiency', data.independent_rounds, n_sims, hline_val=1.0)
    _plot_trajectory_panel(axes[1, 1], df_metrics, 'mean_price', '#d97706',
                            'Mean price', data.independent_rounds, n_sims,
                            hline_val=eq.price, hline_label=f'CE = {eq.price:.2f}')

    ax = axes[1, 2]
    max_seq = int(df_marsh['tx_seq'].quantile(0.95))
    marsh_agg = df_marsh[df_marsh['tx_seq'] <= max_seq].groupby('tx_seq').agg(
        mean_dev=('price_dev', 'mean'), sem_dev=('price_dev', 'sem'))
    ax.errorbar(marsh_agg.index, marsh_agg['mean_dev'], yerr=marsh_agg['sem_dev'],
                fmt='-o', color='#7c3aed', capsize=3, markersize=5)
    ax.set_xlabel('Transaction position within round')
    ax.set_ylabel('Mean |price - CE|')
    ax.set_title('Within-round convergence\n(Marshallian path)')
    plt.tight_layout()
    plt.show()
    return fig


def plot_price_convergence(data: ExperimentData, metrics: ExperimentMetrics,
                           title: str | None = None) -> plt.Figure:
    """Mean transaction price per round across sims, with α annotations."""
    df_metrics, eq = metrics.round_metrics, data.eq
    fig, ax = plt.subplots(figsize=(10, 5))
    sim_color, avg_color = '#1f77b4', '#d62728'
    y_min, y_max = _price_y_bounds(eq)
    alpha_y = y_max - 0.05 * (y_max - y_min)
    avg_alpha = df_metrics.groupby('round')['alpha'].mean()

    if data.independent_rounds:
        ax.scatter(df_metrics['sim'], df_metrics['mean_price'],
                   alpha=0.3, s=15, color=sim_color)
        ax.axhline(df_metrics['mean_price'].mean(), color=avg_color, lw=2,
                   label='Average Price')
        ax.axhline(eq.price, color='grey', ls='--', lw=1.2, alpha=0.6)
        ax.text(df_metrics['sim'].min(), eq.price - 0.15,
                'Competitive Equilibrium Price', color='grey',
                fontsize=10, family='serif', va='bottom')
        ax.text(df_metrics['sim'].median(), alpha_y, f'α={avg_alpha.iloc[0]:.1f}',
                ha='center', va='top', fontsize=10, family='serif',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('Simulation', fontsize=12, family='serif')
    else:
        df_plot = df_metrics.pivot_table(index='round', columns='sim', values='mean_price')
        for sim_id in df_plot.columns:
            ax.plot(df_plot.index, df_plot[sim_id], color=sim_color,
                    alpha=0.3, linewidth=1)
        ax.plot(df_plot.index, df_plot.mean(axis=1), color=avg_color,
                linewidth=2, label='Average Price')
        ax.axhline(eq.price, color='grey', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.text(df_plot.index[0], eq.price - 0.15,
                'Competitive Equilibrium Price', color='grey',
                fontsize=10, family='serif', va='bottom')
        for r, av in avg_alpha.items():
            ax.text(r, alpha_y, f'α={av:.1f}', ha='center', va='top',
                    fontsize=10, family='serif',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
        ax.set_xticks(sorted(df_metrics['round'].unique()))
        ax.set_xlabel('Round', fontsize=12, family='serif')

    ax.set_ylabel('Average Transaction Price per Round', fontsize=12, family='serif')
    ax.set_title(title or 'Average Transaction Price per Round Across Simulations',
                 fontsize=14, family='serif', pad=10)
    _apply_paper_style(ax)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='lower left', frameon=False, prop={'family': 'serif', 'size': 10})
    ax.grid(False); plt.tight_layout(); plt.show()
    return fig


def plot_smith_comparison(data: ExperimentData, metrics: ExperimentMetrics,
                          experiment_id: str | None = None,
                          title: str | None = None) -> plt.Figure:
    """LLM α vs Smith (1962) benchmark."""
    df_metrics = metrics.round_metrics
    eid = experiment_id or data.experiment_id
    smith_key = eid.split('_')[0] if eid else None
    if smith_key not in SMITH_ALPHA:
        smith_key = 'smith1'
    fig, ax = plt.subplots(figsize=(7, 5))
    sr = df_metrics.groupby('round')['alpha'].agg(['mean', 'sem'])
    ax.errorbar(sr.index, sr['mean'], yerr=sr['sem'], fmt='-o',
                color='#1565C0', capsize=3, markersize=6, label='LLM agents')
    smith_data = SMITH_ALPHA[smith_key]
    exp_rounds = set(df_metrics['round'].unique())
    rs = [(r, smith_data[r]) for r in sorted(smith_data.keys()) if r in exp_rounds]
    if rs:
        ax.plot([r for r, _ in rs], [v for _, v in rs], '--s',
                color='#E65100', markersize=6, label=f'Smith (1962) [{smith_key}]')
    ax.set_xlabel('Round'); ax.set_ylabel('Coefficient of convergence (%)')
    ax.set_title(title or 'Coefficient of Convergence\n(LLM agents mean ± SEM vs Smith 1962)')
    ax.set_xticks(sorted(df_metrics['round'].unique())); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


def plot_single_sim_prices(data: ExperimentData, metrics: ExperimentMetrics,
                            sim: int = 1) -> plt.Figure:
    """Smith-style transaction-price chart for one simulation."""
    tx, df_metrics, eq = data.tx, metrics.round_metrics, data.eq
    df_plot = (tx[tx['sim'] == sim].sort_values(['round', 'iteration'])
               .reset_index(drop=True))
    df_plot['transaction_number'] = df_plot.groupby('round').cumcount() + 1

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(df_plot) > 0:
        ax.plot(df_plot.index, df_plot['price'], color='black', linewidth=2,
                label='Transaction Price', zorder=2)
        ax.scatter(df_plot.index, df_plot['price'], marker='s', s=80,
                   color='#1f77b4', zorder=3)

    y_min, y_max = _price_y_bounds(eq)
    alpha_y = y_max - 0.05 * (y_max - y_min)
    # df_plot has a RangeIndex (reset_index above), so first/last index per round
    # is just positional. Avoid groupby().apply() to dodge the FutureWarning.
    round_pos = df_plot.reset_index().groupby('round')['index'].agg(['min', 'max'])
    first_idx = round_pos['min']
    last_idx = round_pos['max']
    for x in last_idx.values[:-1]:
        ax.axvline(x=x + 0.5, color='grey', linestyle='--', linewidth=1.5)

    sim_metrics = df_metrics[df_metrics['sim'] == sim]
    rounds = df_plot['round'].unique()
    for i, rnd in enumerate(rounds):
        start = first_idx.iloc[i] if i == 0 else last_idx.iloc[i - 1] + 0.5
        end = last_idx.iloc[i] + 0.5 if i < len(last_idx) else df_plot.index.max()
        mid = (start + end) / 2
        a = sim_metrics[sim_metrics['round'] == rnd]
        if len(a) > 0:
            ax.text(mid, alpha_y, f'α={a["alpha"].values[0]:.1f}',
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    ax.axhline(eq.price, color='red', linestyle='--')
    ax.set_xticks(df_plot.index)
    ax.set_xticklabels(df_plot['transaction_number'], rotation=0)
    ax.set_title(f'Transaction Prices Across Rounds, simulation {sim}')
    ax.set_xlabel('Transaction within each round'); ax.set_ylabel('Price ($)')
    ax.set_ylim(y_min, y_max); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


# ============================================================
# §8. PLOTS — ORDER FLOW
# ============================================================

def plot_order_flow(data: ExperimentData) -> plt.Figure:
    """Per-sim submitted bid/ask prices with trades highlighted."""
    df_iter, eq = data.iter, data.eq
    ann_rows = df_iter[df_iter['announcement_made'] == True].copy()
    sims = sorted(ann_rows['sim'].unique())
    n_sims = len(sims)
    n_cols = min(2, n_sims); n_rows = (n_sims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(12 * n_cols, 5 * n_rows), squeeze=False)
    flat = axes.flatten()
    y_min, y_max = _price_y_bounds(eq)

    last_i = -1
    for i, sim in enumerate(sims):
        last_i = i
        ax = flat[i]
        df_plot = ann_rows[ann_rows['sim'] == sim]
        for at, st in [('buy', SIDE_STYLES_ALT['buyer']),
                        ('sell', SIDE_STYLES_ALT['seller'])]:
            sub = df_plot[df_plot['announcement_type'] == at]
            ax.plot(sub.index, sub['price'], marker=st['marker'], markersize=8,
                    linestyle='-', label=st['label'], color=st['color'])
        traded = df_plot[df_plot['transaction_made'] == True]
        for at, st in [('buy', SIDE_STYLES_ALT['buyer']),
                        ('sell', SIDE_STYLES_ALT['seller'])]:
            sub = traded[traded['announcement_type'] == at]
            ax.plot(sub.index, sub['price'], marker=st['marker'], markersize=10,
                    linestyle='', markeredgecolor='black', markeredgewidth=1.5,
                    color=st['color'], label=f"{at.title()} → Trade")
        last_idx = (df_plot.reset_index().groupby('round')['index'].max())
        for x in last_idx.values[:-1]:
            ax.axvline(x=x + 0.5, color='grey', linestyle='--', linewidth=1.5)
        ax.axhline(eq.price, color='red', linestyle='--')
        ax.set_xticks([])
        ax.set_title(f'Order Flow, simulation {sim}')
        ax.set_ylabel('Submitted Price'); ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=8)
    for j in range(last_i + 1, len(flat)):
        flat[j].set_visible(False)
    plt.tight_layout(); plt.show()
    return fig


def plot_bid_ask_dispersion(data: ExperimentData) -> plt.Figure:
    """Std of bid/ask prices within round (mean ± std across sims)."""
    df_ann = data.ann
    buy_std = df_ann[df_ann['side'] == 'buyer'].groupby(['sim', 'round'])['price'].std()
    sell_std = df_ann[df_ann['side'] == 'seller'].groupby(['sim', 'round'])['price'].std()
    df_std = pd.concat([buy_std, sell_std], axis=1)
    df_std.columns = ['bid', 'ask']
    mean = df_std.groupby(level=1).mean(); std = df_std.groupby(level=1).std()
    colors = {'ask': SIDE_STYLES_ALT['seller']['color'],
              'bid': SIDE_STYLES_ALT['buyer']['color']}
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df_std.columns:
        ax.plot(mean.index, mean[col], label=col.title(), marker='o', color=colors[col])
        ax.fill_between(mean.index, mean[col] - std[col], mean[col] + std[col],
                        alpha=0.2, color=colors[col])
    ax.set_xticks(mean.index)
    ax.legend(loc='upper right', frameon=False, prop={'family': 'serif', 'size': 10})
    ax.set_xlabel('Round', fontsize=12, family='serif')
    ax.set_ylabel('Mean Standard Deviation', fontsize=12, family='serif')
    ax.set_title('Std of Bids/Asks within Round (mean across simulations)',
                 fontsize=14, family='serif', pad=10)
    _apply_paper_style(ax)
    plt.tight_layout(); plt.show()
    return fig


def plot_order_price_vs_reservation(data: ExperimentData) -> plt.Figure:
    """Scatter: submitted order price vs reservation price."""
    df_ann, eq = data.ann, data.eq
    fig, ax = plt.subplots(figsize=(8, 8))
    for side, marker in [('buyer', 'o'), ('seller', '^')]:
        sub = df_ann[df_ann['side'] == side]
        ax.scatter(sub['announcing_agent_reservation_price'], sub['price'],
                   color=SIDE_STYLES[side]['color'], label=f'{side.title()}s',
                   edgecolor='k', s=70, linewidth=0.5, marker=marker)
    min_v = max(np.concatenate([eq.supply, eq.demand]).min() - 1, 0)
    max_v = np.concatenate([eq.supply, eq.demand]).max() + 1
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1.5, label='45° Line')
    ax.set_xlabel('Reservation Price', fontsize=14, labelpad=10)
    ax.set_ylabel('Submitted Order Price', fontsize=14, labelpad=10)
    ax.set_title('Submitted Order Price vs Reservation Price', fontsize=16, pad=15)
    ax.legend(frameon=True, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_aspect('equal'); ax.set_xlim(min_v, max_v); ax.set_ylim(min_v, max_v)
    plt.tight_layout(); plt.show()
    return fig


def plot_fill_rate(data: ExperimentData) -> plt.Figure | None:
    """Fill rate vs reservation price + by round."""
    df_ann, eq = data.ann, data.eq
    if 'filled' not in df_ann.columns:
        print("No fill rate data available.")
        return None
    agent_fill = (df_ann.groupby(
        ['sim', 'side', 'announcing_agent_id',
         'announcing_agent_reservation_price'])['filled']
        .mean().reset_index(name='fill_rate'))
    agg = agent_fill.groupby(['side', 'announcing_agent_reservation_price']).agg(
        mean_rate=('fill_rate', 'mean'),
        se=('fill_rate', lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
    ).reset_index()
    agg['ci95'] = 1.96 * agg['se']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for side in ['buyer', 'seller']:
        st = SIDE_STYLES[side]
        df_t = agg[agg['side'] == side].sort_values('announcing_agent_reservation_price')
        ax.plot(df_t['announcing_agent_reservation_price'], df_t['mean_rate'],
                marker='o', color=st['color'], label=st['label'])
        ax.fill_between(df_t['announcing_agent_reservation_price'],
                        df_t['mean_rate'] - df_t['ci95'],
                        df_t['mean_rate'] + df_t['ci95'],
                        color=st['color'], alpha=0.3)
    ax.axvline(eq.price, color='grey', ls='--', lw=0.8, label='CE price')
    ax.set_xlabel('Reservation Price', fontsize=12, family='serif')
    ax.set_ylabel('Fill Rate', fontsize=12, family='serif')
    ax.set_title('Fill Rate vs Reservation Price', fontsize=14, family='serif', pad=10)
    ax.legend(frameon=False, prop={'family': 'serif', 'size': 10})
    _apply_paper_style(ax)

    ax = axes[1]
    _errorbar_by_round_by_side(ax, df_ann, 'filled')
    ax.set_xlabel('Round'); ax.set_ylabel('Fill Rate')
    ax.set_title('Fill Rate by Round\n(mean ± SEM across sims)')
    ax.set_xticks(sorted(df_ann['round'].unique()))
    _yax_pct(ax); ax.legend()
    _apply_paper_style(ax)
    plt.tight_layout(); plt.show()
    return fig


# ============================================================
# §9. PLOTS — RENT METRICS
# ============================================================

def _plot_rent_metric_2x2(rent: dict, *, value_col: str, title: str,
                           xlabel: str, hline: float | None = None,
                           hline_label: str | None = None) -> plt.Figure:
    """Standard 2x2 layout used by Metrics 1 (attempted_rent) and 2 (rent_ratio).

    a) histogram by side
    b) errorbar across reservation prices, by side
    c) errorbar across rounds, by side
    d) per-agent horizontal bar chart
    """
    ann = rent['ann']
    sims = sorted(ann['sim'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    _histogram_by_side(axes[0, 0], ann, value_col, sims,
                       title=f'a) Distribution (mean ± SEM)', xlabel=xlabel)

    ax = axes[0, 1]
    _errorbar_by_reservation(ax, ann, value_col)
    if hline is not None:
        ax.axhline(hline, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('Reservation price ($)')
    ax.set_ylabel(f'Mean {xlabel.lower()}')
    ax.set_title('b) By reservation price'); ax.legend()

    ax = axes[1, 0]
    _errorbar_by_round_by_side(ax, ann, value_col)
    if hline is not None:
        ax.axhline(hline, color='grey', ls='--', lw=0.8,
                   label=hline_label) if hline_label else ax.axhline(
                       hline, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('Round'); ax.set_ylabel(f'Mean {xlabel.lower()}')
    ax.set_title('c) Over rounds'); ax.set_xticks(sorted(ann['round'].unique()))
    ax.legend()

    ax = axes[1, 1]
    _per_agent_barh(ax, ann, value_col)
    ax.set_xlabel(f'Mean {xlabel.lower()}'); ax.set_title('d) Per-agent')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()
    return fig


def plot_attempted_rent(rent: dict, eq: Equilibrium) -> plt.Figure:
    return _plot_rent_metric_2x2(
        rent, value_col='attempted_rent',
        title='Metric 1: Attempted Rent Margin (submitted order prices)',
        xlabel='Attempted rent ($)', hline=0)


def plot_rent_ratio(rent: dict, eq: Equilibrium) -> plt.Figure:
    return _plot_rent_metric_2x2(
        rent, value_col='rent_ratio',
        title='Metric 2: Attempted Rent Ratio',
        xlabel='Rent ratio', hline=0)


def plot_realized_rent(rent: dict, eq: Equilibrium) -> plt.Figure:
    """Metric 3 (2x2). Uses execution prices."""
    tx = rent['tx']; sims = sorted(tx['sim'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    all_rents = pd.concat([tx['buyer_rent'], tx['seller_rent']])
    bin_edges = np.linspace(all_rents.min(), all_rents.max(), 26)
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2
    w = bin_edges[1] - bin_edges[0]; h = w / 2
    for col, side, offset in [('buyer_rent', 'buyer', -h / 2),
                               ('seller_rent', 'seller', h / 2)]:
        st = SIDE_STYLES[side]
        counts = np.zeros((len(sims), len(bc)))
        for i, s in enumerate(sims):
            counts[i], _ = np.histogram(tx[tx['sim'] == s][col], bins=bin_edges)
        ax.bar(bc + offset, counts.mean(0), width=h, color=st['color'],
               edgecolor='white', label=st['label'])
        sem = counts.std(0, ddof=1) / np.sqrt(len(sims)) if len(sims) > 1 else np.zeros_like(counts.mean(0))
        ax.errorbar(bc + offset, counts.mean(0), yerr=sem, fmt='none',
                    ecolor='black', capsize=1.5, lw=0.8)
    ax.set_xlabel('Realized rent ($)'); ax.set_ylabel('Mean count')
    ax.set_title('a) Distribution'); ax.legend()

    ax = axes[0, 1]
    tx_pos = tx[tx['total_surplus'] > 0].copy()
    srs = tx_pos.groupby(['sim', 'round'])[['buyer_share', 'seller_share']].mean().reset_index()
    for col, side in [('buyer_share', 'buyer'), ('seller_share', 'seller')]:
        st = SIDE_STYLES[side]
        agg = srs.groupby('round')[col].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'],
                    fmt='-o', color=st['color'], capsize=3, label=st['label'])
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, label='Equal split')
    ax.set_ylabel('Surplus share'); ax.set_xlabel('Round')
    ax.set_title('b) Surplus share by round')
    ax.set_xticks(sorted(tx['round'].unique()))
    _yax_pct(ax); ax.legend(fontsize=8)

    ax = axes[1, 0]
    srm = tx.groupby(['sim', 'round'])['total_surplus'].mean().reset_index()
    agg = srm.groupby('round')['total_surplus'].agg(['mean', 'sem'])
    ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt='-o',
                color='#4CAF50', capsize=3)
    ax.set_xlabel('Round'); ax.set_ylabel('Mean surplus/tx ($)')
    ax.set_title('c) Mean surplus per tx')
    ax.set_xticks(sorted(tx['round'].unique()))

    ax = axes[1, 1]
    srs2 = tx.groupby(['sim', 'round'])['total_surplus'].sum().reset_index()
    agg = srs2.groupby('round')['total_surplus'].agg(['mean', 'sem'])
    ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'], fmt='-o',
                color='#4CAF50', capsize=3, label='Realised')
    ax.axhline(eq.surplus, color='grey', ls='--', lw=0.8,
               label=f'Equilibrium (${eq.surplus:.2f})')
    ax.set_xlabel('Round'); ax.set_ylabel('Total surplus ($)')
    ax.set_title('d) Total surplus vs eq')
    ax.set_xticks(sorted(tx['round'].unique())); ax.legend(fontsize=8)

    fig.suptitle('Metric 3: Realized Rent (execution prices)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()
    return fig


def plot_extraction_efficiency(rent: dict, eq: Equilibrium) -> plt.Figure:
    """Metric 4: Rent Extraction Efficiency (1x3)."""
    eff, ann = rent['extraction_eff'], rent['ann']
    sims = sorted(eff['sim'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _histogram_by_side(axes[0], eff, 'efficiency', sims, n_bins=25,
                       title='a) Distribution', xlabel='Extraction efficiency')
    axes[0].axvline(1.0, color='grey', ls='--', lw=0.8, label='100%')
    axes[0].legend(fontsize=8)

    ax = axes[1]
    _errorbar_by_round_by_side(ax, eff, 'efficiency')
    ax.axhline(1.0, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('Round'); ax.set_ylabel('Mean efficiency')
    ax.set_title('b) Over rounds')
    ax.set_xticks(sorted(ann['round'].unique())); ax.legend()

    ax = axes[2]
    for side in ['buyer', 'seller']:
        st = SIDE_STYLES[side]
        sub = eff[eff['side'] == side]
        ax.scatter(sub['first_attempted_rent'], sub['realized_rent'],
                   alpha=0.2, c=st['color'], marker=st['marker'], s=25,
                   label=st['label'])
    mv = max(eff['first_attempted_rent'].max(), eff['realized_rent'].max()) * 1.05
    ax.plot([0, mv], [0, mv], 'k--', lw=0.8, alpha=0.5, label='100% line')
    ax.set_xlabel('First attempted rent ($)'); ax.set_ylabel('Realised rent ($)')
    ax.set_title('c) Ambition vs outcome'); ax.legend(fontsize=8)

    fig.suptitle('Metric 4: Rent Extraction Efficiency',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout(); plt.show()
    return fig


def plot_concession_rate(rent: dict, eq: Equilibrium) -> plt.Figure:
    """Metric 5: Concession Rate (1x3)."""
    conc, ann, abr = rent['concessions'], rent['ann'], rent['rent_surrendered']
    sims = sorted(conc['sim'].unique()) if len(conc) > 0 else []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    if len(conc) > 0:
        _histogram_by_side(axes[0], conc, 'concession', sims, n_bins=30,
                           title='a) Concession distribution',
                           xlabel='Concession ($)')

    ax = axes[1]
    if len(conc) > 0:
        _errorbar_by_round_by_side(ax, conc, 'frac_concession')
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('Round'); ax.set_ylabel('Frac concession')
    ax.set_title('b) Over rounds')
    ax.set_xticks(sorted(ann['round'].unique()))
    _yax_pct(ax); ax.legend()

    ax = axes[2]
    if len(abr) > 0:
        _errorbar_by_round_by_side(ax, abr, 'frac_surrendered')
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.axhline(1, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Round'); ax.set_ylabel('Frac surrendered')
    ax.set_title('c) Cumulative surrender')
    ax.set_xticks(sorted(ann['round'].unique()))
    _yax_pct(ax); ax.legend()

    fig.suptitle('Metric 5: Concession Rate (submitted prices)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()
    return fig


def plot_zero_profit_orders(rent: dict, eq: Equilibrium,
                             tol: float = 1e-9) -> plt.Figure:
    """Metric 6: % of orders at exactly reservation price (within tolerance)."""
    ann = rent['ann'].copy()
    ann['at_reservation'] = np.isclose(ann['attempted_rent'], 0, atol=tol)

    fig, ax = plt.subplots(figsize=(6, 4))
    _errorbar_by_round_by_side(ax, ann, 'at_reservation')
    ax.set_xlabel('Round'); ax.set_ylabel('% at reservation price')
    ax.set_title('Metric 6: Orders at Exactly Reservation Price')
    ax.set_xticks(sorted(ann['round'].unique()))
    _yax_pct(ax); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


def plot_constraint_violations(rent: dict, eq: Equilibrium) -> plt.Figure:
    """Order-level and transaction-level violation rates."""
    ann, tx = rent['ann'], rent['tx']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ann_viol = (ann.groupby(['sim', 'round', 'side'])
                .agg(n_total=('violation', 'count'), n_viol=('violation', 'sum'))
                .reset_index())
    ann_viol['rate'] = ann_viol['n_viol'] / ann_viol['n_total']

    ax = axes[0]
    _errorbar_by_round_by_side(ax, ann_viol, 'rate')
    ax.set_xlabel('Round'); ax.set_ylabel('Violation rate')
    ax.set_title('Order Violations\n(submitted beyond reservation)')
    ax.set_xticks(sorted(ann['round'].unique()))
    _yax_pct(ax); ax.legend()

    # Transaction-level: a "side" column does not exist on tx, so plot the two
    # rates manually rather than via the helper.
    tx_viol = (tx.groupby(['sim', 'round'])
               .agg(n_tx=('buyer_violation', 'count'),
                    n_bv=('buyer_violation', 'sum'),
                    n_sv=('seller_violation', 'sum'))
               .reset_index())
    tx_viol['bv_rate'] = tx_viol['n_bv'] / tx_viol['n_tx']
    tx_viol['sv_rate'] = tx_viol['n_sv'] / tx_viol['n_tx']
    ax = axes[1]
    for col, side in [('bv_rate', 'buyer'), ('sv_rate', 'seller')]:
        st = SIDE_STYLES[side]
        agg = tx_viol.groupby('round')[col].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'],
                    fmt=f"-{st['marker']}", color=st['color'],
                    capsize=3, label=st['label'])
    ax.set_xlabel('Round'); ax.set_ylabel('Violation rate')
    ax.set_title('Transaction Violations\n(execution at a loss)')
    ax.set_xticks(sorted(tx['round'].unique()))
    _yax_pct(ax); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


def plot_order_frequency(rent: dict, eq: Equilibrium) -> plt.Figure:
    """Mean orders per round, by side."""
    ann = rent['ann']
    ac = ann.groupby(['sim', 'round', 'side']).size().reset_index(name='n')
    fig, ax = plt.subplots(figsize=(6, 4))
    _errorbar_by_round_by_side(ax, ac, 'n', aggregate='sum')
    ax.set_xlabel('Round'); ax.set_ylabel('Orders per round')
    ax.set_title('Order Frequency\n(mean ± SEM)')
    ax.set_xticks(sorted(ann['round'].unique())); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


def plot_who_trades(rent: dict, eq: Equilibrium) -> plt.Figure:
    """Inframarginal participation + transactions by reservation price."""
    tx = rent['tx']; sims = sorted(tx['sim'].unique())
    tw = (tx.groupby(['sim', 'round'])
          .agg(n=('both_inframarginal', 'count'),
               nb=('buyer_inframarginal', 'sum'),
               ns=('seller_inframarginal', 'sum'),
               nboth=('both_inframarginal', 'sum'))
          .reset_index())
    tw['pb'] = tw['nb'] / tw['n']
    tw['ps'] = tw['ns'] / tw['n']
    tw['pboth'] = tw['nboth'] / tw['n']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    series = [
        ('pb', SIDE_STYLES['buyer']['color'], 'o', 'Buyer infra'),
        ('ps', SIDE_STYLES['seller']['color'], 's', 'Seller infra'),
        ('pboth', '#4CAF50', 'D', 'Both infra'),
    ]
    for col, color, marker, label in series:
        agg = tw.groupby('round')[col].agg(['mean', 'sem'])
        ax.errorbar(agg.index, agg['mean'], yerr=agg['sem'],
                    fmt=f'-{marker}', color=color, capsize=3, label=label)
    ax.set_xlabel('Round'); ax.set_ylabel('Fraction')
    ax.set_title('Inframarginal Participation')
    ax.set_xticks(sorted(tx['round'].unique()))
    _yax_pct(ax); ax.legend(fontsize=8)

    ax = axes[1]
    all_res = np.sort(np.union1d(eq.supply, eq.demand))
    x = np.arange(len(all_res)); w = 0.35
    for col, side, offset in [('buyer_val', 'buyer', -w / 2),
                                ('seller_cost', 'seller', w / 2)]:
        st = SIDE_STYLES[side]
        counts = np.zeros((len(sims), len(all_res)))
        for i, s in enumerate(sims):
            rc = tx[tx['sim'] == s][col].value_counts()
            for j, r in enumerate(all_res):
                counts[i, j] = rc.get(r, 0)
        ax.bar(x + offset, counts.mean(0), w, color=st['color'],
               edgecolor='white', label=st['label'])
        sem = counts.std(0, ddof=1) / np.sqrt(len(sims)) if len(sims) > 1 else np.zeros_like(counts.mean(0))
        ax.errorbar(x + offset, counts.mean(0), yerr=sem,
                    fmt='none', ecolor='black', capsize=1.5, lw=0.8)
    eq_idx = np.argmin(np.abs(all_res - eq.price))
    ax.axvline(x=eq_idx + 0.5, color='grey', ls='--', lw=0.8, label='Eq price')
    ax.set_xticks(x)
    ax.set_xticklabels([f'${r:.2f}' for r in all_res], rotation=45, fontsize=8)
    ax.set_xlabel('Reservation price ($)'); ax.set_ylabel('Mean tx/sim')
    ax.set_title('Tx by Reservation Price'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    return fig


def plot_agent_rent_trajectories(rent: dict, eq: Equilibrium) -> plt.Figure:
    """Per-(sim × round) grid of agent rent trajectories."""
    ann = rent['ann']
    sims = sorted(ann['sim'].unique())
    rounds = sorted(ann['round'].unique())
    fig, axes = plt.subplots(len(sims), len(rounds),
                             figsize=(4 * len(rounds), 3.5 * len(sims)),
                             squeeze=False)
    for row, s in enumerate(sims):
        for col, rnd in enumerate(rounds):
            ax = axes[row, col]
            ae = ann[(ann['sim'] == s) & (ann['round'] == rnd)]
            for agent, grp in ae.groupby('announcing_agent_id'):
                grp = grp.sort_values('iteration')
                side = grp['side'].iloc[0]
                color = SIDE_STYLES[side]['color']
                if len(grp) == 1:
                    ax.plot(grp['iteration'].values[0],
                            grp['attempted_rent'].values[0],
                            'o', color=color, alpha=0.35, markersize=4)
                else:
                    ax.plot(grp['iteration'], grp['attempted_rent'], '-o',
                            color=color, alpha=0.5, markersize=3, lw=1.0)
            ax.axhline(0, color='grey', ls='--', lw=0.8)
            if row == 0:
                ax.set_title(f'Round {rnd}', fontsize=10)
            ax.set_xlabel('Iteration')
            if col == 0:
                ax.set_ylabel(f'Sim {s}\nAttempted rent ($)')
    axes[0, 0].legend(handles=[
        Line2D([0], [0], color=SIDE_STYLES['buyer']['color'], marker='o', label='Buyer'),
        Line2D([0], [0], color=SIDE_STYLES['seller']['color'], marker='o', label='Seller'),
    ], fontsize=8)
    fig.suptitle('Agent Rent Trajectories', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.show()
    return fig


# ============================================================
# §10. PLOTS — SPREAD & INITIATION
# ============================================================

def plot_spread_evolution(metrics: ExperimentMetrics, data: ExperimentData,
                          figsize: tuple = (7, 4)) -> plt.Figure:
    """Mean bid-ask spread by round."""
    by_round = metrics.spread_by_round
    tx_per_round = (data.tx.groupby(['sim', 'round']).size()
                    .groupby('round').mean().rename('avg_tx'))
    rounds = sorted(by_round['round'].unique())
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(rounds)))

    fig, ax = plt.subplots(figsize=figsize)
    means = by_round.groupby('round')['mean_spread'].mean().loc[rounds]
    ses = by_round.groupby('round')['mean_spread'].sem().fillna(0).loc[rounds]
    ax.bar(rounds, means, color=palette, alpha=0.75,
           edgecolor='white', linewidth=0.8, zorder=3)
    ax.errorbar(rounds, means, yerr=ses, fmt='none', color='black',
                capsize=4, linewidth=1.2, zorder=4)
    y_max = (means + ses).max()
    for r in rounds:
        n_tx = tx_per_round.get(r, 0)
        ax.text(r, means[r] + ses.get(r, 0) + y_max * 0.03, f'{n_tx:.0f} tx',
                ha='center', va='bottom', fontsize=8, color='#333333')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Mean bid-ask spread ($)', fontsize=11)
    ax.set_title('Bid-Ask Spread by Round', fontsize=13, fontweight='bold')
    ax.set_xticks(rounds)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.2f'))
    ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout(); plt.show()
    return fig


def plot_spread_global(metrics: ExperimentMetrics, data: ExperimentData,
                       figsize: tuple = (14, 5)) -> plt.Figure:
    """Spread across the full timeline, with round boundaries and tx ticks."""
    df = metrics.spread_series.copy()
    df_tx = data.tx
    round_lengths = (df.groupby(['sim', 'round'])['iteration'].count()
                     .groupby(level='round').max()
                     .cumsum().shift(1, fill_value=0).rename('offset'))
    df = df.join(round_lengths, on='round')
    df['local_rank'] = df.sort_values('iteration').groupby(['sim', 'round']).cumcount()
    df['global_tick'] = df['offset'] + df['local_rank']

    agg = (df.groupby('global_tick')['spread']
           .agg(mean_spread='mean', se_spread='sem').reset_index())
    boundaries = round_lengths[round_lengths > 0].to_dict()

    tx_marks = (df_tx[['sim', 'round', 'iteration']]
                .merge(df[['sim', 'round', 'iteration', 'global_tick']],
                       on=['sim', 'round', 'iteration'], how='left')
                .dropna(subset=['global_tick'])
                .groupby('global_tick')['sim'].count().reset_index()
                .rename(columns={'sim': 'n_tx'}))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(agg['global_tick'], agg['mean_spread'],
            color='steelblue', linewidth=1.4, zorder=3)
    ax.fill_between(agg['global_tick'],
                    agg['mean_spread'] - agg['se_spread'].fillna(0),
                    agg['mean_spread'] + agg['se_spread'].fillna(0),
                    color='steelblue', alpha=0.2, zorder=2)
    for r, offset in boundaries.items():
        ax.axvline(offset, color='gray', linestyle='--', linewidth=0.9,
                   alpha=0.7, zorder=1)
        ax.text(offset + agg['global_tick'].max() * 0.005, 1, f'R{r}',
                fontsize=8, color='gray', va='top',
                transform=ax.get_xaxis_transform())
    if not tx_marks.empty:
        ax.vlines(tx_marks['global_tick'], 0, -0.02, color='crimson',
                  linewidth=1.0, transform=ax.get_xaxis_transform(),
                  label='Transaction', clip_on=False)
    ax.set_xlabel('Global iteration (continuous across rounds)', fontsize=11)
    ax.set_ylabel('Bid-ask spread ($)', fontsize=11)
    ax.set_title('Bid-Ask Spread: Full Experiment Timeline',
                 fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.2f'))
    ax.legend(fontsize=9); ax.grid(linestyle=':', alpha=0.5)
    ax.set_axisbelow(True); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout(); plt.show()
    return fig


def plot_trade_initiation(metrics: ExperimentMetrics) -> plt.Figure:
    """Buyer- vs seller-initiated trade fractions by round."""
    summary = metrics.initiation
    rounds = sorted(summary['round'].unique())
    types = ['buy', 'sell']
    labels = {'buy': 'Buyer-initiated (crosses ask)',
              'sell': 'Seller-initiated (crosses bid)'}
    colors = {'buy': 'steelblue', 'sell': 'tomato'}
    bar_w = 0.35; x = np.arange(len(rounds))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, t in enumerate(types):
        sub = summary[summary['announcement_type'] == t].set_index('round')
        means = [sub.loc[r, 'mean_fraction'] if r in sub.index else 0 for r in rounds]
        ses = [sub.loc[r, 'se_fraction'] if r in sub.index else 0 for r in rounds]
        ses = [0 if (s is None or np.isnan(s)) else s for s in ses]
        offset = (i - 0.5) * bar_w
        ax.bar(x + offset, means, bar_w, label=labels[t], color=colors[t],
               alpha=0.75, edgecolor='white', linewidth=0.8, zorder=3)
        ax.errorbar(x + offset, means, yerr=ses, fmt='none', color='black',
                    capsize=3, linewidth=1.0, zorder=4)
        ns = [sub.loc[r, 'mean_n'] if r in sub.index else 0 for r in rounds]
        for xi, (m, n) in enumerate(zip(means, ns)):
            ax.text(xi + offset, m + (max(ses) if ses else 0) + 0.02,
                    f'{n:.0f}', ha='center', va='bottom',
                    fontsize=8, color='#333333')

    ax.set_xticks(x); ax.set_xticklabels([f'Round {r}' for r in rounds])
    ax.set_ylabel('Fraction of trades', fontsize=11)
    ax.set_title('Trade Initiation: Buyer vs Seller', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=9); ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    plt.show()
    return fig


# ============================================================
# §11. PIPELINES
# ============================================================

def render_all_plots(data: ExperimentData, metrics: ExperimentMetrics,
                     title: str | None = None,
                     experiment_id: str | None = None) -> dict:
    """Render every plot. Returns dict of name → Figure."""
    figs = {}
    eid = experiment_id or data.experiment_id

    figs['validation'] = plot_validation(data, metrics, title=title)
    figs['price_convergence'] = plot_price_convergence(data, metrics, title=title)
    if metrics.round_metrics['round'].nunique() > 1:
        figs['smith_comparison'] = plot_smith_comparison(
            data, metrics, experiment_id=eid, title=title)
    figs['single_sim_prices'] = plot_single_sim_prices(data, metrics, sim=1)

    figs['order_flow'] = plot_order_flow(data)
    figs['bid_ask_dispersion'] = plot_bid_ask_dispersion(data)
    figs['order_price_vs_reservation'] = plot_order_price_vs_reservation(data)
    figs['fill_rate'] = plot_fill_rate(data)

    rent, eq = metrics.rent, data.eq
    figs['attempted_rent'] = plot_attempted_rent(rent, eq)
    figs['rent_ratio'] = plot_rent_ratio(rent, eq)
    figs['realized_rent'] = plot_realized_rent(rent, eq)
    figs['extraction_efficiency'] = plot_extraction_efficiency(rent, eq)
    figs['concession_rate'] = plot_concession_rate(rent, eq)
    figs['zero_profit_orders'] = plot_zero_profit_orders(rent, eq)
    figs['constraint_violations'] = plot_constraint_violations(rent, eq)
    figs['order_frequency'] = plot_order_frequency(rent, eq)
    figs['who_trades'] = plot_who_trades(rent, eq)
    figs['agent_trajectories'] = plot_agent_rent_trajectories(rent, eq)

    figs['spread_by_round'] = plot_spread_evolution(metrics, data)
    figs['spread_global'] = plot_spread_global(metrics, data)
    figs['trade_initiation'] = plot_trade_initiation(metrics)

    return figs


def report_all(data: ExperimentData, metrics: ExperimentMetrics) -> None:
    """Print every text summary."""
    report_round_metrics(summarize_round_metrics(
        metrics.round_metrics, data.eq, data.independent_rounds))
    report_marshallian(summarize_marshallian(metrics.marshallian, data.eq))
    report_market_summary_table(metrics.summary_table)
    report_rent(summarize_rent(metrics.rent))


def run_full_analysis(results_path: Path, n_sims: int,
                      title: str | None = None,
                      config: dict | None = None,
                      experiment_id: str | None = None,
                      independent_rounds: bool = False
                      ) -> tuple[ExperimentData, ExperimentMetrics, dict]:
    """End-to-end: load, compute, print, plot. Returns (data, metrics, figs)."""
    data = load_experiment_data(
        results_path, n_sims, config=config,
        experiment_id=experiment_id, independent_rounds=independent_rounds)
    metrics = compute_all_metrics(data)
    metrics.summary_table = build_market_summary_table(
        metrics.round_metrics, data.eq, experiment_id=experiment_id)
    report_all(data, metrics)
    figs = render_all_plots(data, metrics, title=title,
                            experiment_id=experiment_id)
    return data, metrics, figs


# ============================================================
# Standalone usage
# ============================================================
if __name__ == '__main__':
    import sys, yaml
    if len(sys.argv) < 2:
        print("Usage: python results_analysis.py <results_path> [n_sims]")
        sys.exit(1)
    results_path = Path(sys.argv[1])
    cfg_path = results_path / 'config_used.yaml'
    if not cfg_path.exists():
        print(f"No config found at {cfg_path}"); sys.exit(1)
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    n_sims = int(sys.argv[2]) if len(sys.argv) >= 3 else config['experiment']['n_simulations']
    run_full_analysis(results_path, n_sims, config=config,
                      experiment_id=results_path.name)
