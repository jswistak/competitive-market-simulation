"""
Market Experiment Analysis — Supplementary Material
===================================================
Trimmed version of the full analysis pipeline, containing only the figures and
table that appear in the paper:

  1. Supply & demand equilibrium chart       (produced during data loading)
  2. Market summary table (Smith 1962 style)
  3. Per-simulation order-flow charts        (all submitted bids/asks)
  4. Fill rate vs reservation price
  5. Cumulative fraction of quote-improvement sizes (ECDF)
  6. First attempted rent: first vs second mover

Loads three data sources per simulation:
  - iteration_history_{sim}.csv : full order flow (all submitted bids/asks)
  - transactions_{sim}.csv      : completed transactions with execution prices
  - agent_histories_{sim}.csv   : per-agent action log with reservation prices

NOTE: iteration_history.price is the SUBMITTED limit price; transactions.price
is the EXECUTION price. The summary table uses execution prices; the order-flow
and rent charts use submitted prices.

A simulation is loaded whenever its iteration_history and agent_histories files
are present; transactions are optional. Sims with no trades (no transactions
file) still contribute all their submitted orders to the order-flow and rent
figures. The summary table is built from whichever sims did trade, and is
skipped entirely if none did.

Public API:
    data, metrics, figs = run_full_analysis(results_path, n_sims)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


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
# Alternate palette used by the order-flow plot (kept for visual parity with
# the original module).
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
    tx: pd.DataFrame        # enriched transactions (buyer_val, seller_cost)
    agents: pd.DataFrame
    ann: pd.DataFrame       # extracted, deduplicated announcements
    config: dict
    independent_rounds: bool = False  # kept for API compatibility (unused here)
    has_transactions: bool = True     # False if no simulation produced trades


@dataclass
class ExperimentMetrics:
    """Computed metric DataFrames for one experiment."""
    round_metrics: pd.DataFrame
    quote_improvements: pd.DataFrame
    summary_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# ============================================================
# §3. LOADING & ENRICHMENT
# ============================================================

# Canonical schema for an empty transactions frame, used when no simulation
# produced any trades (a transactions_{sim}.csv is never written in that case).
_EMPTY_TX_COLUMNS = [
    'sim', 'round', 'iteration', 'price',
    'buyer_id', 'seller_id', 'announcement_type',
]


def _load_csvs(results_path: Path, n_sims: int
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Load and concat the CSV families across all available sims.

    A simulation is included whenever its iteration_history and agent_histories
    files exist; transactions are OPTIONAL. Simulations with no trades never
    get a transactions_{sim}.csv, so requiring it would silently drop them --
    and every submitted order they contain -- from the order-flow / rent
    figures. Returns (df_iter, df_tx, df_agents, has_transactions), where
    has_transactions is False iff no simulation produced any trades.
    """
    data_dir = results_path / 'data'
    available = [
        s for s in range(1, n_sims + 1)
        if (data_dir / f'iteration_history_{s}.csv').exists()
        and (data_dir / f'agent_histories_{s}.csv').exists()
    ]
    if not available:
        raise FileNotFoundError(f"No matching files found in {data_dir}")

    sims_with_tx = [
        s for s in available
        if (data_dir / f'transactions_{s}.csv').exists()
    ]

    df_iter = pd.concat([
        pd.read_csv(data_dir / f'iteration_history_{s}.csv').assign(sim=s)
        for s in available
    ], ignore_index=True)
    df_agents = pd.concat([
        pd.read_csv(data_dir / f'agent_histories_{s}.csv').assign(sim=s)
        for s in available
    ], ignore_index=True)

    if sims_with_tx:
        df_tx = pd.concat([
            pd.read_csv(data_dir / f'transactions_{s}.csv').assign(sim=s)
            for s in sims_with_tx
        ], ignore_index=True)
        has_transactions = True
    else:
        df_tx = pd.DataFrame(columns=_EMPTY_TX_COLUMNS)
        has_transactions = False

    n_rounds_iter = df_iter.groupby('sim')['round'].nunique().iloc[0]
    print(f"Loaded {len(available)} simulations from {results_path.name}")
    print(f"  Iteration history rows: {len(df_iter)}")
    if has_transactions:
        print(f"  Transaction rows:       {len(df_tx)} "
              f"(from {len(sims_with_tx)}/{len(available)} sims with trades)")
    else:
        print(f"  Transaction rows:       0 (no simulation produced trades)")
    print(f"  Rounds per sim:         {n_rounds_iter}")
    return df_iter, df_tx, df_agents, has_transactions


def _enrich_transactions(df_tx: pd.DataFrame, eq: Equilibrium) -> pd.DataFrame:
    """Attach buyer valuation and seller cost (needed by the summary table).
    Safe on an empty transactions frame (no trades in any simulation)."""
    tx = df_tx.copy()
    if len(tx) == 0:
        tx['buyer_val'] = pd.Series(dtype=float)
        tx['seller_cost'] = pd.Series(dtype=float)
        return tx
    tx['buyer_val'] = tx['buyer_id'].map(eq.buyer_map)
    tx['seller_cost'] = tx['seller_id'].map(eq.seller_map)
    return tx


def _extract_announcements(df_iter: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate submitted orders and tag side + fill status.
    price is the SUBMITTED limit price, not the execution price."""
    ann = (
        df_iter[df_iter['announcement_made'] == True]
        .drop_duplicates(subset=['sim', 'round', 'iteration',
                                 'announcing_agent_id', 'price',
                                 'announcement_type'])
        .copy()
    )
    ann['side'] = ann['announcement_type'].map({'buy': 'buyer', 'sell': 'seller'})

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
    """Visualise S/D curves with equilibrium point. (Paper figure 1.)"""
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
    """Compute competitive equilibrium from config schedules. Renders the
    supply/demand equilibrium chart (paper figure 1) when show_plot=True."""
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
    """Load + enrich one experiment. Reads config_used.yaml if config not given.
    Rendering the equilibrium chart happens here (via _compute_equilibrium)."""
    import yaml
    if config is None:
        cfg_path = results_path / 'config_used.yaml'
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"No config at {cfg_path}. Pass config dict explicitly.")
        with open(cfg_path) as f:
            config = yaml.safe_load(f)

    df_iter, df_tx, df_agents, has_transactions = _load_csvs(results_path, n_sims)
    eq = _compute_equilibrium(config, df_agents, show_plot=show_eq_plot)
    tx = _enrich_transactions(df_tx, eq)
    ann = _extract_announcements(df_iter)

    return ExperimentData(
        experiment_id=experiment_id or results_path.name,
        eq=eq, iter=df_iter, tx=tx, agents=df_agents, ann=ann,
        config=config, independent_rounds=independent_rounds,
        has_transactions=has_transactions,
    )


# ============================================================
# §4. METRICS
# ============================================================

def compute_round_metrics(tx: pd.DataFrame, eq: Equilibrium) -> pd.DataFrame:
    """Per (sim, round): efficiency, alpha, n_trades, mean_price, etc.
    Feeds the market summary table. Uses execution prices."""
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


def compute_quote_improvements(df_iter: pd.DataFrame) -> pd.DataFrame:
    """Per-event improvements in standing bid/ask.

    An "improvement" is when a new order tightens the existing standing quote
    on its side: bid goes up, ask goes down. Returns one row per improvement
    event with the size of the improvement.

    Excludes:
    - Fresh posts where the side was previously empty (no prior quote)
    - Trade events (order_outcome == 'traded'): these clear the book rather
      than improve a resting quote
    - No-announcement ticks
    """
    df = df_iter.copy()
    df['standing_bid'] = pd.to_numeric(df['standing_bid'], errors='coerce')
    df['standing_ask'] = pd.to_numeric(df['standing_ask'], errors='coerce')

    rows = []
    for (sim, rnd), g in df.groupby(['sim', 'round']):
        g = g.sort_values('iteration').reset_index(drop=True)
        prev_bid = np.nan
        prev_ask = np.nan
        for _, r in g.iterrows():
            cur_bid, cur_ask = r['standing_bid'], r['standing_ask']

            if r['announcement_made'] and r['order_outcome'] == 'posted':
                if r['announcement_type'] == 'buy' and pd.notna(prev_bid) \
                        and pd.notna(cur_bid) and cur_bid > prev_bid:
                    rows.append({
                        'sim': sim, 'round': rnd, 'iteration': r['iteration'],
                        'side': 'buyer', 'agent_id': r['announcing_agent_id'],
                        'prev_quote': prev_bid, 'new_quote': cur_bid,
                        'improvement': cur_bid - prev_bid,
                    })
                elif r['announcement_type'] == 'sell' and pd.notna(prev_ask) \
                        and pd.notna(cur_ask) and cur_ask < prev_ask:
                    rows.append({
                        'sim': sim, 'round': rnd, 'iteration': r['iteration'],
                        'side': 'seller', 'agent_id': r['announcing_agent_id'],
                        'prev_quote': prev_ask, 'new_quote': cur_ask,
                        'improvement': prev_ask - cur_ask,
                    })

            prev_bid, prev_ask = cur_bid, cur_ask

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out['improvement'] = out['improvement'].round(4)
    return out


def compute_all_metrics(data: ExperimentData) -> ExperimentMetrics:
    """Compute the metric DataFrames needed by the paper outputs."""
    return ExperimentMetrics(
        round_metrics=compute_round_metrics(data.tx, data.eq),
        quote_improvements=compute_quote_improvements(data.iter),
    )


# ============================================================
# §5. SUMMARY TABLE (paper table 2)
# ============================================================

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

    # format
    summary['n_trades'] = summary['n_trades'].round(1)
    summary['eq_price'] = summary['eq_price'].round(2)
    summary['mean_price'] = summary['mean_price'].round(2)
    summary['alpha'] = summary['alpha'].round(1)
    summary['efficiency'] = summary['efficiency'].round(2)
    summary['n_extramarginal'] = summary['n_extramarginal'].round(1)
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


def _price_y_bounds(eq: Equilibrium) -> tuple[float, float]:
    y_min = max(min(eq.supply[0], eq.demand[-1]) - 1, 0)
    y_max = max(eq.supply[-1], eq.demand[0]) + 1
    return y_min, y_max


# ============================================================
# §7. PLOTS
# ============================================================

def plot_order_flow(data: ExperimentData) -> plt.Figure:
    """Per-sim submitted bid/ask prices with trades highlighted (paper figure 3).
    One figure per simulation."""
    df_iter, eq = data.iter, data.eq
    sims = sorted(df_iter['sim'].unique())
    y_min, y_max = _price_y_bounds(eq)

    fig = None
    for sim in sims:
        fig, ax = plt.subplots(figsize=(12, 5))
        df_sim = df_iter[df_iter['sim'] == sim].copy()

        # Build continuous x-axis: cumulative offset per round + iteration within round.
        round_lengths = df_sim.groupby('round')['iteration'].max()
        rounds_sorted = sorted(round_lengths.index)
        offsets = {}
        cum = 0
        for r in rounds_sorted:
            offsets[r] = cum
            cum += round_lengths[r]
        df_sim['x'] = df_sim.apply(
            lambda row: offsets[row['round']] + row['iteration'], axis=1
        )

        ann_rows = df_sim[df_sim['announcement_made'] == True]

        for at, st in [('buy', SIDE_STYLES_ALT['buyer']),
                       ('sell', SIDE_STYLES_ALT['seller'])]:
            sub = ann_rows[ann_rows['announcement_type'] == at]
            ax.plot(sub['x'], sub['price'], marker=st['marker'], markersize=8,
                    linestyle='-', label=st['label'], color=st['color'])
        traded = ann_rows[ann_rows['transaction_made'] == True]
        for at, st in [('buy', SIDE_STYLES_ALT['buyer']),
                       ('sell', SIDE_STYLES_ALT['seller'])]:
            sub = traded[traded['announcement_type'] == at]
            ax.plot(sub['x'], sub['price'], marker=st['marker'], markersize=10,
                    linestyle='', markeredgecolor='black', markeredgewidth=1.5,
                    color=st['color'], label=f"{at.title()} Trade")

        # Round separators at cumulative round boundaries.
        cum = 0
        for r in rounds_sorted[:-1]:
            cum += round_lengths[r]
            ax.axvline(x=cum + 0.5, color='grey', linestyle='--', linewidth=1.5)

        ax.axhline(eq.price, color='red', linestyle='--')
        ax.set_xticks([])
        # ax.set_title(f'Order Flow, simulation {sim}')
        ax.set_ylabel('Submitted Price'); ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=8)
        plt.tight_layout(); plt.show()

    return fig


def plot_fill_rate(data: ExperimentData) -> plt.Figure | None:
    """Fill rate vs reservation price (paper figure 4)."""
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

    fig, ax = plt.subplots(figsize=(6.5, 5))
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
    # ax.set_title('Fill Rate vs Reservation Price', fontsize=14, family='serif', pad=10)
    ax.legend(frameon=False, prop={'family': 'serif', 'size': 10})
    _apply_paper_style(ax)
    plt.tight_layout(); plt.show()
    return fig


def plot_quote_improvements(metrics: ExperimentMetrics,
                            max_improvement: float | None = None) -> plt.Figure | None:
    """ECDF of per-tick improvements to the standing bid/ask, by side
    (paper figure 5)."""
    imp = metrics.quote_improvements
    if len(imp) == 0:
        print("No improvement events found.")
        return None

    if max_improvement is not None:
        imp = imp[imp['improvement'] <= max_improvement]

    upper = float(imp['improvement'].max())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for side, st in SIDE_STYLES.items():
        sub = np.sort(imp[imp['side'] == side]['improvement'].values)
        if len(sub) == 0:
            continue
        y = np.arange(1, len(sub) + 1) / len(sub)
        ax.step(sub, y, where='post', color=st['color'], lw=1.8,
                label=st['label'])
    for ref in [0.01, 0.05, 0.10, 0.25]:
        if ref > upper:
            continue
        ax.axvline(ref, color='grey', ls=':', lw=0.7, alpha=0.5)
    ax.set_xlabel('Improvement size ($)')
    ax.set_ylabel('Cumulative fraction')
    ax.set_xlim(0, upper)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(linestyle=':', alpha=0.4)

    plt.tight_layout(); plt.show()
    return fig


def plot_first_attempted_rent(data: ExperimentData,
                              figsize: tuple = (6, 4.5)) -> plt.Figure:
    """First attempted rent in round 1: paired within-sim, first vs second mover
    (paper figure 6).

    For each simulation, identifies the first buy and first sell announcement
    in round 1 and computes attempted rent against the announcer's own
    reservation price (buyer: value - price; seller: price - cost). The
    earlier iteration is tagged 'first mover', the later 'second mover'.
    Two dots per sim connected by a line coloured by which role moved first.
    """
    df = data.iter
    r1 = df[(df['round'] == 1) & (df['announcement_made'] == True)].copy()

    # First buy and first sell per sim in round 1.
    first_by_side = (r1.sort_values(['sim', 'iteration'])
                       .groupby(['sim', 'announcement_type'], as_index=False)
                       .first())

    # Attempted rent vs own reservation.
    def _rent(row):
        res = row['announcing_agent_reservation_price']
        p = row['price']
        return res - p if row['announcement_type'] == 'buy' else p - res

    first_by_side['attempted_rent'] = first_by_side.apply(_rent, axis=1)
    first_by_side['role'] = first_by_side['announcement_type'].map(
        {'buy': 'buyer', 'sell': 'seller'})

    # Tag first vs second mover within each sim.
    first_by_side['mover_rank'] = (first_by_side
                                   .groupby('sim')['iteration']
                                   .rank(method='first').astype(int))
    first_by_side['mover'] = first_by_side['mover_rank'].map(
        {1: 'first', 2: 'second'})

    # Drop sims missing one side (no quote from buyer or seller in round 1).
    complete = (first_by_side.groupby('sim')['mover_rank']
                              .max() == 2)
    keep_sims = complete[complete].index
    first_by_side = first_by_side[first_by_side['sim'].isin(keep_sims)]

    if len(first_by_side) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No qualifying round-1 announcements",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#666')
        ax.set_axis_off()
        plt.show()
        return fig

    from matplotlib.lines import Line2D

    pivot = (first_by_side.pivot_table(index='sim', columns='mover',
                                        values='attempted_rent')
                          .dropna(subset=['first', 'second']))
    first_role = (first_by_side[first_by_side['mover'] == 'first']
                  .set_index('sim')['role'])

    fig, ax = plt.subplots(figsize=figsize)
    for sim_id, row in pivot.iterrows():
        col = SIDE_STYLES[first_role.loc[sim_id]]['color']
        f_val, s_val = row['first'], row['second']
        ax.plot([1, 2], [f_val, s_val],
                color=col, alpha=0.45, linewidth=1.0, zorder=2)
        ax.scatter([1], [f_val], s=30, color=col, alpha=0.85, zorder=3)
        ax.scatter([2], [s_val], s=30, facecolor='white',
                   edgecolor=col, linewidth=1.2, alpha=0.85, zorder=3)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['First mover', 'Second mover'])
    ax.set_xlim(0.7, 2.3)
    ax.set_ylabel('First attempted rent ($)', fontsize=11)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.2f'))
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    role_handles = [
        Line2D([0], [0], color=SIDE_STYLES['buyer']['color'],
               linewidth=2, label='Buyer moved first'),
        Line2D([0], [0], color=SIDE_STYLES['seller']['color'],
               linewidth=2, label='Seller moved first'),
    ]
    ax.legend(handles=role_handles, fontsize=9, loc='best')

    fig.tight_layout()
    plt.show()
    return fig


# ============================================================
# §8. PIPELINE
# ============================================================

def render_all_plots(data: ExperimentData, metrics: ExperimentMetrics) -> dict:
    """Render the paper figures (order flow, fill rate, quote improvements,
    first attempted rent). The equilibrium chart is produced earlier, during
    data loading. Returns dict of name -> Figure."""
    figs = {}
    figs['order_flow'] = plot_order_flow(data)
    figs['fill_rate'] = plot_fill_rate(data)
    figs['quote_improvements'] = plot_quote_improvements(metrics)
    figs['first_attempted_rent'] = plot_first_attempted_rent(data)
    return figs


def report_all(data: ExperimentData, metrics: ExperimentMetrics) -> None:
    """Print the market summary table (skipped if no simulation produced trades)."""
    if data.has_transactions:
        report_market_summary_table(metrics.summary_table)
    else:
        print(f"\n{'=' * 60}")
        print("NO TRANSACTIONS IN ANY SIMULATION - skipping summary table")
        print(f"{'=' * 60}")


def run_full_analysis(results_path: Path, n_sims: int,
                      config: dict | None = None,
                      experiment_id: str | None = None,
                      independent_rounds: bool = False
                      ) -> tuple[ExperimentData, ExperimentMetrics, dict]:
    """End-to-end: load (renders equilibrium chart), compute, print summary
    table, plot the remaining paper figures. Returns (data, metrics, figs)."""
    data = load_experiment_data(
        results_path, n_sims, config=config,
        experiment_id=experiment_id, independent_rounds=independent_rounds)
    metrics = compute_all_metrics(data)
    if data.has_transactions:
        metrics.summary_table = build_market_summary_table(
            metrics.round_metrics, data.eq, experiment_id=experiment_id)
    report_all(data, metrics)
    figs = render_all_plots(data, metrics)
    return data, metrics, figs


# ============================================================
# Standalone usage
# ============================================================
if __name__ == '__main__':
    import sys, yaml
    if len(sys.argv) < 2:
        print("Usage: python results_analysis_utils.py <results_path> [n_sims]")
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