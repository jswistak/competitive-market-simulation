# Market Simulation

LangGraph-based market simulation for LLM agents. This project studies how LLM-based agents behave when placed in competitive market environments including continuous double auctions (Smith-style experiments), sealed-bid auctions, English, Dutch, and other mechanisms.

## Features

- **7 Auction Mechanisms**: Continuous double auction (improvement-rule CDA), FPSB, SPSB (Vickrey), English, Dutch, All-Pay, First-Price Open Outcry
- **Multi-Provider LLM Support**: OpenAI, Anthropic (Claude), Google Gemini, DeepSeek
- **Zero-Intelligence Baselines**: Gode & Sunder (1993) ZI-C / ZI-U traders — run without any API key
- **LangGraph Workflow**: Visualizable, editable execution flow with nodes and conditional edges
- **Langfuse Tracing**: Full observability of LLM calls with cost tracking (optional; local stack included)
- **Tool-Augmented Agents**: Optional tools for trade evaluation, market statistics, trader classification, and sandboxed code execution (E2B). Tools are disabled in all paper experiments.
- **Configurable Experiments**: YAML-based configuration for different scenarios and providers
- **CLI Interface**: Easy-to-use commands for running experiments

## Installation

Requires Python 3.11+ and [UV](https://docs.astral.sh/uv/) package manager.

```bash
# Install dependencies (includes analysis-notebook dependencies)
uv sync

# Install package in editable mode
uv pip install -e .
```

## Quick Start

```bash
# Smoke test — Zero-Intelligence traders, no API key required
uv run market-simulation run zi_c_double_auction --no-trace

# Validate a configuration
uv run market-simulation validate configs_final/smith1_openai_small

# Run a paper configuration (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
uv run market-simulation run configs_final/smith1_openai_small --no-trace

# Same run with Langfuse tracing enabled (see Environment Variables)
uv run market-simulation run configs_final/smith1_openai_small
```

## Reproducing the Paper Experiments

The experiments reported in the paper were run with the configs in `configs/configs_final/`. The three reported conditions map to:

| Paper condition | Config | Model |
| --------------- | ------ | ----- |
| GPT Large | `configs_final/smith1_openai_large.yaml` | `gpt-5.4-2026-03-05` |
| GPT Small | `configs_final/smith1_openai_small.yaml` | `gpt-5.4-mini-2026-03-17` |
| Gemini Large | `configs_final/smith1_gemini_large.yaml` | `gemini-3.1-pro-preview` |

All three encode the same market design: 11 buyers and 11 sellers with reservation prices from $0.75 to $3.25 in $0.25 steps (competitive equilibrium p\* = $2.00, q\* = 6), 5 trading rounds, at most 300 ticks per round, and no tools.

Each config file specifies a single simulation (`n_simulations: 1`). The paper runs every condition 10 times; use the CLI override:

```bash
uv run market-simulation run configs_final/smith1_openai_small --sims 10
```

**Determinism.** `random_seed` is `null` in these configs: mechanism-level randomness (which agent acts on each tick) is unseeded, and LLM sampling at temperature 1.0 is inherently non-deterministic (this integration does not pass a sampling seed to the LLM APIs). Results therefore reproduce at the statistical level across repeated runs, not tick-for-tick. Setting `experiment.random_seed` seeds the ZI-trader RNG only.

**LLM parameters.** Provider defaults, except: `temperature: 1.0`, `max_tokens: 1024`, `max_retries: 3` (set in every config). The Gemini provider additionally requests reasoning with `thinking_level="low"` and `include_thoughts=True` (`src/market_simulation/llm/providers/gemini.py`).

**Analysis.** The paper's analysis lives in three files:

- `notebooks/results_analysis_single_aaai27.ipynb` — per-condition summary table and convergence figures. Set `experiment_id` in the second code cell (marked "Manually adjust the experiment_id") to the condition you want to analyze.
- `notebooks/results_analysis_lexical_aaai27.ipynb` — lexical analysis of the agents' reasoning traces; reads `logs/llm_calls_*.jsonl` from the experiment output.
- `notebooks/results_analysis_utils_aaai27.py` — shared helpers, including the Smith (1962) benchmark alpha values.

The notebooks read experiment outputs from `results/<experiment_id>/`, e.g. `results/final/results_configs_final/smith1_openai_small_combined/` containing `config_used.yaml`, `data/*.csv`, and `logs/*.jsonl` (see [Output](#output) for the file layout). The supplementary data package places the collected experiment data at these paths.

**Ablation.** `configs/configs_non_reason/` contains `smith1` variants with the structured reasoning field disabled (`include_reasoning: false`) and one added prompt sentence stating that trading exactly at the reservation price is acceptable (the "zero-profit" variant).

`configs_final/` also contains additional conditions (Smith designs `smith2` and `smith4a`, Anthropic models, `gemini-3-flash-preview`) that are not reported in the paper.

## CLI Commands

### Run Simulation

```bash
uv run market-simulation run <config> [OPTIONS]

Arguments:
  config    Config name relative to configs/ (without .yaml) or a full path

Options:
  -s, --sims INTEGER     Override number of simulations
  -o, --output PATH      Output directory [default: ./results]
  --trace/--no-trace     Override Langfuse tracing [default: tracing.enabled from the config]
  -v, --verbose          Enable verbose logging
```

**Examples:**

```bash
# Zero-Intelligence baselines (no API key required)
uv run market-simulation run zi_c_double_auction --no-trace
uv run market-simulation run zi_u_double_auction --no-trace

# Paper configuration, 10 independent simulations
uv run market-simulation run configs_final/smith1_openai_large --sims 10

# Gemini condition (requires GOOGLE_API_KEY)
uv run market-simulation run configs_final/smith1_gemini_large --sims 10

# Sealed-bid auction demo with verbose output
uv run market-simulation run test_fpsb_gemini --no-trace -v
```

### Validate Configuration

```bash
uv run market-simulation validate <config>
```

### Visualize Graph

```bash
uv run market-simulation visualize [--output graph.png]
```

Prints the LangGraph structure as a Mermaid diagram and additionally saves a PNG rendered via the mermaid.ink web API when network access is available (otherwise only the Mermaid text is printed). Requires no API keys.

## Auction Types

The project supports 7 market mechanisms. The `double_auction` is the default; the remaining 6 are configured via `experiment.auction_type` and the `experiment.auction` block.

| Type | Config value | Description | Payment rule |
|------|-------------|-------------|--------------|
| Double Auction | `double_auction` | Continuous bilateral trading (buyers and sellers) | Standing order's price |
| First-Price Sealed-Bid | `fpsb` | Simultaneous sealed bids; highest wins | Winner pays own bid |
| Second-Price Sealed-Bid (Vickrey) | `spsb` | Simultaneous sealed bids; highest wins | Winner pays second-highest bid |
| All-Pay | `all_pay` | Simultaneous sealed bids; highest wins, all bidders pay | Everyone pays their bid |
| English (ascending) | `english` | Open ascending bids; last bidder standing wins | Winner pays standing bid |
| Dutch (descending) | `dutch` | Price descends from a start price; first to accept wins | Winner pays accepted price |
| First-Price Open Outcry | `first_price_open_outcry` | Open ascending bids (like English) with first-price payment | Winner pays own bid |

### Auction-specific config keys

```yaml
experiment:
  auction_type: fpsb        # One of the values above
  auction:
    n_rounds: 10             # Rounds per simulation
    n_simulations: 10        # Independent simulation runs
    random_seed: 42          # Optional seed for mechanism-level randomness
    bidders:
      num: 5
      value_min: 0.0
      value_max: 10.0
      distribution: linspace # "linspace" or "uniform"

    # English / Open-Outcry only
    min_increment: 0.5
    max_bidding_rounds: 50

    # Dutch only
    dutch_start_price: 12.0
    dutch_decrement: 0.5
    dutch_min_price: 0.0
```

### Auction output files

Each auction simulation produces:

- `auction_results_<sim>.csv` -- per-round winner, payment, surplus, and all bids (as JSON)
- `all_bids_<sim>.csv` -- every individual bid submitted across all rounds
- `bidder_histories_<sim>.csv` -- per-bidder history data (own bids, outcomes, payments)

Pre-built auction configs are located in `configs/` (`auction_fpsb.yaml`, `auction_spsb.yaml`, `auction_allpay.yaml`, `auction_english.yaml`, `auction_dutch.yaml`, `auction_open_outcry.yaml`).

## Configuration

Configuration files live in `configs/` and are validated strictly (unknown keys are rejected), so `market-simulation validate <config>` tells you definitively whether a config is loadable.

**Paper experiment configs** — `configs/configs_final/` (18 files, all validate): `smith{1,2,4a}_{openai,anthropic,gemini}_{large,small}.yaml`. See [Reproducing the Paper Experiments](#reproducing-the-paper-experiments) for the three conditions used in the paper.

**Ablation configs** — `configs/configs_non_reason/` (5 files, all validate): `smith1_*_zeroprofit.yaml` variants with `include_reasoning: false` and zero-profit-acceptable prompt wording.

**Other working configs** (top level of `configs/`, all validate):

| Config | Description |
| ------ | ----------- |
| `zi_c_double_auction.yaml` | Gode–Sunder ZI-C (constrained) traders in the CDA — no API key needed |
| `zi_u_double_auction.yaml` | ZI-U (unconstrained) traders in the CDA — no API key needed |
| `auction_fpsb.yaml`, `auction_spsb.yaml`, `auction_allpay.yaml`, `auction_english.yaml`, `auction_dutch.yaml`, `auction_open_outcry.yaml` | One config per non-CDA auction mechanism |
| `test_fpsb_gemini.yaml`, `test_spsb_gemini.yaml`, `test_allpay_gemini.yaml`, `test_english_gemini.yaml`, `test_dutch_gemini.yaml`, `test_open_outcry_gemini.yaml` | Small Gemini-based auction test runs |

**Legacy configs.** The remaining top-level files (`openai.yaml`, `anthropic.yaml`, `gemini.yaml`, `deepseek.yaml`, the `*_tools` variants, `smith1.yaml`–`smith7.yaml` and their `_t1`/`_cot`/`_personas` variants, `test.yaml`, `test_tools.yaml`, `example_personas.yaml`, `mixed_llm_zi_double_auction.yaml`) predate the current prompt schema (the `system_template`/`user_template` split) and currently **fail validation**. They are kept for reference only — use `configs_final/` as the starting point for new configs.

### Configuration Structure

```yaml
experiment:
  auction_type: double_auction # Mechanism; see Auction Types
  include_reasoning: true # Structured responses include a reasoning field
  n_rounds: 5 # Trading rounds
  n_simulations: 1 # Independent simulation runs (override with --sims)
  max_ticks_per_round: 300 # REQUIRED for double_auction; one tick = one
  #                          randomly chosen agent posts an order
  random_seed: null # Seeds the ZI RNG; LLM sampling is not seedable
  buyers:
    min: 0.75 # Min reservation price
    max: 3.25 # Max reservation price
    num: 11 # Number of buyers
    strategies: llm # "llm", "zi_c", "zi_u", or a per-agent list
  sellers:
    min: 0.75
    max: 3.25
    num: 11
    strategies: llm
  history:
    mode: full # "full" or "summary" (see History Modes below)
    own_history_mode: full # "full" or "summary"
    summary_last_n_events: 3 # Recent raw events appended in summary mode

llm:
  provider: openai # openai | anthropic | gemini | deepseek
  model: gpt-5.4-mini-2026-03-17
  temperature: 1.0
  max_tokens: 1024
  max_retries: 3

tools:
  enabled: false # Enable tool-augmented agents
  enable_simple_tools: false # evaluate_trade, compute_market_stats, classify_trader
  enable_code_interpreter: false # E2B sandboxed Python execution (needs E2B_API_KEY)
  e2b_timeout: 300
  max_tool_iterations: 5

tracing:
  enabled: true # Langfuse tracing (keys from environment)
  llm_call_logging: true # Write logs/llm_calls_<sim>.jsonl per simulation

prompts:
  general:
    # The announcement prompt is split across two messages:
    # system_template holds per-agent constants (cacheable),
    # user_template holds per-tick state.
    system_template: |
      You are a {role} participating in a market...
      {persona}
    user_template: |
      Market history:
      {market_history}
      History of your actions:
      {own_history}
      This is round {round}/{N_ROUNDS}.
      Current standing bid: {standing_bid}
      Current standing ask: {standing_ask}
      Your reservation price is ${reservation_price:.2f}.
      {action_prompt}
    # Own-history and market-history entry templates
    # (announcement_history_template, market_history_accepted_template, ...)
    # have sensible defaults; see src/market_simulation/config/schema.py.
  buyer:
    main_keywords:
      role: buyer
      verb: buy
      preference: lowest
      condition: above
      profit_formula: your reservation price and your transaction price
      order_outcomes: "..." # Side-specific explanation of order handling
    announcement_prompt: |
      Do you want to announce a bid to buy? ...
  seller:
    # ... mirror of buyer with seller keywords

zi: # Zero-intelligence sampling hyperparameters (used when strategies != llm)
  u_low: 0.0 # ZI-U lower sampling bound
  u_high: 10.0 # ZI-U upper sampling bound
  announce_prob: 0.5 # ZI-U probability of acting on a tick
  accept_prob: 0.5
  bid_prob: 0.5
```

See `configs/configs_final/smith1_openai_small.yaml` for a complete working example.

### History Modes

The `experiment.history` block controls how market history and agent history are presented in LLM prompts. This is useful for reducing token usage and prompt length as the simulation progresses.

| Field | Values | Description |
|-------|--------|-------------|
| `mode` | `"full"` (default) / `"summary"` | Controls market-wide history. `"full"` injects the entire raw event log. `"summary"` replaces it with aggregate statistics (transaction count, average price, price trend, bid-ask spread, acceptance rate) plus the last N raw events. |
| `own_history_mode` | `"full"` (default) / `"summary"` | Controls each agent's personal action history. `"full"` shows every past action verbatim. `"summary"` shows counts, success rate, average trade price, and the last action taken. |
| `summary_last_n_events` | integer (default `3`) | In summary mode, this many recent raw event lines are appended after the statistics so the LLM still sees the most recent context. Set to `0` to show only statistics. |

**Example -- summary mode:**

```yaml
experiment:
  n_rounds: 5
  n_simulations: 10
  max_ticks_per_round: 300
  buyers: { min: 0.75, max: 3.25, num: 11 }
  sellers: { min: 0.75, max: 3.25, num: 11 }
  history:
    mode: summary
    own_history_mode: summary
    summary_last_n_events: 3
```

### Agent Personas

The `personas` configuration block allows you to assign behavioral descriptions to individual agents or groups of agents. Persona text is injected at the location of the `{persona}` placeholder, which must be present in `prompts.general.system_template` or `prompts.general.user_template`. If personas are configured but the placeholder is missing, the `run` command logs a warning and the persona text is silently omitted from prompts.

**Fields:**

| Field            | Type              | Description                                                    |
| ---------------- | ----------------- | -------------------------------------------------------------- |
| `buyer_default`  | `string`          | Default persona applied to all buyers (unless overridden)      |
| `seller_default` | `string`          | Default persona applied to all sellers (unless overridden)     |
| `buyers`         | `dict[int, str]`  | Per-buyer overrides, keyed by buyer index (0-based)            |
| `sellers`        | `dict[int, str]`  | Per-seller overrides, keyed by seller index (0-based)          |
| `bidder_default` | `string`          | Default persona for auction bidders (non-CDA mechanisms)       |
| `bidders`        | `dict[int, str]`  | Per-bidder overrides, keyed by bidder index (0-based)          |

**Example:**

```yaml
personas:
  buyer_default: "You are a cautious buyer who carefully evaluates prices before acting."
  seller_default: "You are an assertive seller who aims to maximize profit."
  buyers:
    0: "You are an aggressive buyer who bids boldly and closes deals quickly."
  sellers:
    0: "You are a risk-averse seller who prefers a guaranteed sale."

prompts:
  general:
    system_template: |
      You are a {role} participating in a market...

      {persona}

      There are {N_BUYERS} buyers and {N_SELLERS} sellers...
```

In this example, buyer 0 receives the individual override ("aggressive buyer"), while all other buyers receive the `buyer_default` persona. If no persona is assigned to an agent, the placeholder is replaced with an empty string.

### Reasoning Capture

`experiment.include_reasoning` (default `true`) controls whether agents return a step-by-step `reasoning` field alongside their decision. Responses use provider-native structured output (`src/market_simulation/llm/response_schemas.py`); with reasoning enabled, the `*WithReasoning` schema variants are used and each call's reasoning is recorded in `logs/llm_calls_<sim>.jsonl` (the input to the lexical-analysis notebook).

```yaml
experiment:
  include_reasoning: true

llm:
  max_tokens: 1024 # Leave room for the reasoning text
```

The answer fields are ordered before `reasoning` in the schemas so a truncated response still contains the decision. The paper's main experiments use `include_reasoning: true`; the `configs_non_reason/` ablation disables it.

## Environment Variables

Create a `.env` file (see `.env.example`) or export these variables:

```bash
# LLM Providers (set the ones you need)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=sk-...

# E2B sandbox (optional — only for configs with tools.enable_code_interpreter: true;
# not used in any paper experiment. If unset, the code interpreter is disabled
# with a warning and the run continues.)
E2B_API_KEY=e2b_...

# Langfuse Tracing (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Local Langfuse (optional, no account needed)

A self-contained Langfuse stack is included for local tracing:

```bash
docker compose up -d   # UI: http://localhost:3100 (admin@langfuse.local / password)
```

Then point the environment at it (these values match the auto-initialized project):

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-local-public-key
LANGFUSE_SECRET_KEY=sk-lf-local-secret-key
LANGFUSE_HOST=http://localhost:3100
```

Alternatively, disable tracing entirely with `--no-trace` or `tracing.enabled: false`.

## Project Structure

```
.
├── src/market_simulation/
│   ├── main.py                 # CLI entry point (Typer)
│   ├── config/
│   │   ├── schema.py           # Pydantic config schemas (strict)
│   │   └── settings.py         # Config loader
│   ├── llm/
│   │   ├── factory.py          # LLM provider factory
│   │   ├── response_schemas.py # Structured-output response models
│   │   ├── tool_augmented.py   # Tool-calling agent loop wrapper
│   │   └── providers/          # OpenAI, Anthropic, Gemini, DeepSeek
│   ├── agents/
│   │   ├── factory.py          # Agent creation
│   │   └── zi.py               # Zero-intelligence (ZI-C / ZI-U) strategies
│   ├── graph/
│   │   ├── state.py            # MarketState TypedDict
│   │   ├── workflow.py         # CDA graph builder
│   │   ├── edges.py            # Conditional routing
│   │   ├── history.py          # History summary builder (full/summary modes)
│   │   ├── nodes/
│   │   │   ├── announce.py     # Order generation (LLM or ZI)
│   │   │   ├── apply_order.py  # Order-book matching (improvement-rule CDA)
│   │   │   └── control.py      # History updates, round/tick flow control
│   │   └── auctions/           # Non-CDA auction workflows
│   │       ├── base.py         # Shared helpers
│   │       ├── sealed_bid/     # FPSB, SPSB, All-Pay
│   │       ├── english/        # English ascending auction
│   │       ├── dutch/          # Dutch descending auction
│   │       └── open_outcry/    # First-Price Open Outcry
│   ├── tools/
│   │   ├── definitions.py      # Tool definitions (evaluate_trade, etc.)
│   │   ├── registry.py         # Tool registry
│   │   └── sandbox.py          # E2B sandbox manager
│   ├── tracing/
│   │   ├── langfuse.py         # Langfuse integration
│   │   └── llm_logger.py       # Per-call JSONL logging
│   └── output/
│       └── results.py          # CSV export
├── configs/                    # YAML configurations
│   ├── configs_final/          # Paper experiment configs
│   └── configs_non_reason/     # Ablation configs
├── notebooks/                  # Analysis notebooks (*_aaai27 = paper analysis)
├── tests/                      # Unit tests (no API keys or network needed)
├── analyze_results.py          # Quick results summary script
├── docker-compose.yml          # Local Langfuse stack
├── pyproject.toml
├── .env.example
└── README.md
```

## LangGraph Workflow

The continuous double auction runs a tick-based loop — one graph invocation simulates all rounds of one simulation:

```
START
  │
  ▼
┌──────────────────┐
│ select_announcer │ ◄─────────────────────────────┐
└────────┬─────────┘                               │
         ▼                                         │
┌──────────────────┐                               │
│     announce     │  ← LLM call (or ZI draw):     │
└────────┬─────────┘    bid/ask price or pass      │
         ▼                                         │
┌──────────────────┐                               │
│   apply_order    │  ← Order book: cross /        │
└────────┬─────────┘    improve / reject           │
         ▼                                         │
┌──────────────────┐                               │
│  update_history  │                               │
└────────┬─────────┘                               │
         ▼                                         │
┌──────────────────┐   next tick (same round)      │
│   check_round    ├───────────────────────────────┤
└────────┬─────────┘                               │
         │ round over                              │
         ▼                                         │
┌──────────────────┐   more rounds remain          │
│    next_round    ├───────────────────────────────┘
└────────┬─────────┘
         │ all rounds done
         ▼
        END
```

Each tick, one randomly chosen active agent posts an order; the `apply_order` node applies the improvement rule and executes crossing trades at the standing order's price. There is no separate response-collection loop. The tick loop passes through a small `next_iteration` bookkeeping node (elided above); run `uv run market-simulation visualize` to print the exact graph. Non-CDA mechanisms have their own subgraphs under `graph/auctions/`.

## Output

Results are saved to `./results/<config>_<timestamp>/`:

**Double auction output:**
```
results/configs_final/smith1_openai_small_20260428_160051/
├── config_used.yaml            # Configuration snapshot
├── logs/
│   ├── sim_1.log               # Per-simulation log
│   └── llm_calls_1.jsonl       # One record per LLM call (prompt, response,
│                               #   reasoning); written when
│                               #   tracing.llm_call_logging: true (default)
└── data/
    ├── iteration_history_1.csv
    ├── transactions_1.csv
    ├── agent_histories_1.csv
    └── tool_usage_1.csv        # Only when tools are enabled
```

**Auction output:**
```
results/test_fpsb_gemini_20260226_003623/
├── config_used.yaml
├── logs/
│   └── sim_1.log
└── data/
    ├── auction_results_1.csv   # Per-round winner, payment, surplus, all bids
    ├── all_bids_1.csv          # Every individual bid
    └── bidder_histories_1.csv  # Per-bidder history
```

## Adding a New LLM Provider

1. Create provider class in `src/market_simulation/llm/providers/`:

```python
# src/market_simulation/llm/providers/newprovider.py
from langchain_core.language_models import BaseChatModel
from .base import LLMProvider

class NewProvider(LLMProvider):
    def _create_model(self) -> BaseChatModel:
        return SomeLangChainModel(
            model=self.config.model,
            temperature=self.config.temperature,
        )
```

2. Register in the factory (`src/market_simulation/llm/factory.py`, `create_llm`):

```python
from .providers.newprovider import NewProvider

providers = {
    # ... existing
    "newprovider": NewProvider,
}
```

3. Add to config schema (`src/market_simulation/config/schema.py`):

```python
provider: Literal["openai", "anthropic", "gemini", "deepseek", "newprovider"]
```

## Development

```bash
# Run tests (no API keys or network required)
uv run pytest
```
