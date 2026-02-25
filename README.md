# Market Simulation

LangGraph-based market equilibrium simulation for LLM agents. This project studies how LLM-based agents behave when placed in competitive double auction markets.

## Features

- **Multi-Provider LLM Support**: OpenAI, Anthropic (Claude), Google Gemini, DeepSeek
- **LangGraph Workflow**: Visualizable, editable execution flow with nodes and conditional edges
- **Langfuse Tracing**: Full observability of LLM calls with cost tracking
- **Tool-Augmented Agents**: Optional tools for trade evaluation, market statistics, trader classification, and sandboxed code execution (E2B)
- **Configurable Experiments**: YAML-based configuration for different scenarios and providers
- **CLI Interface**: Easy-to-use commands for running experiments

## Installation

Requires Python 3.11+ and [UV](https://docs.astral.sh/uv/) package manager.

```bash
# Install dependencies
uv sync

# Install package in editable mode
uv pip install -e .
```

## Quick Start

```bash
# Validate configuration
uv run market-simulation validate test

# Run a test simulation (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
uv run market-simulation run test --no-trace

# Run full experiment with Langfuse tracing
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
uv run market-simulation run openai
```

## CLI Commands

### Run Simulation

```bash
uv run market-simulation run <config> [OPTIONS]

Arguments:
  config    Config file name (without .yaml) or full path

Options:
  -s, --sims INTEGER     Override number of simulations
  -o, --output PATH      Output directory [default: ./results]
  --trace/--no-trace     Enable/disable Langfuse tracing [default: trace]
  -v, --verbose          Enable verbose logging
```

**Examples:**

```bash
# Run with OpenAI
uv run market-simulation run openai

# Run 5 simulations with Anthropic, no tracing
uv run market-simulation run anthropic --sims 5 --no-trace

# Run with verbose output
uv run market-simulation run gemini -v

# Run with tool-augmented agents (simple tools)
uv run market-simulation run openai_simple_tools

# Run with simple tools + E2B code interpreter
uv run market-simulation run openai_full_tools
```

### Validate Configuration

```bash
uv run market-simulation validate <config>
```

### Visualize Graph

```bash
uv run market-simulation visualize [--output graph.png]
```

## Auction Types

The project supports 7 market mechanisms. The `double_auction` is the default; the remaining 6 are configured via `experiment.auction_type` and the `experiment.auction` block.

| Type | Config value | Description | Payment rule |
|------|-------------|-------------|--------------|
| Double Auction | `double_auction` | Continuous bilateral trading (buyers and sellers) | Agreed price |
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
    random_seed: 42          # Optional seed for reproducibility
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

- `auction_results_<sim>.csv` -- per-round winner, payment, surplus, and all bids
- `bid_records_<sim>.csv` -- every individual bid submitted across all rounds
- `bidder_histories_<sim>.csv` -- per-bidder history data (own bids, outcomes, payments)
- `tool_usage_<sim>.csv` -- tool invocation log (when tools are enabled)

Pre-built auction configs are located in `configs/` (`auction_fpsb.yaml`, `auction_spsb.yaml`, `auction_allpay.yaml`, `auction_english.yaml`, `auction_dutch.yaml`, `auction_open_outcry.yaml`).

## Configuration

Configuration files are located in `configs/`. Available configs:

| Config                        | Provider  | Model                   | Tools        |
| ----------------------------- | --------- | ----------------------- | ------------ |
| `openai.yaml`                 | OpenAI    | gpt-4o-mini             | No           |
| `anthropic.yaml`              | Anthropic | claude-3-5-haiku-latest | No           |
| `gemini.yaml`                 | Google    | gemini-2.0-flash        | No           |
| `deepseek.yaml`               | DeepSeek  | deepseek-chat           | No           |
| `test.yaml`                   | OpenAI    | gpt-4o-mini (minimal)   | No           |
| `openai_simple_tools.yaml`    | OpenAI    | gpt-4o-mini             | Simple       |
| `openai_full_tools.yaml`      | OpenAI    | gpt-4o-mini             | Simple + E2B |
| `anthropic_simple_tools.yaml` | Anthropic | claude-3-5-haiku-latest | Simple       |
| `anthropic_full_tools.yaml`   | Anthropic | claude-3-5-haiku-latest | Simple + E2B |
| `gemini_simple_tools.yaml`    | Google    | gemini-2.0-flash        | Simple       |
| `gemini_full_tools.yaml`      | Google    | gemini-2.0-flash        | Simple + E2B |
| `deepseek_simple_tools.yaml`  | DeepSeek  | deepseek-chat           | Simple       |
| `deepseek_full_tools.yaml`    | DeepSeek  | deepseek-chat           | Simple + E2B |
| `test_tools.yaml`             | OpenAI    | gpt-4o-mini (minimal)   | Simple       |

### Configuration Structure

```yaml
experiment:
  n_rounds: 5 # Trading rounds
  n_iterations: 10 # Max iterations per round
  n_simulations: 10 # Independent simulation runs
  buyers:
    min: 0.8 # Min reservation price
    max: 3.2 # Max reservation price
    num: 11 # Number of buyers
  sellers:
    min: 0.8
    max: 3.2
    num: 11

llm:
  provider: openai # openai | anthropic | gemini | deepseek
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 10
  max_retries: 5

tools:
  enabled: false # Enable tool-augmented agents
  enable_simple_tools: true # evaluate_trade, compute_market_stats, classify_trader
  enable_code_interpreter: false # E2B sandboxed Python execution
  e2b_timeout: 300
  max_tool_iterations: 5

tracing:
  enabled: true
  # Keys from environment: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

prompts:
  general:
    main_template: |
      You are a {role} participating in a market...
  buyer:
    main_keywords:
      role: buyer
      verb: buy
      preference: lowest
      condition: above
    response_prompt: |
      Someone is offering to sell at ${price:.2f}. Do you buy?
    announcement_prompt: |
      Do you want to announce a bid to buy?
  seller:
    # ... similar structure
```

## Environment Variables

Create a `.env` file or export these variables:

```bash
# LLM Providers (set the ones you need)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=sk-...

# Langfuse Tracing (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Project Structure

```
master-thesis/
├── src/market_simulation/
│   ├── __init__.py
│   ├── main.py              # CLI entry point (Typer)
│   ├── config/
│   │   ├── schema.py        # Pydantic config schemas
│   │   └── settings.py      # Config loader
│   ├── llm/
│   │   ├── factory.py       # LLM provider factory
│   │   ├── tool_augmented.py # Tool-calling agent loop wrapper
│   │   └── providers/       # OpenAI, Anthropic, Gemini, DeepSeek
│   ├── agents/
│   │   └── factory.py       # Agent creation
│   ├── graph/
│   │   ├── state.py         # MarketState TypedDict
│   │   ├── nodes/           # LangGraph nodes
│   │   │   ├── announce.py  # Price announcement
│   │   │   ├── respond.py   # Response handling
│   │   │   ├── transaction.py
│   │   │   └── control.py   # Flow control
│   │   ├── edges.py         # Conditional routing
│   │   └── workflow.py      # Graph builder
│   ├── tools/
│   │   ├── definitions.py   # Tool definitions (evaluate_trade, etc.)
│   │   ├── registry.py      # Tool registry
│   │   └── sandbox.py       # E2B sandbox manager
│   ├── tracing/
│   │   └── langfuse.py      # Langfuse integration
│   └── output/
│       └── results.py       # CSV export
├── configs/                  # YAML configurations
├── tests/                    # Unit tests
├── pyproject.toml
├── .env.example
└── README.md
```

## LangGraph Workflow

The simulation uses a LangGraph StateGraph with the following flow:

```
START
  │
  ▼
┌─────────────────┐
│ select_announcer│  ← Pick random active agent
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    announce     │  ← LLM call: generate price
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
 price     no price
    │         │
    ▼         │
┌─────────────────┐    │
│select_responders│    │
└────────┬────────┘    │
         │             │
         ▼             │
┌─────────────────┐    │
│    respond      │  ← LLM call: yes/no
└────────┬────────┘    │
         │             │
    ┌────┴────┐        │
    │         │        │
accepted  rejected     │
    │         │        │
    ▼         │        │
┌────────┐    │        │
│record  │    │        │
│transact│    │        │
└───┬────┘    │        │
    │         │        │
    └────┬────┘        │
         │             │
         ▼             │
┌─────────────────┐◄───┘
│ update_history  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  check_round    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
continue     end
    │         │
    ▼         ▼
  LOOP       END
```

## Output

Results are saved to `./results/<config>_<timestamp>/`:

```
results/openai_20250121_143052/
├── config_used.yaml           # Configuration snapshot
└── data/
    ├── iteration_history_1.csv
    ├── iteration_history_2.csv
    ├── transactions_1.csv
    ├── transactions_2.csv
    ├── agent_histories_1.csv
    └── agent_histories_2.csv
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

2. Register in factory (`src/market_simulation/llm/factory.py`):

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
# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Format code
uv run ruff format src/
```
