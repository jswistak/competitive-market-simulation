"""CLI entry point for market simulation."""

import logging
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.logging import RichHandler

from .config import load_config
from .config.schema import AuctionType
from .llm.factory import create_tool_augmented_llm
from .graph import build_market_graph, build_auction_graph
from .agents import create_initial_state, create_auction_initial_state
from .tracing import create_tracing_manager, LLMCallLogger
from .output import ResultsSaver
from .tools.sandbox import SandboxManager

app = typer.Typer(
    name="market-simulation",
    help="LangGraph-based market equilibrium simulation for LLM agents",
)
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def run(
    config: str = typer.Argument(
        ...,
        help="Config file name (without .yaml) or full path to config file",
    ),
    sims: int | None = typer.Option(
        None,
        "--sims",
        "-s",
        help="Override number of simulations",
    ),
    output_dir: Path = typer.Option(
        Path("./results"),
        "--output",
        "-o",
        help="Output directory for results",
    ),
    trace: bool | None = typer.Option(
        None,
        "--trace/--no-trace",
        help="Enable/disable Langfuse tracing",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Run market simulation experiment."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    console.print("[bold green]Market Simulation[/]")
    console.print(f"Loading config: {config}")

    # Load configuration
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)

    # Override settings from CLI
    if sims is not None:
        cfg.experiment.n_simulations = sims
    if trace is not None:
        cfg.tracing.enabled = trace

    # Determine if this is an auction run
    is_auction = cfg.experiment.auction_type != AuctionType.DOUBLE_AUCTION
    auction_config = cfg.experiment.auction

    # Display configuration
    console.print(f"Provider: [cyan]{cfg.llm.provider}[/]")
    console.print(f"Model: [cyan]{cfg.llm.model}[/]")
    if is_auction:
        console.print(f"Auction type: [cyan]{cfg.experiment.auction_type.value}[/]")
        if auction_config:
            console.print(f"Simulations: [cyan]{auction_config.n_simulations}[/]")
            console.print(f"Rounds: [cyan]{auction_config.n_rounds}[/]")
            console.print(f"Bidders: [cyan]{auction_config.bidders.num}[/]")
            console.print(
                f"Values: [cyan]{auction_config.bidders.value_min}"
                f"–{auction_config.bidders.value_max} "
                f"({auction_config.bidders.distribution})[/]"
            )
            if cfg.experiment.auction_type in (AuctionType.ENGLISH, AuctionType.FIRST_PRICE_OPEN_OUTCRY):
                console.print(f"Min increment: [cyan]{auction_config.min_increment}[/]")
            elif cfg.experiment.auction_type == AuctionType.DUTCH:
                console.print(
                    f"Dutch: [cyan]start={auction_config.dutch_start_price}, "
                    f"decrement={auction_config.dutch_decrement}, "
                    f"floor={auction_config.dutch_min_price}[/]"
                )
    else:
        console.print(f"Auction type: [cyan]double_auction[/]")
        console.print(f"Simulations: [cyan]{cfg.experiment.n_simulations}[/]")
        console.print(f"Rounds: [cyan]{cfg.experiment.n_rounds}[/]")
        console.print(f"Max ticks/round: [cyan]{cfg.experiment.max_ticks_per_round}[/]")
    console.print(f"Tracing: [cyan]{'enabled' if trace else 'disabled'}[/]")
    if cfg.tools.enabled:
        tool_types = []
        if cfg.tools.enable_simple_tools:
            tool_types.append("simple")
        if cfg.tools.enable_code_interpreter:
            tool_types.append("E2B code interpreter")
        console.print(f"Tools: [cyan]{', '.join(tool_types)}[/]")
    else:
        console.print("Tools: [cyan]disabled[/]")
    console.print()

    # Determine whether any configured agent uses an LLM strategy.
    # If all agents are zero-intelligence we can skip creating the LLM
    # provider (and its API client) entirely.
    def _uses_llm(strategies) -> bool:
        if isinstance(strategies, list):
            return any(s == "llm" for s in strategies)
        return strategies == "llm"

    if is_auction and auction_config:
        any_llm = _uses_llm(auction_config.bidders.strategies)
    else:
        any_llm = _uses_llm(cfg.experiment.buyers.strategies) or _uses_llm(
            cfg.experiment.sellers.strategies
        )

    # Create sandbox manager if E2B is enabled
    sandbox_manager: SandboxManager | None = None
    if any_llm and cfg.tools.enabled and cfg.tools.enable_code_interpreter:
        sandbox_manager = SandboxManager(
            enabled=True,
            timeout=cfg.tools.e2b_timeout,
        )

    # Create LLM provider only if at least one agent uses it.
    llm = None
    if any_llm:
        llm = create_tool_augmented_llm(cfg.llm, cfg.tools, sandbox_manager)
        logger.info(f"Created {llm.provider_name} provider with model {llm.model_name}")
    else:
        console.print("[yellow]All agents are zero-intelligence — skipping LLM provider creation[/]")

    # Seed the ZI RNG. Prefer experiment.random_seed; fall back to auction
    # random_seed for auction runs so existing auction configs keep working.
    seed = cfg.experiment.random_seed
    if seed is None and is_auction and auction_config is not None:
        seed = auction_config.random_seed
    zi_rng = np.random.default_rng(seed)

    # Create tracing manager
    tracing = create_tracing_manager(cfg.tracing)
    if tracing.enabled:
        console.print("[green]Langfuse tracing enabled[/]")

    # Create results saver
    results_saver = ResultsSaver(
        output_dir=output_dir,
        experiment_name=config,
        config=cfg,
    )
    console.print(f"Results will be saved to: [cyan]{results_saver.output_dir}[/]")
    console.print()

    # Extract experiment_id from output folder name for Langfuse tracking
    experiment_id = results_saver.output_dir.name
    tracing.set_experiment_context(
        experiment_id=experiment_id,
        experiment_name=config,
    )

    # Warn if personas are configured but placeholder is missing from template
    has_da_personas = cfg.personas and (
        cfg.personas.buyer_default or cfg.personas.seller_default
        or cfg.personas.buyers or cfg.personas.sellers
    )
    has_auction_personas = cfg.personas and (
        cfg.personas.bidder_default or cfg.personas.bidders
    )
    if has_da_personas and not is_auction:
        templates = (
            cfg.prompts.general.system_template
            + "\n"
            + cfg.prompts.general.user_template
        )
        if "{persona}" not in templates:
            logger.warning(
                "Personas configured but {persona} placeholder missing from both "
                "system_template and user_template. Persona text will not appear "
                "in prompts."
            )
    if has_auction_personas and is_auction and cfg.prompts.auction:
        if "{persona}" not in cfg.prompts.auction.system_template:
            logger.warning(
                "Bidder personas configured but {persona} placeholder missing from "
                "auction system_template. Persona text will not appear in prompts."
            )

    # Build graph and calculate recursion limit
    if is_auction and auction_config:
        if any_llm and not cfg.prompts.auction:
            raise typer.BadParameter(
                f"Auction type '{cfg.experiment.auction_type.value}' with "
                f"strategy='llm' requires 'prompts.auction' to be configured."
            )
        graph = build_auction_graph(
            cfg.experiment.auction_type, llm, cfg.prompts.auction,
            # Use the resolved `seed` (experiment.random_seed first,
            # auction.random_seed fallback) so mechanism-level randomness
            # (e.g. Dutch bidder shuffle) follows the same precedence as
            # the ZI RNG. Without this, setting only experiment.random_seed
            # would leave the shuffle non-deterministic.
            random_seed=seed,
            include_reasoning=cfg.experiment.include_reasoning,
            zi_config=cfg.zi,
            rng=zi_rng,
        )
        n_bidders = auction_config.bidders.num
        n_rounds = auction_config.n_rounds

        if cfg.experiment.auction_type in (AuctionType.ENGLISH, AuctionType.FIRST_PRICE_OPEN_OUTCRY):
            recursion_limit = n_rounds * auction_config.max_bidding_rounds * n_bidders * 3 + 100
        elif cfg.experiment.auction_type == AuctionType.DUTCH:
            price_steps = int((auction_config.dutch_start_price - auction_config.dutch_min_price) / max(auction_config.dutch_decrement, 0.01))
            recursion_limit = n_rounds * price_steps * n_bidders * 2 + 100
        else:  # sealed-bid
            recursion_limit = n_rounds * n_bidders * 3 + 50

        n_sims = auction_config.n_simulations
    elif is_auction and not auction_config:
        raise typer.BadParameter(
            f"Auction type '{cfg.experiment.auction_type.value}' requires "
            f"an 'experiment.auction' block in the config file."
        )
    else:
        logger.info("No auction type specified — defaulting to double-auction mode")
        graph = build_market_graph(
            llm, cfg.prompts,
            include_reasoning=cfg.experiment.include_reasoning,
            zi_config=cfg.zi,
            rng=zi_rng,
        )
        # CDA recursion bound: each tick runs 6 nodes
        # (select_announcer, announce, apply_order, update_history,
        # check_round, next_iteration). Plus a next_round cycle per
        # round. Use a generous multiplier for safety.
        max_ticks = cfg.experiment.max_ticks_per_round
        nodes_per_tick = 6
        nodes_per_round = max_ticks * nodes_per_tick + 4
        recursion_limit = cfg.experiment.n_rounds * nodes_per_round + 50
        n_sims = cfg.experiment.n_simulations

    if cfg.experiment.recursion_limit is not None:
        logger.info(
            f"Recursion limit overridden by config: {cfg.experiment.recursion_limit} "
            f"(calculated: {recursion_limit})"
        )
        recursion_limit = cfg.experiment.recursion_limit
    else:
        recursion_limit = max(100, min(recursion_limit, 500_000))
    logger.info(f"Using recursion limit: {recursion_limit}")

    # Run simulations
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running simulations...", total=n_sims)

        for sim_id in range(1, n_sims + 1):
            progress.update(task, description=f"Simulation {sim_id}/{n_sims}")

            # Start file logging for this simulation
            results_saver.start_simulation_logging(sim_id)

            # Create LLM call logger for this simulation
            llm_logger: LLMCallLogger | None = None
            if cfg.tracing.llm_call_logging:
                llm_log_path = results_saver.logs_dir / f"llm_calls_{sim_id}.jsonl"
                llm_logger = LLMCallLogger(llm_log_path)

            # Create initial state
            if is_auction and auction_config:
                initial_state = create_auction_initial_state(
                    cfg.experiment, auction_config, simulation_id=sim_id,
                    personas=cfg.personas,
                )
            else:
                initial_state = create_initial_state(
                    cfg.experiment, simulation_id=sim_id, personas=cfg.personas,
                )

            try:
                # Use trace_simulation context manager for proper Langfuse tracing
                with tracing.trace_simulation(
                    simulation_id=sim_id,
                    recursion_limit=recursion_limit,
                    initial_state=initial_state,
                ) as graph_config:
                    if llm_logger:
                        graph_config.setdefault("callbacks", []).append(llm_logger)
                    final_state = graph.invoke(initial_state, config=graph_config)

                    # Update trace output with results
                    if is_auction:
                        n_results = len(final_state.get("auction_results", []))
                        tracing.update_trace_output(
                            auction_results=final_state.get("auction_results", []),
                            n_rounds_completed=n_results,
                        )
                    else:
                        n_transactions = len(final_state["transactions"])
                        tracing.update_trace_output(
                            transactions=final_state["transactions"],
                            n_transactions=n_transactions,
                        )

                # Save results
                if is_auction:
                    results_saver.save_auction_simulation(final_state, sim_id)
                else:
                    results_saver.save_simulation(final_state, sim_id)

                n_violations = final_state.get("constraint_violations", 0)

                if is_auction:
                    n_results = len(final_state.get("auction_results", []))
                    logger.info(
                        f"Simulation {sim_id} complete: {n_results} auction rounds, "
                        f"{n_violations} constraint violations"
                    )
                else:
                    n_transactions = len(final_state["transactions"])
                    logger.info(
                        f"Simulation {sim_id} complete: {n_transactions} transactions, "
                        f"{n_violations} constraint violations"
                    )

            except Exception as e:
                logger.error(f"Simulation {sim_id} failed: {e}")
                if verbose:
                    console.print_exception()

            finally:
                # Close LLM call logger
                if llm_logger:
                    llm_logger.close()

                # Stop file logging for this simulation
                results_saver.stop_simulation_logging(sim_id)

                # Close sandbox between simulations (fresh sandbox per sim)
                if sandbox_manager is not None:
                    sandbox_manager.close()

            progress.advance(task)

    # Cleanup
    if sandbox_manager is not None:
        sandbox_manager.close()
    tracing.flush()

    console.print()
    console.print("[bold green]Experiment complete![/]")
    console.print(f"Results saved to: [cyan]{results_saver.output_dir}[/]")


@app.command()
def visualize(
    output: Path = typer.Option(
        Path("graph.png"),
        "--output",
        "-o",
        help="Output file for graph visualization",
    ),
) -> None:
    """Visualize the market simulation graph structure."""
    from .config.schema import PromptConfig

    console.print("[bold]Generating graph visualization...[/]")

    # Build graph without LLM (for visualization only)
    # We need a minimal mock
    class MockLLM:
        def invoke(self, prompt, callbacks=None):
            return "1.5"

        def invoke_structured(self, prompt, schema, callbacks=None):
            # Build kwargs from schema fields with sensible defaults
            defaults = {"price": 1.5, "accept": True, "bid": 1.5,
                        "action": "bid", "reasoning": "mock"}
            kwargs = {k: defaults[k] for k in schema.model_fields if k in defaults}
            return schema(**kwargs)

    mock_llm = MockLLM()
    prompts = PromptConfig()

    try:
        graph = build_market_graph(mock_llm, prompts)

        # Get mermaid diagram
        mermaid = graph.get_graph().draw_mermaid()
        console.print("[green]Graph structure (Mermaid):[/]")
        console.print(mermaid)

        # Try to save as PNG if graphviz available
        try:
            png_data = graph.get_graph().draw_mermaid_png()
            with open(output, "wb") as f:
                f.write(png_data)
            console.print(f"[green]Graph saved to:[/] {output}")
        except Exception as e:
            console.print(
                f"[yellow]Could not save PNG (graphviz may not be installed):[/] {e}"
            )

    except Exception as e:
        console.print(f"[red]Error generating visualization:[/] {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    config: str = typer.Argument(
        ...,
        help="Config file to validate",
    ),
) -> None:
    """Validate a configuration file."""
    console.print(f"[bold]Validating config:[/] {config}")

    try:
        cfg = load_config(config)
        console.print("[green]Configuration is valid![/]")
        console.print()
        console.print(f"  LLM Provider: {cfg.llm.provider}")
        console.print(f"  Model: {cfg.llm.model}")
        console.print(f"  Auction type: {cfg.experiment.auction_type.value}")

        if cfg.experiment.auction_type != AuctionType.DOUBLE_AUCTION and cfg.experiment.auction:
            ac = cfg.experiment.auction
            console.print(f"  Simulations: {ac.n_simulations}")
            console.print(f"  Rounds: {ac.n_rounds}")
            console.print(f"  Bidders: {ac.bidders.num}")
            console.print(f"  Values: {ac.bidders.value_min}–{ac.bidders.value_max} ({ac.bidders.distribution})")
            if cfg.experiment.auction_type in (AuctionType.ENGLISH, AuctionType.FIRST_PRICE_OPEN_OUTCRY):
                console.print(f"  Min increment: {ac.min_increment}")
                console.print(f"  Max bidding rounds: {ac.max_bidding_rounds}")
            elif cfg.experiment.auction_type == AuctionType.DUTCH:
                console.print(f"  Dutch start: {ac.dutch_start_price}")
                console.print(f"  Dutch decrement: {ac.dutch_decrement}")
                console.print(f"  Dutch floor: {ac.dutch_min_price}")
        else:
            console.print(f"  Simulations: {cfg.experiment.n_simulations}")
            console.print(f"  Rounds: {cfg.experiment.n_rounds}")
            console.print(f"  Max ticks/round: {cfg.experiment.max_ticks_per_round}")
            console.print(f"  Buyers: {cfg.experiment.buyers.num}")
            console.print(f"  Sellers: {cfg.experiment.sellers.num}")
    except Exception as e:
        console.print(f"[red]Invalid configuration:[/] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
