"""CLI entry point for market simulation."""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler

from .config import load_config, SimulationConfig
from .llm import create_llm
from .graph import build_market_graph
from .agents import create_initial_state
from .tracing import create_tracing_manager
from .output import ResultsSaver

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
        "--sims", "-s",
        help="Override number of simulations",
    ),
    output_dir: Path = typer.Option(
        Path("./results"),
        "--output", "-o",
        help="Output directory for results",
    ),
    trace: bool = typer.Option(
        True,
        "--trace/--no-trace",
        help="Enable/disable Langfuse tracing",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Run market simulation experiment."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    console.print(f"[bold green]Market Simulation[/]")
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
    cfg.tracing.enabled = trace

    # Display configuration
    console.print(f"Provider: [cyan]{cfg.llm.provider}[/]")
    console.print(f"Model: [cyan]{cfg.llm.model}[/]")
    console.print(f"Simulations: [cyan]{cfg.experiment.n_simulations}[/]")
    console.print(f"Rounds: [cyan]{cfg.experiment.n_rounds}[/]")
    console.print(f"Iterations: [cyan]{cfg.experiment.n_iterations}[/]")
    console.print(f"Tracing: [cyan]{'enabled' if trace else 'disabled'}[/]")
    console.print()

    # Create LLM provider
    llm = create_llm(cfg.llm)
    logger.info(f"Created {llm.provider_name} provider with model {llm.model_name}")

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

    # Run simulations
    n_sims = cfg.experiment.n_simulations

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

            # Create callbacks factory for this simulation
            callbacks_factory = tracing.create_callbacks_factory(
                simulation_id=sim_id,
                experiment_name=config,
            )

            # Build graph with tracing
            graph = build_market_graph(llm, cfg.prompts, callbacks_factory)

            # Create initial state
            initial_state = create_initial_state(cfg.experiment, simulation_id=sim_id)

            # Run simulation
            try:
                final_state = graph.invoke(initial_state)

                # Save results
                results_saver.save_simulation(final_state, sim_id)

                n_transactions = len(final_state["transactions"])
                logger.info(f"Simulation {sim_id} complete: {n_transactions} transactions")

            except Exception as e:
                logger.error(f"Simulation {sim_id} failed: {e}")
                if verbose:
                    console.print_exception()

            progress.advance(task)

    # Flush tracing
    tracing.flush()

    console.print()
    console.print("[bold green]Experiment complete![/]")
    console.print(f"Results saved to: [cyan]{results_saver.output_dir}[/]")


@app.command()
def visualize(
    output: Path = typer.Option(
        Path("graph.png"),
        "--output", "-o",
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

    mock_llm = MockLLM()
    prompts = PromptConfig()

    try:
        graph = build_market_graph(mock_llm, prompts, None)

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
            console.print(f"[yellow]Could not save PNG (graphviz may not be installed):[/] {e}")

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
        console.print(f"  Simulations: {cfg.experiment.n_simulations}")
        console.print(f"  Rounds: {cfg.experiment.n_rounds}")
        console.print(f"  Iterations: {cfg.experiment.n_iterations}")
        console.print(f"  Buyers: {cfg.experiment.buyers.num}")
        console.print(f"  Sellers: {cfg.experiment.sellers.num}")
    except Exception as e:
        console.print(f"[red]Invalid configuration:[/] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
