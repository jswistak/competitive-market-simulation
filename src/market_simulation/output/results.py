"""Results saving and export functionality."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..graph.state import MarketState
from ..config.schema import SimulationConfig


class ResultsSaver:
    """Manages saving simulation results to disk."""

    def __init__(
        self, output_dir: Path, experiment_name: str, config: SimulationConfig
    ):
        """Initialize results saver.

        Args:
            output_dir: Base output directory.
            experiment_name: Name of the experiment.
            config: Simulation configuration.
        """
        self.experiment_name = experiment_name
        self.config = config
        self._file_handlers: dict[int, logging.FileHandler] = {}

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir / f"{experiment_name}_{timestamp}"
        self.data_dir = self.output_dir / "data"
        self.logs_dir = self.output_dir / "logs"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Save config
        self._save_config()

    def _save_config(self) -> None:
        """Save the configuration used for this experiment."""
        config_path = self.output_dir / "config_used.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config.model_dump(), f, default_flow_style=False)

    def start_simulation_logging(self, simulation_id: int) -> None:
        """Start logging to a file for this simulation.

        Args:
            simulation_id: Identifier for this simulation.
        """
        log_path = self.logs_dir / f"sim_{simulation_id}.log"
        handler = logging.FileHandler(log_path, mode="w")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        self._file_handlers[simulation_id] = handler

    def stop_simulation_logging(self, simulation_id: int) -> None:
        """Stop logging to file for this simulation.

        Args:
            simulation_id: Identifier for this simulation.
        """
        if simulation_id in self._file_handlers:
            handler: logging.FileHandler = self._file_handlers.pop(simulation_id)
            handler.close()
            logging.getLogger().removeHandler(handler)

    def save_simulation(self, state: MarketState, simulation_id: int) -> None:
        """Save results from a single simulation.

        Args:
            state: Final state from the simulation.
            simulation_id: Identifier for this simulation.
        """
        # Save iteration history
        if state["iteration_records"]:
            df_iterations = pd.DataFrame(state["iteration_records"])
            df_iterations.to_csv(
                self.data_dir / f"iteration_history_{simulation_id}.csv",
                index=False,
            )

        # Save transactions
        if state["transactions"]:
            df_transactions = pd.DataFrame(state["transactions"])
            df_transactions.to_csv(
                self.data_dir / f"transactions_{simulation_id}.csv",
                index=False,
            )

        # Save agent histories
        agent_records = []
        for agent in state["agents"]:
            for record in agent["own_history_data"]:
                agent_records.append(
                    {
                        "agent_id": agent["id"],
                        "agent_type": agent["type"],
                        "reservation_price": agent["reservation_price"],
                        "persona": agent.get("persona", ""),
                        **record,
                    }
                )

        if agent_records:
            df_agents = pd.DataFrame(agent_records)
            df_agents.to_csv(
                self.data_dir / f"agent_histories_{simulation_id}.csv",
                index=False,
            )

        # Save tool usage log
        tool_log = state.get("tool_usage_log", [])
        if tool_log:
            df_tools = pd.DataFrame(tool_log)
            df_tools.to_csv(
                self.data_dir / f"tool_usage_{simulation_id}.csv",
                index=False,
            )

    def save_auction_simulation(self, state: dict, simulation_id: int) -> None:
        """Save results from an auction simulation (non-double-auction).

        Args:
            state: Final state dict from the auction graph.
            simulation_id: Identifier for this simulation.
        """
        # Save auction results (per-round outcomes)
        auction_results = state.get("auction_results", [])
        if auction_results:
            df_results = pd.DataFrame(auction_results)
            # all_bids column is a list of dicts — serialize to JSON for CSV
            if "all_bids" in df_results.columns:
                df_results["all_bids"] = df_results["all_bids"].apply(json.dumps)
            df_results.to_csv(
                self.data_dir / f"auction_results_{simulation_id}.csv",
                index=False,
            )

        # Save all individual bids
        all_bids = state.get("all_bid_records", [])
        if all_bids:
            df_bids = pd.DataFrame(all_bids)
            df_bids.to_csv(
                self.data_dir / f"all_bids_{simulation_id}.csv",
                index=False,
            )

        # Save bidder histories
        bidder_records = []
        for bidder in state.get("bidders", []):
            for record in bidder.get("own_history_data", []):
                bidder_records.append(
                    {
                        "bidder_id": bidder["id"],
                        "private_value": bidder["private_value"],
                        **record,
                    }
                )

        if bidder_records:
            df_bidders = pd.DataFrame(bidder_records)
            df_bidders.to_csv(
                self.data_dir / f"bidder_histories_{simulation_id}.csv",
                index=False,
            )

        # Save tool usage log
        tool_log = state.get("tool_usage_log", [])
        if tool_log:
            df_tools = pd.DataFrame(tool_log)
            df_tools.to_csv(
                self.data_dir / f"tool_usage_{simulation_id}.csv",
                index=False,
            )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of saved results.

        Returns:
            Dictionary with summary statistics.
        """
        return {
            "output_dir": str(self.output_dir),
            "experiment_name": self.experiment_name,
            "data_files": list(self.data_dir.glob("*.csv")),
        }


def save_simulation_results(
    state: MarketState,
    output_dir: Path,
    simulation_id: int,
    config: SimulationConfig | None = None,
) -> Path:
    """Convenience function to save simulation results.

    Args:
        state: Final simulation state.
        output_dir: Output directory path.
        simulation_id: Simulation identifier.
        config: Optional configuration to save.

    Returns:
        Path to the data directory.
    """
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save iteration history
    if state["iteration_records"]:
        df_iterations = pd.DataFrame(state["iteration_records"])
        df_iterations.to_csv(
            data_dir / f"iteration_history_{simulation_id}.csv",
            index=False,
        )

    # Save transactions
    if state["transactions"]:
        df_transactions = pd.DataFrame(state["transactions"])
        df_transactions.to_csv(
            data_dir / f"transactions_{simulation_id}.csv",
            index=False,
        )

    return data_dir
