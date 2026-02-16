"""First-Price Open-Outcry auction (ascending bids, first-price settlement)."""

from .workflow import build_open_outcry_graph

__all__ = ["build_open_outcry_graph"]
