"""Tests for graph node functions (announce, control flow)."""

import pytest
from unittest.mock import MagicMock, patch

from market_simulation.graph.nodes.announce import (
    make_select_announcer_node,
    make_announce_node,
    _render_announcement_prompt,
)
from market_simulation.config.schema import (
    PromptConfig,
    PromptTemplates,
    AgentPromptConfig,
    AgentKeywords,
)
from market_simulation.graph.history import build_market_history_for_prompt
from market_simulation.graph.nodes.control import (
    make_update_history_node,
    make_check_iteration_node,
    make_check_round_node,
    make_next_iteration_node,
    make_next_round_node,
    _update_agent_histories,
)
from market_simulation.llm.response_schemas import (
    AnnouncementResponse,
    AnnouncementResponseWithReasoning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(callbacks=None):
    """Create a minimal RunnableConfig-like dict."""
    return {"callbacks": callbacks or []}


# ===========================================================================
# TestSelectAnnouncerNode
# ===========================================================================


class TestSelectAnnouncerNode:
    """Tests for select_announcer node."""

    def test_selects_active_agent(self, base_market_state):
        node = make_select_announcer_node()
        result = node(base_market_state)
        assert result["announcing_agent_id"] in base_market_state["active_agent_ids"]

    def test_returns_none_when_no_active_agents(self, base_market_state):
        state = {**base_market_state, "active_agent_ids": []}
        node = make_select_announcer_node()
        result = node(state)
        assert result["announcing_agent_id"] is None
        assert result["announcement_made"] is False

    def test_selected_from_active_ids(self, base_market_state):
        # Only agents 0 and 3 are active
        state = {**base_market_state, "active_agent_ids": [0, 3]}
        node = make_select_announcer_node()
        result = node(state)
        assert result["announcing_agent_id"] in [0, 3]


# ===========================================================================
# TestAnnounceNode
# ===========================================================================


class TestAnnounceNode:
    """Tests for announce node."""

    def test_valid_announcement(self, base_market_state, mock_llm, prompt_config):
        state = {**base_market_state, "announcing_agent_id": 0}  # buyer
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is True
        assert result["announced_price"] == 1.50
        assert result["announcement_type"] == "buy"

    def test_seller_announcement_type(self, base_market_state, mock_llm, prompt_config):
        state = {**base_market_state, "announcing_agent_id": 3}  # seller
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_type"] == "sell"

    def test_none_agent_id(self, base_market_state, mock_llm, prompt_config):
        state = {**base_market_state, "announcing_agent_id": None}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is False
        assert result["announced_price"] is None

    def test_none_price_means_no_announcement(self, base_market_state, mock_llm, prompt_config):
        """Structured output with price=None is a valid response meaning no announcement."""
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(price=None, reasoning="")
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is False

    def test_llm_exception(self, base_market_state, mock_llm, prompt_config):
        mock_llm.invoke_structured.side_effect = RuntimeError("API error")
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is False
        assert result["announced_price"] is None
        assert "API error" in result["last_error"]

    def test_tool_usage_log_captured(self, base_market_state, mock_llm, prompt_config):
        mock_llm.last_tool_log = [{"tool": "calculator", "input": "1+1"}]
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert len(result["tool_usage_log"]) == 1
        assert result["tool_usage_log"][0]["agent_id"] == 0
        assert result["tool_usage_log"][0]["action"] == "announce"

    def test_buyer_above_reservation_increments_violations(self, base_market_state, mock_llm, prompt_config):
        # Buyer 0 has reservation_price=2.0, announcing $3.00 is a violation
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(price=3.00, reasoning="")
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is True
        assert result["announced_price"] == 3.00
        assert result["constraint_violations"] == 1

    def test_seller_below_reservation_increments_violations(self, base_market_state, mock_llm, prompt_config):
        # Seller 3 has reservation_price=1.0, announcing $0.50 is a violation
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(price=0.50, reasoning="")
        state = {**base_market_state, "announcing_agent_id": 3}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is True
        assert result["announced_price"] == 0.50
        assert result["constraint_violations"] == 1

    def test_no_violation_when_within_bounds(self, base_market_state, mock_llm, prompt_config):
        # Buyer 0 has reservation_price=2.0, announcing $1.50 is fine
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(price=1.50, reasoning="")
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is True
        assert "constraint_violations" not in result

    def test_violation_counter_accumulates(self, base_market_state, mock_llm, prompt_config):
        # Start with 2 existing violations
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(price=3.00, reasoning="")
        state = {**base_market_state, "announcing_agent_id": 0, "constraint_violations": 2}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["constraint_violations"] == 3


# ===========================================================================
# TestUpdateHistoryNode
# ===========================================================================


class TestUpdateHistoryNode:
    """Tests for update_history node."""

    def test_transaction_accepted_history(self, base_market_state):
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "counterparty_agent_id": 3,
            "current_responder_index": 1,
            "potential_responder_ids": [3],
            "transactions": [
                {"round": 1, "iteration": 1, "price": 1.50,
                 "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
            ],
        }
        node = make_update_history_node()
        result = node(state)

        assert "accepted" in result["market_history_text"]
        assert "$1.50" in result["market_history_text"]
        assert len(result["iteration_records"]) == 1

    def test_posted_announcement_history(self, base_market_state):
        """An improving order that posted to the book (no cross yet) renders
        the posted-but-not-traded line, distinguishable from a non_improving
        drop."""
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": False,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "last_order_outcome": "posted",
        }
        node = make_update_history_node()
        result = node(state)

        text = result["market_history_text"]
        assert "$1.50" in text
        assert "posted" in text.lower()

    def test_no_announcement_history(self, base_market_state):
        state = {
            **base_market_state,
            "announcement_made": False,
            "transaction_made": False,
            "iteration_complete": True,
            "announced_price": None,
            "announcement_type": None,
            "announcing_agent_id": None,
            "counterparty_agent_id": None,
            "current_responder_index": 0,
            "potential_responder_ids": [],
        }
        node = make_update_history_node()
        result = node(state)

        assert "no announcement was made" in result["market_history_text"]

    def test_announcing_agent_history_updated_on_transaction(self, base_market_state):
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "counterparty_agent_id": 3,
            "current_responder_index": 1,
            "potential_responder_ids": [3],
            "transactions": [
                {"round": 1, "iteration": 1, "price": 1.50,
                 "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
            ],
        }
        node = make_update_history_node()
        result = node(state)

        announcing_agent = next(a for a in result["agents"] if a["id"] == 0)
        assert len(announcing_agent["own_history_data"]) == 1
        assert announcing_agent["own_history_data"][0]["action"] == "announce"
        assert announcing_agent["own_history_data"][0]["outcome"] == "accepted"
        assert "accepted" in announcing_agent["own_history_prompt"]

    def test_counterparty_history_back_annotated_on_cross(self, base_market_state):
        """When buyer 0 crosses seller 3's standing ask, seller 3 (the
        resting-order owner) gets an 'accepted' entry at the trade tick
        — same wording as the announcer side. Without this, the
        seller's history shows only their original 'posted' line and
        they never learn the order traded.

        Buyer announces $1.60 (their ceiling); standing ask was $1.50;
        trade executes at $1.50. The seller's accepted entry must
        record $1.50 (the trade price), not $1.60.
        """
        # Seller 3 has a prior 'posted at $1.50' entry from when they
        # placed the standing ask on an earlier tick.
        agents = []
        for a in base_market_state["agents"]:
            if a["id"] == 3:
                agents.append({
                    **a,
                    "own_history_prompt": (
                        "In round 1 at iteration 1, your offer to sell "
                        "for $1.50 was posted.\n"
                    ),
                    "own_history_data": [{
                        "round": 1, "iteration": 1, "action": "announce",
                        "price": 1.50, "outcome": "posted",
                    }],
                })
            else:
                agents.append(a)

        state = {
            **base_market_state,
            "agents": agents,
            "iteration": 5,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.60,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "counterparty_agent_id": 3,
            "last_order_outcome": "traded",
            "transactions": [
                {"round": 1, "iteration": 5, "price": 1.50,
                 "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
            ],
        }
        node = make_update_history_node()
        result = node(state)

        seller_3 = next(a for a in result["agents"] if a["id"] == 3)
        # Original 'posted' line must remain — history is append-only.
        assert any(
            e.get("outcome") == "posted" and e.get("iteration") == 1
            for e in seller_3["own_history_data"]
        )
        # New 'accepted' entry at the trade tick.
        accepted_entries = [
            e for e in seller_3["own_history_data"]
            if e.get("outcome") == "accepted" and e.get("iteration") == 5
        ]
        assert accepted_entries, (
            "seller 3's resting ask was crossed at iteration 5 but their "
            "own_history_data has no 'accepted' entry"
        )
        # Trade price ($1.50), not the buyer's announced ceiling ($1.60).
        assert accepted_entries[-1]["price"] == 1.50, (
            "back-annotated 'accepted' entry must record the trade price "
            f"($1.50, the matched standing ask); got "
            f"${accepted_entries[-1]['price']:.2f}"
        )
        # Rendered prompt: original 'posted' line plus the new
        # 'accepted' line, both for the seller's own offer.
        prompt = seller_3["own_history_prompt"]
        assert "for $1.50 was posted" in prompt
        # The 'accepted' line is the seller's perspective on their
        # original sell offer being filled, so announcement_type=sell.
        assert "your offer to sell for $1.50 was accepted" in prompt, (
            f"counterparty's accepted line missing or wrong; got: {prompt!r}"
        )
        # Must not show the buyer's announced ceiling as a fill price.
        assert "$1.60" not in prompt, (
            f"buyer's announced $1.60 leaked into seller's history: {prompt!r}"
        )

    def test_iteration_record_fields(self, base_market_state):
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "counterparty_agent_id": 3,
            "current_responder_index": 1,
            "potential_responder_ids": [3],
            "transactions": [
                {"round": 1, "iteration": 1, "price": 1.50,
                 "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
            ],
        }
        node = make_update_history_node()
        result = node(state)

        record = result["iteration_records"][0]
        assert record["round"] == 1
        assert record["iteration"] == 1
        assert record["price"] == 1.50
        assert record["announcement_made"] is True
        assert record["transaction_made"] is True
        assert record["announcing_agent_id"] == 0
        assert record["counterparty_agent_id"] == 3

    def test_non_improving_renders_distinct_market_history_line(
        self, base_market_state
    ):
        """A non-improving order must produce its own market-history entry,
        not silently fall through to the no-announcement template."""
        state = {
            **base_market_state,
            # announcement_made stays True for non_improving — the agent
            # emitted a price; the order was just dropped from the book.
            "announcement_made": True,
            "transaction_made": False,
            "announced_price": 0.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "last_order_outcome": "non_improving",
        }
        node = make_update_history_node()
        result = node(state)

        text = result["market_history_text"]
        assert "$0.50" in text
        assert "rejected" in text.lower()
        assert "no announcement was made" not in text

    def test_non_improving_records_announcer_attempt(self, base_market_state):
        """The announcer's own_history must record a non-improving
        attempt with the rejection *reason* (not just the word
        'rejected'), so the agent has the same signal in their
        own_history that the market_history broadcasts to others."""
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": False,
            "announced_price": 0.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "last_order_outcome": "non_improving",
        }
        node = make_update_history_node()
        result = node(state)

        announcer = next(a for a in result["agents"] if a["id"] == 0)
        assert len(announcer["own_history_data"]) == 1
        entry = announcer["own_history_data"][0]
        assert entry["action"] == "announce"
        assert entry["price"] == 0.50
        # Structured field stays as the short label for analysis filters.
        assert entry["outcome"] == "rejected"
        # Rendered prompt text must include the rejection reason.
        prompt_history = announcer["own_history_prompt"]
        assert "rejected" in prompt_history
        assert "did not improve the standing book" in prompt_history

    def test_posted_outcome_distinct_from_traded_in_own_history(
        self, base_market_state
    ):
        """A posted-but-not-traded order must be labelled 'posted' in the
        announcer's own_history, distinguishable from 'accepted' (traded)
        and 'rejected' (non_improving)."""
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": False,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "last_order_outcome": "posted",
        }
        node = make_update_history_node()
        result = node(state)

        announcer = next(a for a in result["agents"] if a["id"] == 0)
        assert len(announcer["own_history_data"]) == 1
        entry = announcer["own_history_data"][0]
        assert entry["outcome"] == "posted"
        assert "posted" in announcer["own_history_prompt"]
        # Must NOT use the old conflated "rejected" wording for posts.
        assert "rejected" not in announcer["own_history_prompt"]

    def test_non_improving_iteration_record_captures_attempted_price(
        self, base_market_state
    ):
        """The IterationRecord must capture the price the agent
        attempted on a non_improving outcome. Since announcement_made
        consistently means 'agent emitted a price', the field should be
        True here even though the order was dropped from the book."""
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": False,
            "announced_price": 0.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "last_order_outcome": "non_improving",
        }
        node = make_update_history_node()
        result = node(state)

        record = result["iteration_records"][0]
        assert record["price"] == 0.50
        assert record["announcement_type"] == "buy"
        assert record["order_outcome"] == "non_improving"
        assert record["announcing_agent_id"] == 0

    def test_legacy_fallback_for_traded(self, base_market_state):
        """Pre-PR-#18 callers that don't set last_order_outcome should
        still resolve to a 'traded' outcome via the (transaction_made,
        announcement_made) fallback. Locks in backward compatibility."""
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            # last_order_outcome intentionally unset
        }
        node = make_update_history_node()
        result = node(state)

        # Market history renders the accepted template.
        assert "accepted" in result["market_history_text"].lower()
        # Announcer's own_history records the trade.
        announcer = next(a for a in result["agents"] if a["id"] == 0)
        assert len(announcer["own_history_data"]) == 1
        assert announcer["own_history_data"][0]["outcome"] == "accepted"

    def test_no_announcer_skips_history_update(self, base_market_state):
        """When announcing_agent_id is None, no agent should receive a
        history entry — even if a populated outcome was passed in. Locks
        in the announcer_id guard so callers can't accidentally write
        ghost actions to all agents."""
        agents_before = base_market_state["agents"]

        updated = _update_agent_histories(
            {**base_market_state, "announcing_agent_id": None,
             "announced_price": 0.50},
            outcome="non_improving",
        )

        for before, after in zip(agents_before, updated):
            assert after["own_history_data"] == before["own_history_data"]
            assert after["own_history_prompt"] == before["own_history_prompt"]

    def test_all_four_outcomes_appear_in_rendered_llm_prompt(
        self, sample_agents, base_market_state
    ):
        """End-to-end: each of the four order_outcome values must produce
        a rendered line in the LLM's prompt string. This is the regression
        test for the whole point of PR #21 — every case must reach the
        agent's view of the market.

        The chain is: update_history writes a line into market_history_text
        based on outcome → build_market_history_for_prompt(mode='full')
        returns the text unchanged → _render_announcement_prompt
        substitutes it into {market_history}. Each link is tested
        individually elsewhere; this test asserts the whole chain.
        """
        prompts = PromptConfig(
            general=PromptTemplates(
                user_template=(
                    "Reservation: {reservation_price}. "
                    "Rounds: {N_ROUNDS}. Iters: {N_ITER}. "
                    "Buyers: {N_BUYERS}. Sellers: {N_SELLERS}. "
                    "MARKET=[{market_history}] OWN=[{own_history}] "
                    "{action_prompt}"
                ),
            ),
            buyer=AgentPromptConfig(
                main_keywords=AgentKeywords(
                    role="buyer", verb="buy",
                    preference="lowest", condition="above",
                ),
                announcement_prompt="Announce.",
            ),
            seller=AgentPromptConfig(
                main_keywords=AgentKeywords(
                    role="seller", verb="sell",
                    preference="highest", condition="below",
                ),
                announcement_prompt="Announce.",
            ),
        )
        node = make_update_history_node(prompts)

        # Run update_history once per outcome, threading the
        # accumulating market_history_text through each call.
        state = {**base_market_state}

        # Case 1 — traded
        state = {
            **state, "iteration": 1,
            "announcement_made": True, "transaction_made": True,
            "announced_price": 2.00, "announcement_type": "buy",
            "announcing_agent_id": 0,
            "last_order_outcome": "traded",
            "transactions": [
                {"round": 1, "iteration": 1, "price": 2.00,
                 "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
            ],
        }
        out = node(state)
        state = {**state, **out, "agents": out["agents"]}

        # Case 2 — posted
        state = {
            **state, "iteration": 2,
            "announcement_made": True, "transaction_made": False,
            "announced_price": 1.60, "announcement_type": "buy",
            "announcing_agent_id": 1,
            "last_order_outcome": "posted",
        }
        out = node(state)
        state = {**state, **out, "agents": out["agents"]}

        # Case 3 — non_improving (announcement_made=True post-fix)
        state = {
            **state, "iteration": 3,
            "announcement_made": True, "transaction_made": False,
            "announced_price": 2.10, "announcement_type": "sell",
            "announcing_agent_id": 3,
            "last_order_outcome": "non_improving",
        }
        out = node(state)
        state = {**state, **out, "agents": out["agents"]}

        # Case 4 — no_announcement
        state = {
            **state, "iteration": 4,
            "announcement_made": False, "transaction_made": False,
            "announced_price": None, "announcement_type": None,
            "announcing_agent_id": None,
            "last_order_outcome": "no_announcement",
        }
        out = node(state)
        state = {**state, **out, "agents": out["agents"]}

        # Render the prompt for an agent on the next tick.
        _, rendered = _render_announcement_prompt(
            agent=state["agents"][0],
            state=state,
            prompts=prompts,
            agent_prompts=prompts.buyer,
        )

        # All four outcome wordings must appear in the rendered prompt
        # (these assertions match the market_history block).
        assert "for $2.00 was accepted" in rendered, "Case 1 (traded) missing"
        assert (
            "for $1.60 was posted as the new best buy" in rendered
        ), "Case 2 (posted) missing"
        assert (
            "for $2.10 was rejected because it did not improve"
            in rendered
        ), "Case 3 (non_improving) missing in market_history"
        assert "no announcement was made" in rendered, "Case 4 missing"

        # Case 3's announcer (seller id=3) must also see the rejection
        # *reason* in their own_history when their prompt is rendered —
        # the symmetric counterpart to the market_history broadcast.
        seller_3 = next(a for a in state["agents"] if a["id"] == 3)
        _, rendered_seller = _render_announcement_prompt(
            agent=seller_3,
            state=state,
            prompts=prompts,
            agent_prompts=prompts.seller,
        )
        # OWN=[...] block contains seller 3's own_history; the
        # market_history block contains everyone's, so we slice on the
        # explicit delimiter to test the own block specifically.
        own_block = rendered_seller.split("OWN=[", 1)[1].split("]", 1)[0]
        assert (
            "for $2.10 was rejected because it did not improve" in own_block
        ), "Case 3 reason missing from own_history"

    def test_non_improving_contributes_to_summary_mode_aggregates(
        self, base_market_state
    ):
        """After Commit 1 (announcement_made=True for non_improving),
        summary-mode market history must include non_improving prices in
        the bid/ask aggregates and the acceptance-rate denominator.
        Before the fix, non_improving rows had announcement_made=False
        and were silently filtered out by build_market_history_for_prompt's
        summary-mode aggregator at history.py:76,98.
        """
        records = [
            # One traded buy at $1.50
            {"round": 1, "iteration": 1, "announcement_made": True,
             "transaction_made": True, "price": 1.50,
             "announcement_type": "buy", "order_outcome": "traded"},
            # One non_improving sell at $2.10 (the case in question)
            {"round": 1, "iteration": 2, "announcement_made": True,
             "transaction_made": False, "price": 2.10,
             "announcement_type": "sell", "order_outcome": "non_improving"},
        ]
        state = {
            **base_market_state,
            "iteration_records": records,
            "transactions": [
                {"round": 1, "iteration": 1, "price": 1.50,
                 "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
            ],
        }

        summary = build_market_history_for_prompt(state, mode="summary",
                                                  last_n_events=0)

        # The non_improving sell at $2.10 must surface in the bid/ask spread.
        assert "$2.10" in summary, "non_improving sell price missing from summary"
        # Acceptance rate denominator includes the non_improving attempt.
        # 1 traded out of 2 announcements = 50%.
        assert "50%" in summary or "(1/2)" in summary, (
            "non_improving must count toward acceptance-rate denominator"
        )

    # ----------------------------------------------------------------
    # Trade-price rendering: on a cross, the trade executes at the
    # matched standing price (NYSE / G&S convention; see apply_order
    # ``_record_trade``). Both the market-history broadcast and the
    # announcer's own-history line must show the executed trade price,
    # not the announcer's emitted ceiling/floor (state["announced_price"]).
    # ----------------------------------------------------------------

    def test_market_history_shows_trade_price_not_announced_price_on_cross(
        self, base_market_state
    ):
        """Buyer 0 announces $1.60. Seller 3 had a standing ask at
        $1.50, so the trade executes at $1.50. market_history must
        broadcast the actual trade price, not the buyer's announced
        ceiling — otherwise third-party agents reading the market
        believe it cleared at $1.60.
        """
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.60,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "counterparty_agent_id": 3,
            "last_order_outcome": "traded",
            "transactions": [
                {"round": 1, "iteration": 1, "price": 1.50,
                 "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
            ],
        }
        node = make_update_history_node()
        result = node(state)

        text = result["market_history_text"]
        assert "$1.50" in text, (
            "market_history must broadcast the actual trade price "
            f"($1.50, the matched standing ask); got: {text!r}"
        )
        assert "$1.60" not in text, (
            "market_history shows the buyer's announced $1.60 as if it "
            "were the trade price — third-party agents reading this "
            f"will misread the market clearing level; got: {text!r}"
        )

    def test_announcer_own_history_shows_trade_price_not_announced_price(
        self, base_market_state
    ):
        """Same scenario from the announcer's perspective. Buyer 0
        announced $1.60 and crossed at a $1.50 standing ask. Their
        own_history must say "$1.50 was accepted" (what they actually
        paid), not "$1.60" (their max willingness-to-pay). Profit
        reasoning depends on the trade price.
        """
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.60,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "counterparty_agent_id": 3,
            "last_order_outcome": "traded",
            "transactions": [
                {"round": 1, "iteration": 1, "price": 1.50,
                 "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
            ],
        }
        node = make_update_history_node()
        result = node(state)

        announcer = next(a for a in result["agents"] if a["id"] == 0)
        accepted_entries = [
            e for e in announcer["own_history_data"]
            if e.get("outcome") == "accepted"
        ]
        assert accepted_entries, "announcer should have an 'accepted' entry"
        assert accepted_entries[-1]["price"] == 1.50, (
            "announcer's accepted own_history_data entry stored "
            f"${accepted_entries[-1]['price']:.2f} but the trade was at "
            f"$1.50 (the standing ask)"
        )
        prompt = announcer["own_history_prompt"]
        assert "$1.50" in prompt, (
            "announcer's own_history prompt must show the actual trade "
            f"price ($1.50); got: {prompt!r}"
        )
        assert "$1.60" not in prompt, (
            "announcer's own_history prompt shows their announced "
            "$1.60 — but they actually traded at $1.50; their profit "
            f"reasoning will be off; got: {prompt!r}"
        )


# ===========================================================================
# TestCheckIterationNode
#
# check_iteration is retained as a no-op stub for any legacy caller —
# the improvement-rule CDA does not wire it into the graph. The node
# simply returns iteration_complete=True.
# ===========================================================================


class TestCheckIterationNode:
    def test_always_returns_iteration_complete(self, base_market_state):
        node = make_check_iteration_node()
        assert node(base_market_state)["iteration_complete"] is True


# ===========================================================================
# TestCheckRoundNode
# ===========================================================================


class TestCheckRoundNode:
    """Tests for check_round node."""

    def test_complete_at_max_iterations(self, base_market_state):
        state = {**base_market_state, "iteration": 3, "max_iterations": 3}
        node = make_check_round_node()
        assert node(state)["round_complete"] is True

    def test_complete_with_too_few_agents(self, base_market_state):
        state = {**base_market_state, "active_agent_ids": [0]}
        node = make_check_round_node()
        assert node(state)["round_complete"] is True

    def test_complete_with_zero_agents(self, base_market_state):
        state = {**base_market_state, "active_agent_ids": []}
        node = make_check_round_node()
        assert node(state)["round_complete"] is True

    def test_not_complete(self, base_market_state):
        state = {**base_market_state, "iteration": 1, "max_iterations": 3, "active_agent_ids": [0, 3]}
        node = make_check_round_node()
        assert node(state)["round_complete"] is False


# ===========================================================================
# TestNextIterationNode
# ===========================================================================


class TestNextIterationNode:
    """Tests for next_iteration node."""

    def test_increments_iteration(self, base_market_state):
        state = {**base_market_state, "iteration": 1}
        node = make_next_iteration_node()
        result = node(state)
        assert result["iteration"] == 2

    def test_resets_flags(self, base_market_state):
        state = {
            **base_market_state,
            "iteration": 1,
            "transaction_made": True,
            "announcement_made": True,
            "announcing_agent_id": 0,
            "announced_price": 1.50,
        }
        node = make_next_iteration_node()
        result = node(state)

        assert result["transaction_made"] is False
        assert result["announcement_made"] is False
        assert result["announcing_agent_id"] is None
        assert result["announced_price"] is None
        assert result["announcement_type"] is None
        assert result["counterparty_agent_id"] is None
# ===========================================================================
# TestNextRoundNode
# ===========================================================================


class TestNextRoundNode:
    """Tests for next_round node."""

    def test_increments_round(self, base_market_state):
        state = {**base_market_state, "round": 1, "max_rounds": 2}
        node = make_next_round_node()
        result = node(state)
        assert result["round"] == 2

    def test_simulation_complete_at_max_rounds(self, base_market_state):
        state = {**base_market_state, "round": 2, "max_rounds": 2}
        node = make_next_round_node()
        result = node(state)
        assert result["simulation_complete"] is True

    def test_reactivates_all_agents(self, base_market_state):
        # Deactivate some agents
        agents = [
            {**a, "active": False} if a["id"] in (0, 3) else a
            for a in base_market_state["agents"]
        ]
        state = {
            **base_market_state,
            "round": 1,
            "max_rounds": 3,
            "agents": agents,
            "active_agent_ids": [1, 2, 4, 5],
        }
        node = make_next_round_node()
        result = node(state)

        assert all(a["active"] is True for a in result["agents"])
        assert set(result["active_agent_ids"]) == {0, 1, 2, 3, 4, 5}

    def test_resets_iteration_to_one(self, base_market_state):
        state = {**base_market_state, "round": 1, "max_rounds": 3}
        node = make_next_round_node()
        result = node(state)
        assert result["iteration"] == 1

    def test_resets_state_flags(self, base_market_state):
        state = {
            **base_market_state,
            "round": 1,
            "max_rounds": 3,
            "transaction_made": True,
            "announcement_made": True,
        }
        node = make_next_round_node()
        result = node(state)

        assert result["transaction_made"] is False
        assert result["announcement_made"] is False
        assert result["round_complete"] is False

    def test_boundary_new_round_equals_max_rounds_not_complete(self, base_market_state):
        """When round 1 -> 2 and max_rounds=2, simulation is NOT complete (round 2 still runs)."""
        state = {**base_market_state, "round": 1, "max_rounds": 2}
        node = make_next_round_node()
        result = node(state)

        assert result["round"] == 2
        assert result.get("simulation_complete", False) is False
