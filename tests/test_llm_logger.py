"""Tests for LLMCallLogger callback handler."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from market_simulation.tracing.llm_logger import LLMCallLogger, load_llm_logs


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "llm_calls.jsonl"


@pytest.fixture
def logger(log_path):
    lg = LLMCallLogger(log_path)
    yield lg
    if not lg._file.closed:
        lg.close()


def _make_serialized(model_class="ChatOpenAI"):
    return {"id": ["langchain", "chat_models", model_class]}


def _make_llm_result(content="Hello", input_tokens=10, output_tokens=5, model_name="gpt-4o-mini", finish_reason="stop"):
    message = AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={
            "model_name": model_name,
            "finish_reason": finish_reason,
        },
    )
    generation = ChatGeneration(message=message)
    return LLMResult(generations=[[generation]])


class TestLLMCallLogger:

    def test_basic_call_logged(self, logger, log_path):
        run_id = uuid4()

        logger.on_chat_model_start(
            _make_serialized(),
            [[HumanMessage(content="What is 2+2?")]],
            run_id=run_id,
        )
        logger.on_llm_end(
            _make_llm_result(content="4"),
            run_id=run_id,
        )
        logger.close()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["prompt"] == "What is 2+2?"
        assert record["response"] == "4"
        assert record["model_name"] == "gpt-4o-mini"
        assert record["model_class"] == "ChatOpenAI"
        assert record["input_tokens"] == 10
        assert record["output_tokens"] == 5
        assert record["total_tokens"] == 15
        assert record["finish_reason"] == "stop"
        assert record["error"] is None
        assert record["latency_seconds"] >= 0
        assert record["timestamp"] is not None
        assert record["run_id"] == str(run_id)

    def test_multiple_calls_logged(self, logger, log_path):
        for i in range(3):
            run_id = uuid4()
            logger.on_chat_model_start(
                _make_serialized(),
                [[HumanMessage(content=f"Question {i}")]],
                run_id=run_id,
            )
            logger.on_llm_end(
                _make_llm_result(content=f"Answer {i}"),
                run_id=run_id,
            )

        logger.close()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["prompt"] == f"Question {i}"
            assert record["response"] == f"Answer {i}"

    def test_error_logged(self, logger, log_path):
        run_id = uuid4()

        logger.on_chat_model_start(
            _make_serialized(),
            [[HumanMessage(content="Will fail")]],
            run_id=run_id,
        )
        logger.on_llm_error(
            ValueError("API rate limit"),
            run_id=run_id,
        )
        logger.close()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["prompt"] == "Will fail"
        assert record["error"] == "API rate limit"
        assert record["response"] is None
        assert record["input_tokens"] is None

    def test_parent_run_id_captured(self, logger, log_path):
        run_id = uuid4()
        parent_id = uuid4()

        logger.on_chat_model_start(
            _make_serialized(),
            [[HumanMessage(content="test")]],
            run_id=run_id,
            parent_run_id=parent_id,
        )
        logger.on_llm_end(
            _make_llm_result(),
            run_id=run_id,
        )
        logger.close()

        record = json.loads(log_path.read_text().strip())
        assert record["parent_run_id"] == str(parent_id)

    def test_no_parent_run_id(self, logger, log_path):
        run_id = uuid4()

        logger.on_chat_model_start(
            _make_serialized(),
            [[HumanMessage(content="test")]],
            run_id=run_id,
        )
        logger.on_llm_end(
            _make_llm_result(),
            run_id=run_id,
        )
        logger.close()

        record = json.loads(log_path.read_text().strip())
        assert record["parent_run_id"] is None

    def test_multiple_messages_joined(self, logger, log_path):
        run_id = uuid4()

        logger.on_chat_model_start(
            _make_serialized(),
            [[HumanMessage(content="System msg"), HumanMessage(content="User msg")]],
            run_id=run_id,
        )
        logger.on_llm_end(
            _make_llm_result(),
            run_id=run_id,
        )
        logger.close()

        record = json.loads(log_path.read_text().strip())
        assert record["prompt"] == "System msg\nUser msg"

    def test_on_llm_end_without_start_ignored(self, logger, log_path):
        logger.on_llm_end(
            _make_llm_result(),
            run_id=uuid4(),
        )
        logger.close()

        assert log_path.read_text().strip() == ""

    def test_on_llm_error_without_start_ignored(self, logger, log_path):
        logger.on_llm_error(
            ValueError("oops"),
            run_id=uuid4(),
        )
        logger.close()

        assert log_path.read_text().strip() == ""

    def test_missing_usage_metadata(self, logger, log_path):
        run_id = uuid4()
        message = AIMessage(content="no usage")
        generation = ChatGeneration(message=message)
        result = LLMResult(generations=[[generation]])

        logger.on_chat_model_start(
            _make_serialized(),
            [[HumanMessage(content="test")]],
            run_id=run_id,
        )
        logger.on_llm_end(result, run_id=run_id)
        logger.close()

        record = json.loads(log_path.read_text().strip())
        assert record["input_tokens"] is None
        assert record["output_tokens"] is None
        assert record["total_tokens"] is None

    def test_close_idempotent(self, logger):
        logger.close()
        logger.close()  # should not raise

    def test_ignore_flags(self, logger):
        assert logger.ignore_chain is True
        assert logger.ignore_agent is True
        assert logger.ignore_retriever is True

    def test_metadata_captured(self, logger, log_path):
        run_id = uuid4()
        metadata = {
            "agent_id": 2,
            "agent_type": "seller",
            "action": "announce",
            "round": 1,
            "iteration": 1,
            "simulation_id": 5,
        }

        logger.on_chat_model_start(
            _make_serialized(),
            [[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata=metadata,
        )
        logger.on_llm_end(_make_llm_result(), run_id=run_id)
        logger.close()

        record = json.loads(log_path.read_text().strip())
        assert record["agent_id"] == 2
        assert record["agent_type"] == "seller"
        assert record["action"] == "announce"
        assert record["round"] == 1
        assert record["iteration"] == 1
        assert record["simulation_id"] == 5

    def test_metadata_captured_on_error(self, logger, log_path):
        run_id = uuid4()
        metadata = {"agent_id": 0, "action": "respond"}

        logger.on_chat_model_start(
            _make_serialized(),
            [[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata=metadata,
        )
        logger.on_llm_error(ValueError("boom"), run_id=run_id)
        logger.close()

        record = json.loads(log_path.read_text().strip())
        assert record["agent_id"] == 0
        assert record["action"] == "respond"
        assert record["error"] == "boom"

    def test_no_metadata_omits_keys(self, logger, log_path):
        run_id = uuid4()

        logger.on_chat_model_start(
            _make_serialized(),
            [[HumanMessage(content="hi")]],
            run_id=run_id,
        )
        logger.on_llm_end(_make_llm_result(), run_id=run_id)
        logger.close()

        record = json.loads(log_path.read_text().strip())
        assert "agent_id" not in record
        assert "action" not in record


class TestLoadLLMLogs:

    def test_load_multiple_simulations(self, tmp_path):
        for sim_id in [1, 2]:
            path = tmp_path / f"llm_calls_{sim_id}.jsonl"
            lg = LLMCallLogger(path)
            run_id = uuid4()
            lg.on_chat_model_start(
                _make_serialized(),
                [[HumanMessage(content=f"sim {sim_id}")]],
                run_id=run_id,
            )
            lg.on_llm_end(_make_llm_result(), run_id=run_id)
            lg.close()

        df = load_llm_logs(tmp_path)
        assert len(df) == 2
        assert set(df["simulation_id"].tolist()) == {1, 2}
        assert "prompt" in df.columns
        assert "input_tokens" in df.columns

    def test_load_empty_directory(self, tmp_path):
        df = load_llm_logs(tmp_path)
        assert len(df) == 0
