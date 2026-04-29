"""File-based LLM call logger using LangChain callback mechanism."""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class LLMCallLogger(BaseCallbackHandler):
    """Logs every LLM call to a JSONL file for offline analysis.

    Captures prompt text, response, token usage, latency, and model metadata.
    Designed to coexist with other callback handlers (e.g., Langfuse).

    Usage:
        logger = LLMCallLogger(Path("llm_calls.jsonl"))
        # pass as callback alongside other handlers
        model.invoke(messages, config={"callbacks": [langfuse_handler, logger]})
        logger.close()
    """

    ignore_chain = True
    ignore_agent = True
    ignore_retriever = True

    def __init__(self, log_path: Path) -> None:
        super().__init__()
        self._log_path = log_path
        self._file = open(log_path, "a")
        self._pending: dict[UUID, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Walk the messages list and split System / Human content into
        # separate buckets so the JSONL row can record them as distinct
        # fields. This is how the system+user prompt split surfaces in
        # offline analysis: ``system_prompt`` carries the per-agent
        # constants, ``user_prompt`` carries the per-tick state, and
        # ``prompt`` is kept as the concatenation for back-compat with
        # any existing notebook that joined on the single field.
        system_parts: list[str] = []
        user_parts: list[str] = []
        all_parts: list[str] = []
        for batch in messages:
            for msg in batch:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                all_parts.append(content)
                if isinstance(msg, SystemMessage):
                    system_parts.append(content)
                elif isinstance(msg, HumanMessage):
                    user_parts.append(content)
                # ToolMessage / AIMessage from tool-calling loops fall
                # through into ``prompt`` but not into either named
                # bucket; the loop's internal back-and-forth is not
                # part of the per-tick prompt design.

        system_prompt = "\n".join(system_parts) if system_parts else ""
        user_prompt = "\n".join(user_parts) if user_parts else ""
        prompt_text = "\n".join(all_parts)

        model_class = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"

        with self._lock:
            self._pending[run_id] = {
                "start_ts": time.perf_counter(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt_text,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model_class": model_class,
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "metadata": dict(metadata) if metadata else {},
            }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            pending = self._pending.pop(run_id, None)

        if pending is None:
            return

        latency = time.perf_counter() - pending["start_ts"]

        # Extract data from the first generation
        generation = response.generations[0][0] if response.generations and response.generations[0] else None
        response_text = ""
        input_tokens = None
        output_tokens = None
        total_tokens = None
        reasoning_tokens = None
        thinking_text = None
        model_name = None
        finish_reason = None

        if generation:
            message = getattr(generation, "message", None)
            response_text = generation.text or ""

            if message:
                # Token usage from AIMessage.usage_metadata
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                    total_tokens = usage.get("total_tokens")
                    output_details = usage.get("output_token_details") or {}
                    reasoning_tokens = output_details.get("reasoning")

                # Gemini thinking parts live in message.content as
                # {"type": "thinking", "thinking": "..."} blocks when
                # include_thoughts=True. Extract them separately.
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    thoughts = [
                        part.get("thinking", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "thinking"
                    ]
                    if thoughts:
                        thinking_text = "\n".join(t for t in thoughts if t)

                # Model name and finish reason from response_metadata
                meta = getattr(message, "response_metadata", None) or {}
                model_name = meta.get("model_name") or meta.get("model")
                finish_reason = meta.get("finish_reason")

        record = {
            "timestamp": pending["timestamp"],
            "run_id": str(run_id),
            "parent_run_id": pending["parent_run_id"],
            "model_name": model_name,
            "model_class": pending["model_class"],
            "prompt": pending["prompt"],
            "system_prompt": pending.get("system_prompt", ""),
            "user_prompt": pending.get("user_prompt", ""),
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "thinking": thinking_text,
            "latency_seconds": round(latency, 4),
            "finish_reason": finish_reason,
            "error": None,
            **pending.get("metadata", {}),
        }

        self._write(record)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            pending = self._pending.pop(run_id, None)

        if pending is None:
            return

        latency = time.perf_counter() - pending["start_ts"]

        record = {
            "timestamp": pending["timestamp"],
            "run_id": str(run_id),
            "parent_run_id": pending["parent_run_id"],
            "model_name": None,
            "model_class": pending["model_class"],
            "prompt": pending["prompt"],
            "system_prompt": pending.get("system_prompt", ""),
            "user_prompt": pending.get("user_prompt", ""),
            "response": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "thinking": None,
            "latency_seconds": round(latency, 4),
            "finish_reason": None,
            "error": str(error),
            **pending.get("metadata", {}),
        }

        self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()


def load_llm_logs(log_dir: Path) -> "pd.DataFrame":
    """Load all LLM call JSONL files from a results directory into a DataFrame.

    Args:
        log_dir: Path to the logs/ directory of an experiment.

    Returns:
        DataFrame with all LLM calls across simulations.
    """
    import pandas as pd

    frames = []
    for jsonl_file in sorted(log_dir.glob("llm_calls_*.jsonl")):
        df = pd.read_json(jsonl_file, lines=True)
        sim_id = int(jsonl_file.stem.rsplit("_", 1)[-1])
        df["simulation_id"] = sim_id
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
