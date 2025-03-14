from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from src.travel_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""
    max_loops: int = field(
        default=1,
        metadata={
            "description": "The maximum number of interaction loops allowed before the agent terminates."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Load configuration w/ defaults for the given invocation."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
