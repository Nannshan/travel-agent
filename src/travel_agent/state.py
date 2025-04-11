"""
旅游规划Agent专用状态管理模块
"""
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic.v1 import Field
from scripts.regsetup import description


# --------------------------
# 状态机定义
# --------------------------

@dataclass(kw_only=True)
class InputState:
    """初始输入状态"""
    chat_id : str = Field(
        description="用户id",
        example="d0ff21fc-0505-4655-9cc5-971731d6d666",
    )
    initial_input: str = Field(
        description="初始输入文本",
        example = "hello"
    )

@dataclass(kw_only=True)
class State(InputState):
    """规划过程状态机"""
    city: str = Field(example="北京", min_length=2, default="")
    start_date: str = Field(example="2024-10-01", description="开始日期，格式为yyyy-mm-dd", default="")
    days: Annotated[int, Field(gt=1, le=7)] = Field(default=0)
    preferences: List[str] = Field(
        example={"文化, 自然, 娱乐, 美食"},
        default_factory=list
    )

    current_date: str = field(
        default="",
        metadata={"description": "当前日期"}
    )
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    weather_data: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "多日天气预报"}
    )
    candidate_pool: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={"description": "候选景点池"}
    )
    plans: str = field(
        default="",
        metadata={"description": "当前行程草案"}
    )
    feed_back: List[str] = field(
        default_factory=list,
        metadata={"description": "用户反馈历史"}
    )

@dataclass(kw_only=True)
class OutputState:
    """最终输出状态"""
    optimized_itinerary: str = Field(
        description="包含每日详细行程",
    )