"""
旅游规划Agent专用状态管理模块
"""
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic.v1 import BaseModel, Field

# --------------------------
# 状态机定义
# --------------------------

@dataclass(kw_only=True)
class InputState:
    """初始输入状态"""
    city: str = Field(example="北京", min_length=2)
    start_date: str = Field(example="2024-10-01", description="开始日期，格式为yyyy-mm-dd")
    days: Annotated[int, Field(gt=1, le=7)]
    preferences: List[str] = Field(
        example={"文化, 自然, 娱乐, 美食"}
    )

@dataclass(kw_only=True)
class State(InputState):
    """规划过程状态机"""
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    # messages: List[Dict[str, Any]] = field(
    #     default_factory=list,
    #     metadata={"description": "完整对话记录"}
    # )
    weather_data: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "多日天气预报"}
    )
    candidate_pool: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={"description": "候选景点池"}
    )
    daily_recommendations: str = field(
        default= "",
        metadata={"description": "当前行程草案"}
    )
    updated_plan: str = None
    feedback: List[str] = field(
        default_factory=list,
        metadata={"description": "用户反馈"}
    )

@dataclass(kw_only=True)
class OutputState:
    """最终输出状态"""
    optimized_itinerary: str = Field(
        description="包含每日详细行程",
    )