"""
旅游规划Agent专用状态管理模块
"""
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages
from pydantic.v1 import BaseModel, Field


# @dataclass(kw_only=True)
# class InputState:
#     """Input state defines the interface between the graph and the user (external API)."""
#
#     topic: str
#     "The topic for which the agent is tasked to gather information."
#
#     extraction_schema: dict[str, Any]
#     "The json schema defines the information the agent is tasked with filling out."
#
#     info: Optional[dict[str, Any]] = field(default=None)
#     "The info state tracks the current extracted data for the given topic, conforming to the provided schema. This is primarily populated by the agent."
#
#
# @dataclass(kw_only=True)
# class State(InputState):
#     """A graph's State defines three main things.
#
#     1. The structure of the data to be passed between nodes (which "channels" to read from/write to and their types)
#     2. Default values for each field
#     3. Reducers for the state's fields. Reducers are functions that determine how to apply updates to the state.
#     See [Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) for more information.
#     """
#
#     messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
#     """
#     Messages track the primary execution state of the agent.
#
#     Typically accumulates a pattern of:
#
#     1. HumanMessage - user input
#     2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
#         information
#     3. ToolMessage(s) - the responses (or errors) from the executed tools
#
#         (... repeat steps 2 and 3 as needed ...)
#     4. AIMessage without .tool_calls - agent responding in unstructured
#         format to the user.
#
#     5. HumanMessage - user responds with the next conversational turn.
#
#         (... repeat steps 2-5 as needed ... )
#
#     Merges two lists of messages, updating existing messages by ID.
#
#     By default, this ensures the state is "append-only", unless the
#     new message has the same ID as an existing message.
#
#     Returns:
#         A new list of messages with the messages from `right` merged into `left`.
#         If a message in `right` has the same ID as a message in `left`, the
#         message from `right` will replace the message from `left`.
#         """
#
#     loop_step: Annotated[int, operator.add] = field(default=0)
#
#     # Feel free to add additional attributes to your state as needed.
#     # Common examples include retrieved documents, extracted entities, API connections, etc.
#
#
# @dataclass(kw_only=True)
# class OutputState:
#     """The response object for the end user.
#
#     This class defines the structure of the output that will be provided
#     to the user after the graph's execution is complete.
#     """
#
#     info: dict[str, Any]
#     """
#     A dictionary containing the extracted and processed information
#     based on the user's query and the graph's execution.
#     This is the primary output of the enrichment process.
#     """

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
    # messages: Annotated[List[BaseMessage], add_messages] = field(
    #     default_factory=list,
    #     metadata={"description": "完整对话记录"}
    # )
    messages: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={"description": "完整对话记录"}
    )
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