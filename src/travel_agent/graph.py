import asyncio
import json
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field

from src.travel_agent import prompts
from src.travel_agent.configuration import Configuration
from src.travel_agent.state import InputState, OutputState, State
from src.travel_agent.tools import scrape_website, search, get_weather, rag_retrieval
from src.travel_agent.utils import init_model, use_deepseek, is_last_feedback_satisfied


async def call_agent_model(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Call the primary Language Model (LLM) to decide on the next research action.

    This asynchronous function performs the following steps:
    1. Initializes configuration and sets up the 'Info' tool, which is the user-defined extraction schema.
    2. Prepares the prompt and message history for the LLM.
    3. Initializes and configures the LLM with available tools.
    4. Invokes the LLM and processes its response.
    5. Handles the LLM's decision to either continue research or submit final info.
    """
    # Load configuration from the provided RunnableConfig
    configuration = Configuration.from_runnable_config(config)

    # Define the 'Info' tool, which is the user-defined extraction schema
    info_tool = {
        "name": "Info",
        "description": "Call this when you have gathered all the relevant info",
        "parameters": state.extraction_schema,
    }

    # Format the prompt defined in prompts.py with the extraction schema and topic
    p = configuration.prompt.format(
        info=json.dumps(state.extraction_schema, indent=2), topic=state.topic
    )

    # Create the messages list with the formatted prompt and the previous messages
    messages = [HumanMessage(content=p)] + state.messages

    # Initialize the raw model with the provided configuration and bind the tools
    raw_model = init_model(config)
    model = raw_model.bind_tools([scrape_website, search, info_tool], tool_choice="any")
    response = cast(AIMessage, await model.ainvoke(messages))

    # Initialize info to None
    info = None

    # Check if the response has tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "Info":
                info = tool_call["args"]
                break
    if info is not None:
        # The agent is submitting their answer;
        # ensure it isn't erroneously attempting to simultaneously perform research
        response.tool_calls = [
            next(tc for tc in response.tool_calls if tc["name"] == "Info")
        ]
    response_messages: List[BaseMessage] = [response]
    if not response.tool_calls:  # If LLM didn't respect the tool_choice
        response_messages.append(
            HumanMessage(content="Please respond by calling one of the provided tools.")
        )
    return {
        "messages": response_messages,
        "info": info,
        # Add 1 to the step count
        "loop_step": 1,
    }


class InfoIsSatisfactory(BaseModel):
    """Validate whether the current extracted info is satisfactory and complete."""

    reason: List[str] = Field(
        description="First, provide reasoning for why this is either good or bad as a final result. Must include at least 3 reasons."
    )
    is_satisfactory: bool = Field(
        description="After providing your reasoning, provide a value indicating whether the result is satisfactory. If not, you will continue researching."
    )
    improvement_instructions: Optional[str] = Field(
        description="If the result is not satisfactory, provide clear and specific instructions on what needs to be improved or added to make the information satisfactory."
        " This should include details on missing information, areas that need more depth, or specific aspects to focus on in further research.",
        default=None,
    )


async def reflect(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Validate the quality of the data enrichment agent's output.

    This asynchronous function performs the following steps:
    1. Prepares the initial prompt using the main prompt template.
    2. Constructs a message history for the model.
    3. Prepares a checker prompt to evaluate the presumed info.
    4. Initializes and configures a language model with structured output.
    5. Invokes the model to assess the quality of the gathered information.
    6. Processes the model's response and determines if the info is satisfactory.
    """
    p = prompts.MAIN_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), topic=state.topic
    )
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"{reflect.__name__} expects the last message in the state to be an AI message with tool calls."
            f" Got: {type(last_message)}"
        )
    messages = [HumanMessage(content=p)] + state.messages[:-1]
    presumed_info = state.info
    checker_prompt = """I am thinking of calling the info tool with the info below. \
Is this good? Give your reasoning as well. \
You can encourage the Assistant to look at specific URLs if that seems relevant, or do more searches.
If you don't think it is good, you should be very specific about what could be improved.

{presumed_info}"""
    p1 = checker_prompt.format(presumed_info=json.dumps(presumed_info or {}, indent=2))
    messages.append(HumanMessage(content=p1))
    raw_model = init_model(config)
    bound_model = raw_model.with_structured_output(InfoIsSatisfactory)
    response = cast(InfoIsSatisfactory, await bound_model.ainvoke(messages))
    if response.is_satisfactory and presumed_info:
        return {
            "info": presumed_info,
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content="\n".join(response.reason),
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="success",
                )
            ],
        }
    else:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content=f"Unsatisfactory response:\n{response.improvement_instructions}",
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="error",
                )
            ]
        }


def route_after_agent(
    state: State,
) -> Literal["reflect", "tools", "call_agent_model", "__end__"]:
    """Schedule the next node after the agent's action.

    This function determines the next step in the research process based on the
    last message in the state. It handles three main scenarios:

    1. Error recovery: If the last message is unexpectedly not an AIMessage.
    2. Info submission: If the agent has called the "Info" tool to submit findings.
    3. Continued research: If the agent has called any other tool.
    """
    last_message = state.messages[-1]

    # "If for some reason the last message is not an AIMessage (due to a bug or unexpected behavior elsewhere in the code),
    # it ensures the system doesn't crash but instead tries to recover by calling the agent model again.
    if not isinstance(last_message, AIMessage):
        return "call_agent_model"
    # If the "Into" tool was called, then the model provided its extraction output. Reflect on the result
    if last_message.tool_calls and last_message.tool_calls[0]["name"] == "Info":
        return "reflect"
    # The last message is a tool call that is not "Info" (extraction output)
    else:
        return "tools"


def route_after_checker(
    state: State, config: RunnableConfig
) -> Literal["__end__", "call_agent_model"]:
    """Schedule the next node after the checker's evaluation.

    This function determines whether to continue the research process or end it
    based on the checker's evaluation and the current state of the research.
    """
    configurable = Configuration.from_runnable_config(config)
    last_message = state.messages[-1]

    if state.loop_step < configurable.max_loops:
        if not state.info:
            return "call_agent_model"
        if not isinstance(last_message, ToolMessage):
            raise ValueError(
                f"{route_after_checker.__name__} expected a tool messages. Received: {type(last_message)}."
            )
        if last_message.status == "error":
            # Research deemed unsatisfactory
            return "call_agent_model"
        # It's great!
        return "__end__"
    else:
        return "__end__"


# 核心处理节点
async def generate(
        state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """处理用户输入并生成旅行推荐方案"""
    # 初始化消息列表
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个热心的旅行计划师，擅长根据天气情况、兴趣爱好推荐旅游景点。"
                "请根据提供的城市、开始时间、旅行天数、旅行偏好等信息推荐合理的旅行规划。"
                f"""无论是初次生成计划还是后续调整都请严格按照如下旅行规划json格式：{prompts.TRAVEL_PLAN_EXAMPLE}"""
            ),
        },
    ]

    configuration = Configuration.from_runnable_config(config)

    # 1. 从state获取用户输入参数
    weather_data = await get_weather(state.city, state.start_date, state.days)

    # 2. 调用RAG模型获取候选景点数据(根据地点和用户偏好，考虑景点的评分)
    # 根据天数决定景点数量，一天选三个
    candidate_pool = await rag_retrieval(state.city, state.preferences, state.days)

    # 2. 获取候选景点数据, 优先从 state 中获取 candidate_pool，否则构造一个示例列表
    candidate_pool: List[Dict[str, Any]] = getattr(state, "candidate_pool", [])
    if not candidate_pool:
        candidate_pool = [
            {"name": "故宫博物院",  "category": "历史"},
            {"name": "颐和园",  "category": "自然"},
            {"name": "天坛公园",  "category": "文化"},
            {"name": "798艺术区",  "category": "艺术"},
            {"name": "王府井大街",  "category": "购物"},
            {"name": "南锣鼓巷",  "category": "文化"},
            {"name": "什刹海",  "category": "休闲"},
            {"name": "北京动物园", "category": "家庭游"},
        ]

    # 3. 初始化大模型
    # raw_model = init_model(config)
    # model = raw_model.bind_tools([])

    daily_recommendations = {}
    attraction_index = 0  # 用于遍历候选景点

    # 4. 生成旅行规划
    prompt = f"""
    - 开始日期：{state.start_date}
    - 持续天数：{state.days}
    - 天气状况：{weather_data}
    - 用户偏好：{state.preferences}
    - 候选景点信息：
    {candidate_pool}
    请整合以上信息，从候选景点选择最大化满足用户偏好的景点，生成一个连贯且详细的旅行规划，包括各景点的推荐理由及合理的行程安排，景点不要重复选择，要求描述清晰且具有吸引力。"""

    messages.append({"role": "user", "content": prompt})
    # 调用大模型生成旅行规划
    response = use_deepseek(messages)
    messages.append({"role": "assistant", "content": response})

    # 保存当天规划及对应的候选景点
    daily_recommendations =  response

    return {
        "messages": messages,
        "weather_data": weather_data,
        "candidate_pool": candidate_pool,
        "daily_recommendations": daily_recommendations,
    }


def handle_feedback(state: State, *, config: Optional[RunnableConfig] = None) -> State:
    """
    根据用户反馈调整旅游推荐规划。
    """
    # 初始化消息列表，若 state 中已有对话记录则复用，否则初始化一个新的列表
    # messages: List[Dict[str, Any]] = getattr(state, "messages", [])
    messages = state.messages

    # 记录用户反馈
    messages.append({"role": "user", "content": f"用户反馈：{state.feedback}"})

    # 获取候选景点列表
    # candidate_pool: List[Dict[str, Any]] = getattr(state, "candidate_pool", [])
    candidate_pool = state.candidate_pool

    # 构造大模型调用的提示词，要求生成更新后的旅行规划
    prompt = f"""请根据以下反馈信息和候选景点状态为我调整旅行规划：
- 用户反馈：{state.feedback}"""

    messages.append({"role": "user", "content": prompt})
    # 调用大模型（假设 use_deepseek 为调用大模型的函数）
    response = use_deepseek(messages)
    messages.append({"role": "assistant", "content": response})

    # 更新 state 中的候选景点和消息记录
    state.candidate_pool = candidate_pool
    state.messages = messages
    state.updated_plan = response

    return state

def human_assistance(state: State) -> State:
    """咨询用户进行下一步."""
    res = interrupt("请问对当前旅行计划是否满意？若不满意，请告诉我你想去哪些景点或者不想去哪些景点。")
    state.feedback.append(res)
    return state

def finalize(state: State) -> OutputState:
    # 优先使用updated_plan，其次使用daily_recommendations
    itinerary = (
        state.updated_plan
        if state.updated_plan and state.updated_plan.strip()
        else state.daily_recommendations
    )
    return OutputState(optimized_itinerary=itinerary)


# 路由逻辑
def route(state: State) -> Literal["handle_feedback", "finalize"]:
    return "finalize" if is_last_feedback_satisfied(state.feedback) else "handle_feedback"

# 构建工作流
workflow = StateGraph(
    State, input=InputState, output=OutputState, config_schema=Configuration
)

# 添加节点
workflow.add_node(generate)
workflow.add_node(handle_feedback)
workflow.add_node(finalize)
workflow.add_node(human_assistance)

# 设置边
workflow.add_edge("__start__", "generate")
workflow.add_edge("generate", "human_assistance")
workflow.add_edge("handle_feedback", "human_assistance")
workflow.add_edge("finalize", "__end__")
workflow.add_conditional_edges(
    "human_assistance",
    route,
)
memory = MemorySaver()
# 初始化
graph = workflow.compile(checkpointer=memory)
graph.name = "TravelRecommendation"

async def run_stream():
    # 确保这个代码在异步函数中
    async for chunk in graph.astream(initial_input, thread, stream_mode="updates"):
        # 处理每个 chunk
        print(chunk)
    while True:
        ans = input("是否满意？")
        for chunk in graph.stream(Command(resume = ans), thread, stream_mode="updates"):
            # 处理每个 chunk
            print(chunk)
        # 根据反馈判断是否进入 finalze 节点，退出 while 循环
        if "满意" in ans:
            print("反馈满意，退出循环。")
            break  # 退出 while 循环


# 在事件循环中运行异步函数
if __name__ == "__main__":
    initial_input = {"city": "北京", "start_date": "2025-2-27", "days": 1, "preferences": ["文化"]}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}
    asyncio.run(run_stream())  # 运行异步任务