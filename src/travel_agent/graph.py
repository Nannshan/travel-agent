import asyncio
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

from src.travel_agent.configuration import Configuration
from src.travel_agent.prompts import PRE_MESSAGE_PROMPT, USER_PLAN_PROMPT
from src.travel_agent.state import InputState, OutputState, State
from src.travel_agent.tools import get_weather, rag_retrieval
from src.travel_agent.utils import use_deepseek, is_last_feedback_satisfied

# # 处理用户输入
# def handle_input(state: State) -> Dict[str, Any]:

# 核心处理节点
async def generate(
        state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """处理用户输入并生成旅行推荐方案"""
    # 初始化消息列表
    messages = [PRE_MESSAGE_PROMPT]

    configuration = Configuration.from_runnable_config(config)

    # 1. 从state获取用户输入参数
    weather_data = await get_weather(state.city, state.start_date, state.days)

    # 2. 调用RAG模型获取候选景点数据(根据地点和用户偏好，考虑景点的评分)
    # 根据天数决定景点数量，一天选三个
    candidate_pool = await rag_retrieval(state.city, state.preferences, state.days)

    # 3. 获取候选景点数据, 优先从 state 中获取 candidate_pool，否则构造一个示例列表
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

    # 4. 生成旅行规划
    prompt = USER_PLAN_PROMPT.format(
        start_date=state.start_date,
        days=state.days,
        weather_data=weather_data,
        preferences=state.preferences,
        candidate_pool=candidate_pool
    )
    messages.append(HumanMessage(content=prompt))
    # 调用大模型生成旅行规划
    response = cast(AIMessage, await llm.ainvoke(messages))
    # response = use_deepseek(messages)
    messages.append(AIMessage(content=response.content))

    # 保存当天规划及对应的候选景点
    daily_recommendations =  response.content

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
    interrupt("对当前旅程满意吗？如果有其他想法，请随时与我沟通。")
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

llm = ChatDeepSeek(
    model="deepseek-chat",
    timeout=None,
    max_retries=2,
    temperature=1.3
)

# 添加节点
workflow.add_node(generate)
workflow.add_node(handle_feedback)
workflow.add_node(finalize)
workflow.add_node(human_assistance)

# 设置边
workflow.add_edge("__start__", "generate")
workflow.add_edge("generate", "human_assistance")
workflow.add_edge("human_assistance", "__end__")
# workflow.add_edge("handle_feedback", "human_assistance")
# workflow.add_edge("finalize", "__end__")
# workflow.add_conditional_edges(
#     "human_assistance",
#     route,
# )
memory = MemorySaver()
# 初始化
graph = workflow.compile(checkpointer=memory)
graph.name = "travel-agent"

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
        # 根据反馈判断是否进入 finalize 节点，退出 while 循环
        if "满意" in ans:
            print("反馈满意，退出循环。")
            break  # 退出 while 循环


# 在事件循环中运行异步函数
if __name__ == "__main__":
    initial_input = {"city": "北京", "start_date": "2025-03-13", "days": 1, "preferences": ["文化"]}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}
    asyncio.run(run_stream())  # 运行异步任务