from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from src.travel_agent.configuration import Configuration
from src.travel_agent.prompts import PRE_MESSAGE_PROMPT, USER_PLAN_PROMPT, SYSTEM_MESSAGE, FEEDBACK_PROMPT
from src.travel_agent.state import InputState, OutputState, State
from src.travel_agent.tools import  rag_retrieval, extract_info, ready_to_generate
from src.travel_agent.utils import use_deepseek, is_satisfied, get_weather, get_current_date


# 回应用户输入
async def handle_input(state: State) -> State:
    # 初始化消息列表
    if not state.messages:
        date = get_current_date()
        # 更新 current_date
        state.current_date = date
        sres = f"当前日期：{state.current_date}, 对话id：{state.chat_id}"
        messages = [SYSTEM_MESSAGE, SystemMessage(content=sres), HumanMessage(content=state.initial_input)]
    else:
        messages = state.messages

    # 调用大模型获取规划条件
    # response = cast(AIMessage, await llm.ainvoke(messages))
    response = use_deepseek(messages)
    messages.append(AIMessage(content=response))

    state.messages = messages
    return state

# 用户输入
def user_input(state: State) -> State:
    res = interrupt(state.messages[-1].content)
    state.messages.append(HumanMessage(content=res))

    return state

# 核心处理节点
async def generate(state: State) -> State:
    # 初始化消息列表
    messages = [PRE_MESSAGE_PROMPT]

    # 解析用户输入
    res = extract_info(state.messages)
    state.city = res["city"]
    state.days = res["days"]
    state.start_date = res["start_date"]
    state.preferences = res["preferences"]

    # 从state获取用户输入参数
    weather_data = await get_weather(state.city, state.start_date, state.days)
    state.weather_data = weather_data

    # 调用RAG模型获取候选景点数据(根据地点和用户偏好，考虑景点的评分)
    # 根据天数决定景点数量，一天选三个
    candidate_pool = rag_retrieval(state.city, state.preferences, state.days)
    state.candidate_pool = candidate_pool

    # 生成旅行规划
    prompt = USER_PLAN_PROMPT.format(
        start_date=state.start_date,
        days=state.days,
        weather_data=weather_data,
        preferences=state.preferences,
        candidate_pool=candidate_pool
    )
    messages.append(HumanMessage(content=prompt))
    # 调用大模型生成旅行规划
    # response = cast(AIMessage, await llm.ainvoke(messages))
    response = use_deepseek(messages)
    messages.append(AIMessage(content=response))

    # 保存当天规划及对应的候选景点
    state.messages = messages
    state.plans = response

    return state


def handle_feedback(state: State) -> State:
    """
    根据用户反馈调整旅游推荐规划。
    """

    # 获取候选景点列表
    # candidate_pool: List[Dict[str, Any]] = getattr(state, "candidate_pool", [])

    # 调用大模型（假设 use_deepseek 为调用大模型的函数）
    response = use_deepseek(state.messages)
    state.messages.append(AIMessage(content=response))

    # 更新
    state.plans = response

    return state

def human_assistance(state: State) -> State:
    """咨询用户进行下一步."""
    # 创建新的消息列表，只包含反馈相关的消息
    state.messages = []
    # 获取用户反馈
    res = interrupt("对当前旅程满意吗？如果有其他想法，请随时与我沟通。")
    state.feed_back.append(res)

    feedback_messages = [SystemMessage(content=FEEDBACK_PROMPT["content"].format(
        feed_back=state.feed_back,
        pre_plan=state.plans
    )), AIMessage(content="对当前旅程满意吗？如果有其他想法，请随时与我沟通。"), HumanMessage(content=res)]

    # 更新state中的消息
    state.messages = feedback_messages
    return state

def finalize(state: State) -> OutputState:
    itinerary = state.plans
    return OutputState(optimized_itinerary=itinerary)


# 路由逻辑
def route_after_human_assistance(state: State) -> Literal["handle_feedback", "finalize"]:
    return "finalize" if is_satisfied(state.messages) else "handle_feedback"

# 判断继续获取或者生成规划
def route_after_handle_input(state: State) -> Literal["generate", "user_input"]:
    if ready_to_generate(state.messages):
        return "generate"
    else:
        return "user_input"

# 构建工作流
workflow = StateGraph(
    State, input=InputState, output=OutputState, config_schema=Configuration
)

DEEPSEEK_API_KEY="sk-6bf57ceb23e9467cb5e77f81b57b8c84"
llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    temperature=1.5
)

# 添加节点
workflow.add_node(handle_input)
workflow.add_node(user_input)
workflow.add_node(generate)
workflow.add_node(handle_feedback)
workflow.add_node(finalize)
workflow.add_node(human_assistance)

# 设置边
workflow.add_edge("__start__", "handle_input")
workflow.add_edge("user_input", "handle_input")
workflow.add_edge("generate", "human_assistance")
workflow.add_edge("handle_feedback", "human_assistance")
workflow.add_edge("finalize", "__end__")
workflow.add_conditional_edges(
    "handle_input",
    route_after_handle_input,
)
workflow.add_conditional_edges(
    "human_assistance",
    route_after_human_assistance,
)
memory = MemorySaver()
# 初始化
graph = workflow.compile(checkpointer=memory)
graph.name = "travel-agent"




# async def run_stream():
#     # 确保这个代码在异步函数中
#     async for chunk in graph.astream(initial_input, thread, stream_mode="updates"):
#         # 处理每个 chunk
#         print(chunk)
#     while True:
#         ans = input("是否满意？")
#         for chunk in graph.stream(Command(resume = ans), thread, stream_mode="updates"):
#             # 处理每个 chunk
#             print(chunk)
#         # 根据反馈判断是否进入 finalize 节点，退出 while 循环
#         if "满意" in ans:
#             print("反馈满意，退出循环。")
#             break  # 退出 while 循环
#
#
# # 在事件循环中运行异步函数
# if __name__ == "__main__":
#     initial_input = {"city": "北京", "start_date": "2025-03-13", "days": 1, "preferences": ["文化"]}
#
#     # Thread
#     thread = {"configurable": {"thread_id": "1"}}
#     asyncio.run(run_stream())  # 运行异步任务