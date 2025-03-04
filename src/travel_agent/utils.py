from typing import Optional, List, Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from openai import OpenAI

from src.travel_agent.configuration import Configuration


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def init_model(config: Optional[RunnableConfig] = None) -> BaseChatModel:
    """Initialize the configured chat model."""
    configuration = Configuration.from_runnable_config(config)
    fully_specified_name = configuration.model
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)


def is_last_feedback_satisfied(feedback: Optional[List[str]]) -> bool:
        # 检查 feedback 列表是否为空
        if not feedback:
            return False
        # 获取最后一条反馈
        last_feedback = feedback[-1]
        # 判断最后一条反馈是否包含“满意”
        return "满意" in last_feedback


def use_deepseek(msgs: List[Dict[str, Any]]) -> str:
    client = OpenAI(api_key="sk-6bf57ceb23e9467cb5e77f81b57b8c84", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=1.3,
        messages=msgs,
        stream=False
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "你是一个热心的旅行计划师, 擅长根据天气情况、兴趣爱好推荐旅游景点。请根据用户提供的城市，开始时间，旅行天数，旅行偏好等信息，为用户推荐旅游景点。"},
        {"role": "assistant", "content": "请问你想去哪旅行？什么时间？有什么旅行偏好吗？"},
        {"role": "user", "content": "你好"},
    ]
    use_deepseek(messages)