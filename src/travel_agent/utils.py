from typing import Optional, List, Dict, Any
from openai import OpenAI


def is_last_feedback_satisfied(feedback: Optional[List[str]]) -> bool:
        # 检查 feedback 列表是否为空
        if not feedback:
            return False
        # 获取最后一条反馈
        last_feedback = feedback[-1]
        # 判断最后一条反馈是否包含"满意"
        return "满意" in last_feedback


def use_deepseek(msgs: List[Dict[str, Any]]) -> str:
    client = OpenAI(api_key="sk-6bf57ceb23e9467cb5e77f81b57b8c84", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=msgs,
        # response_format={
        #     'type': 'json_object'
        # },
        frequency_penalty = -0.8,
        max_tokens = 8000,
        temperature=1.3,
        stream=True
    )
    
    # 处理流式响应
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            # 使用 data: 前缀输出，符合 SSE 格式
            # print(f"data: {content}")
            yield content
    
    # 输出结束标记
    print("data: [DONE]")
    return full_response

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "你是一个热心的旅行计划师, 擅长根据天气情况、兴趣爱好推荐旅游景点。请根据用户提供的城市，开始时间，旅行天数，旅行偏好等信息，为用户推荐旅游景点。"},
        {"role": "assistant", "content": "请问你想去哪旅行？什么时间？有什么旅行偏好吗？"},
        {"role": "user", "content": "你好"},
    ]
    # 测试流式响应
    for response_chunk in use_deepseek(messages):
        print(response_chunk, end="", flush=True)