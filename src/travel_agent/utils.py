import asyncio
from typing import List, Dict, Any

import aiohttp
import datetime

from xpinyin import Pinyin
from langchain_core.messages import BaseMessage, AIMessage
from openai import OpenAI

from src.travel_agent.prompts import SYSTEM_MESSAGE

# 天气api
WEATHER_API_KEY = "63c6f9bef927de53784494ec41141c7f"
BASE_WEATHER_URL = "http://api.openweathermap.org/data/2.5/forecast/daily"
PROXY_URL = "http://127.0.0.1:7890"

def get_current_date() -> str:
    """
    获取当前日期，格式为 YYYY-MM-DD
    """
    return datetime.datetime.now().strftime("%Y-%m-%d")


def is_satisfied(messages: List[BaseMessage]) -> bool:
        # 检查是否为空
        if not messages:
            return False
        # 判断最后一条是否包含"满意"
        return "满意" in messages[-1].content


async def get_weather(city: str, date: str, days: int):
    """
    获取从指定日期开始连续 days 天的天气情况，
    返回结果为 { "YYYY-MM-DD": 天气情况 } 字典。
    天气情况包含：天气描述、最低温度、最高温度和平均温度。
    若请求日期超过当前日期起16天，则返回 "超出查询范围"。
    """
    try:
        city_pinyin = Pinyin().get_pinyin(city, "")
        start_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        today = datetime.date.today()
        max_available_date = today + datetime.timedelta(days=16)

        # 构建请求参数
        params = {
            "q": f"{city_pinyin},CN",
            "appid": WEATHER_API_KEY,
            "cnt": 16,  # 获取16天预报
            "units": "metric",
            "lang": "zh_cn"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(BASE_WEATHER_URL, params=params, proxy=PROXY_URL) as resp:
                if resp.status != 200:
                    print(f"天气API请求失败: {resp.status}")
                    return {(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"): "天气数据获取失败"
                            for i in range(days)}

                data = await resp.json()

        # 构造日期到天气的映射字典
        forecast_dict = {}
        for day_data in data.get("list", []):
            forecast_date = datetime.date.fromtimestamp(day_data["dt"])
            weather_main = day_data["weather"][0]["description"]
            temp_min = day_data["temp"]["min"]
            temp_max = day_data["temp"]["max"]
            temp_avg = day_data["temp"]["day"]
            forecast_dict[forecast_date] = f"{weather_main} 最低{temp_min:.1f}°C 最高{temp_max:.1f}°C 平均{temp_avg:.1f}°C"

        # 根据用户请求的日期区间组装结果
        result = {}
        for i in range(days):
            current_day = start_date + datetime.timedelta(days=i)
            date_str = current_day.strftime("%Y-%m-%d")
            if current_day > max_available_date:
                result[date_str] = "超出查询范围"
            else:
                result[date_str] = forecast_dict.get(current_day, "超出查询范围")

        return result

    except Exception as e:
        print(f"天气API调用异常: {str(e)}")
        return {(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"): "天气数据获取失败"
                for i in range(days)}

def use_deepseek(msgs: List[Any]) -> str:
    # 转换消息格式
    formatted_messages = []
    for msg in msgs:
        if isinstance(msg, dict):
            formatted_messages.append(msg)
        else:
            # 处理 langchain 消息类型
            role = msg.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            formatted_messages.append({
                "role": role,
                "content": msg.content
            })
    
    client = OpenAI(api_key="sk-6bf57ceb23e9467cb5e77f81b57b8c84", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=formatted_messages,
        response_format={
            'type': 'json_object'
        },
        temperature=1.3,
        stream = False
    )
    return response.choices[0].message.content
    
    # # 处理流式响应
    # full_response = ""
    # for chunk in response:
    #     if chunk.choices[0].delta.content is not None:
    #         content = chunk.choices[0].delta.content
    #         full_response += content
    #         # 使用 data: 前缀输出，符合 SSE 格式
    #         # print(f"data: {content}")
    #         yield content
    #
    # # 输出结束标记
    # print("data: [DONE]")
    # return full_response

# 示例运行代码
async def main():
    city = "威海"
    start_date = "2025-03-17"  # 使用未来日期进行测试
    days = 2
    weather_data = await get_weather(city, start_date, days)
    for date_str, weather in weather_data.items():
        print(f"日期: {date_str} : 天气: {weather}")



# 运行示例
if __name__ == "__main__":
    print("---------------------")
    asyncio.run(main())
    # messages = [
    #     SYSTEM_MESSAGE,
    #     {"role": "user", "content": "我想去北京"},
    # ]
    # print(use_deepseek(messages))