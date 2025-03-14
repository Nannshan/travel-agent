from typing import List

import aiohttp
import asyncio
import datetime

from xpinyin import Pinyin

# 天气api
WEATHER_API_KEY = "63c6f9bef927de53784494ec41141c7f"
BASE_WEATHER_URL = "http://api.openweathermap.org/data/2.5/forecast/daily"
PROXY_URL = "http://127.0.0.1:7890"


async def rag_retrieval(city: str, date: str, days: int) -> List[dict]:
    """RAG景点检索示例"""
    # 实际应连接向量数据库进行检索
    return [{
        "name": "示例景点",
        "description": "著名旅游景点",
        "tags": ["文化", "历史"]
    }]


async def get_weather(city: str, date: str, days: int):
    """
    获取从指定日期开始连续 days 天的天气情况，
    返回结果为 { "YYYY-MM-DD": 天气情况 } 字典。
    若请求日期超过当前日期起15天，则返回 "超出查询范围"。
    """
    # 解析起始日期，假设格式为 "YYYY-MM-DD"
    city_pinyin = Pinyin().get_pinyin(city, "")
    start_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    today = datetime.date.today()
    max_available_date = today + datetime.timedelta(days=15)

    # 请求 API，获取最多 16 天的预报数据
    cnt = 16  # OpenWeatherMap 16天预报
    params = {
        "q": city_pinyin,
        "appid": WEATHER_API_KEY,
        "cnt": cnt,
        "lang": "zh_cn"  # 返回中文描述（若需要）
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(BASE_WEATHER_URL, params=params, proxy=PROXY_URL) as resp:
            data = await resp.json()

    # 构造日期到天气的映射字典
    forecast_dict = {}
    for day_data in data.get("list", []):
        # dt 字段为 Unix 时间戳，转换为日期
        forecast_date = datetime.date.fromtimestamp(day_data["dt"])
        # 获取天气主状况
        weather_main = day_data["weather"][0]["description"]
        forecast_dict[forecast_date] = weather_main

    # 根据用户请求的日期区间组装结果
    result = {}
    for i in range(days):
        current_day = start_date + datetime.timedelta(days=i)
        date_str = current_day.strftime("%Y-%m-%d")
        # 如果超出可查询范围，则返回 “超出查询范围”
        if current_day > max_available_date:
            result[date_str] = "超出查询范围"
        else:
            # 若 API 返回数据中没有该日期，则也返回 “超出查询范围”
            result[date_str] = forecast_dict.get(current_day, "超出查询范围")
    return result


# 示例运行代码
async def main():
    city = "威海"
    start_date = "2025-03-05"  # 示例起始日期
    days = 2
    weather_data = await get_weather(city, start_date, days)
    for date_str, weather in weather_data.items():
        print(f"日期: {date_str} : 天气: {weather}")


# 运行示例
if __name__ == "__main__":
    asyncio.run(main())