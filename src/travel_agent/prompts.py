TRAVEL_PLAN_EXAMPLE = {
    "travel_plan": [
        {
            "subject": "",
            "date": "2025年2月26日",
            "weather_condition": "晴",
            "user_preference": "自然",
            "itinerary": {
                "morning": {
                    "attraction": "",
                    "recommendation_reason": "",
                    "arrangement": ""
                },
                "afternoon": {
                    "attraction": "",
                    "recommendation_reason": "",
                    "arrangement": ""
                }
            },
            "summary": ""
        },
        {
            "subject": "",
            "date": "2025年2月27日",
            "weather_condition": "小雨",
            "user_preference": "自然",
            "itinerary": {
                "morning": {
                    "attraction": "",
                    "recommendation_reason": "",
                    "arrangement": ""
                },
                "afternoon": {
                    "attraction": "",
                    "recommendation_reason": "",
                    "arrangement": ""
                }
            },
            "summary": ""
        }
    ]
}

PRE_MESSAGE_PROMPT = {
        "role": "system",
        "content": (
        "你是一个热心的旅行计划师，擅长根据天气情况、兴趣爱好推荐旅游景点。"
        "请根据提供的城市、开始时间、旅行天数、旅行偏好等信息推荐合理的旅行规划, 主题为一天的简要总结7字左右，每个景点推荐理由110字左右，行程安排60字左右, 总结40字左右。"
        f"""格式参考：{TRAVEL_PLAN_EXAMPLE},不要输出MarkDown格式。"""
    ),
}

USER_PLAN_PROMPT = """
    - 开始日期：{start_date}
    - 持续天数：{days}
    - 天气状况：{weather_data}
    - 用户偏好：{preferences}
    - 候选景点信息：
    {candidate_pool}
    请整合以上信息，从候选景点选择最大化满足用户偏好的景点，生成一个连贯且详细的旅行规划，包括各景点的推荐理由及合理的行程安排，景点不要重复选择，要求描述清晰且具有吸引力，请输出严格json格式。"""