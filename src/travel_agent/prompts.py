MAIN_PROMPT = """You are doing web research on behalf of a user. You are trying to figure out this information:

<info>
{info}
</info>

You have access to the following tools:

- `Search`: call a search tool and get back some results
- `ScrapeWebsite`: scrape a website and get relevant notes about the given request. This will update the notes above.
- `Info`: call this when you are done and have gathered all the relevant info

Here is the information you have about the topic you are researching:

Topic: {topic}"""

TRAVEL_PLAN_EXAMPLE = {
    "旅行计划": [
        {
            "日期": "2025年2月26日（第1天）",
            "天气状况": "晴",
            "用户偏好": "人文",
            "行程安排": {
                "上午": {
                    "景点": "",
                    "推荐理由": "",
                    "行程安排": ""
                },
                "下午": {
                    "景点": "",
                    "推荐理由": "",
                    "行程安排": ""
                }
            },
            "总结": ""
        },
        {
            "日期": "2025年2月27日（第2天）",
            "天气状况": "",
            "用户偏好": "",
            "行程安排": {
                "上午": {
                    "景点": "",
                    "推荐理由": "",
                    "行程安排": ""
                },
                "下午": {
                    "景点": "",
                    "推荐理由": "",
                    "行程安排": ""
                }
            },
            "总结": ""
        }
    ]
}

