from typing import List, Dict, Any
import json
from langchain_core.messages import BaseMessage


async def rag_retrieval(city: str, date: str, days: int) -> List[dict]:
    """RAG景点检索示例"""
    # 实际应连接向量数据库进行检索
    return [{
        "name": "示例景点",
        "description": "著名旅游景点",
        "tags": ["文化", "历史"]
    }]


def ready_to_generate(messages: List[BaseMessage]) -> bool:
    """
    检查是否已收集齐所有必要信息
    """
    try:
        if not messages:
            return False

        content = messages[-1].content.strip()
        # 跳过非 JSON 内容
        if not (content.startswith('{') and content.endswith('}')):
            return False

        # 处理单引号 JSON
        if content.startswith("{'"):
            content = content.replace("'", '"')

        info = json.loads(content)
        # 检查所有必需字段是否存在且有效
        if not all(key in info for key in ["city", "start_date", "days", "preferences"]):
            return False

        # 检查值是否有效
        if not info["city"] or not info["start_date"] or not info["days"] or not info["preferences"]:
            return False

        return True
    except (json.JSONDecodeError, AttributeError, KeyError):
        return False


def extract_info(messages: List[BaseMessage]) -> Dict[str, Any]:
    """
    从消息中提取旅行相关信息
    """
    default_info = {
        "city": "",
        "start_date": "",
        "days": 0,
        "preferences": []
    }

    try:
        if not messages:
            return default_info

        content = messages[-1].content.strip()
        # 跳过非 JSON 内容
        if not (content.startswith('{') and content.endswith('}')):
            return default_info

        # 处理单引号 JSON
        if content.startswith("{'"):
            content = content.replace("'", '"')

        info = json.loads(content)
        return {
            "city": str(info.get("city", "")),
            "start_date": str(info.get("start_date", "")),
            "days": int(info.get("days", 0)),
            "preferences": list(info.get("preferences", []))
        }
    except (json.JSONDecodeError, AttributeError, KeyError, ValueError, TypeError):
        print(f"解析输入时出错: {messages[-1].content if messages else 'No messages'}")
        return default_info
