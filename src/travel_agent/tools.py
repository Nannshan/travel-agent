from typing import List, Dict, Any
import json
from langchain_core.messages import BaseMessage
import chromadb
from pathlib import Path
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def get_embedding_function():
    """获取embedding函数，优先使用GPU"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    

    model = SentenceTransformer("shibing624/text2vec-base-chinese-sentence")
    model.to(device)
    
    # 创建embedding函数
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="shibing624/text2vec-base-chinese-sentence",
        device=device
    )

def rag_retrieval(city: str, preferences: List[str], days: int) -> List[dict]:
    """使用Chroma进行景点检索
    
    Args:
        city: 城市名称
        preferences: 偏好列表，如["自然", "人文"]
        days: 旅行天数
        
    Returns:
        List[dict]: 景点信息列表
    """
    try:
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent.parent.parent
        print(f"当前目录: {current_dir}")
        
        # 确保数据目录存在
        data_dir = current_dir / "static" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"数据目录: {data_dir}")
        
        # 检查CSV文件是否存在
        csv_path = data_dir / "scene.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        print(f"CSV文件路径: {csv_path}")
        
        # 检查CSV文件内容
        try:
            df = pd.read_csv(csv_path)
            print(f"CSV文件列名: {df.columns.tolist()}")
            print(f"CSV文件行数: {len(df)}")
        except Exception as e:
            print(f"CSV文件读取错误: {str(e)}")
            raise
        
        # 初始化Chroma客户端，使用持久化存储
        chroma_db_path = data_dir / "chroma_db"
        print(f"ChromaDB路径: {chroma_db_path}")
        
        # 确保ChromaDB目录存在
        chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
        
        # 使用shibing624/text2vec-base-chinese-sentence
        embedding_function = get_embedding_function()
        
        # 创建或获取collection
        collection_name = "travel_scenes"
        
        try:
            # 尝试获取已存在的collection
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print("成功获取已存在的collection")
        except chromadb.errors.InvalidCollectionException:
            print("正在创建新的向量数据库...")
            # 如果collection不存在，创建新的collection
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            print("Collection创建成功")
            
            # 加载CSV数据
            print(f"正在加载CSV文件: {csv_path}")
            try:
                # 使用pandas读取CSV
                df = pd.read_csv(csv_path)
                documents = []
                
                # 将DataFrame转换为文档格式
                for _, row in df.iterrows():
                    doc = {
                        'page_content': f"{row['name']}。{row['description']}",
                        'metadata': {
                            'name': row['name'],
                            'city': row['city'],
                            'address': row['address'],
                            'score': row['score'],
                            'tags': row['tags'],
                            'features': row['features']
                        }
                    }
                    documents.append(doc)
                
                print(f"成功加载{len(documents)}条数据")
                
                # 准备数据
                texts = []
                metadatas = []
                ids = []
                
                for i, doc in enumerate(documents):
                    # 处理tags字段，移除引号和方括号
                    tags = doc['metadata'].get('tags', '[]')
                    tags = tags.strip('[]').replace('"', '').split(',')
                    tags = [tag.strip() for tag in tags]
                    
                    # 处理features字段
                    features = doc['metadata'].get('features', '[]')
                    features = features.strip('[]').replace('"', '').split(',')
                    features = [feature.strip() for feature in features]
                    
                    # 构建文档内容
                    content = doc['page_content']
                    
                    # 构建元数据（将列表转换为字符串）
                    metadata = {
                        'name': doc['metadata'].get('name', ''),
                        'city': doc['metadata'].get('city', ''),
                        'address': doc['metadata'].get('address', ''),
                        'score': doc['metadata'].get('score', ''),
                        'tags': ','.join(tags),  # 将列表转换为逗号分隔的字符串
                        'features': ','.join(features)  # 将列表转换为逗号分隔的字符串
                    }
                    
                    texts.append(content)
                    metadatas.append(metadata)
                    ids.append(f"doc_{i}")
                
                print(f"正在添加{len(documents)}条数据到向量数据库...")
                # 分批添加数据到Chroma
                batch_size = 1000  # 设置每批数据的大小
                for i in range(0, len(documents), batch_size):
                    batch_end = min(i + batch_size, len(documents))
                    print(f"正在处理第{i+1}到{batch_end}条数据...")
                    
                    # 获取当前批次的数据
                    batch_texts = texts[i:batch_end]
                    batch_metadatas = metadatas[i:batch_end]
                    batch_ids = ids[i:batch_end]
                    
                    # 添加到Chroma
                    collection.add(
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                print("向量数据库创建完成！")
                
            except Exception as e:
                print(f"CSV数据处理错误: {str(e)}")
                raise
        
        # 构建查询
        preferences_text = "、".join(preferences)
        query_text = f"在{city}的特点为{preferences_text}类的旅游景点"
        print(f"查询文本: {query_text}")
        
        # 计算需要返回的景点数量
        n_results = days * 3
        
        # 执行检索，降低相似度阈值
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results * 3,  # 增加返回数量，以便后续过滤
            where={"city": city} if city else None,  # 添加城市过滤
            include=["documents", "metadatas", "distances"]  # 包含相似度分数
        )
        
        print(f"查询到 {len(results['documents'][0])} 个结果")
        
        # 处理结果
        scenes = []
        for i in range(len(results['documents'][0])):
            if results['distances'][0][i] < 0.4:
                metadata = results['metadatas'][0][i]
                try:
                    score = float(metadata.get('score', 0))
                except (ValueError, TypeError):
                    score = 0
                    
                scene = {
                    "name": metadata.get('name', '未知景点'),
                    "score": score,
                    "tags": metadata.get('tags', '').split(',')  # 将字符串转回列表
                }
                scenes.append(scene)
        
        print(f"过滤后剩余 {len(scenes)} 个结果")
        
        # 按评分从高到低排序
        scenes.sort(key=lambda x: x['score'], reverse=True)
        
        # 只返回需要的数量（每天3个景点）
        max_places = days * 3
        scenes = scenes[:max_places]
        
        print("\n最终推荐景点：")
        for scene in scenes:
            print(f"景点: {scene['name']}")
            print(f"评分: {scene['score']}")
            print(f"标签: {', '.join(scene['tags'])}")
            print("-------------------")
        
        return scenes
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return []


def ready_to_generate(messages: List[BaseMessage]) -> bool:
    """
    检查是否已收集齐所有必要信息
    """
    content = messages[-1].content
    jsonRes = json.loads(content)
    finish = jsonRes.get("finish", False)
    return finish

def extract_info(messages: List[BaseMessage]) -> Dict[str, Any]:
    """
    从消息中提取旅行相关信息
    返回包含 city, preferences, start_date, days 的字典
    """
    content = messages[-1].content
    try:
        data = json.loads(content)
        info = data.get("info", {})
        
        return {
            "city": info.get("city", ""),
            "preferences": info.get("preferences", []),
            "start_date": info.get("start_date", ""),
            "days": info.get("days", 1)
        }
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        print(f"解析消息内容失败: {str(e)}")
        # 返回默认值
        return {
            "city": "",
            "preferences": [],
            "start_date": "",
            "days": 1
        }


if __name__ == "__main__":
    print("----------------------")
    print(rag_retrieval("济南",["自然"],2))