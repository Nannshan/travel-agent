from typing import List, Dict, Any
import json
from langchain_core.messages import BaseMessage
import chromadb
from pathlib import Path
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd


def rag_retrieval(city: str, preferences: List[str], days: int) -> List[dict]:
    """使用Chroma进行景点检索
    
    Args:
        city: 城市名称
        preferences: 偏好列表，如["自然", "人文"]
        days: 旅行天数，返回景点数量为天数的4倍
        
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
        
        chroma_client = chromadb.Client(Settings(
            persist_directory=str(chroma_db_path),
            anonymized_telemetry=False
        ))
        
        # 使用sentence-transformers进行向量化
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 创建或获取collection
        collection_name = "travel_scenes"
        
        # 检查collection是否存在
        collections = chroma_client.list_collections()
        collection_exists = any(col.name == collection_name for col in collections)
        
        if not collection_exists:
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
                    
                    # 构建元数据
                    metadata = {
                        'name': doc['metadata'].get('name', ''),
                        'city': doc['metadata'].get('city', ''),
                        'address': doc['metadata'].get('address', ''),
                        'score': doc['metadata'].get('score', ''),
                        'tags': tags,
                        'features': features
                    }
                    
                    texts.append(content)
                    metadatas.append(metadata)
                    ids.append(f"doc_{i}")
                
                print(f"正在添加{len(documents)}条数据到向量数据库...")
                # 添加到Chroma，使用批量添加以提高效率
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                print("向量数据库创建完成！")
                
            except Exception as e:
                print(f"CSV数据处理错误: {str(e)}")
                raise
        else:
            print("获取已存在的collection...")
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print("成功获取collection")
        
        # 构建查询
        preferences_text = "、".join(preferences)
        query_text = f"在{city}的{preferences_text}类旅游景点"
        print(f"查询文本: {query_text}")
        
        # 计算需要返回的景点数量
        n_results = days * 3
        
        # 执行检索，添加相似度阈值
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"city": city} if city else None,  # 添加城市过滤
            where_document={"$contains": preferences_text} if preferences else None,  # 添加偏好过滤
            include=["documents", "metadatas", "distances"]  # 包含相似度分数
        )
        
        # 处理结果
        scenes = []
        for i in range(len(results['documents'][0])):
            # 只添加相似度大于0.7的结果
            if results['distances'][0][i] < 0.7:
                metadata = results['metadatas'][0][i]
                scene = {
                    "name": metadata.get('name', '未知景点'),
                    "description": results['documents'][0][i],
                    "address": metadata.get('address', ''),
                    "score": metadata.get('score', ''),
                    "tags": metadata.get('tags', []),
                    "features": metadata.get('features', []),
                    "similarity": 1 - results['distances'][0][i]  # 转换为相似度分数
                }
                scenes.append(scene)
        
        # 按相似度排序
        scenes.sort(key=lambda x: x['similarity'], reverse=True)
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
    print(rag_retrieval("北京",["自然"],1))