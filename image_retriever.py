"""
图片检索工具模块

功能：
1. 加载图片向量数据库
2. 将文本 query 映射到图片向量空间
3. 返回图片 ID 与其原始 chunk 关联信息
"""

import json
import os
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


CHROMA_PERSIST_DIR = "./data/chroma_image_db"
COLLECTION_NAME = "manual_images"
EMBEDDING_MODEL_NAME = os.getenv("IMAGE_EMBEDDING_MODEL_NAME", "clip-ViT-B-32")


class ImageRetriever:
    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        print("🔧 初始化图片检索器...")
        print(f"   📥 加载图片 Embedding 模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"   📂 连接图片向量数据库: {persist_dir}")
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"   ✓ 图片 Collection '{collection_name}' 加载成功")
            print(f"   ✓ 总记录数: {self.collection.count()}")
        except Exception as exc:
            raise RuntimeError(
                f"无法加载图片 Collection '{collection_name}'，请先运行 build_image_vector_db.py。错误: {exc}"
            )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        parsed_results = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            try:
                source_chunk_ids = json.loads(metadata.get("source_chunk_ids", "[]"))
            except json.JSONDecodeError:
                source_chunk_ids = []
            try:
                source_products = json.loads(metadata.get("source_products", "[]"))
            except json.JSONDecodeError:
                source_products = []

            parsed_results.append(
                {
                    "image_id": metadata.get("image_id", results["ids"][0][i]),
                    "image_path": metadata.get("image_path", ""),
                    "product": metadata.get("product", "unknown"),
                    "source_products": source_products,
                    "source_chunk_ids": source_chunk_ids,
                    "source_preview": metadata.get("source_preview", ""),
                    "distance": results["distances"][0][i],
                }
            )

        return parsed_results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 图片检索器测试")
    print("=" * 60)

    retriever = ImageRetriever()
    test_queries = [
        "电钻指示灯闪烁是什么意思？",
        "How do I turn on or off the water supply button on my boat?",
        "尺码表怎么看？",
    ]
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        for index, item in enumerate(retriever.search(query, top_k=3), 1):
            print(
                f"   {index}. image_id={item['image_id']} "
                f"distance={item['distance']:.4f} "
                f"product={item['product']}"
            )
