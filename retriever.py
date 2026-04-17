"""
向量检索工具模块

功能：
1. 加载已构建的 ChromaDB 向量数据库
2. 提供 `ManualRetriever` 类封装检索逻辑
3. 支持语义相似度检索，返回结构化结果

使用示例：
    retriever = ManualRetriever()
    results = retriever.search("电钻指示灯闪烁是什么意思？", top_k=5)

作者：Claude Code
"""

import json
import os
from typing import List, Dict, Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer


# ============================================
# 配置常量
# ============================================

# ChromaDB 持久化目录
# ⚠️ 重要：必须与 build_vector_db.py 中的配置保持一致！
CHROMA_PERSIST_DIR = "./data/chroma_db_m3"

# Collection 名称
COLLECTION_NAME = "manuals_qa_m3"

# Embedding 模型名称（多语言 SOTA 模型）
# BAAI/bge-m3: 多语言支持，向量维度 1024，中英文效果俱佳
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "18"))


# ============================================
# 检索器类
# ============================================

class ManualRetriever:
    """
    产品手册向量检索器

    封装 ChromaDB 向量检索功能，提供语义相似度搜索接口

    Attributes:
        model: SentenceTransformer Embedding 模型
        collection: ChromaDB Collection
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
        enable_rerank: bool = ENABLE_RERANK,
        reranker_model_name: str = RERANKER_MODEL_NAME
    ):
        """
        初始化检索器

        Args:
            persist_dir: ChromaDB 持久化目录
            collection_name: Collection 名称
            model_name: Embedding 模型名称
        """
        print(f"🔧 初始化检索器...")
        self.enable_rerank = enable_rerank
        self.reranker = None

        # 加载 Embedding 模型
        print(f"   📥 加载 Embedding 模型: {model_name}")
        self.model = SentenceTransformer(model_name)

        if self.enable_rerank:
            try:
                print(f"   📥 加载 Reranker 模型: {reranker_model_name}")
                self.reranker = CrossEncoder(reranker_model_name)
                print("   ✓ Reranker 加载成功")
            except Exception as e:
                print(f"   ⚠️  Reranker 加载失败，回退为纯向量检索: {e}")
                self.enable_rerank = False
                self.reranker = None

        # 连接 ChromaDB
        print(f"   📂 连接向量数据库: {persist_dir}")
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # 获取 Collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"   ✓ Collection '{collection_name}' 加载成功")
            print(f"   ✓ 总记录数: {self.collection.count()}")
        except Exception as e:
            raise RuntimeError(
                f"无法加载 Collection '{collection_name}'，"
                f"请先运行 build_vector_db.py 构建向量数据库。错误: {e}"
            )

    def _parse_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        parsed_results = []

        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            images_str = metadata.get('images', '[]')

            try:
                images = json.loads(images_str)
            except json.JSONDecodeError:
                images = []

            parsed_results.append(
                {
                    'chunk_id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'images': images,
                    'product': metadata.get('product', 'unknown'),
                    'distance': results['distances'][0][i]
                }
            )

        return parsed_results

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        执行语义相似度检索

        Args:
            query: 用户查询文本
            top_k: 返回结果数量
            where_filter: ChromaDB where 过滤条件（可选）
                          例如: {"product": "电钻"} 只搜索电钻相关内容

        Returns:
            List[Dict]: 检索结果列表，每项包含:
                - content: 文本内容
                - images: 图片ID列表
                - product: 产品名称
                - distance: 相似度距离（越小越相似）
        """
        query_embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )

        return self._parse_results(results)

    def rerank_results(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if not self.enable_rerank or self.reranker is None:
            sorted_candidates = sorted(candidates, key=lambda x: x.get("distance", 999.0))
            return sorted_candidates[:top_k] if top_k is not None else sorted_candidates

        pairs = [[query, item.get("content", "")] for item in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        reranked = []
        for item, score in zip(candidates, scores):
            enriched = dict(item)
            enriched["rerank_score"] = float(score)
            reranked.append(enriched)

        reranked.sort(
            key=lambda x: (
                -x.get("rerank_score", float("-inf")),
                x.get("distance", 999.0)
            )
        )
        return reranked[:top_k] if top_k is not None else reranked

    def search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict] = None,
        candidate_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        candidate_k = max(top_k, candidate_k or RERANK_CANDIDATE_K)
        candidates = self.search_semantic(
            query=query,
            top_k=candidate_k,
            where_filter=where_filter
        )
        return self.rerank_results(query, candidates, top_k=top_k)

    def search_by_product(
        self,
        query: str,
        product: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        在指定产品范围内检索

        Args:
            query: 用户查询文本
            product: 产品名称（如 "电钻"、"空调"）
            top_k: 返回结果数量

        Returns:
            List[Dict]: 检索结果列表
        """
        return self.search(
            query=query,
            top_k=top_k,
            where_filter={"product": product},
            candidate_k=max(top_k, RERANK_CANDIDATE_K)
        )

    def get_all_products(self) -> List[str]:
        """
        获取所有产品名称列表

        Returns:
            List[str]: 产品名称列表
        """
        # 从所有记录中提取不重复的产品名称
        results = self.collection.get(
            include=['metadatas']
        )

        products = set()
        for metadata in results['metadatas']:
            if 'product' in metadata:
                products.add(metadata['product'])

        return sorted(list(products))


# ============================================
# 测试入口
# ============================================

if __name__ == "__main__":
    import pprint

    print("\n" + "=" * 60)
    print("🧪 检索器测试（多语言支持）")
    print("=" * 60)

    # 初始化检索器
    retriever = ManualRetriever()

    # 显示所有产品
    print(f"\n📋 可用产品: {retriever.get_all_products()}")

    # 测试查询（中英文混合）
    test_queries = [
        # 中文测试
        "我的DCB107指示灯闪烁代表什么含义？",
        "空调遥控器怎么安装电池？",
        "洗碗机如何添加洗涤剂？",
        # 英文测试（验证多语言能力）
        "How do I turn on or off the water supply button on my boat?",
        "How to flush the cooling system of the boat?",
        "What are the steps to charge an electric toothbrush?",
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"🔍 查询: {query}")
        print(f"{'─' * 60}")

        results = retriever.search(query, top_k=5)

        for i, result in enumerate(results, 1):
            print(f"\n📍 结果 {i} (相似度距离: {result['distance']:.4f})")
            print(f"   产品: {result['product']}")
            print(f"   图片: {result['images']}")
            print(f"   内容: {result['content'][:150]}...")

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60 + "\n")
