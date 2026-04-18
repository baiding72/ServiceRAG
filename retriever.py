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
import math
import os
import re
from collections import Counter
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
BM25_CANDIDATE_K = int(os.getenv("BM25_CANDIDATE_K", "12"))
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))
ENGLISH_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "be", "to", "of", "on", "in", "at", "for",
    "and", "or", "if", "my", "your", "their", "what", "how", "do", "does", "did",
    "can", "could", "should", "would", "i", "you", "we", "they", "it", "this", "that",
    "these", "those", "according", "manual", "position"
}


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

        self._corpus_cache = self._load_corpus_cache()
        self._corpus_index_map = {
            item["chunk_id"]: item["corpus_index"] for item in self._corpus_cache
        }
        self._build_bm25_index()

    def _load_corpus_cache(self) -> List[Dict[str, Any]]:
        """
        载入完整语料，用于轻量关键词召回。
        对 2324 条 chunk 的规模来说，内存成本可接受。
        """
        results = self.collection.get(include=['documents', 'metadatas'])
        corpus = []

        for corpus_index, (chunk_id, content, metadata) in enumerate(zip(
            results.get("ids", []),
            results.get("documents", []),
            results.get("metadatas", []),
        )):
            images_str = metadata.get('images', '[]')
            try:
                images = json.loads(images_str)
            except json.JSONDecodeError:
                images = []

            raw_content = metadata.get("raw_content", content)
            bm25_text = metadata.get("bm25_text", content)
            corpus.append(
                {
                    "chunk_id": chunk_id,
                    "content": raw_content,
                    "retrieval_text": content,
                    "images": images,
                    "product": metadata.get("product", "unknown"),
                    "sub_manual": metadata.get("sub_manual", ""),
                    "section_title": metadata.get("section_title", ""),
                    "language": metadata.get("language", ""),
                    "content_type": metadata.get("content_type", ""),
                    "distance": 999.0,
                    "normalized_content": self._normalize_text(bm25_text),
                    "bm25_tokens": self._tokenize_for_bm25(bm25_text),
                    "corpus_index": corpus_index,
                }
            )

        return corpus

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    def _extract_query_terms(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        terms = set()
        english_tokens = [
            token for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", normalized)
            if len(token) >= 2 and token not in ENGLISH_STOPWORDS
        ]

        for token in english_tokens:
            terms.add(token)

        for size in (3, 2):
            for idx in range(0, len(english_tokens) - size + 1):
                phrase = " ".join(english_tokens[idx: idx + size])
                if len(phrase) >= 8:
                    terms.add(phrase)

        for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
            terms.add(phrase)
            if len(phrase) >= 4:
                for idx in range(0, len(phrase) - 1):
                    terms.add(phrase[idx:idx + 2])

        return sorted(terms, key=len, reverse=True)

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        tokens: List[str] = []

        for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", normalized):
            if token not in ENGLISH_STOPWORDS:
                tokens.append(token)

        for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
            tokens.append(phrase)
            if len(phrase) >= 2:
                for idx in range(0, len(phrase) - 1):
                    tokens.append(phrase[idx: idx + 2])
            if len(phrase) >= 3:
                for idx in range(0, len(phrase) - 2):
                    tokens.append(phrase[idx: idx + 3])

        return tokens

    def _build_bm25_index(self) -> None:
        self._bm25_doc_freqs: List[Counter] = []
        self._bm25_lengths: List[int] = []
        self._bm25_df = Counter()

        for item in self._corpus_cache:
            term_freq = Counter(item.get("bm25_tokens", []))
            self._bm25_doc_freqs.append(term_freq)
            self._bm25_lengths.append(sum(term_freq.values()))
            for token in term_freq.keys():
                self._bm25_df[token] += 1

        doc_count = max(1, len(self._corpus_cache))
        total_length = sum(self._bm25_lengths)
        self._bm25_avgdl = total_length / doc_count if total_length else 1.0
        self._bm25_doc_count = doc_count

    def _bm25_idf(self, term: str) -> float:
        doc_freq = self._bm25_df.get(term, 0)
        return math.log(1 + (self._bm25_doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    def search_keyword(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        轻量关键词召回，补足界面名、目录词、固定短语这类精确匹配场景。
        """
        terms = self._extract_query_terms(query)
        if not terms:
            return []

        query_norm = self._normalize_text(query)
        scored = []
        wants_steps = any(token in query_norm for token in ("步骤", "step", "steps", "first", "three"))
        wants_screen = "screen" in query_norm or "界面" in query_norm
        wants_assembly = "assembly" in query_norm or "组装" in query_norm or "安装" in query_norm
        wants_grill = "grill" in query_norm

        for item in self._corpus_cache:
            if where_filter and item.get("product") != where_filter.get("product"):
                continue

            content_norm = item["normalized_content"]
            hits = [term for term in terms if term in content_norm]
            if not hits:
                continue

            coverage = len(set(hits)) / max(1, len(set(terms)))
            exact_phrase_bonus = 2.2 if query_norm and query_norm in content_norm else 0.0
            strong_phrase_hits = sum(1 for term in hits if " " in term or len(term) >= 4)
            phrase_bonus = min(2.0, strong_phrase_hits * 0.35)
            heading_bonus = 0.0
            if wants_screen and "screen" in content_norm:
                heading_bonus += 0.6
            if wants_assembly and "assembly" in content_norm:
                heading_bonus += 0.8
            if wants_assembly and content_norm.startswith("# assembly"):
                heading_bonus += 3.0
            if wants_steps and re.search(r"(?:^|\s)(?:#\s*)?\d{1,2}[.)]?\s", content_norm):
                heading_bonus += 0.7
            if wants_steps and re.match(r"^\s*(?:#\s*)?[123][.)]?\b", content_norm):
                heading_bonus += 1.0
            if wants_steps and wants_assembly:
                if re.match(r"^\s*#\s*assembly\b", content_norm):
                    heading_bonus += 2.5
                if re.match(r"^\s*#\s*1\b", content_norm):
                    heading_bonus += 2.2
                elif re.match(r"^\s*#\s*2\b", content_norm):
                    heading_bonus += 1.8
                elif re.match(r"^\s*#\s*3\b", content_norm):
                    heading_bonus += 1.6

            toc_penalty = 0.0
            if "table of contents" in content_norm or content_norm.count("...") >= 8:
                toc_penalty += 2.2
            if len(content_norm) > 2500:
                toc_penalty += 0.4
            if wants_steps and wants_assembly:
                has_step_marker = bool(re.match(r"^\s*(?:#\s*)?\d{1,2}\b", content_norm))
                has_assembly_marker = "assembly" in content_norm
                if not has_step_marker and not has_assembly_marker:
                    toc_penalty += 1.0
            if wants_assembly and wants_grill and "grill" not in content_norm:
                toc_penalty += 2.5

            score = (
                coverage
                + exact_phrase_bonus
                + phrase_bonus
                + heading_bonus
                + math.log1p(len(hits)) * 0.12
                - toc_penalty
            )

            enriched = dict(item)
            enriched["lexical_score"] = float(score)
            scored.append(enriched)

        scored.sort(
            key=lambda x: (
                -x.get("lexical_score", 0.0),
                x.get("distance", 999.0),
            )
        )
        return scored[:top_k]

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = []
        for idx, item in enumerate(self._corpus_cache):
            if where_filter and item.get("product") != where_filter.get("product"):
                continue

            term_freq = self._bm25_doc_freqs[idx]
            doc_len = self._bm25_lengths[idx] or 1
            score = 0.0
            for token in query_tokens:
                freq = term_freq.get(token, 0)
                if not freq:
                    continue
                idf = self._bm25_idf(token)
                denom = freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self._bm25_avgdl)
                score += idf * (freq * (BM25_K1 + 1)) / denom

            if score <= 0:
                continue

            enriched = dict(item)
            enriched["bm25_score"] = float(score)
            scores.append(enriched)

        scores.sort(
            key=lambda x: (
                -x.get("bm25_score", 0.0),
                x.get("distance", 999.0),
            )
        )
        return scores[:top_k]

    def expand_with_neighbors(
        self,
        results: List[Dict[str, Any]],
        window: int = 1
    ) -> List[Dict[str, Any]]:
        """
        按语料顺序补充相邻 chunk，适合步骤题、界面题。
        """
        expanded = {}

        for item in results:
            key = (item.get("chunk_id", ""), item.get("product", ""))
            expanded[key] = item

            center_idx = self._corpus_index_map.get(item.get("chunk_id", ""))
            if center_idx is None:
                continue

            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor_idx = center_idx + offset
                if neighbor_idx < 0 or neighbor_idx >= len(self._corpus_cache):
                    continue

                neighbor = self._corpus_cache[neighbor_idx]
                if neighbor.get("product") != item.get("product"):
                    continue

                neighbor_item = dict(neighbor)
                neighbor_item["distance"] = item.get("distance", neighbor_item.get("distance", 999.0))
                neighbor_item["lexical_score"] = max(
                    float(item.get("lexical_score", 0.0)) * 0.85,
                    float(neighbor_item.get("lexical_score", 0.0))
                )
                expanded[(neighbor_item.get("chunk_id", ""), neighbor_item.get("product", ""))] = neighbor_item

        return list(expanded.values())

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
                    'content': metadata.get('raw_content', results['documents'][0][i]),
                    'retrieval_text': results['documents'][0][i],
                    'images': images,
                    'product': metadata.get('product', 'unknown'),
                    'sub_manual': metadata.get('sub_manual', ''),
                    'section_title': metadata.get('section_title', ''),
                    'language': metadata.get('language', ''),
                    'content_type': metadata.get('content_type', ''),
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
                -x.get("retrieval_score", float("-inf")),
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
