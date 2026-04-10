"""
向量数据库构建脚本

功能：
1. 读取结构化的知识库 JSON 文件
2. 使用 sentence-transformers 模型进行文本向量化
3. 将向量和元数据存入 ChromaDB（本地持久化）

注意事项：
- ChromaDB 的 metadata 不支持 List 或 Dict 类型
- images 列表必须序列化为 JSON 字符串后存储

作者：Claude Code
"""

import json
import os
import time
import gc
from pathlib import Path
from typing import List, Dict, Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ============================================
# 配置常量
# ============================================

# 输入文件路径
INPUT_FILE = Path("./data/structured_knowledge.json")

# ChromaDB 持久化目录
# ⚠️ 重要：bge-m3 向量维度为 1024，必须使用新目录，不能复用旧目录！
CHROMA_PERSIST_DIR = "./data/chroma_db_m3"

# Collection 名称
COLLECTION_NAME = "manuals_qa_m3"

# Embedding 模型名称（多语言 SOTA 模型）
# BAAI/bge-m3: 多语言支持，向量维度 1024，中英文效果俱佳
# BAAI/bge-small-zh-v1.5: 仅中文，向量维度 512（已弃用）
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# 批量处理大小（避免内存溢出）
BATCH_SIZE = 10  # bge-m3 模型较大，使用更小批次


# ============================================
# 数据加载函数
# ============================================

def load_knowledge_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    加载结构化知识库数据

    Args:
        file_path: JSON 文件路径

    Returns:
        List[Dict]: 知识库数据列表
    """
    print(f"\n📂 加载知识库数据: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"   ✓ 加载完成，共 {len(data)} 条记录")
    return data


# ============================================
# 向量数据库构建函数
# ============================================

def build_vector_database(
    data: List[Dict[str, Any]],
    persist_dir: str,
    collection_name: str,
    model_name: str
) -> chromadb.Collection:
    """
    构建向量数据库

    Args:
        data: 知识库数据列表
        persist_dir: ChromaDB 持久化目录
        collection_name: Collection 名称
        model_name: Embedding 模型名称

    Returns:
        chromadb.Collection: 创建的 Collection
    """
    print(f"\n🔧 初始化 Embedding 模型: {model_name}")
    print("   首次运行需要下载模型，请耐心等待...")

    # 加载 Embedding 模型
    model = SentenceTransformer(model_name)
    print(f"   ✓ 模型加载完成")

    # 初始化 ChromaDB 客户端（持久化模式）
    print(f"\n💾 初始化 ChromaDB: {persist_dir}")

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(
            anonymized_telemetry=False  # 禁用遥测
        )
    )

    # 检查是否已存在同名 Collection，存在则删除重建
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        print(f"   ⚠️  Collection '{collection_name}' 已存在，将删除重建")
        client.delete_collection(collection_name)

    # 创建新的 Collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )
    print(f"   ✓ Collection '{collection_name}' 创建成功")

    # 准备批量数据
    print(f"\n📤 开始向量化并写入数据库...")

    total_count = len(data)
    processed_count = 0

    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]

        # 提取批量数据
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for item in batch:
            # ID
            ids.append(item['chunk_id'])

            # 文档内容
            documents.append(item['content'])

            # 元数据（注意：images 列表必须序列化为字符串）
            metadata = {
                'product': item['product'],
                'images': json.dumps(item['images'], ensure_ascii=False)  # 序列化列表
            }
            metadatas.append(metadata)

        # 批量生成向量（内存优化）
        with torch.no_grad():
            batch_embeddings = model.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=BATCH_SIZE
            ).tolist()

        # 写入 ChromaDB
        collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas
        )

        # 清理内存
        del batch_embeddings
        del ids, documents, metadatas
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        processed_count += len(batch)
        progress = processed_count / total_count * 100
        print(f"   进度: {processed_count}/{total_count} ({progress:.1f}%)")

    print(f"   ✓ 全部写入完成")

    return collection


# ============================================
# 验证函数
# ============================================

def verify_collection(collection: chromadb.Collection) -> None:
    """
    验证 Collection 数据

    Args:
        collection: ChromaDB Collection
    """
    print("\n🔍 验证数据库内容...")

    # 获取记录总数
    count = collection.count()
    print(f"   ✓ 总记录数: {count}")

    # 随机查看一条记录
    result = collection.peek(limit=1)
    print(f"\n   📋 示例记录:")
    print(f"      ID: {result['ids'][0]}")
    print(f"      Document: {result['documents'][0][:80]}...")
    print(f"      Metadata: {result['metadatas'][0]}")


# ============================================
# 主函数
# ============================================

def main():
    """主函数入口"""
    print("\n" + "=" * 50)
    print("🚀 向量数据库构建脚本")
    print("=" * 50)

    start_time = time.time()

    # 1. 检查输入文件是否存在
    if not INPUT_FILE.exists():
        print(f"\n❌ 错误: 输入文件不存在: {INPUT_FILE}")
        print("   请先运行 parse_manuals.py 生成知识库文件")
        return

    # 2. 加载知识库数据
    data = load_knowledge_data(INPUT_FILE)

    if not data:
        print("\n❌ 错误: 知识库数据为空")
        return

    # 3. 构建向量数据库
    collection = build_vector_database(
        data=data,
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        model_name=EMBEDDING_MODEL_NAME
    )

    # 4. 验证结果
    verify_collection(collection)

    # 5. 打印统计信息
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("📊 构建统计")
    print("=" * 50)
    print(f"   📁 持久化目录: {CHROMA_PERSIST_DIR}")
    print(f"   📝 Collection: {COLLECTION_NAME}")
    print(f"   📊 总记录数: {collection.count()}")
    print(f"   🤖 Embedding 模型: {EMBEDDING_MODEL_NAME}")
    print(f"   ⏱️  耗时: {elapsed_time:.2f} 秒")
    print("=" * 50)
    print("\n✅ 向量数据库构建完成！\n")


if __name__ == "__main__":
    main()
