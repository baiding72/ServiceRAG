"""
图片向量数据库构建脚本

功能：
1. 读取 structured_knowledge.json 中的图片 ID
2. 将图片文件编码成向量并写入 ChromaDB
3. 保留图片与原始 chunk / 产品的映射，供主服务做视觉候选补充
"""

import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
import torch
from chromadb.config import Settings
from PIL import Image
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("./data/structured_knowledge.json")
IMAGE_DIR = Path("./data/手册/插图")
CHROMA_PERSIST_DIR = "./data/chroma_image_db"
COLLECTION_NAME = "manual_images"
EMBEDDING_MODEL_NAME = os.getenv("IMAGE_EMBEDDING_MODEL_NAME", "clip-ViT-B-32")
BATCH_SIZE = 16


def load_knowledge_data(file_path: Path) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_path_map(image_dir: Path) -> Dict[str, str]:
    image_map: Dict[str, str] = {}
    for path in image_dir.iterdir():
        if path.is_file():
            image_map[path.stem] = str(path.resolve())
    return image_map


def collect_image_records(data: List[dict], image_map: Dict[str, str]) -> List[dict]:
    grouped_chunks = defaultdict(list)

    for item in data:
        for image_id in item.get("images", []):
            image_path = image_map.get(image_id)
            if image_path is None:
                continue
            grouped_chunks[image_id].append(
                {
                    "chunk_id": item["chunk_id"],
                    "product": item["product"],
                    "content": item["content"],
                    "image_path": image_path,
                }
            )

    records = []
    for image_id, refs in grouped_chunks.items():
        first_ref = refs[0]
        records.append(
            {
                "image_id": image_id,
                "image_path": first_ref["image_path"],
                "product": first_ref["product"],
                "source_products": sorted({ref["product"] for ref in refs}),
                "source_chunk_ids": [ref["chunk_id"] for ref in refs],
                "source_preview": " ".join(ref["content"][:100] for ref in refs[:2]).strip(),
            }
        )

    records.sort(key=lambda x: x["image_id"])
    return records


def load_images(image_paths: List[str]) -> List[Image.Image]:
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            images.append(img.convert("RGB"))
    return images


def create_collection(client: chromadb.PersistentClient, collection_name: str) -> chromadb.Collection:
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        print(f"   ⚠️  Collection '{collection_name}' 已存在，将删除重建")
        client.delete_collection(collection_name)

    return client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_image_vector_database(records: List[dict]) -> chromadb.Collection:
    print(f"\n🔧 初始化图片 Embedding 模型: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("   ✓ 模型加载完成")

    print(f"\n💾 初始化图片 ChromaDB: {CHROMA_PERSIST_DIR}")
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = create_collection(client, COLLECTION_NAME)
    print(f"   ✓ Collection '{COLLECTION_NAME}' 创建成功")

    total_count = len(records)
    processed_count = 0

    print("\n📤 开始图片向量化并写入数据库...")
    for start in range(0, total_count, BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        ids = [record["image_id"] for record in batch]
        image_paths = [record["image_path"] for record in batch]
        documents = [
            f"图片ID: {record['image_id']}\n产品: {record['product']}\n关联内容: {record['source_preview']}"
            for record in batch
        ]
        metadatas = [
            {
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "product": record["product"],
                "source_products": json.dumps(record["source_products"], ensure_ascii=False),
                "source_chunk_ids": json.dumps(record["source_chunk_ids"], ensure_ascii=False),
                "source_preview": record["source_preview"],
            }
            for record in batch
        ]

        images = load_images(image_paths)
        with torch.no_grad():
            embeddings = model.encode(
                images,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        for image in images:
            image.close()
        del images, embeddings, ids, image_paths, documents, metadatas
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        processed_count += len(batch)
        progress = processed_count / total_count * 100
        print(f"   进度: {processed_count}/{total_count} ({progress:.1f}%)")

    return collection


def verify_collection(collection: chromadb.Collection) -> None:
    print("\n🔍 验证图片数据库内容...")
    print(f"   ✓ 总记录数: {collection.count()}")
    sample = collection.peek(limit=1)
    print(f"   ✓ 示例 ID: {sample['ids'][0]}")
    print(f"   ✓ 示例 Metadata: {sample['metadatas'][0]}")


def main() -> None:
    print("\n" + "=" * 50)
    print("🚀 图片向量数据库构建脚本")
    print("=" * 50)
    start_time = time.time()

    if not INPUT_FILE.exists():
        print(f"\n❌ 错误: 输入文件不存在: {INPUT_FILE}")
        return
    if not IMAGE_DIR.exists():
        print(f"\n❌ 错误: 图片目录不存在: {IMAGE_DIR}")
        return

    data = load_knowledge_data(INPUT_FILE)
    image_map = build_image_path_map(IMAGE_DIR)
    records = collect_image_records(data, image_map)

    print(f"\n📂 共收集 {len(records)} 张知识库图片")
    if not records:
        print("❌ 未找到可索引图片")
        return

    collection = build_image_vector_database(records)
    verify_collection(collection)

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("📊 构建统计")
    print("=" * 50)
    print(f"   🖼️  图片数量: {len(records)}")
    print(f"   📁 持久化目录: {CHROMA_PERSIST_DIR}")
    print(f"   📝 Collection: {COLLECTION_NAME}")
    print(f"   🤖 模型: {EMBEDDING_MODEL_NAME}")
    print(f"   ⏱️  耗时: {elapsed:.2f} 秒")
    print("=" * 50)


if __name__ == "__main__":
    main()
