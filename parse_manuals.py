"""
产品手册数据清洗与解析脚本

功能：
1. 遍历输入目录下的所有 .txt 手册文件
2. 解析 JSON 格式的手册数据（文本 + 图片ID列表）
3. 按 <PIC> 标记和段落进行智能切分
4. 将图片ID与对应的文本段落对齐
5. 输出结构化的 JSON 数据

作者：Claude Code
"""

import os
import re
import json
import uuid
import hashlib
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict


# ============================================
# 配置常量
# ============================================

# 输入目录（存放手册 txt 文件）
INPUT_DIR = Path("./data/manuals")

# 输出文件路径
OUTPUT_FILE = Path("./data/structured_knowledge.json")

# 最小 Chunk 长度（过滤过短的无意义片段）
MIN_CHUNK_LENGTH = 5

# 图片占位符标记
IMAGE_PLACEHOLDER = "<PIC>"


# ============================================
# 数据类定义
# ============================================

@dataclass
class TextChunk:
    """文本块数据结构"""
    chunk_id: str           # 唯一标识符
    product: str            # 产品名称（来源文件名）
    content: str            # 文本内容
    images: List[str]       # 关联的图片ID列表


# ============================================
# 工具函数
# ============================================

def generate_chunk_id(product: str, content: str) -> str:
    """
    生成唯一的 Chunk ID

    使用 UUID + 内容哈希的方式，确保唯一性和可追溯性

    Args:
        product: 产品名称
        content: 文本内容

    Returns:
        str: 唯一的 Chunk ID
    """
    # 使用内容哈希作为基础，避免重复内容的 ID 不同
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    return f"{product}_{content_hash}_{uuid.uuid4().hex[:8]}"


def read_file_with_encoding(file_path: Path) -> Optional[str]:
    """
    尝试多种编码读取文件内容

    优先使用 utf-8，失败后尝试 gbk，确保兼容性

    Args:
        file_path: 文件路径

    Returns:
        Optional[str]: 文件内容，读取失败返回 None
    """
    # 编码尝试顺序：优先 utf-8，然后尝试中文编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return content
        except (UnicodeDecodeError, UnicodeError):
            continue

    return None


def extract_images_from_text(text: str) -> List[str]:
    """
    从文本中提取图片ID

    支持多种格式：
    - <PIC> 占位符（在手册中按顺序对应图片列表）
    - 直接嵌入的图片ID（如 drill0_04, Manual11_0, air_conditioner_01）
    - 方括号包裹格式 [drill0_04]
    - 尖括号格式 <图片:drill0_04>

    Args:
        text: 待提取的文本

    Returns:
        List[str]: 提取到的图片ID列表
    """
    images = []

    # 定义多种图片ID匹配模式
    patterns = [
        # 模式1: 字母开头 + 数字 + 下划线 + 数字（如 drill0_04, Manual11_0）
        r'[a-zA-Z][a-zA-Z0-9_]*\d+_\d+',
        # 模式2: 字母 + 下划线 + 多个字母/数字 + 下划线 + 数字（如 air_conditioner_01）
        r'[a-zA-Z][a-zA-Z0-9_]+_\d+',
        # 模式3: 方括号包裹 [xxx]
        r'\[([a-zA-Z0-9_]+)\]',
        # 模式4: 尖括号格式 <图片:xxx> 或 <PIC:xxx>
        r'<(?:图片|PIC):([a-zA-Z0-9_]+)>',
        # 模式5: Manual + 数字 + _ + 数字（如 Manual16_51）
        r'Manual\d+_\d+',
        # 模式6: Dish_washer_XX 格式
        r'Dish_washer_\d+',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # 如果是元组（来自捕获组），取第一个元素
            img_id = match if isinstance(match, str) else match[0]
            # 过滤掉方括号等包裹符号
            img_id = img_id.strip('[]<>')
            if img_id and img_id not in images and len(img_id) > 2:
                images.append(img_id)

    return images


def clean_text(text: str) -> str:
    """
    清洗文本内容

    - 去除首尾空白
    - 去除多余的连续换行（超过2个换行符压缩为2个）
    - 去除行首行尾多余空白

    Args:
        text: 待清洗的文本

    Returns:
        str: 清洗后的文本
    """
    if not text:
        return ""

    # 去除首尾空白
    text = text.strip()

    # 压缩多个连续换行为最多2个
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 去除行首行尾多余空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text


# ============================================
# 核心解析逻辑
# ============================================

def split_text_by_pic_tags(text: str, image_list: List[str]) -> List[Tuple[str, List[str]]]:
    """
    按 <PIC> 标记分割文本，并关联对应的图片ID

    手册格式说明：
    - 文本中 <PIC> 标记按顺序出现
    - image_list 中的图片ID按顺序对应每个 <PIC>

    Args:
        text: 完整的手册文本
        image_list: 图片ID列表（来自 JSON 数组的第二个元素）

    Returns:
        List[Tuple[str, List[str]]]: [(文本段落, 图片ID列表), ...]
    """
    # 按 <PIC> 分割文本
    parts = text.split(IMAGE_PLACEHOLDER)

    chunks = []
    image_index = 0  # 当前图片索引

    for i, part in enumerate(parts):
        # 清洗文本
        part = clean_text(part)

        # 过滤过短的片段
        if len(part) < MIN_CHUNK_LENGTH:
            continue

        # 当前段落的图片列表
        chunk_images = []

        # 如果这不是最后一段，说明后面有 <PIC> 标记
        # 需要关联下一个图片ID
        if i < len(parts) - 1 and image_index < len(image_list):
            chunk_images.append(image_list[image_index])
            image_index += 1

        chunks.append((part, chunk_images))

    return chunks


def smart_split_paragraph(text: str) -> List[str]:
    """
    智能段落分割

    对于没有 <PIC> 标记的长文本，按照自然段落分割：
    - 双换行符（\n\n）作为主要分割点
    - 单换行符（\n）作为次要分割点
    - 标题标记（如 # ）也作为分割参考

    Args:
        text: 待分割的文本

    Returns:
        List[str]: 分割后的段落列表
    """
    # 首先按双换行分割
    paragraphs = text.split('\n\n')

    result = []
    for para in paragraphs:
        para = clean_text(para)

        # 如果段落过长（超过500字符），尝试按单换行分割
        if len(para) > 500:
            sub_paras = para.split('\n')
            for sub in sub_paras:
                sub = clean_text(sub)
                if len(sub) >= MIN_CHUNK_LENGTH:
                    result.append(sub)
        elif len(para) >= MIN_CHUNK_LENGTH:
            result.append(para)

    return result


def parse_manual_file(file_path: Path) -> List[TextChunk]:
    """
    解析单个手册文件

    Args:
        file_path: 手册文件路径

    Returns:
        List[TextChunk]: 解析得到的文本块列表
    """
    chunks = []

    # 提取产品名称（去掉后缀）
    product = file_path.stem.replace('手册', '')

    # 读取文件内容
    raw_content = read_file_with_encoding(file_path)
    if raw_content is None:
        print(f"  ⚠️  无法读取文件: {file_path.name}")
        return chunks

    # 初始化变量
    text_content = ""
    image_list = []

    # 尝试解析 JSON 格式
    try:
        data = json.loads(raw_content)

        # 检查数据格式
        if isinstance(data, list) and len(data) >= 2:
            # 格式: [文本内容, 图片ID列表]
            text_content = data[0]
            image_list = data[1] if len(data) > 1 else []
        elif isinstance(data, str):
            # 纯文本格式
            text_content = data
            image_list = []
        else:
            print(f"  ⚠️  未知数据格式: {file_path.name}")
            return chunks

    except json.JSONDecodeError:
        # JSON 解析失败，尝试使用 ast.literal_eval（更宽松的解析）
        try:
            data = ast.literal_eval(raw_content)
            if isinstance(data, list) and len(data) >= 2:
                text_content = data[0]
                image_list = data[1] if len(data) > 1 else []
            else:
                text_content = raw_content
                image_list = []
        except (ValueError, SyntaxError):
            # 最终回退：作为纯文本处理
            text_content = raw_content
            image_list = []

    # 确保 text_content 是字符串
    if not isinstance(text_content, str):
        text_content = str(text_content)

    # 确保图片列表是列表类型
    if not isinstance(image_list, list):
        image_list = []

    # 按 <PIC> 标记分割并关联图片
    if IMAGE_PLACEHOLDER in text_content:
        # 有图片标记的情况
        split_chunks = split_text_by_pic_tags(text_content, image_list)
    else:
        # 没有图片标记，按段落分割
        paragraphs = smart_split_paragraph(text_content)
        split_chunks = [(p, []) for p in paragraphs]

    # 创建 TextChunk 对象
    for text, images in split_chunks:
        # 清洗文本
        text = clean_text(text)

        # 再次检查长度
        if len(text) < MIN_CHUNK_LENGTH:
            continue

        # 生成唯一 ID
        chunk_id = generate_chunk_id(product, text)

        # 创建 TextChunk
        chunk = TextChunk(
            chunk_id=chunk_id,
            product=product,
            content=text,
            images=images
        )
        chunks.append(chunk)

    return chunks


def process_all_manuals(input_dir: Path) -> Tuple[List[TextChunk], Dict[str, Any]]:
    """
    处理所有手册文件

    Args:
        input_dir: 输入目录

    Returns:
        Tuple[List[TextChunk], Dict]: (所有文本块, 统计信息)
    """
    all_chunks = []
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'failed_files': 0,
        'total_chunks': 0,
        'chunks_with_images': 0,
        'products': {}
    }

    # 获取所有 txt 文件
    txt_files = list(input_dir.glob('*.txt'))
    stats['total_files'] = len(txt_files)

    print(f"\n📂 开始处理目录: {input_dir}")
    print(f"📄 找到 {len(txt_files)} 个手册文件\n")

    for file_path in sorted(txt_files):
        print(f"  📖 处理: {file_path.name}")

        # 解析文件
        chunks = parse_manual_file(file_path)

        if chunks:
            stats['processed_files'] += 1
            stats['total_chunks'] += len(chunks)

            # 统计包含图片的 chunk
            chunks_with_img = sum(1 for c in chunks if c.images)
            stats['chunks_with_images'] += chunks_with_img

            # 记录产品统计
            product = chunks[0].product if chunks else 'unknown'
            stats['products'][product] = {
                'chunks': len(chunks),
                'with_images': chunks_with_img
            }

            all_chunks.extend(chunks)
            print(f"      ✓ 生成 {len(chunks)} 个 Chunk（含图片: {chunks_with_img}）")
        else:
            stats['failed_files'] += 1
            print(f"      ✗ 解析失败")

    return all_chunks, stats


def save_to_json(chunks: List[TextChunk], output_path: Path) -> None:
    """
    保存结果到 JSON 文件

    Args:
        chunks: 文本块列表
        output_path: 输出文件路径
    """
    # 转换为字典列表
    data = [asdict(chunk) for chunk in chunks]

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果已保存到: {output_path}")


def print_statistics(stats: Dict[str, Any]) -> None:
    """
    打印处理统计信息

    Args:
        stats: 统计信息字典
    """
    print("\n" + "=" * 50)
    print("📊 处理统计报告")
    print("=" * 50)
    print(f"  📁 输入文件总数:  {stats['total_files']}")
    print(f"  ✅ 成功处理文件:  {stats['processed_files']}")
    print(f"  ❌ 处理失败文件:  {stats['failed_files']}")
    print(f"  📝 总生成 Chunk:  {stats['total_chunks']}")
    print(f"  🖼️  含图片 Chunk:  {stats['chunks_with_images']}")

    if stats['total_chunks'] > 0:
        coverage = stats['chunks_with_images'] / stats['total_chunks'] * 100
        print(f"  📈 图片覆盖率:    {coverage:.1f}%")

    print("\n📋 各产品统计:")
    print("-" * 45)
    print(f"  {'产品名称':<14} | {'Chunk数':>8} | {'含图片':>8}")
    print("-" * 45)
    for product, info in sorted(stats['products'].items()):
        print(f"  {product:<14} | {info['chunks']:>8} | {info['with_images']:>8}")

    print("=" * 50)


# ============================================
# 主函数
# ============================================

def main():
    """主函数入口"""
    print("\n" + "=" * 50)
    print("🔧 产品手册数据清洗与解析脚本")
    print("=" * 50)

    # 检查输入目录是否存在
    if not INPUT_DIR.exists():
        print(f"\n❌ 错误: 输入目录不存在: {INPUT_DIR}")
        print("   请确保目录路径正确，或创建目录后放入手册文件")
        return

    # 处理所有手册
    chunks, stats = process_all_manuals(INPUT_DIR)

    # 保存结果
    if chunks:
        save_to_json(chunks, OUTPUT_FILE)
    else:
        print("\n⚠️  未生成任何 Chunk，请检查输入文件")

    # 打印统计信息
    print_statistics(stats)

    print("\n✅ 处理完成！\n")


if __name__ == "__main__":
    main()
