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
CHILD_TARGET_LENGTH = 560
CHILD_OVERLAP = 40


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
    level: str = "child"    # chunk 层级：parent / child
    parent_id: Optional[str] = None
    section_title: str = ""
    step_no: str = ""
    chunk_index: int = 0
    prev_chunk_id: str = ""
    next_chunk_id: str = ""
    source_file: str = ""
    sub_manual: str = ""
    language: str = ""
    content_type: str = ""
    embedding_text: str = ""
    bm25_text: str = ""


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


def preprocess_text_for_structure(text: str, product: str) -> str:
    """
    对结构较差的原文做轻量预标准化，帮助 section / step 识别。
    """
    text = text or ""
    if product == "汇总英文":
        text = re.sub(r'(?<!\n)\s+(#\s*[A-Z][A-Za-z/&\'’ -]{1,80})', r'\n\1', text)
        text = re.sub(r'(?<=[.?!])\s+(?=#\s*[A-Z])', '\n', text)
    return text


def infer_sub_manual(text: str, product: str, record_index: int) -> str:
    """
    为聚合手册推断更细的子手册标题。
    """
    normalized = clean_text(text)
    if not normalized:
        return f"{product}_{record_index:02d}"

    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    heading_candidate = ""
    for line in lines[:12]:
        if line.startswith("#"):
            heading_candidate = re.sub(r'^#+\s*', '', line).strip()
            break

    candidate = heading_candidate or lines[0]
    candidate = candidate.replace(IMAGE_PLACEHOLDER, " ")
    candidate = re.sub(r'\s+', ' ', candidate)
    candidate = re.sub(r'[|•·]+', ' ', candidate)
    candidate = re.split(r'(?<=[.?!:])\s+', candidate, maxsplit=1)[0]
    words = candidate.split()
    if len(words) > 12:
        candidate = " ".join(words[:12])
    candidate = candidate.strip(" -_\t")
    if len(candidate) > 80:
        candidate = candidate[:80].rstrip()
    return candidate or f"{product}_{record_index:02d}"


def infer_language(product: str, text: str) -> str:
    if product == "汇总英文":
        return "en"
    cjk_hits = len(re.findall(r'[\u4e00-\u9fff]', text))
    ascii_hits = len(re.findall(r'[A-Za-z]', text))
    return "zh" if cjk_hits >= ascii_hits else "en"


def infer_content_type(section_title: str, content: str, images: List[str], step_no: str) -> str:
    normalized_title = clean_text(section_title).lower()
    normalized_content = clean_text(content).lower()
    haystack = f"{normalized_title}\n{normalized_content}"

    if step_no:
        return "steps"
    if is_toc_section(content, section_title):
        return "toc"
    if "warning" in haystack or "caution" in haystack or "注意" in haystack or "警告" in haystack:
        return "warning"
    if "screen" in haystack or "menu" in haystack or "button" in haystack or "setting" in haystack:
        return "ui"
    if "specification" in haystack or "specifications" in haystack or "参数" in haystack:
        return "spec"
    if images:
        return "image_section"
    return "general"


def build_embedding_text(
    product: str,
    sub_manual: str,
    section_title: str,
    content_type: str,
    language: str,
    content: str
) -> str:
    """
    构造供 Dense Embedding 使用的富化文本。
    """
    content = clean_text(content)
    section_title = clean_text(section_title)
    sub_manual = clean_text(sub_manual) or clean_text(product)
    content_type = clean_text(content_type) or "general"

    if language == "en":
        parts = [
            "This is an English product manual.",
            f"Product: {sub_manual}.",
        ]
        if section_title:
            parts.append(f"Section: {section_title}.")
        parts.append(f"Content type: {content_type}.")
        parts.append(f"Details: {content}")
        return " ".join(parts)

    parts = [
        "这是中文产品说明书内容。",
        f"产品：{sub_manual or product}。",
    ]
    if section_title:
        parts.append(f"章节：{section_title}。")
    parts.append(f"内容类型：{content_type}。")
    parts.append(f"详细内容：{content}")
    return "".join(parts)


def build_bm25_text(
    product: str,
    sub_manual: str,
    section_title: str,
    content_type: str,
    language: str,
    content: str
) -> str:
    """
    构造供 BM25 使用的高密度关键词文本。
    """
    fields = [
        language,
        clean_text(product),
        clean_text(sub_manual),
        clean_text(section_title),
        clean_text(content_type),
        clean_text(content),
    ]
    return " ".join(part for part in fields if part)


def parse_raw_manual_records(raw_content: str, file_name: str) -> List[Tuple[str, List[str]]]:
    """
    解析手册原始内容，兼容三种格式：
    1. 单个 JSON 数组 [text, images]
    2. 单个 Python 字面量数组
    3. 按行拼接的多条 JSON/字面量记录（汇总英文手册）
    """
    def normalize_record(data: Any) -> Optional[Tuple[str, List[str]]]:
        if isinstance(data, list) and len(data) >= 1:
            text_content = data[0]
            image_list = data[1] if len(data) > 1 and isinstance(data[1], list) else []
            return str(text_content), image_list
        if isinstance(data, str):
            return data, []
        return None

    try:
        parsed = json.loads(raw_content)
        record = normalize_record(parsed)
        if record:
            return [record]
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(raw_content)
        record = normalize_record(parsed)
        if record:
            return [record]
    except (ValueError, SyntaxError):
        pass

    records: List[Tuple[str, List[str]]] = []
    for line in raw_content.splitlines():
        line = line.strip()
        if not line:
            continue

        parsed_line = None
        try:
            parsed_line = json.loads(line)
        except json.JSONDecodeError:
            try:
                parsed_line = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                parsed_line = None

        record = normalize_record(parsed_line) if parsed_line is not None else None
        if record:
            records.append(record)

    if records:
        return records

    return [(raw_content, [])]


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


STEP_LINE_PATTERN = re.compile(
    r'^\s*(?:#\s*)?(?:step\s*)?(?P<step>\d{1,2}|[一二三四五六七八九十]+)[.)、：:]?\s*',
    re.IGNORECASE
)


def is_step_line(line: str) -> bool:
    return bool(STEP_LINE_PATTERN.match(line.strip()))


def extract_step_no(line: str) -> str:
    match = STEP_LINE_PATTERN.match(line.strip())
    return match.group("step") if match else ""


def is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return False
    return not is_step_line(stripped)


def is_major_heading_line(line: str) -> bool:
    stripped = clean_text(line)
    if not is_heading_line(stripped):
        return False

    title = re.sub(r'^#+\s*', '', stripped).strip()
    if not title or title == IMAGE_PLACEHOLDER:
        return False

    if re.fullmatch(r'[#\s]+', stripped):
        return False

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9/&'’.-]*", title)
    word_count = len(words)
    upper_chars = sum(1 for ch in title if ch.isalpha() and ch.isupper())
    alpha_chars = sum(1 for ch in title if ch.isalpha())
    upper_ratio = upper_chars / alpha_chars if alpha_chars else 0.0

    if "table of contents" in title.lower() or title.lower().startswith("contents"):
        return True
    if re.match(r'^(chapter|appendix)\b', title, flags=re.IGNORECASE):
        return True
    if re.search(r'\.{4,}', title):
        return True
    if len(title) <= 70 and word_count <= 9 and not re.search(r'[.?!]', title):
        return True
    if len(title) <= 90 and word_count <= 12 and upper_ratio >= 0.65:
        return True

    return False


def attach_section_title(section_title: str, text: str) -> str:
    text = clean_text(text)
    title = clean_text(section_title)
    if not title:
        return text
    if text.startswith(title):
        return text
    return f"{title}\n{text}" if text else title


def split_into_sections(text: str, image_list: List[str]) -> List[Tuple[str, List[str], str]]:
    """
    按显式 section 标题切块，并将图片顺序映射回各 section。
    """
    lines = text.split("\n")
    sections: List[Tuple[str, str]] = []
    current_title = ""
    current_lines: List[str] = []

    for line in lines:
        if is_major_heading_line(line):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = clean_text(line)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    if not sections:
        return [(text, image_list, "")]

    image_index = 0
    resolved = []
    for section_title, section_text in sections:
        pic_count = section_text.count(IMAGE_PLACEHOLDER)
        section_images = image_list[image_index:image_index + pic_count]
        image_index += pic_count
        resolved.append((section_text, section_images, section_title))

    return resolved


def split_step_blocks(text: str) -> List[Tuple[str, str]]:
    """
    识别连续步骤块，返回 [(step_no, block_text), ...]。
    """
    lines = [line.rstrip() for line in text.split("\n")]
    blocks: List[Tuple[str, List[str]]] = []
    current_step = ""
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append(line)
            continue

        if is_step_line(stripped):
            if current_lines:
                blocks.append((current_step, current_lines))
            current_step = extract_step_no(stripped)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append((current_step, current_lines))

    normalized = []
    valid_steps = 0
    for step_no, block_lines in blocks:
        block_text = clean_text("\n".join(block_lines))
        if len(block_text) >= MIN_CHUNK_LENGTH:
            normalized.append((step_no, block_text))
            if step_no:
                valid_steps += 1

    return normalized if valid_steps >= 2 else []


def is_toc_section(text: str, section_title: str) -> bool:
    normalized = clean_text(text).lower()
    title = clean_text(section_title).lower()
    if "table of contents" in normalized or "content/ contenu" in normalized:
        return True
    if title.startswith("# table of contents") or title.startswith("# content"):
        return True
    dot_leader_hits = len(re.findall(r'\.{4,}', normalized))
    page_like_hits = len(re.findall(r'\bpage\s+\d+\b|\b\d{1,3}\s*$', normalized, flags=re.MULTILINE))
    return dot_leader_hits >= 5 or page_like_hits >= 8


def split_long_text_with_overlap(
    text: str,
    target_len: int = CHILD_TARGET_LENGTH,
    overlap: int = CHILD_OVERLAP
) -> List[str]:
    """
    对过长文本做轻量滑动窗口。
    """
    text = clean_text(text)
    if len(text) <= target_len:
        return [text] if text else []

    units = [unit.strip() for unit in re.split(r'(?<=[。！？.!?])\s+|\n+', text) if unit.strip()]
    if not units:
        return [text]

    windows: List[str] = []
    start = 0
    while start < len(units):
        current = []
        current_len = 0
        idx = start
        while idx < len(units):
            unit = units[idx]
            projected = current_len + len(unit) + (1 if current else 0)
            if current and projected > target_len:
                break
            current.append(unit)
            current_len = projected
            idx += 1

        if not current:
            current = [units[start]]
            idx = start + 1

        windows.append(clean_text(" ".join(current)))
        if idx >= len(units):
            break

        overlap_chars = 0
        next_start = idx
        while next_start > start:
            overlap_chars += len(units[next_start - 1]) + 1
            if overlap_chars >= overlap:
                break
            next_start -= 1
        start = max(start + 1, next_start)

    return windows


def merge_pic_children(pic_units: List[Tuple[str, List[str]]], target_len: int = CHILD_TARGET_LENGTH) -> List[Tuple[str, List[str]]]:
    """
    将连续的短图文单元合并，避免一张图切成一个过短 child。
    """
    merged: List[Tuple[str, List[str]]] = []
    current_texts: List[str] = []
    current_images: List[str] = []
    current_len = 0

    for text, images in pic_units:
        text = clean_text(text)
        if not text and not images:
            continue

        unit_len = len(text)
        if current_texts and current_len + unit_len > target_len:
            merged.append((clean_text("\n".join(current_texts)), current_images))
            current_texts = []
            current_images = []
            current_len = 0

        if text:
            current_texts.append(text)
            current_len += unit_len
        for image_id in images:
            if image_id not in current_images:
                current_images.append(image_id)

    if current_texts or current_images:
        merged.append((clean_text("\n".join(current_texts)), current_images))

    return merged


def merge_text_units(units: List[str], target_len: int = CHILD_TARGET_LENGTH) -> List[str]:
    """
    将连续短段落合并，避免普通正文一段一个 child。
    """
    merged: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for unit in units:
        unit = clean_text(unit)
        if not unit:
            continue

        projected = current_len + len(unit) + (1 if current_parts else 0)
        if current_parts and projected > target_len:
            merged.append(clean_text("\n".join(current_parts)))
            current_parts = [unit]
            current_len = len(unit)
            continue

        current_parts.append(unit)
        current_len = projected

    if current_parts:
        merged.append(clean_text("\n".join(current_parts)))

    return merged


def build_child_units(section_text: str, section_images: List[str], section_title: str) -> List[Tuple[str, List[str], str]]:
    """
    根据 section 结构生成 child 单元：
    1. 连续步骤块
    2. <PIC> 图文块
    3. 普通段落 + 滑动窗口
    """
    if is_toc_section(section_text, section_title):
        return []

    step_blocks = split_step_blocks(section_text)
    if step_blocks:
        units = []
        for step_no, block_text in step_blocks:
            for piece in split_long_text_with_overlap(block_text):
                units.append((attach_section_title(section_title, piece), [], step_no))
        return units

    if IMAGE_PLACEHOLDER in section_text:
        units = []
        merged_pic_units = merge_pic_children(split_text_by_pic_tags(section_text, section_images))
        for piece, images in merged_pic_units:
            for child_text in split_long_text_with_overlap(piece, target_len=max(CHILD_TARGET_LENGTH, 520)):
                units.append((attach_section_title(section_title, child_text), images, ""))
        return units

    units = []
    paragraphs = [
        paragraph for paragraph in smart_split_paragraph(section_text)
        if len(clean_text(paragraph)) >= 24 or section_images
    ]
    for paragraph in merge_text_units(paragraphs):
        if len(clean_text(paragraph)) < 24 and not section_images:
            continue
        for piece in split_long_text_with_overlap(paragraph):
            units.append((attach_section_title(section_title, piece), [], ""))
    return units


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

    records = parse_raw_manual_records(raw_content, file_path.name)

    global_section_index = 0
    for record_index, (record_text, record_images) in enumerate(records, start=1):
        text_content = preprocess_text_for_structure(str(record_text), product)
        image_list = record_images if isinstance(record_images, list) else []
        sub_manual = infer_sub_manual(text_content, product, record_index)
        language = infer_language(product, text_content)

        sections = split_into_sections(text_content, image_list)

        for section_text, section_images, section_title in sections:
            parent_text = clean_text(section_text)
            if len(parent_text) < MIN_CHUNK_LENGTH:
                continue

            parent_id = generate_chunk_id(product, f"parent::{parent_text}")
            parent_type = infer_content_type(section_title, parent_text, section_images, "")
            chunks.append(
                TextChunk(
                    chunk_id=parent_id,
                    product=product,
                    content=parent_text,
                    images=section_images,
                    level="parent",
                    parent_id=None,
                    section_title=clean_text(section_title),
                    chunk_index=global_section_index,
                    source_file=file_path.name,
                    sub_manual=sub_manual,
                    language=language,
                    content_type=parent_type,
                    embedding_text=build_embedding_text(
                        product=product,
                        sub_manual=sub_manual,
                        section_title=section_title,
                        content_type=parent_type,
                        language=language,
                        content=parent_text,
                    ),
                    bm25_text=build_bm25_text(
                        product=product,
                        sub_manual=sub_manual,
                        section_title=section_title,
                        content_type=parent_type,
                        language=language,
                        content=parent_text,
                    ),
                )
            )

            child_units = build_child_units(section_text, section_images, section_title)
            child_ids = []
            child_chunks = []

            for child_index, (child_text, child_images, step_no) in enumerate(child_units):
                child_text = clean_text(child_text)
                if len(child_text) < MIN_CHUNK_LENGTH:
                    continue

                child_id = generate_chunk_id(product, f"child::{child_text}")
                child_ids.append(child_id)
                child_chunks.append(
                    TextChunk(
                        chunk_id=child_id,
                        product=product,
                        content=child_text,
                        images=child_images,
                        level="child",
                        parent_id=parent_id,
                        section_title=clean_text(section_title),
                        step_no=step_no,
                        chunk_index=child_index,
                        source_file=file_path.name,
                        sub_manual=sub_manual,
                        language=language,
                        content_type=infer_content_type(section_title, child_text, child_images, step_no),
                    )
                )

            for child_chunk in child_chunks:
                child_chunk.embedding_text = build_embedding_text(
                    product=child_chunk.product,
                    sub_manual=child_chunk.sub_manual,
                    section_title=child_chunk.section_title,
                    content_type=child_chunk.content_type,
                    language=child_chunk.language,
                    content=child_chunk.content,
                )
                child_chunk.bm25_text = build_bm25_text(
                    product=child_chunk.product,
                    sub_manual=child_chunk.sub_manual,
                    section_title=child_chunk.section_title,
                    content_type=child_chunk.content_type,
                    language=child_chunk.language,
                    content=child_chunk.content,
                )

            for idx, child_chunk in enumerate(child_chunks):
                child_chunk.prev_chunk_id = child_ids[idx - 1] if idx > 0 else ""
                child_chunk.next_chunk_id = child_ids[idx + 1] if idx < len(child_ids) - 1 else ""
                chunks.append(child_chunk)

            global_section_index += 1

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
