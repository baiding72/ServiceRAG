from app.query_analyzer import analyze_query, detect_language, split_sub_questions


def test_detect_language_en_and_zh():
    assert detect_language("How do I reset it?") == "en"
    assert detect_language("怎么重置？") == "zh"


def test_analyze_image_related_entities():
    analysis = analyze_query("我的DCB107电钻指示灯闪烁是什么意思？")
    assert analysis.intent in {"image_related", "troubleshooting"}
    assert analysis.is_image_related
    assert "DCB107" in analysis.entities["models"]


def test_split_sub_questions():
    parts = split_sub_questions("能送到乡镇吗？要运费吗？多久到？")
    assert len(parts) >= 3

