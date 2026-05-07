from app.retrieval.types import normalize_result


def test_normalize_retrieval_result_schema():
    result = normalize_result({
        "chunk_id": "c1",
        "content": "hello",
        "product": "manual",
        "images": ["img1"],
        "bm25_score": 2.0,
    })
    expected_keys = {
        "chunk_id", "content", "manual", "product", "section_title", "source_path",
        "score", "dense_score", "bm25_score", "rerank_score", "image_ids",
        "matched_terms", "metadata",
    }
    assert expected_keys.issubset(result.keys())
    assert result["image_ids"] == ["img1"]

