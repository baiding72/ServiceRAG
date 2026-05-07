from app.query_analyzer import analyze_query
from app.retrieval.hybrid import HybridRetriever


class FakeManualRetriever:
    def search_semantic(self, query, top_k=10, where_filter=None):
        return [
            {"chunk_id": "a", "content": "DCB107 battery light <PIC>", "product": "电钻", "distance": 0.2, "images": ["img_a"]},
            {"chunk_id": "b", "content": "shipping faq", "product": "FAQ", "distance": 0.3},
        ]

    def search_bm25(self, query, top_k=10, where_filter=None):
        return [
            {"chunk_id": "a", "content": "DCB107 battery light <PIC>", "product": "电钻", "bm25_score": 3.0, "images": ["img_a"]},
            {"chunk_id": "c", "content": "charger flashing indicator", "product": "电钻", "bm25_score": 2.0},
        ]

    def rerank_results(self, query, candidates, top_k=10):
        return candidates[:top_k]


def test_hybrid_rrf_records_route_ranks_and_trace():
    retriever = HybridRetriever(FakeManualRetriever())
    results = retriever.search(analyze_query("DCB107 battery light flashing?"), top_k=3)

    assert results[0]["chunk_id"] == "a"
    assert results[0]["route_ranks"] == {"dense": 1, "bm25": 1}
    assert results[0]["rrf_score"] > results[1]["rrf_score"]
    assert retriever.last_trace["fusion"]["mode"] == "rrf"
    assert retriever.last_trace["routes"][0]["dense_top"][0]["chunk_id"] == "a"
    assert retriever.last_trace["final_results"][0]["route_ranks"] == {"dense": 1, "bm25": 1}
