from app.memory import SessionMemoryStore
from app.query_analyzer import analyze_query


def test_session_memory_isolated_and_followup():
    store = SessionMemoryStore(max_turns=2, ttl_seconds=60)
    analysis = analyze_query("DCB107 指示灯闪烁怎么办？")
    store.add_turn("s1", "DCB107 指示灯闪烁怎么办？", "answer", analysis)

    enriched = store.enrich_question("s1", "继续")
    assert "DCB107" in enriched
    assert store.enrich_question("s2", "继续") == "继续"

