# Evaluation

This project has two eval layers.

## Local sample eval

`eval/eval_dataset.jsonl` is a small reproducible regression set. Run:

```bash
LLM_MOCK_MODE=true python -m app.evaluation.run_eval --limit 20
```

Outputs:

- `eval_reports/latest.json`
- `eval_reports/latest.md`

Metrics:

- `recall@1/3/5`
- `mrr`
- `image_id_recall`
- `answer_keyword_hit_rate`
- `fallback_rate`
- `average_latency_ms`

These metrics are generated from real local retrieval. They are not official leaderboard scores.

## Pooled eval

The existing scripts under `scripts/` support heavier pool-based candidate construction and LLM-as-a-judge labeling. Use them when comparing retrieval strategy changes. Do not hand-edit metric outputs.

