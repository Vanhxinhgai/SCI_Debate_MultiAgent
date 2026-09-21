# SciFact dataset files

Place the local SciFact JSONL files here:

- `corpus.jsonl`
- `claims_train.jsonl`
- `claims_dev.jsonl`
- `claims_test.jsonl`

The app uses gold SciFact evidence only when the input claim exactly matches a
claim in one of the `claims_*.jsonl` files. For arbitrary claims, it performs
lexical retrieval over `corpus.jsonl` and treats the results as retrieved
documents, not gold labels.

Expected document shape:

```json
{"doc_id": 123, "title": "...", "abstract": ["sentence 0", "sentence 1"]}
```

Expected claim shape:

```json
{"id": 1, "claim": "...", "evidence": {"123": [{"sentences": [0], "label": "SUPPORT"}]}}
```
