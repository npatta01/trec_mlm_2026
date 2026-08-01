# Audited pointwise judgments GitHub Release

The primary automated-use labels and full audited judgments are also persisted
in the private-repository GitHub Release:

- Release: <https://github.com/npatta01/trec_mlm_2026/releases/tag/pointwise-judgments-v1>
- Tag: `pointwise-judgments-v1`
- Title: `Audited pointwise judgments v1`
- Privacy: inherited from the private `npatta01/trec_mlm_2026` repository

## Judgment assets

| Asset | Purpose | SHA-256 |
|---|---|---|
| `pointwise-labels.jsonl` | Compact automated-use labels | `7cde37d0d913b97ede43c4791b2f808e6d1b0c863bdcac729efde78480113c48` |
| `final-results.json` | Full audited judgments and explanations | `8b6ae7d2d8b0a9e3ed9e92de43006aa9f4d4f5ca90af72905f153d19383462e2` |

The release also contains the aggregate audit, structured checkpoint, restore
guide, and full private-archive checksum manifest. GitHub intentionally stores
the two canonical judgment artifacts rather than duplicating the complete 2 GB
controller/provider evidence bundle.

## Complete evidence bundle

The controller state, raw provider outputs, prompts, batch receipts, shards,
and audit packets remain in the local archive and private Hugging Face Bucket:

```text
hf://buckets/Npatta01/trec_mlm_2026/trec_mlm_2026/experiments/openai-luna-medium-v1-500-20260731-8b6ae7d2d8b0
```

Use `reports/openai-luna-pointwise-production-restore.md` for both GitHub and
Hugging Face restore commands and the expected content hashes.
