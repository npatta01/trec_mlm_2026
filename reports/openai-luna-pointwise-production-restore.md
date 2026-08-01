# Restore the audited OpenAI Luna pointwise judgments

The complete private run is stored locally and in the private Hugging Face Bucket. Git contains only the controller, frozen contract, audit tooling, aggregate report, and this restore pointer.

## Primary artifacts

- `results/pointwise-labels.jsonl` is the compact automated-use artifact. Treat `supported=true` as the primary label.
- `results/final-results.json` is the full audited result with verdict explanations and provider metadata.
- `run/final-results.json` is the immutable raw controller/provider output.

## Private locations

Local archive:

```text
/home/npatta01/data/artifacts/trec_mlm_2026/openai_pointwise_judgments/openai-luna-medium-v1-500-20260731-8b6ae7d2d8b0
```

HF Bucket:

```text
hf://buckets/Npatta01/trec_mlm_2026/trec_mlm_2026/experiments/openai-luna-medium-v1-500-20260731-8b6ae7d2d8b0
```

## Restore and verify

```bash
restore_dir=$(mktemp -d /tmp/openai-pointwise-restore.XXXXXX)
chmod 700 "$restore_dir"
HF_XET_FIXED_DOWNLOAD_CONCURRENCY=4 \
  HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS=4 \
  hf buckets sync \
  hf://buckets/Npatta01/trec_mlm_2026/trec_mlm_2026/experiments/openai-luna-medium-v1-500-20260731-8b6ae7d2d8b0 \
  "$restore_dir"
(cd "$restore_dir" && sha256sum -c SHA256SUMS)
```

Expected primary hashes:

- Audited full results: `8b6ae7d2d8b0a9e3ed9e92de43006aa9f4d4f5ca90af72905f153d19383462e2`
- Compact labels: `7cde37d0d913b97ede43c4791b2f808e6d1b0c863bdcac729efde78480113c48`
- Raw results: `033edf940d31299340ebb3283774fbf0f77e4fe75d834f67e7d36f9caca65436`
- `SHA256SUMS`: `c6acf2615e12788d37e2c7dd83f06f4c615cda0b7b8a0b808e4deb3162598edb`

The 2026-08-01 acceptance restore downloaded 1,128 files / 2,121,195,834 bytes, passed all 1,127 payload checksums, and reproduced the audited results and compact-label hashes. The bucket was verified private.
