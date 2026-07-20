# Grounded LLM discovery responses

This directory documents the Perplexity Sonar response cache used as a grounded
reference-answer artifact for the TREC Million LLMs experiments.

The full cache is distributed as a GitHub Release asset rather than committed to
Git. The raw JSON is 114 MB, while the ZIP archive is 25.1 MB. The cache retains
the answer text, inline citation markers, citation URLs, search results, usage
metadata, and retry history.

## Download

```bash
gh release download trec-2026-llm-discovery-sonar-2026-07-20 \
  --repo npatta01/trec_mlm_2026 \
  --pattern trec_2026_llm_discovery_sonar_cache_2026-07-20.zip
unzip trec_2026_llm_discovery_sonar_cache_2026-07-20.zip
```

The extracted file is `perplexity_full_cache.json`. Verify the raw file with:

```text
cde4ca41c555e29533961bc41496361b00f73b70264899597573362e355a4ee0
```

The archive and raw-file checksums, generation settings, and quality audit are
recorded in [`manifest.json`](manifest.json).

## Provenance and limitations

- 7,759 questions were answered with the Perplexity `sonar` model.
- The initial run used low search context and excluded Reddit, YouTube,
  Facebook, Quora, Pinterest, and Medium.
- 121 records were retried with medium search context to repair missing or
  malformed citations. A retry replaced the original only when it contained
  citations and all inline markers were within the returned citation list.
- The final cache still contains 3 records without citations and 48 records with
  out-of-range citation markers. These should be treated as unresolved rather
  than silently accepted as gold answers.
- Citation markers are intentionally preserved for auditability. Consumers that
  compare answer text should remove citation markup in a separate derived view,
  not modify this source artifact.
