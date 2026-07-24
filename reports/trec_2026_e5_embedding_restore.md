# Restore the private E5 inference artifact

Date verified: 2026-07-24

The private Hugging Face Bucket contains a complete, loader-ready copy of the
qrel-free E5 inference artifact:

```text
hf://buckets/Npatta01/trec_mlm_2026/trec_mlm_2026/artifacts/e5-raw-response-refinement-2026-07-23
```

It contains 13 files required by `load_profile_inference_artifacts()`, including
the ordered `records.json` mapping for all 182,764 response vectors. The remote
prefix also contains four checkpoint/construction metadata files that are not
part of the inference loader inventory.

This was a user-authorized, post-experiment durability backup on 2026-07-23,
not part of experiment construction, ranking, or evaluation. The private copy
contains the 13 loader files, including embeddings, ordered record metadata,
and the full historical question/response text in `records.json`, plus four
checkpoint/construction metadata files.

## Restore

Authenticate with an account authorized for the private Bucket, choose an
empty destination, and run:

```bash
E5_REMOTE="hf://buckets/Npatta01/trec_mlm_2026/trec_mlm_2026/artifacts/e5-raw-response-refinement-2026-07-23"
E5_RESTORE_DIR="${E5_RESTORE_DIR:?set an empty restore destination}"

uvx hf auth whoami
uvx hf buckets sync "$E5_REMOTE" "$E5_RESTORE_DIR" \
  --include audits.json \
  --include config.json \
  --include e5_rankings.json \
  --include formatted_inputs.json \
  --include inputs.json \
  --include label_embeddings.npy \
  --include label_question_embeddings.npy \
  --include manifest.json \
  --include metadata.json \
  --include query_embeddings.npy \
  --include record_embeddings.npy \
  --include records.json \
  --include validation.json
```

Do not use `--delete` at the Bucket or repository root.

## Verify with the project loader

From a checkout containing the matching inference loader:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python - "$E5_RESTORE_DIR" <<'PY'
import json
import sys

from trec_mlm.profile_inference_artifact import load_profile_inference_artifacts

artifact = load_profile_inference_artifacts(sys.argv[1])
expected_aggregate = "d9ef709db6642d3f7d9f62f25186939378c130ac17a407ee736e383b2f7d7054"
assert artifact.manifest["aggregate_sha256"] == expected_aggregate
assert artifact.query_embeddings.shape == (322, 4096)
assert artifact.record_embeddings.shape == (182764, 4096)
assert len(artifact.records) == 182764
print(json.dumps({
    "aggregate_sha256": artifact.manifest["aggregate_sha256"],
    "query_shape": list(artifact.query_embeddings.shape),
    "record_shape": list(artifact.record_embeddings.shape),
    "record_count": len(artifact.records),
}, sort_keys=True))
PY
```

Expected:

```json
{"aggregate_sha256": "d9ef709db6642d3f7d9f62f25186939378c130ac17a407ee736e383b2f7d7054", "query_shape": [322, 4096], "record_count": 182764, "record_shape": [182764, 4096]}
```

The 2026-07-24 verification downloaded the 13 files to a separate filesystem
and passed the real loader's inventory, manifest, hash, shape, ordering, and
semantic-reconstruction checks.
