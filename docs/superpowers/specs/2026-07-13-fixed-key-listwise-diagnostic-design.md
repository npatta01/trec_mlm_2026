# Fixed-Key Listwise Diagnostic Design

## Goal

Test whether Qwen3-4B can jointly rank the twenty frozen expert labels for each
Stage 2 question when the output contract does not require an unenforceable
permutation.

This is a small feasibility diagnostic, not a full sweep or deployment decision.

## Frozen Scope

- Use the existing twelve Stage 2 questions and their twenty-candidate manifests.
- Use `Qwen/Qwen3-4B-Instruct-2507` at revision
  `cdbee75f17c01a7cc42f958dc650907174af0554` in the pinned ROCm vLLM image.
- Preserve FP32, seed 2026, the official chat path, the validated pilot memory limit,
  and all existing input/provenance checks.
- Make one deterministic listwise request per question: twelve requests total.
- Do not rerun the old direct-permutation method or alter its terminal failure
  artifact.

## Output Contract

Each request shows the question and all twenty handle/label/capability rows. The
model returns a JSON object with exactly the twenty handles as required keys and
integer relevance scores from 0 through 100 as values. Higher is better.

Example shape:

```json
{"C00": 87, "C01": 12, "C02": 61}
```

The real schema contains all twenty handles, requires every property, rejects
additional properties, and bounds every integer. It does not require distinct
scores. Rankings sort by descending score and then candidate ID for exact ties.

Parsing must detect duplicate JSON object keys with `object_pairs_hook`; ordinary
`json.loads` behavior is insufficient because it silently overwrites duplicates.
Missing, extra, duplicate, non-integer, boolean, or out-of-range values make the
attempt unscorable. No fallback rankings are synthesized.

This remains listwise: the model sees and scores all candidates jointly in one
request. It changes only the output representation.

## Data Flow and Artifacts

1. Validate the frozen Stage 2 selection, smoke receipt, model/runtime metadata,
   and prompt/schema hashes without opening qrels.
2. Before launch, verify `/dev/kfd` ownership, available memory, memory/IO pressure,
   and running containers. Defer inference if the host is busy.
3. Persist prompts, schemas, raw text, token counts, finish reasons, runtime data,
   and validation errors before parsing or scoring.
4. Validate twelve complete twenty-candidate rankings and a qrel-free ranking
   receipt.
5. Only after that validation, open the sealed judgments and compare this method on
   the same restricted candidate sets with Stage 2 pointwise, contextual Qwen4B,
   and contextual E5.
6. Write a separate JSON and Markdown result receipt. Preserve the previous Stage 1
   and Stage 2 reports byte-for-byte.

Expected artifact directory:

`artifacts/llm_label_ranker/pilot/stage2/listwise_fixed_scores/`

Expected report:

`docs/results/2026-07-13-llm-ranker-fixed-score-listwise-pilot.md`

## Evaluation and Decision

Report nDCG@10, MRR, relevant-label top-three count, per-query best relevant rank,
score ties, and comparisons with the three captured methods. Clearly label the
evidence qrel-assisted and exploratory because the Stage 2 questions/candidates
were selected with development qrels.

The diagnostic passes its exploratory expansion gate only if:

- all 12 outputs satisfy the fixed-key contract;
- its top-three relevant count is at least the best embedding baseline; and
- its overall nDCG@10 is not below the best embedding baseline.

No larger run, fusion, model selection, or deployment action starts automatically.

## Tests

Use test-driven development for:

- exact fixed-key prompt and schema formatting;
- pinned-backend CPU schema compilation where available;
- duplicate-key, missing-key, extra-key, type, and range rejection;
- deterministic score sorting and tie behavior;
- raw-output persistence before parse failure;
- qrel mutation independence for prompts/schemas/rankings;
- qrels remaining unopened until the qrel-free receipt validates;
- exact twelve-query/twenty-candidate inventory and deterministic report writing.

The GPU run is authorized only after these CPU tests and the resource preflight
pass.
