# Issue 18 viability-receipt repair design

Status: approved by the user on 2026-07-15 after qrel-free advisor review.

## Problem

Pass one froze a valid multi-resolution artifact with aggregate
`2a3909cfea23af246eed59565557d57ec4c5b8eb1e2f1d5c2b11d2f9f8c6d5b9`.
Qrels remain sealed. The receipt correctly found localized candidate dominance,
but it has three procedural defects:

1. it labels every non-integrity failure `stop-no-incremental-value`, although
   every incremental-distinctness threshold passed;
2. it compares new systems only with C0, not with both required external
   controls (query-kNN and contextual E5);
3. evaluation authorization is incorrectly coupled to repeat authorization.

The artifact provenance also omits the actual execution-code commit, exact
command, dependency versions, and construction timestamp.

## Frozen boundary

The existing profiles, embeddings, component scores, system scores, rankings,
runs, manifest, and aggregate are immutable. The repair may read those files
and the qrel-free issue-12 control rankings, but it may not reconstruct or
replace a ranking. The second construction repeat remains prohibited.

## Supplemental receipt

Regenerate the qrel-free viability receipt from the strictly validated frozen
artifact. Preserve all existing integrity, cluster, ranking, cost, and
threshold evidence, and add:

- a decision of `stop-query-sensitivity` when integrity and incremental
  distinctness pass but dominance/query-sensitivity thresholds fail;
- exact source identities and hashes for query-kNN (`dense_reference`) and
  contextual E5 (`contextual_e5_control`);
- for every one of the eight frozen systems versus each control: mean top-10
  Jaccard, top-10 difference-query share, mean top-10 overlap count, and mean
  absolute rank displacement across the common top-10 candidates;
- execution provenance: actual code commit, exact command and deterministic
  thread environment, Python/NumPy/SciPy/scikit-learn versions, construction
  completion timestamp, artifact aggregate, and receipt-generation timestamp.

The contextual-E5 ranking must be strictly bound to the already validated
issue-12 source aggregate and candidate/query inventories. Receipt validation
recomputes the decision, baseline comparisons, provenance schema, and artifact
binding rather than trusting editable JSON fields.

## Authorization semantics

- `reject-invalid`: neither repeat nor evaluation is permitted.
- `stop-query-sensitivity` or `stop-no-incremental-value`: repeat is forbidden,
  but the one preregistered post-freeze diagnostic evaluation is permitted.
- `continue`: evaluation is permitted; repeat additionally requires the
  existing wall-time and storage limits.

Evaluation must validate the complete receipt and recheck its byte hash before
the first qrel hash/read. It still accepts no tuning options.

## Tests and review

Tests must fail before implementation and cover control identity/hash mismatch,
inventory mismatch, exact comparison recomputation, decision classification,
provenance completeness, valid-stop evaluation authorization, invalid-artifact
rejection, repeat prohibition, and qrel access ordering. The full suite and
changed-file lint must pass. An independent advisor reviews the repair before
qrels are opened.

## Frozen-snapshot loader correction

Final combined review rejected the original repair because it reused the
semantic reconstruction loader. The user approved a loader redesign before
qrels were opened.

The authoritative receipt and evaluator must use a dedicated frozen-snapshot
loader that never calls `construct_multiresolution`, `canonical_kmeans`,
`score_profiles`, `_freeze_systems`, or reciprocal-rank fusion. It must:

- accept only the canonical artifact root
  `artifacts/multiresolution_profiles/issue18_pr17head`;
- require exact aggregate
  `2a3909cfea23af246eed59565557d57ec4c5b8eb1e2f1d5c2b11d2f9f8c6d5b9`;
- verify every manifest file hash/size, exact schema and config, source binding,
  322-query/150-candidate identities, finite arrays, normalized centroids,
  complete rankings, and ranking/score consistency;
- compare frozen C0 profiles, scores, and rankings directly with their exact
  issue-12 source files without reconstructing them;
- trust the earlier semantic validation only because the exact immutable
  aggregate is pinned.

Receipt regeneration is locked to
`reports/trec_2026_issue18_pr17head_viability.json`, which must resolve outside
the artifact tree and must not already exist. Evaluation is locked to that
receipt, canonical qrels `data/dev/llm_dev_qrels.txt` with SHA-256
`3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35`, and the
single non-overwriting output
`reports/trec_2026_issue18_pr17head_evaluation.json`. These identities eliminate
arbitrary alternate evaluation paths; deleting sealed evidence is outside the
authorized workflow and must not be automated.

