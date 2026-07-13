# LLM Expert-Label Ranker Feasibility Pilots

## Goal

Determine cheaply whether an instruction-tuned LLM contains useful signal for ranking
expert topic labels for a question. This is a qrel-assisted diagnostic, not a zero-shot
benchmark or deployment-model selection exercise.

The completed 322×150 listwise run remains preserved as exploratory evidence. Do not
repeat it or start any other full sweep during these pilots.

## Frozen Ranker

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Runtime: the existing pinned vLLM ROCm container and validated official chat-template
  path
- Representations: cleaned expert labels plus the frozen expert-capability sentence
- Methods: pointwise expected relevance grade and listwise ranking
- Seed: 2026

## Stage 1: Small Pilot

Run six questions with twelve candidate expert labels per question.

### Question set

Select six questions before inference:

1. the two issue #11 examples;
2. two questions where contextual label embeddings performed well; and
3. two questions where contextual label embeddings failed or confused related topics.

Prefer distinct question types and subject areas. Record the selection rule and exact
query IDs. Selection may inspect existing embedding results and qrels because this pilot
is diagnostic.

### Candidate sets

For each question, freeze twelve unique candidates:

- up to two highest-grade qrel-relevant experts;
- four hard confounders: nonrelevant labels that are semantically or lexically close to
  the question or relevant label; and
- enough deterministic negative labels to reach twelve candidates.

Write two separately hashed manifests. `selection.json` contains only query IDs, question
text, candidate IDs, and non-qrel source hashes; it is the only manifest Phase A may
open. `selection_judgments.json` contains qrel-derived roles and grades and remains
unavailable until reporting. Do not include qrel grades, responses, discovery profiles,
or expert exemplars in prompts.

### Inference

- Pointwise: score all 72 question-candidate pairs using the existing exact grade-token
  probability method.
- Listwise: show the same twelve candidates, request a complete ordering of all twelve,
  and run three deterministic balanced presentation rotations per question (18 calls).
- Preserve prompts, raw outputs, parsed outputs, scores, rankings, runtime metadata, and
  hashes.
- Do not run repeats in Stage 1.

### Evidence and decision

Compare both LLM methods against contextual Qwen Embedding 4B and E5-Mistral scores on
the identical restricted candidate sets. Report every question in a compact table with:

- relevant-label ranks;
- top three predictions;
- rank differences versus both embedding baselines;
- listwise rotation agreement and any parse/fallback event; and
- a short manual error classification.

Stage 1 shows useful signal only if at least one LLM method places a relevant expert in
the top three for at least four of six questions and the wins are not all caused by one
subject area or fallback behavior. This threshold is a feasibility heuristic, not a
statistical test.

## Stage 2: Conditional Larger Pilot

Run Stage 2 only if Stage 1 shows useful signal. Otherwise stop and discuss the failure
examples.

Stage 2 uses twelve questions with twenty candidates per question. Preserve the same
question-selection strata, candidate construction, ranker, prompts, three listwise
rotations, baselines, and evidence table. Add different questions rather than reusing
the Stage 1 six so the second stage tests whether the observed signal generalizes.

Stage 2 is still diagnostic. Do not trigger a full run, fusion, model sweep, or deployment
decision automatically.

## Validation and Failure Handling

- Freeze question and candidate manifests before inference.
- Verify prompts contain no grades or qrel-derived prose.
- Apply the official chat template exactly once.
- Pointwise grades remain distinct constrained single tokens.
- Listwise output must contain every pilot handle exactly once; reject duplicates and
  unknown handles. Record failures explicitly rather than silently inventing ranks.
- Qrel mutation must not change prompts, outputs, scores, or rankings for an already
  frozen pilot.
- The advisor-requested provenance hardening applies to pilot receipts: exact prompt,
  input, smoke, model, runtime, and code hashes are required.

## Deliverables

- one frozen qrel-free Stage 1 selection manifest and one sealed judgment manifest;
- qrel-free raw pointwise and listwise artifacts;
- one question-level feasibility report and plain-language decision receipt;
- Stage 2 artifacts only if the Stage 1 rule passes; and
- no automatic continuation beyond Stage 2.
