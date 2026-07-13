# Corrected label-embedding sweep design

## Goal

Replace the invalid Qwen and E5 Stage 1 results with reference-compatible runs, then
measure whether a fixed expert-capability sentence represents candidate labels better
than the original bare names.

## Frozen conditions

Each executed encoder uses two qrel-free document representations:

1. `bare`: the existing cleaned candidate label.
2. `expert_capability`: `This LLM is most likely to correctly answer questions about
   {cleaned_label}.`

Queries retain the existing frozen instruction. Scores remain cosine similarity, exact
ties remain candidate-ID ordered, and qrels remain unavailable until rankings are fixed.

## Reference-compatible preprocessing

Qwen uses the pinned checkpoint's Transformers reference tokenization: tokenize without
padding at `max_length - 2`, append `<|endoftext|>` to every query and document, then
left-pad and apply normalized last-token pooling. Token-ID prompts are passed to pinned
vLLM so its generic tokenizer cannot omit the terminal token.

E5 uses the pinned checkpoint's legacy Transformers 4.34 semantics: tokenize with a
beginning token and terminal `</s>`, left-pad, then apply normalized last-token pooling.
The implementation constructs and validates the terminal token explicitly so behavior
does not depend on Transformers 5 tokenizer migration rules.

## Validation

- Unit tests freeze both label templates and exact terminal-token construction.
- Boundary tests verify truncation leaves room for the terminal token.
- Qwen must reproduce the official China/gravity diagonal score structure.
- E5 token IDs must match a Transformers 4.34 reference fixture.
- Each of the four encoder/representation conditions runs twice. Rankings and metrics
  must be identical and scores must agree within `1e-5`.

## Decision

The formal advancement gate is reported separately for every condition against MPNet.
Stage 2 becomes eligible if a corrected Stage 1 condition passes both performance rules
and repeat validation. No deployment model or fusion is selected automatically.
