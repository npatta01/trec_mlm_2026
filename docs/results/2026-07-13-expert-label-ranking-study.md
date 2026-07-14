# Expert-label ranking study: what worked, what did not, and what to keep

[Open the standalone accessible HTML version](./2026-07-13-expert-label-ranking-study.html).

## Decision

Carry forward **contextual E5** as the stable, qrel-free question-to-expert-label
method. Do not carry the experimental pointwise, generated-listwise, GroupRank,
FirstMistral, sweep, or fusion harnesses into the production path.

This is a selection for stability and maintainability, not a claim that E5 is a
statistically proven winner. On all 322 development questions, contextual E5
reached **0.3347 nDCG@10**, **0.3866 MRR**, and **120 zero-nDCG questions**.
Qwen3-Embedding-4B had a slightly higher point estimate at 0.3420 nDCG@10, but
the difference was small and its strict all-150 repeat check failed on two
adjacent tail orders. E5's paired 95% nDCG@10 interval against MPNet also
included zero, so it should be described as the selected clean control rather
than a confirmed improvement.

The LLM-ranker line is closed for now. A stable purpose-trained listwise pilot
was competitive on 12 questions, but it remained below contextual E5 on the same
restricted candidate sets. Scaling it to all 322 questions is not justified.

## Where we started

The original hypothesis was deliberately simple: embed a user question and a
cleaned topic label, then use cosine similarity to rank labels representing the
LLM experts most likely to answer the question correctly. This is attractive
because it is zero-shot, needs no qrel-trained expert representation, returns
multiple topics naturally, and can be run before any judgments exist.

The formal MPNet raw-label control was **0.3195 nDCG@10**, **0.3674 MRR**, and
**131 zero-nDCG questions**. A query-neighbor reference, which represents a new
question using other questions rather than label names, was stronger at 0.3599
nDCG@10 and 0.4041 MRR. That gap suggested that short labels omit useful
semantic context and motivated two bounded questions:

1. Can instruction-aware encoders understand expert labels more effectively?
2. If embeddings plateau, can an LLM explicitly rerank a small label set?

## What we tried

| Step | Scope | Outcome | What it established |
| --- | --- | --- | --- |
| MPNet raw-label control | 322 questions × 150 labels | 0.3195 nDCG@10 / 0.3674 MRR / 131 zeros | Direct question-to-label matching is viable but limited. |
| Initial Qwen8B and E5 sweep | 322 × 150 | Invalid low scores | Required model-specific terminal-token behavior was omitted. The runtime does not invent each embedding checkpoint's asymmetric contract. |
| Corrected large sweep | Qwen3-Embedding-8B and E5-Mistral-7B | Contextual E5: 0.3347 / 0.3866 / 120 zeros | Explicit formatting restored valid results; E5 improved modestly but did not pass the statistical MPNet gate. |
| Small diverse sweep | Qwen 0.6B/4B, Jina, BGE-M3, Nomic | Contextual Qwen4B: 0.3420 / 0.3902 / 115 zeros | Best point estimate, only +0.0073 nDCG over E5, with two tail-order repeat failures. |
| Six-question LLM pilot | Pointwise and generated listwise | Pointwise 5/6 top-three; listwise 3/6 | Pointwise earned one larger diagnostic; generic listwise formatting was unreliable. |
| Twelve-question pointwise pilot | 12 questions × 20 labels | 0.4820 / 0.3209 / 5 of 12 top-three | The early signal did not scale; E5 remained much stronger. |
| Generic generated-list pilot | 12 × 20 | Unscorable | The output was not a complete machine-valid permutation, so no quality result was claimed. |
| GroupRank 7B smoke | Two groupwise prompt protocols | No quality result | Both exact-output contracts failed. This was an operational format result, not evidence that the ranking algorithm was poor. |
| FirstMistral listwise closure | Purpose-trained first-token logit ranker, 12 × 20 | 0.6843 / 0.5847 / 8 of 12 top-three | Stable via Transformers and genuinely useful, but below E5 on both headline metrics. Stop rather than scale. |

## Full embedding sweep

These are the strongest contextual or capability-label conditions for the small
and diverse encoders, evaluated on the same 322 questions and 150 labels.

| Encoder | nDCG@10 | MRR | Zero nDCG | Strict repeat |
| --- | ---: | ---: | ---: | --- |
| MPNet raw-label control | 0.3195 | 0.3674 | 131 | Pass |
| E5-Mistral 7B contextual | 0.3347 | 0.3866 | 120 | Pass |
| Qwen3-Embedding 8B contextual | 0.3281 | 0.3796 | 131 | Pass |
| Qwen3-Embedding 4B contextual | **0.3420** | **0.3902** | **115** | Fail: two adjacent tail swaps |
| Qwen3-Embedding 0.6B contextual | 0.3000 | 0.3412 | 131 | Pass |
| Jina v5 small contextual | 0.3122 | 0.3711 | 132 | Pass |
| BGE-M3 contextual | 0.2576 | 0.3197 | 150 | Pass |
| Nomic Embed v1.5 contextual | 0.2591 | 0.3259 | 154 | Pass |

Contextual E5 improved over MPNet by 0.0152 mean nDCG@10, but its paired
10,000-resample 95% interval was **[-0.0125, 0.0424]**. It satisfied the zero
count limit but failed the requirement that the full interval be above zero.
Its win/tie/loss count against MPNet was 110/126/86. The correct conclusion is
"stable selected method," not "confirmed MPNet replacement."

Capability sentences helped E5 by 0.0240 nDCG over its bare-label condition,
with a paired 95% interval of [0.0004, 0.0479]. They also improved contextual
Qwen4B, Qwen0.6B, and Jina in exploratory within-model comparisons, but Qwen8B
was essentially unchanged and BGE-M3 decreased. Richer label text is therefore
a tested representation choice, not a universal rule.

## LLM ranker findings

The decisive pilot used the same 12 qrel-assisted questions and 20 candidate
labels per question for every method below. These restricted metrics are not
comparable to the 322×150 full-sweep numbers.

| Method | nDCG@10 | MRR | Relevant in top 3 | Decision |
| --- | ---: | ---: | ---: | --- |
| Contextual E5 | **0.7232** | **0.6361** | **8/12** | Best frozen baseline |
| FirstMistral listwise | 0.6843 | 0.5847 | **8/12** | Stable; fails E5 quality gate |
| Contextual Qwen4B | 0.6648 | 0.5585 | **8/12** | Complete restricted baseline |
| Pointwise Qwen | 0.4820 | 0.3209 | 5/12 | Does not scale from Stage 1 |

FirstMistral was not discarded because of output formatting. Its Transformers
fallback produced byte-identical scores and rankings in two fresh processes.
Across the 12 questions it ranked the best relevant label higher than E5 on 4,
tied on 3, and ranked it lower on 5. It recovered useful cases—`Computing` was
rank 1 versus E5 rank 5, and one `Natural_environment` case was rank 2 versus E5
rank 5—but placed `Sports_events` at rank 10 versus E5 rank 5 and lost on several
clear labels. The aggregate quality gate reflects that uneven trade-off.

| Question | Relevant label | FirstMistral rank | E5 rank | FirstMistral vs E5 |
| --- | --- | ---: | ---: | --- |
| Who sets individual degree requirements at Northwestern? | Educational organizations | 3 | 1 | Worse |
| Who wrote “A Kind of Alaska”? | Literature | 1 | 1 | Tie |
| Paloma and Monkey Gland are what type of item? | Food and drink | 1 | 1 | Tie |
| Do Sinningia or Elodea have more species? | Flora | 1 | 3 | Better |
| What was the Swiss quota-system rejection rate? | Politics | 4 | 5 | Better |
| Which former dog track was in Wood Village, Oregon? | Sports venues | 4 | 1 | Worse |
| Where were Dinosaur Jr. formed and M83 based? | Popular music | 1 | 1 | Tie |
| In what ocean was the final 1991 hurricane-season cyclone? | Natural environment | 3 | 1 | Worse |
| When was USB Type-C Specification 1.0 finalized? | Computing | 1 | 5 | Better |
| Juan Sebastián Cabal and John Lloyd play what sport? | Sports events | 10 | 5 | Worse |
| What disorder did “You're Only Human” refer to? | Human behavior | 4 | 2 | Worse |
| What vegetation do Lythrum and Erythrina share? | Natural environment | 2 | 5 | Better |

## Bugs and invalid results

The study separates implementation defects from negative method results:

- **Initial embedding prompt/token bug.** The original Qwen8B and E5 runs did
  not reproduce each model's required terminal-token construction. Those scores
  are superseded and must not be cited.
- **Server templating misconception.** vLLM handles token execution and batching;
  it does not guarantee a checkpoint's query instruction, query/document
  asymmetry, EOS handling, or pooling semantics unless that model integration
  explicitly provides them.
- **Generated-list failures.** Generic listwise and both tested GroupRank
  protocols failed strict complete-permutation validation. They have no quality
  result. Few-shot examples might improve formatting, but would not rescue a
  method that has not yet produced a valid ranking.
- **FirstMistral vLLM repeat failure.** Both vLLM runs completed, but 8 of 12
  rankings changed; 240/240 scores exceeded `1e-5`, with a maximum absolute
  difference of 0.1858. The evaluator rejected the result before opening qrels.
- **FirstMistral stable resolution.** The pinned Transformers path produced
  identical rankings and scores in two fresh processes. It then failed the
  quality gate, making the stop decision about measured quality rather than
  formatting.
- **Retained-artifact validator bugs.** The clean E5 smoke test caught that
  Transformers 5.12.1 does not expose the tokenizer commit in
  `tokenizer.init_kwargs`. The implementation now resolves the cached immutable
  snapshot path directly. A second test caught cross-BLAS tail-order drift when
  recomputing float32 dot products on a different host library. The final format
  persists the exact float32 score matrix, verifies a recomputation within
  `1e-6`, and validates rankings against the exact persisted scores. The final
  advisor review also removed embedding injection from the public runner,
  required runtime metadata to match the exact formatted-input hash, verified a
  Torch HIP build on an AMD `gfx` device, and added semantic validation of the
  human-readable prediction ranks.

## What is retained in the final branch

Only the contextual E5 solution, its focused tests, the thin generic adapter,
and this Markdown/HTML report are retained. Experimental model registries,
sweeps, pointwise/listwise rankers, GroupRank, FirstMistral, fusion, and planning
harnesses are intentionally excluded.

The authoritative configuration remains Python:

```python
from trec_mlm.label_experiment import (
    LabelExperimentConfig,
    run_label_experiment,
    write_label_experiment_artifacts,
)

config = LabelExperimentConfig(
    query_path="data/dev/llm_dev_data.tsv",
    candidate_path="reports/trec_2026_discovery_profile_audit.json",
    batch_size=32,
    seed=2026,
    run_id="contextual_e5_labels",
)
result = run_label_experiment(config)
write_label_experiment_artifacts(result, "artifacts/contextual_e5")
```

The existing generic CLI is only a thin adapter over that configuration:

```bash
trec-mlm label-retrieval \
  --queries data/dev/llm_dev_data.tsv \
  --candidates reports/trec_2026_discovery_profile_audit.json \
  --batch-size 32 \
  --seed 2026 \
  --output-dir artifacts/contextual_e5
```

There is no encoder switch, representation switch, sweep command, qrel argument,
or public embedding-injection hook in this ranking path. If development
judgments are available, the resulting frozen TREC run can be inspected
separately:

```bash
trec-mlm score-run \
  --qrels data/dev/llm_dev_qrels.txt \
  --run artifacts/contextual_e5/run.trec
```

That separate command is for evaluation and failure inspection. It is not a
post-ranking transformation and cannot alter embeddings, scores, or rankings.

## Frozen contextual E5 contract

| Item | Frozen value |
| --- | --- |
| Model | `intfloat/e5-mistral-7b-instruct` |
| Revision | `d62cc4417be0e3289c03dd459d1ef2d137e2554e` |
| License | MIT |
| Task instruction | `Given a user question, retrieve topic labels describing the LLM experts most likely to answer it correctly.` |
| Query template | `Instruct: {instruction}\nQuery: {question}` |
| Document template | `This LLM is most likely to correctly answer questions about {cleaned_label}.` |
| Label cleaning | Replace underscores and parentheses with spaces; collapse whitespace |
| Token contract | Left padding; append exactly one EOS token ID 2 (`</s>`); maximum length 512 |
| Numeric contract | BF16 inference; full 4096 dimensions; normalized last-token pooling; cosine similarity |
| Ranking contract | All 150 candidates; descending score; exact ties by candidate ID |
| Seed and batch | Seed 2026; batch size 32 |
| Required container image | `vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28` |

Although the image includes vLLM, E5 uses its official Transformers
last-token-pooling implementation inside that digest-pinned ROCm environment.
The artifact records this digest as a required reproducibility contract rather
than pretending it can cryptographically attest its own OCI image from inside
the container. Runtime generation separately verifies a Torch HIP build and an
AMD `gfx` device.

## Final clean-run verification

Two fresh processes ran the advisor-reviewed implementation from clean artifact
commit `5bb68a24b17f33e937bd84f4c692c983a6f49afa` (implementation fix commit
`a142b3f`). Both artifacts recorded a clean worktree. The independent loader
validated both complete directories.

| Check | Result |
| --- | --- |
| Queries × candidates | 322 × 150 |
| Complete scores / TREC rows | 48,300 / 48,300 |
| Query embedding shape | 322 × 4096 |
| Label embedding shape | 150 × 4096 |
| Rankings identical | Yes |
| Query embeddings identical | Yes |
| Label embeddings identical | Yes |
| Scores identical | Yes |
| Maximum absolute score difference | 0.0 |
| Post-hoc nDCG@10 / MRR | 0.3347215379757303 / 0.38656649577390506 |
| Automated suite | 126 tests and 11 subtests passed |

Runtime provenance was identical except for timing: 36.73 and 37.26 seconds.
Both runs used Transformers 5.12.1, PyTorch `2.11.0+gitd0c8b1f`, HIP
7.2.53211, BF16, and gfx1151 ROCm. The model had 7,110,660,096 parameters. Peak
allocated/reserved memory was 15,065,729,536 / 16,252,928,000 bytes. All 472
formatted inputs fit without truncation; the longest was 144 tokens.

Input and implementation hashes:

| Object | SHA-256 |
| --- | --- |
| Questions TSV | `862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c` |
| Candidate manifest | `4a770b919024dffd1cffe85edd2f0219aeec31cdbb2a96d1005861fa1bf03f91` |
| All formatted E5 inputs | `c0b8f381a064a4f4a361830ecaad363d936c85f711622b0352ced27dbef7653e` |
| `label_retrieval.py` | `5228a70af7f60105af4317530377f29aaed5b4e9e895fea67a27e8ed3a2f44c6` |
| `label_inference.py` | `c7def72cb291a72d8033c02cbbd32f94c28f5d3afd9ffa5bc028f3719ca8e9e4` |
| `label_experiment.py` | `4aba3cad56eff57e89c8d8d7c4b9f009f371d7a69cce36d174b63a295b711279` |

Hashes common to both final runs:

| Artifact | SHA-256 |
| --- | --- |
| Input inventory | `8a87030512adecf5766e95368e9f7cdd60eea97fea9d078d0de46b8f836ebbd7` |
| Query embeddings | `8394dfc3f66093941d39f19a658b5744709917d0adcd220ef7b494b140e0f55d` |
| Label embeddings | `1687b0e80a17cb2fc38977405881041e1bfdb0016936f67f9ef46a8aeb51c36a` |
| Binary score matrix | `f56a69a66ad12ab3cb7166ca75bab9a4489ce803ed3b37df3803c27eae805546` |
| Human-readable predictions | `583f61d7fb1550bd54a79c9fbd154d5502f05cb35efb1189696cb448eef27950` |
| Rankings | `a88a0454494098b683eb745bd9ec909b9f4266c94ac2a538b2b552d95754681f` |

Run-specific receipts differ only where run ID or timing is recorded:

| Artifact | Final run 1 | Final run 2 |
| --- | --- | --- |
| Config | `941c0df09fdb72ada87e06b9228e848220ae01a47436b7e0af7ed6bc08536bf4` | `558647e683f3535347eef26d1918a7aa4eb1f6fcae65a0893d39a9966fa8e52f` |
| Encoder metadata | `1bbbdc3b53111d3c8a1386beb159c6af761fb3e61d0286f0ca06a11a4fbea147` | `477be83960ad87e806a715027e47bbed49306e124ba636c0ae965324d85dfb75` |
| TREC run | `2ecefabf448c1b99a19a69a7157407751620c0660196b8897519b785a69646c0` | `262d095fd066b683517f43d10fc4f513d9c8b4b81a5110260dabc2d46111dae2` |
| Validation receipt | `1bc7a6a666279f20a5f03bcbf9ccfdb4f8161d68dd14e046451ceefd350a98a0` | `c929738b3bc7952a48ea042328549877ae760319fb7289d9b30141b5ee27d5ec` |

## Historical E5 parity and limitations

The retained implementation reproduces the corrected historical E5 embeddings
exactly. Its rankings match the historical TREC ranking, and the maximum
difference against the historical text score file is
`5.000444502911705e-13`—only decimal serialization.

The historical runner stored the qrel path in its experiment config because it
performed evaluation in the same workflow. Its embedding and ranking stage was
qrel-free, but that API boundary was too easy to misuse. The retained runner
does not accept qrels at all. The historical config also recorded
`code_revision: null` and `git_worktree_status: null`; therefore the exact
historical source checkout cannot be recovered from that manifest alone. The
clean-run source commit and per-file hashes above close that provenance gap.

Historical corrected E5 validation receipts:

| Artifact | Historical run 1 | Historical run 2 |
| --- | --- | --- |
| Config | `22a9326cb640f239791c3a2f74cefc91a8de87f3a5c2be93eab6acdd3aadd467` | `22a9326cb640f239791c3a2f74cefc91a8de87f3a5c2be93eab6acdd3aadd467` |
| Encoder metadata | `c0a7cc2ad188b0172e83e64c40ee924ec57d9cf4e178c213aad423464946d408` | `ee1c1d5b5db9e813a76d66ba26bb248a96c9aa780fe9b3a16df3cad1b34bce86` |
| Query embeddings | `8394dfc3f66093941d39f19a658b5744709917d0adcd220ef7b494b140e0f55d` | same |
| Label embeddings | `1687b0e80a17cb2fc38977405881041e1bfdb0016936f67f9ef46a8aeb51c36a` | same |
| Scores TSV | `a57aadbca83aaeb87091dc77c68fe982e0ed84cfe35f57aeee18abf7bbfdeb7c` | same |
| Per-query metrics | `859ebbb73d760c1f0d4cc677e77e9b883453c4e89cf9344b002a3606ae7a43fa` | same |
| TREC run | `98b5a2d7e90a76b4e39e435a83735389717f1d5f071eef43ac85a8a464c54156` | same |
| Summary | `367bb6127bceb2f7d19f8ed8dbafe32295f1ed4b3fc262e0a9c3961325e1050c` | same |
| Validation file | `e3497d3eff4e19d5bb1549cf9fd28a9a3a5158573d13a03c288a999d0214c296` | `9c14324dc30f6161501b812fbb43db5ab1eed668de453edd4fb081cb10ae0aad` |

The corrected combined capability-sweep summary has SHA-256
`856e2865902a4999ef40beaf1e81c3d8dc46dafa30235ab5824c05b06205b31c`.

## What to carry forward

1. **Use contextual E5 as the clean zero-shot control.** It is stable, simple,
   fully reproduced, and the strongest method that satisfies the strict
   operational contract.
2. **Keep ranking qrel-free by construction.** Evaluation and failure inspection
   consume a frozen run; they never participate in representation or ranking.
3. **Preserve exact model-specific contracts.** Prompt asymmetry, EOS handling,
   padding, pooling, dtype, revision, and tie rules are part of the method.
4. **Use richer descriptions selectively.** Capability text helped several
   families but hurt or did nothing for others.
5. **Treat full-ranking repeatability separately from top-rank quality.** Qwen4B's
   two swaps were in the tail and did not change its headline metrics, but the
   frozen contract required all 150 positions to repeat.
6. **Represent multi-topic expertise with multiple descriptors or vectors.** A
   single averaged embedding can blur an expert that is strong in unrelated
   domains. Several human-readable descriptors are the cleanest next zero-shot
   hypothesis.
7. **Do not retain experimental ranker code.** The negative and invalid results
   are preserved here with hashes; the disposable experiment worktree may be
   deleted after this branch is reviewed.

## Recommended next experiments

1. Build **multiple qrel-free natural-language descriptors per expert label**,
   covering meaning, aliases, scope, and representative subtopics. Score a query
   against every descriptor and aggregate by maximum or a pre-frozen softmax.
2. Freeze descriptor creation, candidate inventory, aggregation, and the
   comparison gate before looking at evaluation judgments.
3. Compare against contextual E5, MPNet raw labels, and the query-neighbor
   reference on a held-out split. Report both top-rank metrics and strict
   repeatability.
4. If later scope permits success-query exemplars, represent each expert with
   several held-out topic clusters rather than one centroid. Keep profile-building
   questions disjoint from evaluation questions.
5. Revisit an LLM ranker only with materially better evidence—for example richer
   expert profiles or a ranker trained for this taxonomy—not just more examples
   teaching the same output format.

## Suggested GitHub issue #11 update after merge

Do not post this automatically. After the final PR is merged, adapt the following
outline with the merge link:

> We completed the zero-shot question-to-expert-label study. The original low
> Qwen8B/E5 results were invalid because the embedding token contract was not
> reproduced. Corrected contextual E5 reached 0.3347 nDCG@10, 0.3866 MRR, and
> 120 zero-nDCG questions on 322×150. Contextual Qwen4B had the highest point
> estimate (0.3420) but failed the strict full-ranking repeat rule on two tail
> swaps and was not significantly better than E5. We are carrying contextual E5
> forward as the stable qrel-free control, not as a statistically proven MPNet
> winner. Pointwise and listwise LLM pilots did not beat E5 on the shared
> 12-question diagnostic; GroupRank never produced a valid output, while stable
> FirstMistral reached 0.6843 versus E5's 0.7232 nDCG@10. The next recommended
> experiment is a frozen multi-descriptor or multi-vector expert representation,
> evaluated on held-out questions. Report and implementation: `<merged PR URL>`.

## Scope and interpretation

- The full embedding sweep used all 322 questions and 150 labels. The LLM
  diagnostics used qrel-assisted selections of 6 or 12 questions and restricted
  candidate sets. Their absolute metrics are not directly comparable.
- The sweep and ranker pilots were designed after inspecting development results;
  all model comparisons are exploratory development evidence.
- GroupRank has no quality result. Its tested output protocols were unreliable;
  this does not prove that every possible GroupRank implementation is poor.
- The conclusion is "abandon this listwise line for now," not "listwise ranking
  is impossible."
- No deployment model is selected automatically, and no fusion work begins from
  this report. Discuss the next representation experiment first.

## Other preserved evidence hashes

These receipts identify the broader experimental evidence summarized above;
they are not required by the retained runtime.

| Evidence | SHA-256 |
| --- | --- |
| Corrected large embedding report | `408d20eb8a8a76be121d9b858cd494680be0afc278cad70577e313e16be8bae5` |
| Small diverse embedding report | `d30fb64e5cd0a604fe716b1c01761245fa0d7aad13023180f71e27b6421f20b5` |
| Stage 1 ranker report | `d1582ca85388b2447fa28539f1f3077120d787e984a33ce637d713370e5e38c7` |
| Stage 2 baseline report | `14fda5fdae7d3bc25fbb138f89332e62bc12871d89a4c9b3aa61e510880980a1` |
| GroupRank result report | `b22fce67b105b061f63712f939daa6ab2a40bae3dd3cf5e275f36a5f14b64296` |
| FirstMistral sealed report | `89cd3b811e85d2937152ad8b1e4b069c01aabf2fed6788c81a4149017b3a677a` |
| FirstMistral repeatability receipt | `cb21835572e867c646b875958ab1769619d0db7932b4c30576bccb333a4afead` |
| FirstMistral vLLM failure receipt | `759e7216b96a6ab85ecfe8c1a9f0dc2f1e891ebae27ce73663c624764aec6be9` |
| FirstMistral selection | `668515c30a80473aaaa2dc530f50dfd3b7731711e9f9adf70074ebb95aa98b9d` |
| Sealed FirstMistral judgments | `8ced65041023f25ab157fdfa9e6a34397505fed875c1e0687527638f4285e83c` |

This report is the durable record. No essential finding depends on retaining the
experimental worktree.
