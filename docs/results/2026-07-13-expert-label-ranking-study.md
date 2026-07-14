# Expert-label ranking study: what worked, what did not, and what to keep

[Open the standalone accessible HTML version](./2026-07-13-expert-label-ranking-study.html).

## Technical summary

**Decision: stop the current listwise-ranking line and do not run it on all 322
questions.** The stable FirstMistral listwise pilot reached 0.6843 nDCG@10,
0.5847 MRR, and a relevant label in the top three for 8 of 12 questions. That is
useful signal—it beat the pointwise pilot and contextual Qwen4B on the first two
metrics—but it did not beat the clean contextual E5 baseline at 0.7232 nDCG@10,
0.6361 MRR, and 8 of 12 top-three hits.

The final result is valid. FirstMistral's vLLM path was not repeatable on this
ROCm host, but the frozen Transformers fallback produced byte-identical rankings
and scores in two fresh processes. The maximum score difference was 0.0 across
all 240 question–label pairs. This separates the earlier runtime problem from the
quality result: formatting and repeatability were fixed, and the stable method
still did not clear the E5 gate.

For the original zero-shot question-to-label idea, **contextual E5 is the best
clean carry-forward control**. Contextual Qwen4B had a slightly higher full-run
point estimate (0.3420 versus 0.3347 nDCG@10), but the difference was only 0.0073,
its paired interval included zero, and it failed the strict full-ranking repeat
rule because two adjacent tail orders changed. In practical top-rank behavior it
did not differ much.

## Where we started

The starting hypothesis was simple: map a question directly to a cleaned expert
topic label with an embedding model. This is a reasonable zero-shot formulation
because it requires no qrel-trained expert representation and naturally returns
more than one topic when a question or expert spans multiple areas.

The formal MPNet raw-label control on 322 questions was 0.3195 nDCG@10, 0.3674
MRR, and 131 zero-nDCG questions. A separate query-neighbor reference—representing
questions with other questions rather than label names—was stronger at 0.3599
nDCG@10 and 0.4041 MRR. That gap motivated two questions:

1. Can instruction-aware encoders understand label semantics better?
2. If embeddings plateau, can an LLM explicitly rank a small candidate set?

## What we tried

| Step | Scope | Outcome | What it established |
| --- | --- | --- | --- |
| Raw-label baseline | 322 questions × 150 labels | MPNet: 0.3195 nDCG@10 | Direct question-to-label matching is viable but limited. |
| Large instruction-aware sweep | Qwen3-Embedding-8B and E5-Mistral-7B | Initial low scores were invalid | Newer generic runtime behavior omitted required terminal tokens; vLLM does not invent each embedding model's asymmetric formatting. |
| Corrected large sweep | 322 × 150, explicit model formatting | Contextual E5: 0.3347 / 0.3866 / 120 zeros | Richer capability text helped E5, but not enough to prove a gain over MPNet. |
| Small diverse sweep | Qwen 0.6B/4B, Jina, BGE-M3, Nomic | Contextual Qwen4B: 0.3420 / 0.3902 / 115 zeros | Best point estimate, but only +0.0073 nDCG over E5 and strict repeatability failed on two tail swaps. |
| Six-question LLM pilot | Pointwise and generic listwise | Pointwise 5/6 top-three; listwise 3/6 | Pointwise looked feasible on a tiny qrel-assisted sample, so a larger diagnostic was allowed. |
| Twelve-question pointwise pilot | 12 questions × 20 labels | 0.4820 / 0.3209 / 5 of 12 | The early signal did not scale; E5 remained much stronger. |
| Generic listwise pilot | 12 × 20 | Unscorable | The generated permutation failed strict parsing; no ranking or quality claim was valid. |
| GroupRank 7B smoke | Purpose-trained groupwise model | No quality result | Two prompt protocols failed their exact output contract. This was a format result, not an algorithm-quality result. |
| FirstMistral listwise closure | Purpose-trained first-token logit ranker, 12 × 20 | 0.6843 / 0.5847 / 8 of 12 | Stable and useful, but below E5 on both headline metrics; stop rather than scale. |

## The embedding sweep plateaued near the baseline

These are the strongest contextual/capability-label condition for each small or
diverse encoder, evaluated on all 322 questions and 150 labels. Higher is better;
the zero count is the number of questions with no gain at nDCG@10.

| Encoder | nDCG@10 | MRR | Zero nDCG | Strict repeat |
| --- | ---: | ---: | ---: | --- |
| MPNet raw-label control | 0.3195 | 0.3674 | 131 | Pass |
| E5-Mistral 7B contextual | 0.3347 | 0.3866 | 120 | Pass |
| Qwen3-Embedding 4B contextual | **0.3420** | **0.3902** | **115** | Fail: two adjacent tail swaps |
| Qwen3-Embedding 0.6B contextual | 0.3000 | 0.3412 | 131 | Pass |
| Jina v5 small contextual | 0.3122 | 0.3711 | 132 | Pass |
| BGE-M3 contextual | 0.2576 | 0.3197 | 150 | Pass |
| Nomic Embed v1.5 contextual | 0.2591 | 0.3259 | 154 | Pass |

Capability sentences helped E5 by 0.0240 nDCG over its bare-label condition,
with a paired 95% interval of [0.0004, 0.0479]. They also improved Qwen4B,
Qwen0.6B, and Jina in exploratory within-model comparisons. They did not help
every family: Qwen8B was essentially unchanged and BGE-M3 decreased. The lesson
is to carry richer label text forward as a tested representation option, not as
a universal rule.

## The stable listwise method was competitive, but not the best method

The decisive pilot used the same twelve qrel-assisted questions and twenty
candidate labels as the captured Stage 2 baselines.

| Method | nDCG@10 | MRR | Relevant in top 3 | Status |
| --- | ---: | ---: | ---: | --- |
| Contextual E5 | **0.7232** | **0.6361** | **8/12** | Best frozen baseline |
| FirstMistral listwise | 0.6843 | 0.5847 | **8/12** | Stable; fails E5 quality gate |
| Contextual Qwen4B | 0.6648 | 0.5585 | **8/12** | Complete restricted baseline |
| Pointwise Qwen | 0.4820 | 0.3209 | 5/12 | Does not scale from Stage 1 |

FirstMistral was not useless. On the four clear, issue-like questions chosen for
the diagnostic, its best relevant-label ranks were 3, 1, 1, and 1; E5's were 1,
1, 1, and 3. In other words, both methods handled three of those four very well,
but they disagreed on which remaining example was easiest. Across all twelve
questions, FirstMistral ranked the relevant label higher than E5 on four
questions, tied on three, and ranked it lower on five. It also recovered two
examples where the embedding methods struggled: `Computing` was rank 1 versus
E5 rank 5, and `Natural_environment` was rank 2 versus E5 rank 5. But it placed
`Sports_events` at rank 10 versus E5 rank 5 and missed other clear labels that E5
placed first. The aggregate gate correctly reflects that uneven trade-off.

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
| What disorder did “You’re Only Human” refer to? | Human behavior | 4 | 2 | Worse |
| What vegetation do Lythrum and Erythrina share? | Natural environment | 2 | 5 | Better |

## Runtime bugs were separated from method quality

Several low or missing results were genuine implementation problems, and the
study treats them as such:

- **Embedding prompt/token bug:** the original Qwen8B and E5 scores omitted each
  checkpoint's required terminal token. Those scores are superseded. Corrected
  runs explicitly constructed the pinned model format.
- **Generated-list format failures:** generic listwise and GroupRank did not
  produce a machine-valid complete ranking under the frozen parsers. No quality
  metric is attributed to those failures.
- **FirstMistral vLLM runtime-path repeatability failure:** both full vLLM
  executions were complete, but
  8 of 12 rankings differed. All 240 scores exceeded the `1e-5` tolerance and the
  maximum absolute difference was 0.1858. The evaluator rejected that backend
  result before loading judgments. The evidence does not isolate a lower-level
  kernel or numerical root cause.
- **FirstMistral Transformers resolution:** the same pinned checkpoint, prompt,
  BF16 precision, first-token handle logits, and candidate sets were run twice in
  fresh Transformers processes. Rankings and raw score files were byte-identical;
  maximum score difference was 0.0. The evaluator loaded judgments only after
  that repeatability check passed.

This is why the final conclusion is stronger than “the model could not format an
answer.” We obtained stable listwise rankings and measured them. They were good
enough to beat two alternatives on this diagnostic, but not good enough to beat
the best clean baseline.

## What to carry forward

1. **Keep contextual E5 as the clean zero-shot control.** It is stable, simple,
   and remains the strongest eligible method across the full label sweep and the
   restricted pilot.
2. **Preserve exact model-specific embedding contracts.** A server handles token
   execution and batching; it does not necessarily supply the query instruction,
   document asymmetry, terminal token, or pooling semantics required by a model.
3. **Use richer label descriptions selectively.** Capability text helped E5 and
   several smaller models, but the effect was model-dependent.
4. **Separate operational and quality gates.** Qrel-free prompt, score, ranking,
   and repeat validation should finish before judgments are opened.
5. **Represent multi-topic expertise with multiple descriptors or vectors.** One
   averaged embedding can blur an expert that is strong in unrelated areas. A
   future expert profile should retain several topic descriptions or clusters of
   successful questions and score a new question against the best-matching
   component.
6. **Do not merge experimental ranker code into the production path.** The
   durable output from this line of work is the evidence and decision report.

## Recommended next steps

1. **Close the current listwise experiment.** Do not run FirstMistral over all 322
   questions, and do not spend more cycles repairing GroupRank prompts.
2. **Improve the representation before adding another ranker.** The most useful
   next hypothesis is a multi-vector expert profile: several human-readable topic
   descriptors and, if later allowed, clusters of held-out questions the expert
   answered well. This directly addresses experts with multiple specialties.
3. **Evaluate representation changes on a held-out split.** Do not select examples
   and judge them on the same qrels. Freeze the profile-building data, candidate
   set, and comparison gate before evaluation.
4. **Only revisit an LLM ranker with a new reason it should win.** A new attempt
   should change the evidence available to the model or use a ranker trained for
   this taxonomy—not merely add few-shot formatting examples to the same method.

## Scope and limitations

- The full embedding sweep used 322 questions and all 150 labels. The LLM pilots
  used qrel-assisted selections of 6 or 12 questions and restricted candidate
  sets of 12 or 20. Their absolute metrics are not directly comparable.
- Stage 1 and Stage 2 ranker samples were selected after inspecting development
  qrels. They are feasibility diagnostics, not confirmatory generalization
  estimates.
- With only twelve questions, no claim of statistical superiority is made for
  FirstMistral over Qwen4B or pointwise Qwen.
- GroupRank has no quality result; its failure only establishes that the tested
  output protocols were unreliable under strict parsing.
- The final receipt records the exact Transformers backend, model and container
  revisions, run directories, and config/runtime/validation hashes. It does not
  fully embed GPU identity, peak memory, truncation counters, or resolved dtype.
  The inspected prompts were only 531–603 tokens and both fresh-process outputs
  were byte-identical, so this metadata gap does not change the pilot decision.
- The conclusion is “abandon this listwise line for now,” not “all listwise
  ranking is impossible.” Reopening it requires a materially new hypothesis and
  a newly frozen evaluation.

## Evidence and reproducibility

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

The frozen FirstMistral selection contains 12 questions × 20 candidate labels and
has SHA-256 `668515c30a80473aaaa2dc530f50dfd3b7731711e9f9adf70074ebb95aa98b9d`.
The sealed judgment manifest has SHA-256
`8ced65041023f25ab157fdfa9e6a34397505fed875c1e0687527638f4285e83c`.

The raw receipts, checkpoints, prompts, and experimental harness remain on the
disposable experiment branch and local artifact area. They are deliberately not
part of the reports-only branch.
