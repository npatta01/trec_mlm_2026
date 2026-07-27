# DeepSeek Pointwise Production Run Design

## Objective

Produce one validated DeepSeek V4 Flash pointwise judgment for each of the
182,764 filtered and exactly deduplicated discovery response records. Review
the production prompt with Claude Opus before launch, validate every block of
100 exhaustively, and use periodic Opus audits to detect semantic surprises
that structural validation cannot detect.

This run evaluates response quality. It does not establish that DeepSeek or the
provisional references are ground truth.

## Frozen inputs

- Response inventory:
  `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/inference/records.json`
- Expected response count: exactly 182,764.
- Provisional references: the existing grounded Gemini reference artifact,
  joined through the discovery-question identity already used by the pointwise
  bake-off.
- Judge model: `deepseek/deepseek-v4-flash`, direct/non-thinking mode.
- Prompt and Pydantic decision contract: a new production revision derived
  from pointwise v2 only after the Opus prompt review is adjudicated.

Before any hosted judgment call, seal the source, reference, prompt, JSON
schema, model alias, provider-routing policy, and ordered record inventory by
SHA-256. Resume must reject any identity drift.

## Privacy and external transmission

The user authorizes transmitting the prompt, questions, provisional
references, candidate responses, and derived review packets to Anthropic
Claude and OpenRouter-hosted DeepSeek for this task. Raw inputs, outputs,
review packets, ledgers, and failure diagnostics remain private local artifacts
with directory mode `0700` and file mode `0600`. Sanitized aggregate reports
must not contain API keys or raw private records.

## Prompt review gate

Claude Opus reviews the exact proposed production prompt and schema before the
first DeepSeek call. The review must cover:

- the boundary between candidate defects, question ambiguity, and reference
  conflicts;
- evidence-source rules for rationales, including cases where the defect is in
  the question or provisional reference rather than the candidate;
- refusal, negation, relevance, uncertainty, and truncation behavior;
- all label, score, flag, confidence, and reference-status invariants;
- resistance to invented facts or rationales;
- deterministic JSON/schema behavior and repair instructions;
- whether any requested field is unnecessary or ambiguous.

The review is advisory. Every accepted change is recorded with a reason; every
rejected recommendation is also recorded. The final prompt, schema, Opus
review, and adjudication are sealed before launch.

## Runner and checkpoint architecture

The runner processes the ordered inventory in blocks of exactly 100, except
for the final block of 64. It writes each provider response and accounting
event durably before parsing. A completed record is never requested again on
resume unless its earlier attempt is explicitly recorded as invalid and the
bounded repair policy permits another physical call.

Concurrency may be introduced only after the first 100-record gate passes.
Concurrency must preserve deterministic record identity, bounded in-flight
requests, provider rate limits, durable per-call accounting, and ordered block
completion. A block is not promoted merely because all HTTP requests returned.

The runner uses:

- an immutable run manifest;
- an ordered inventory commitment;
- a durable physical-attempt ledger;
- a conservative spend ledger with a hard combined cap of USD 60;
- per-record raw-response checkpoints;
- block manifests and validation receipts;
- a resumable terminal-results artifact;
- private anomaly and Opus-audit artifacts;
- a sanitized aggregate progress and final report.

## Exhaustive 100-record block gate

Every block must satisfy all hard checks before the next block can begin:

1. Expected coverage: 100 scheduled identities, or 64 in the final block.
2. Exactly one terminal result for every expected identity.
3. No missing, unexpected, or duplicate identities.
4. Every terminal result parses as JSON and validates through the exact
   Pydantic production model.
5. Every cross-field invariant passes.
6. No empty output, whitespace-only output, capped/truncated output, or invalid
   completion reason is promoted.
7. Every physical call, retry, repair, token count, latency, and provider cost
   is present in the durable ledgers.
8. Ledger counts reconcile with raw checkpoints and terminal records.
9. Source, prompt, schema, model, and runtime receipt hashes match the run
   manifest.
10. Files and directories satisfy the private permission policy, and a secret
    scan is clean.

Any hard-check failure stops the run before the next block. The run remains
resumable after correction.

## Bounded failure policy

- Transient provider failures use bounded backoff and a finite retry count.
- A syntactically or semantically invalid model result may receive at most one
  targeted repair call.
- Repair prompts contain only sanitized validation findings and never Python
  exception internals, filesystem paths, secrets, or unrelated records.
- A record that still fails is terminal-invalid. The block fails closed and
  the next block is not sent.
- No retry, repair, or resume may bypass the physical-call or spend ledger.

## Statistical anomaly checks

Each clean block also receives soft checks against the cumulative and recent
history:

- label, score, flag, confidence, and reference-status distributions;
- output-token and rationale-length distributions;
- repair, retry, invalid, and provider-error rates;
- cost, latency, and cached-versus-uncached input behavior;
- repeated identical decisions or rationales;
- abrupt shifts associated with provider, model, prompt, or runtime drift.

Soft anomalies pause the run for review even if all records are structurally
valid. Thresholds are frozen before launch and must combine absolute safety
limits with distribution-shift checks; they cannot be changed post hoc merely
to allow a surprising block through.

## Semantic audit schedule

### First block

Claude Opus reviews all 100 first-block cases using the question, provisional
reference, candidate response, and DeepSeek decision. The audit checks label
support, field support, evidence fidelity, invented claims, reference handling,
and cross-field coherence. We then pause for the user to review the aggregate
and representative failures before scaling.

### Production milestones

After the user approves the first block, Opus audits a deterministic stratified
sample at each 10,000-record milestone. The sample includes:

- all terminal-invalid or repaired records since the prior milestone;
- all low-confidence and rare-label/rare-flag records up to a fixed cap;
- records near configured anomaly thresholds;
- a seeded random sample of ordinary `good` and non-`good` decisions.

Every anomalous block receives an additional targeted Opus audit. The final run
receives a seeded random 100-record audit. Audit sample identities are committed
before their DeepSeek decisions are exposed to the Opus reviewer whenever the
sampling stratum permits this.

Opus audits are diagnostic and cannot silently rewrite DeepSeek results. Any
proposed correction is preserved as a separate review field. Systematic errors
pause the run and require a new versioned prompt/run rather than mutating earlier
results in place.

## Cost and operational gates

- Combined DeepSeek and Claude hard spend cap: USD 60.
- The first external phase is limited to one Opus prompt review and 100
  DeepSeek judgments plus bounded repairs/retries.
- The runner records provider-reported usage and cost whenever supplied and
  retains a conservative reservation for unreconciled calls.
- Execution pauses if projected total spend exceeds USD 50 or if the USD 60
  hard cap would be crossed.
- Cost projections are refreshed after the first 100, 1,000, and 10,000
  completed records.

No batch larger than 100 is submitted as an opaque provider batch because the
required safety boundary is the locally validated 100-record block.

## First-block acceptance criteria

The first 100 may be presented for user approval only when:

- all 100 identities have valid terminal results;
- no hard block gate failed;
- all accounting reconciles or any conservative reservation is explicitly
  isolated and explained;
- Opus completed its 100-case semantic audit;
- a sanitized report shows distributions, failures, repairs, cost, latency,
  representative audit findings, and projected full-run cost;
- no claim exceeds the evidence from those 100 records.

The remaining 182,664 responses require explicit user approval after this
first-block report.

## Verification

Implementation uses test-driven development. Tests must cover frozen inventory
identity, exact block boundaries, crash-safe resume, duplicate/missing IDs,
schema and invariant failures, capped responses, bounded repairs/retries,
ledger reconciliation, cost caps, anomaly pauses, permission enforcement,
secret scanning, audit sampling, and the final 64-record block. Before the first
external call, the focused suite, adjacent pointwise suite, lint, formatting,
and a no-network dry run must pass.

## Deliverables

- Versioned prompt, schema, Opus review, and adjudication.
- Reproducible production runner and tests.
- Private sealed run directory with checkpoints and audits.
- Sanitized first-100 report for the user review gate.
- If later approved, sanitized milestone and final reports for the full run.

