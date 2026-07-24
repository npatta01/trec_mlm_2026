# TREC 2026 Reference-Answer Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible rough-pass comparison of the 7,759-answer Perplexity cache against Gemini 3.6 Flash, Qwen3.6-27B, and Gemma 4 31B on 400 discovery questions, with Codex qualitative review of every comparison and deeper adjudication of informative disagreements.

**Architecture:** Keep raw provider outputs in resumable, provider-specific caches; derive a deterministic clustered sample, blinded comparison packet, Codex review ledger, selected-reference artifact, and summarized report. Pure validation and selection logic lives in small `trec_mlm` modules, while scripts provide checkpointed CLIs. Candidate responses and organizer qrels never enter reference generation or selection.

**Tech Stack:** Python 3.11+, standard-library HTTP/JSON, NumPy, pinned scikit-learn 1.9 and sentence-transformers through the existing `dense` extra, pytest, rootless Podman, ROCm, vLLM's OpenAI-compatible server, Gemini Developer API, and standalone accessible HTML.

## Global Constraints

- Discovery has 7,759 questions and no organizer correctness/relevance judgments; the separate 322 development qrels do not validate discovery reference answers.
- Use the downloaded `data/llm_discovery/perplexity_full_cache.json` as the full-coverage baseline and verify its committed SHA-256 before consuming it.
- Select approximately 400 questions reproducibly across 192 non-overlapping semantic clusters, question forms, entity-density and historical disagreement/refusal difficulty proxies, ranking-failure analogues, time sensitivity, and all resolvable Perplexity retry gaps.
- Generate challenger answers from the question only. Never expose Perplexity answers, candidate responses, other challengers, or qrels to generators.
- Codex qualitatively reviews every sampled comparison; approximately 30–50 informative disagreements receive deeper evidence-backed adjudication.
- Automatic comparison is triage, not ground truth. Selected references are helpful pivots, not gold answers.
- Preserve `uncertain_needs_check` and `unanswerable_or_underspecified`; never silently promote failures.
- Do not perform exhaustive citation-entailment annotation.
- Do not download unusually large models or make paid calls until the preflight records exact storage, memory, compatibility, expected cost, and user authorization.
- Run local LLMs through rootless Podman with ROCm when compatible. Do not expose inference services beyond loopback.
- Raw caches and full review packets stay under ignored `artifacts/`; only sanitized summaries and representative examples may be copied to the tailnet-only rendered portal.
- The Perplexity full-cache deadline is already satisfied. Challenger pilot execution must remain resumable and must not jeopardize the existing source cache.

## Current execution status (2026-07-21)

- Gemini was repinned from 3.5 Flash to stable `gemini-3.6-flash` after its
  2026-07-21 release. Requests use temperature `1.0`, low thinking, a
  16,384-token output/reasoning ceiling, and no grounding or tool field.
- A two-question corrected smoke and a 20-question representative canary
  completed without failures or grounding metadata.
- The 400-question cache is complete by ID at
  `artifacts/reference_answer_validation/generations/gemini36_low_400.json`:
  400 expected/400 observed, no missing or extra IDs, no provider failures,
  400 `STOP` responses after the four original `MAX_TOKENS` records were
  retried at the larger ceiling. Length-terminated cache records are now
  automatically considered incomplete and resumable.
- Final observed usage was 37,890 prompt tokens, 92,018 candidate tokens, and
  98,623 thinking tokens. Estimated standard API cost is $1.487. The final
  cache SHA-256 is
  `1d80b8d870207f63611c87f0ee65f2ab99b5aa15cd311863bf23dee5289d5860`.
- Gemma 4 31B passed Google's two-question high-thinking canary. The Google
  free-tier run produced 131 successful records but stalled under quota; that
  partial cache is preserved and excluded from the comparison. The fresh exact
  `google/gemma-4-31B-it` DeepInfra comparison source completed 400/400 with
  zero failures.
- DeepInfra's exact `Qwen/Qwen3.6-27B` run produced 396 normal-stop answers.
  Six multi-hop/entity questions required a 32,768-token retry; two recovered
  and four remained explicit generation failures after the bounded retry. The
  final records used 42,384 input and 628,249 completion tokens, for an
  estimated $2.024 at the verified listed rates. The final cache SHA-256 is
  `786a9cd2d0c798fbb3fe4618f0a7fec3614694b101f199b673d93fe893292264`.
- DeepInfra Gemma used 45,077 input and 81,643 completion tokens, for an
  estimated $0.037 at the verified listed rates. The original Gemini run and
  four repairs remain below the authorized $10 cap; its prior receipt estimated
  $1.48 before the small repair increment.
- The 400-row blind four-model packet and strict all-six-pairs review
  validation are implemented. The packet contains four explicit anonymous
  failure slots, and the sealed provider map is mode 0600. Advisor methodology is
  recorded in `reports/trec_2026_four_model_review_advisor_2026-07-21.md`.
- The independent advisor completed 400/400 blind reviews. Codex then
  inspected all 400 and revised 34 judgments, primarily replacing forced
  fine-grained ordering with substantive ties. Exact review coverage passed
  before the provider map was opened.
- The unblinded result selects Gemini 3.6 Flash: 82.3% top-tier share and 7.0%
  serious-error rate, versus Perplexity at 62.3% and 21.5%, Qwen at 58.8% and
  31.3%, and Gemma at 50.8% and 36.3%. The result remains first in every
  inspected challenge slice.
- Actual Gemini token use projected to $28.84 for all 7,759 or about $27.35 for
  the 7,359 uncached questions. The full grounded run subsequently completed at
  7,759/7,759 records and is preserved in the private `reference-corpus-v1`
  archive. The earlier $10-cap statement is historical pre-run context.
- Canonical report:
  `reports/trec_2026_four_model_reference_review_2026-07-21.md`. Portable
  report artifact:
  `reports/trec_2026_four_model_reference_review.artifact.json`. The sanitized
  generated HTML is listed in the private tailnet portal.

---

## File structure

- `trec_mlm/reference_data.py`: frozen JSON contracts, cache loading, JSONL reading/writing, SHA checks, and provenance validation.
- `trec_mlm/reference_sampling.py`: question strata, deterministic clustering, challenge inclusion, and exact-size sample construction.
- `trec_mlm/reference_generation.py`: provider-neutral generation records, Gemini and OpenAI-compatible request adapters, retry policy, and atomic checkpoints.
- `trec_mlm/reference_adjudication.py`: blinding, comparator result validation, Codex review-ledger validation, and selected-reference construction.
- `scripts/build_reference_validation_sample.py`: create embeddings, the 192-cluster assignment, and the 400-question sample manifest.
- `scripts/preflight_reference_models.py`: inspect credentials, host/ROCm/Podman state, pinned Hub metadata, estimated downloads, and model-serving compatibility without exposing secrets.
- `scripts/generate_reference_challengers.py`: resumable generation CLI for Gemini or one local OpenAI-compatible Qwen server.
- `scripts/build_reference_comparisons.py`: create blinded packets and optionally run the automatic triage comparator.
- `scripts/reference_review.py`: print unreviewed batches and atomically record Codex decisions.
- `scripts/finalize_reference_validation.py`: validate complete review coverage and build selected references plus summary metrics.
- `scripts/build_reference_validation_report.py`: build the durable Markdown report, sanitized progressive-disclosure HTML, and full local artifact explorer.
- `tests/test_reference_data.py`, `tests/test_reference_sampling.py`, `tests/test_reference_generation.py`, `tests/test_reference_adjudication.py`, `tests/test_reference_validation_report.py`: focused unit and CLI tests.

## Frozen artifact layout

```text
artifacts/reference_answer_validation/
├── preflight.json
├── sample_manifest.json
├── cluster_assignments.tsv
├── generations/
│   ├── gemini36_low_400.json
│   ├── qwen36_27b_400.json
│   └── gemma4_31b_deepinfra_400.json
├── blind_system_map.json
├── blind_comparisons.jsonl
├── automatic_comparisons.jsonl
├── codex_reviews.jsonl
├── selected_references.jsonl
├── summary.json
├── manifest.json
└── full_review_explorer.html
```

---

### Task 1: Freeze reference-data contracts and source validation

**Files:**
- Create: `trec_mlm/reference_data.py`
- Create: `tests/test_reference_data.py`

**Interfaces:**
- Consumes: `trec_mlm.artifact_io.sha256_file`, Perplexity cache schema from `scripts/fetch_perplexity_sonar.py`.
- Produces: `QuestionRecord`, `GenerationRecord`, `ReviewDecision`, `load_perplexity_questions()`, `read_jsonl()`, `write_jsonl_atomic()`, and `validate_exact_ids()`.

- [ ] **Step 1: Write failing contract tests**

```python
from pathlib import Path
import json
import pytest

from trec_mlm.reference_data import (
    EXPECTED_PERPLEXITY_SHA256,
    load_perplexity_questions,
    read_jsonl,
    validate_exact_ids,
    write_jsonl_atomic,
)


def test_load_perplexity_questions_preserves_ids_answers_and_gap_state(tmp_path: Path) -> None:
    payload = {
        "total_questions": 2,
        "records": {
            "q2": {"question": "Second?", "response": {"answer": "B", "citations": []}},
            "q1": {"question": "First?", "response": {"answer": "A [1]", "answer_no_citations": "A", "citations": ["https://a.test"]}},
        },
        "retry_summary": {"unresolved": ["q2"]},
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_perplexity_questions(path, expected_sha256=None)

    assert [record.question_id for record in records] == ["q1", "q2"]
    assert records[0].perplexity_answer == "A [1]"
    assert records[0].perplexity_answer_no_citations == "A"
    assert records[1].perplexity_unresolved is True


def test_jsonl_round_trip_is_sorted_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl_atomic(path, [{"question_id": "q2"}, {"question_id": "q1"}])
    assert read_jsonl(path) == [{"question_id": "q1"}, {"question_id": "q2"}]
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_validate_exact_ids_reports_missing_and_extra() -> None:
    with pytest.raises(ValueError, match=r"missing=\['q2'\].*extra=\['q3'\]"):
        validate_exact_ids(["q1", "q2"], ["q1", "q3"], label="reviews")


def test_committed_perplexity_hash_is_frozen() -> None:
    assert EXPECTED_PERPLEXITY_SHA256 == "cde4ca41c555e29533961bc41496361b00f73b70264899597573362e355a4ee0"
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `uv run pytest -q tests/test_reference_data.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'trec_mlm.reference_data'`.

- [ ] **Step 3: Implement the frozen dataclasses and loaders**

```python
"""Validated contracts for reference-answer sampling, generation, and review."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from trec_mlm.artifact_io import sha256_file

EXPECTED_PERPLEXITY_SHA256 = "cde4ca41c555e29533961bc41496361b00f73b70264899597573362e355a4ee0"


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    question_id: str
    question: str
    perplexity_answer: str
    perplexity_answer_no_citations: str
    perplexity_citations: tuple[str, ...]
    perplexity_unresolved: bool


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    question_id: str
    provider: str
    model: str
    answer: str
    elapsed_seconds: float
    usage: Mapping[str, Any]
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    question_id: str
    outcome: str
    selected_systems: tuple[str, ...]
    selected_answer: str | None
    rationale: str
    uncertainty: str
    deep_adjudication: bool
    evidence_urls: tuple[str, ...]


def load_perplexity_questions(path: Path, *, expected_sha256: str | None = EXPECTED_PERPLEXITY_SHA256) -> list[QuestionRecord]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Perplexity cache SHA-256 does not match the frozen manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, dict) or payload.get("total_questions") != len(records):
        raise ValueError("Perplexity cache record count is inconsistent")
    unresolved = set(payload.get("retry_summary", {}).get("unresolved", []))
    result = []
    for question_id in sorted(records):
        item = records[question_id]
        response = item.get("response") or {}
        result.append(QuestionRecord(
            question_id=str(question_id),
            question=str(item["question"]),
            perplexity_answer=str(response.get("answer") or ""),
            perplexity_answer_no_citations=str(response.get("answer_no_citations") or response.get("answer") or ""),
            perplexity_citations=tuple(str(url) for url in response.get("citations") or []),
            perplexity_unresolved=str(question_id) in unresolved,
        ))
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row["question_id"]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
    temporary.replace(path)


def validate_exact_ids(expected: Iterable[str], observed: Iterable[str], *, label: str) -> None:
    expected_set, observed_set = set(expected), set(observed)
    if expected_set != observed_set:
        raise ValueError(f"{label}: missing={sorted(expected_set-observed_set)!r} extra={sorted(observed_set-expected_set)!r}")
```

- [ ] **Step 4: Run focused and artifact tests**

Run: `uv run pytest -q tests/test_reference_data.py tests/test_artifact_io.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the contracts**

```bash
git add trec_mlm/reference_data.py tests/test_reference_data.py
git commit -m "feat: add reference validation data contracts"
```

---

### Task 2: Build the deterministic 400-question clustered sample

**Files:**
- Create: `trec_mlm/reference_sampling.py`
- Create: `scripts/build_reference_validation_sample.py`
- Create: `tests/test_reference_sampling.py`

**Interfaces:**
- Consumes: `QuestionRecord`, frozen MPNet embeddings, aggregate-only features streamed from `data/dev/llm_discovery_data.json`, and optional sparse-failure analogue TSV.
- Produces: `question_strata(text) -> tuple[str, ...]`, `historical_difficulty_features(path)`, `build_clustered_sample(...) -> list[dict]`, `sample_manifest.json`, and `cluster_assignments.tsv`.

- [ ] **Step 1: Write failing tests for strata and exact deterministic selection**

```python
import numpy as np

from trec_mlm.reference_data import QuestionRecord
from trec_mlm.reference_sampling import build_clustered_sample, question_strata


def _record(i: int, *, unresolved: bool = False) -> QuestionRecord:
    return QuestionRecord(str(i), f"What happened in {2000+i}?", "answer [1]", "answer", (), unresolved)


def test_question_strata_marks_numeric_time_sensitive_and_underspecified() -> None:
    assert set(question_strata("What is the latest population in 2026?")) >= {"factoid", "numeric", "time_sensitive"}
    assert "underspecified" in question_strata("What did the company do next?")


def test_sample_contains_every_cluster_and_required_id_and_is_repeatable() -> None:
    records = [_record(i, unresolved=i == 7) for i in range(12)]
    embeddings = np.asarray([[i, i % 3 + 1] for i in range(12)], dtype=np.float32)
    labels = np.asarray([0] * 4 + [1] * 4 + [2] * 4)

    first = build_clustered_sample(records, embeddings, labels, sample_size=7, seed=2026, required_ids={"7"})
    second = build_clustered_sample(records, embeddings, labels, sample_size=7, seed=2026, required_ids={"7"})

    assert first == second
    assert len(first) == 7
    assert {row["cluster_id"] for row in first} == {0, 1, 2}
    assert "7" in {row["question_id"] for row in first}
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest -q tests/test_reference_sampling.py`

Expected: collection fails because `trec_mlm.reference_sampling` does not exist.

- [ ] **Step 3: Implement deterministic strata and selection**

Implement these exact rules:

```python
QUESTION_PREFIX = {
    "why": "explanation", "how": "explanation", "what": "factoid",
    "who": "factoid", "when": "factoid", "where": "factoid", "which": "factoid",
}
TIME_RE = re.compile(r"\b(latest|current|currently|today|now|recent|as of)\b", re.I)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
UNDERSPECIFIED_RE = re.compile(r"\b(the company|the city|the three|that determination|what happened next)\b", re.I)


def question_strata(text: str) -> tuple[str, ...]:
    lowered = text.strip().lower()
    first = lowered.split(maxsplit=1)[0] if lowered else ""
    labels = {QUESTION_PREFIX.get(first, "other")}
    if NUMBER_RE.search(text): labels.add("numeric")
    if TIME_RE.search(text): labels.add("time_sensitive")
    if UNDERSPECIFIED_RE.search(text): labels.add("underspecified")
    if len(text.split()) >= 25: labels.add("long_or_multihop")
    return tuple(sorted(labels))
```

`build_clustered_sample()` must validate aligned finite embeddings and labels, choose the cosine-nearest medoid to each cluster centroid, add all `required_ids`, then fill to `sample_size` by round-robin across clusters using descending `(challenge_score, centroid_distance, stable_sha256_tiebreak)`. `challenge_score` is the sum of unresolved, known-failure analogue, time-sensitive, numeric, underspecified, and long/multi-hop indicators. Reject `sample_size < cluster_count` and missing required IDs.

Before selection, `historical_difficulty_features()` streams the official 150-response matrix without retaining response text. For each discovery question it records only: response count, exact-refusal share, non-refusal share, number of distinct normalized non-refusal answers, and normalized answer-disagreement entropy. Define the entity-density proxy as the number of title-cased token spans after the first token divided by query token count. Add capped high-refusal, high-disagreement, and high-entity-density indicators to `challenge_score`; record the numeric aggregates and source hash in the private sample manifest. Never copy historical response text into the manifest, generator input, comparison packet, report, or rendered artifact.

- [ ] **Step 4: Add the sampling CLI**

The CLI must:

```bash
uv run --extra dense python scripts/build_reference_validation_sample.py \
  --perplexity-cache data/llm_discovery/perplexity_full_cache.json \
  --discovery-responses data/dev/llm_discovery_data.json \
  --analogue-cases reports/trec_2026_sparse_failure_analogue_cases.tsv \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --revision e8c3b32edf5434bc2275fc9bab85f82640a19130 \
  --clusters 192 --sample-size 400 --seed 2026 \
  --output-dir artifacts/reference_answer_validation
```

The script initializes `SentenceTransformer(args.encoder, revision=args.revision, device="cpu")`, normalizes embeddings, calls `sklearn.cluster.KMeans(n_clusters=192, random_state=2026, n_init=10)`, maps each `nearest_discovery_query_text` in the analogue TSV to exactly one normalized discovery question when possible, and records unmatched/ambiguous analogues rather than guessing.

- [ ] **Step 5: Test the CLI with a six-question fixture**

Monkeypatch embedding creation and KMeans so the test requires no model download. Use a small streamed response fixture and assert its aggregate refusal/disagreement features while confirming no raw candidate response appears in the manifest. Also assert exact record count, cluster coverage, source/cache hashes, selection reasons, seed, encoder ID/revision, and unmatched analogue diagnostics.

Run: `uv run pytest -q tests/test_reference_sampling.py`

Expected: all tests pass.

- [ ] **Step 6: Commit sampling**

```bash
git add trec_mlm/reference_sampling.py scripts/build_reference_validation_sample.py tests/test_reference_sampling.py
git commit -m "feat: build clustered reference validation sample"
```

---

### Task 3: Implement resumable Gemini and local-Qwen generation

**Files:**
- Create: `trec_mlm/reference_generation.py`
- Create: `scripts/generate_reference_challengers.py`
- Create: `tests/test_reference_generation.py`

**Interfaces:**
- Consumes: `sample_manifest.json`; `GEMINI_API_KEY` for Gemini; an explicitly supplied loopback OpenAI-compatible endpoint for Qwen.
- Produces: provider caches under `generations/` with raw answer, model, usage, latency, prompt hash, request metadata, and checkpointed failures.

- [ ] **Step 1: Write failing tests for prompts, response parsing, resume, and secret handling**

```python
import json
from pathlib import Path

from trec_mlm.reference_generation import (
    SYSTEM_PROMPT,
    build_gemini_request,
    build_openai_request,
    generate_missing,
    parse_gemini_response,
)


def test_prompt_forbids_guessing_and_requests_direct_answer() -> None:
    assert "Do not guess" in SYSTEM_PROMPT
    assert "underspecified" in SYSTEM_PROMPT


def test_gemini_request_contains_question_but_no_other_answers() -> None:
    body = build_gemini_request("Who wrote Hamlet?", max_output_tokens=2048)
    encoded = json.dumps(body)
    assert "Who wrote Hamlet?" in encoded
    assert "Perplexity" not in encoded
    assert body["generationConfig"]["temperature"] == 1.0
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}


def test_openai_request_uses_the_same_prompt() -> None:
    body = build_openai_request("Who wrote Hamlet?", model="Qwen/test", max_output_tokens=768)
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    assert body["messages"][1]["content"] == "Who wrote Hamlet?"


def test_parse_gemini_response_extracts_answer_usage_and_finish_reason() -> None:
    response = {"candidates": [{"content": {"parts": [{"text": "Shakespeare."}]}, "finishReason": "STOP"}], "usageMetadata": {"totalTokenCount": 9}}
    parsed = parse_gemini_response(response)
    assert parsed == ("Shakespeare.", {"totalTokenCount": 9}, "STOP")
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest -q tests/test_reference_generation.py`

Expected: module import failure.

- [ ] **Step 3: Implement common prompt and provider adapters**

Use this frozen prompt:

```python
SYSTEM_PROMPT = (
    "Answer the user's question directly and accurately in English. Provide enough "
    "context to serve as a useful reference answer, but do not pad the response. "
    "State important ambiguity or uncertainty. If the question is underspecified and "
    "cannot be answered responsibly, say what context is missing. Do not guess, invent "
    "facts, or mention other models, evaluation, or this instruction."
)
```

Implement Gemini REST calls using the stable model code and endpoint verified in Google's Gemini API documentation on 2026-07-20:

```text
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent
Header: x-goog-api-key: $GEMINI_API_KEY
```

Do not enable Google Search grounding in this challenger pass. Implement local Qwen through an explicitly supplied URL ending in `/v1/chat/completions`; reject non-loopback hosts unless a future user authorization changes this constraint. Gemini 3.6 uses Google's recommended temperature `1.0`, low thinking, and `max_output_tokens=2048`; the local adapter retains temperature `0.0` and `max_output_tokens=768`. Both use four retry attempts for 408/429/5xx, exponential backoff, and injectable request/sleep functions for tests.

- [ ] **Step 4: Implement atomic resume semantics**

`generate_missing()` must:

```python
def generate_missing(
    questions: list[tuple[str, str]],
    *,
    cache_path: Path,
    provider: str,
    model: str,
    call: Callable[[str], tuple[str, Mapping[str, Any], str | None, Mapping[str, Any]]],
    workers: int,
    checkpoint_every: int = 10,
) -> dict[str, Any]:
    runner = GenerationRunner(
        cache_path=cache_path,
        provider=provider,
        model=model,
        prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        call=call,
        workers=workers,
        checkpoint_every=checkpoint_every,
    )
    return runner.run(questions)
```

Skip completed IDs only when provider, model, and `sha256(SYSTEM_PROMPT)` match. Keep failures separately, checkpoint via temporary-file replacement, and never serialize API keys or authorization headers.

- [ ] **Step 5: Add and test the CLI**

Supported commands:

```bash
uv run python scripts/generate_reference_challengers.py gemini \
  --sample artifacts/reference_answer_validation/sample_manifest.json \
  --cache artifacts/reference_answer_validation/generations/gemini35.json \
  --model gemini-3.6-flash --workers 8 --requests-per-minute 120

uv run python scripts/generate_reference_challengers.py openai-compatible \
  --sample artifacts/reference_answer_validation/sample_manifest.json \
  --cache artifacts/reference_answer_validation/generations/qwen35.json \
  --model Qwen/Qwen3.6-35B-A3B \
  --base-url http://127.0.0.1:8000/v1/chat/completions \
  --workers 4
```

Tests must verify a missing `GEMINI_API_KEY` fails before network access, non-loopback Qwen URLs fail closed, interrupted runs resume without duplicate calls, and cache JSON contains no key-shaped values.

Run: `uv run pytest -q tests/test_reference_generation.py`

Expected: all tests pass.

- [ ] **Step 6: Commit generation**

```bash
git add trec_mlm/reference_generation.py scripts/generate_reference_challengers.py tests/test_reference_generation.py
git commit -m "feat: add resumable challenger generation"
```

---

### Task 4: Add a no-download model and cost preflight

**Files:**
- Create: `scripts/preflight_reference_models.py`
- Modify: `tests/test_reference_generation.py`

**Interfaces:**
- Consumes: host memory/ROCm/Podman state, `hf models info`/`hf download --dry-run`, credential presence, model metadata.
- Produces: `artifacts/reference_answer_validation/preflight.json`; exits nonzero when a required gate fails.

- [ ] **Step 1: Add failing preflight tests**

Expose a pure `build_preflight(environ, command_results, disk_free_bytes, commit, dirty)` function behind the CLI. Use fixed fixtures so tests assert credential redaction, hardware gating, and exact model pins without touching the host:

```python
import json
import pytest

from scripts.preflight_reference_models import MODELS, build_preflight


def _commands(*, rocminfo: str = "GPU agent gfx1151") -> dict[str, str]:
    return {
        "rocminfo": rocminfo,
        "podman_version": "podman version 5.4.0",
        "podman_images": "vllm/vllm-openai-rocm:latest sha256:image",
        "meminfo": "MemAvailable: 120000000 kB\nSwapFree: 8000000 kB\n",
    }


def test_preflight_records_presence_not_secret_value() -> None:
    receipt = build_preflight(
        environ={"GEMINI_API_KEY": "super-secret-value"},
        command_results=_commands(),
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["credentials"]["gemini_api_key"] == "configured"
    assert "super-secret-value" not in json.dumps(receipt)


def test_preflight_rejects_missing_rocm_device_for_local_models() -> None:
    receipt = build_preflight(
        environ={}, command_results=_commands(rocminfo=""),
        disk_free_bytes=500_000_000_000, commit="abc123", dirty=False,
    )
    assert receipt["gates"]["local_model_smoke_ready"] is False
    assert "gfx1151" in receipt["gates"]["local_model_reason"]


def test_preflight_pins_verified_qwen_revisions() -> None:
    assert MODELS["qwen35"]["base_revision"] == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert MODELS["qwen35"]["fp8_revision"] == "95a723d08a9490559dae23d0cff1d9466213d989"
    assert MODELS["qwen27"]["base_revision"] == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert MODELS["qwen27"]["fp8_revision"] == "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"
```

The expected model table is:

```python
MODELS = {
    "qwen35": {
        "base": "Qwen/Qwen3.6-35B-A3B",
        "base_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
        "fp8_revision": "95a723d08a9490559dae23d0cff1d9466213d989",
    },
    "qwen27": {
        "base": "Qwen/Qwen3.6-27B",
        "base_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "fp8": "Qwen/Qwen3.6-27B-FP8",
        "fp8_revision": "e89b16ebf1988b3d6befa7de50abc2d76f26eb09",
    },
}
```

- [ ] **Step 2: Implement the preflight**

The script records:

- current commit and dirty status;
- available RAM/swap from `/proc/meminfo`;
- ROCm device and `gfx1151` evidence from `rocminfo`;
- rootless Podman version and presence of `vllm/vllm-openai-rocm:latest`;
- Hub model SHA, license tag, parameter count, and dry-run download bytes;
- whether `GEMINI_API_KEY` is configured, never its value;
- free disk space in the Hugging Face cache and artifact directory;
- a projected Gemini pilot cap supplied as `--gemini-cost-cap-usd`;
- pass/fail gates for smoke-test readiness.

Use `subprocess.run(argv, shell=False, check=True, text=True, capture_output=True)` exclusively, where `argv` is a validated list of literal arguments. Do not invoke `hf download` without `--dry-run`.

- [ ] **Step 3: Run tests and a real read-only preflight**

Run:

```bash
uv run pytest -q tests/test_reference_generation.py
uv run python scripts/preflight_reference_models.py \
  --output artifacts/reference_answer_validation/preflight.json \
  --gemini-cost-cap-usd 10
```

Expected: tests pass; the real preflight may report `GEMINI_API_KEY=not_configured`, but it must still complete as a readiness receipt and perform no downloads or paid calls.

- [ ] **Step 4: Stop for authorization before downloads or paid calls**

Present the receipt with exact Qwen download sizes, selected representation, expected peak memory, and Gemini cap. Proceed only after the user authorizes the downloads and supplies the key through their shell environment.

- [ ] **Step 5: Commit preflight**

```bash
git add scripts/preflight_reference_models.py tests/test_reference_generation.py
git commit -m "feat: add reference model preflight"
```

---

### Task 5: Smoke-test and generate the three challenger caches

**Files:**
- Create after execution: `artifacts/reference_answer_validation/generations/*.json`
- Update after execution: `artifacts/reference_answer_validation/preflight.json`

**Interfaces:**
- Consumes: authorized model downloads, configured Gemini key, Task 3 CLI.
- Produces: complete, validated answer caches for 400 sample IDs plus throughput/cost receipts.

- [ ] **Step 1: Run a two-question Gemini smoke test with a one-dollar stop**

Create a two-record derived sample from the manifest, run the Gemini CLI, and verify answers, usage metadata, no search grounding, and projected 400-question cost. Abort if any secret appears in the cache or if projected cost exceeds the approved cap.

- [ ] **Step 2: Download only the authorized Qwen representation**

Use the pinned revision from the approved preflight:

```bash
hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0
```

or the explicitly approved FP8 alternative. Verify it with `hf cache verify`. Repeat for Qwen27 only after Qwen35's two-question smoke completes and memory is released.

- [ ] **Step 3: Start Qwen35 on loopback and smoke-test ROCm**

Use rootless Podman, the existing vLLM ROCm image, `/dev/kfd`, `/dev/dri`, host IPC, the Hugging Face cache mounted read-only, and `--host 127.0.0.1`. Record the exact image digest and command in `preflight.json`. Do not expose or publish port 8000 beyond loopback.

- [ ] **Step 4: Benchmark 20 questions before the full Qwen35 run**

Record load time, answers/second, output tokens/second, peak unified memory, failures, and projected wall time for 400. Stop if output is malformed, repeated, empty, or the projection threatens the agreed execution window.

- [ ] **Step 5: Generate and validate all 400 Qwen35 answers**

Run the generation CLI, stop the container, verify exact sample ID coverage, then repeat Steps 3–5 for Qwen27 on a fresh server process.

- [ ] **Step 6: Generate and validate all 400 Gemini answers**

Run with the predeclared cost cap and verify exact ID coverage, usage totals, finish reasons, and zero accidental grounding/tool use.

- [ ] **Step 7: Write a generation receipt**

Extend `preflight.json` with actual model revisions, server image digest, runtime configuration, elapsed time, peak memory, token counts, API cost, failure/retry counts, and exact cache SHA-256 values.

---

### Task 6: Build blind automatic comparisons and the all-400 Codex review ledger

**Files:**
- Create: `trec_mlm/reference_adjudication.py`
- Create: `scripts/build_reference_comparisons.py`
- Create: `scripts/reference_review.py`
- Create: `tests/test_reference_adjudication.py`

**Interfaces:**
- Consumes: sample manifest and four complete answer sources.
- Produces: sealed system map, randomized blind packet, optional automatic comparator records, and an append-safe Codex review ledger.

- [ ] **Step 1: Write failing tests for blinding and review validation**

```python
import pytest

from trec_mlm.reference_adjudication import (
    ALLOWED_OUTCOMES,
    blind_system_order,
    select_reference,
    validate_review_coverage,
)


def test_blind_order_is_repeatable_and_not_global() -> None:
    q1 = blind_system_order("q1", ("perplexity", "gemini", "qwen35", "qwen27"), seed=2026)
    assert q1 == blind_system_order("q1", ("perplexity", "gemini", "qwen35", "qwen27"), seed=2026)
    assert q1 != blind_system_order("q2", ("perplexity", "gemini", "qwen35", "qwen27"), seed=2026)


def test_review_coverage_requires_all_400_exactly_once() -> None:
    expected = [f"q{i:03d}" for i in range(400)]
    complete = [{"question_id": question_id} for question_id in expected]
    validate_review_coverage(expected, complete)

    duplicate = complete + [{"question_id": "q000"}]
    with pytest.raises(ValueError, match="duplicate.*q000"):
        validate_review_coverage(expected, duplicate)

    missing = complete[:-1]
    with pytest.raises(ValueError, match="missing.*q399"):
        validate_review_coverage(expected, missing)


def test_uncertain_selection_never_promotes_an_answer() -> None:
    result = select_reference({"outcome": "uncertain_needs_check", "selected_answer": None}, {})
    assert result["status"] == "uncertain"
    assert result["answer"] is None
```

- [ ] **Step 2: Implement per-question deterministic blinding**

Generate a system permutation from `sha256(f"{seed}\0{question_id}\0{system}")`; expose only labels `response_a` through `response_d` in the review packet. Save the reverse mapping separately with mode `0600`. Use Perplexity's frozen `answer_no_citations` derived field in the blind packet so inline markers do not reveal the provider; retain its raw answer and citation URLs only in the source cache for deep checks. The packet contains the question, anonymized answers, question strata, and Perplexity gap flag, but no candidate answers or qrels.

- [ ] **Step 3: Implement strict comparator-result parsing**

Require JSON with:

```json
{
  "question_id": "q017",
  "outcome": "equivalent",
  "understands_question": {"response_a": true},
  "correctness": {"response_a": "likely_correct"},
  "completeness": {"response_a": "complete"},
  "rationale": "All four responses identify the same person and give compatible supporting context.",
  "uncertainty": "low"
}
```

Reject unknown labels, missing systems, empty rationale, duplicate IDs, or prose outside the JSON object. Automatic comparison remains optional and is labeled `triage_only`.

- [ ] **Step 4: Implement the Codex review CLI**

Commands:

```bash
uv run python scripts/reference_review.py next \
  --packet artifacts/reference_answer_validation/blind_comparisons.jsonl \
  --reviews artifacts/reference_answer_validation/codex_reviews.jsonl \
  --count 20

uv run python scripts/reference_review.py record \
  --packet artifacts/reference_answer_validation/blind_comparisons.jsonl \
  --reviews artifacts/reference_answer_validation/codex_reviews.jsonl \
  --decision-file /tmp/reference-decisions.jsonl
```

`record` validates IDs, outcomes, anonymized selected systems, rationale, uncertainty, deep-adjudication flag, evidence URLs, and explicit synthesized answer for `combine`. It replaces an existing decision only with `--replace` and always writes atomically.

- [ ] **Step 5: Test and commit adjudication tooling**

Run: `uv run pytest -q tests/test_reference_adjudication.py`

Expected: all tests pass.

```bash
git add trec_mlm/reference_adjudication.py scripts/build_reference_comparisons.py scripts/reference_review.py tests/test_reference_adjudication.py
git commit -m "feat: add blind reference adjudication workflow"
```

- [ ] **Step 6: Run automatic triage over all 400**

Use `Qwen/Qwen3.6-35B-A3B` as the default local triage comparator after its generation run is frozen. Give it randomized anonymized answers and require structured per-dimension JSON. Repeat every non-`equivalent` result with a second deterministic permutation; mark reversals or materially changed rationales as `triage_disagreement` for Codex priority. Record the comparator model/revision, prompt hash, both permutations, and the limitation that one compared answer came from the same model family. Ensure the comparator never sees the sealed system map, historical responses, candidate answers, or qrels.

- [ ] **Step 7: Codex reviews all 400 in batches of 20**

For every item, Codex confirms or revises the triage outcome and records concise rationale and uncertainty. Mark approximately 30–50 informative disagreements for deep adjudication. Use Perplexity citations or targeted web lookup only for material disputes, surprising claims, numeric facts, and time-sensitive answers. Continue until `reference_review.py status` reports `400/400 reviewed` and zero invalid decisions.

---

### Task 7: Finalize selected references and decision evidence

**Files:**
- Create: `scripts/finalize_reference_validation.py`
- Modify: `trec_mlm/reference_adjudication.py`
- Modify: `tests/test_reference_adjudication.py`

**Interfaces:**
- Consumes: blind map, four answer sources, automatic triage, and complete Codex reviews.
- Produces: `selected_references.jsonl`, `summary.json`, and integrity `manifest.json`.

- [ ] **Step 1: Add failing selection tests**

Cover every allowed outcome:

- each provider winner resolves to that provider's raw answer;
- `equivalent` records all equivalent systems and chooses the shortest nonempty answer deterministically for the pivot;
- `combine` requires the reviewed synthesized answer;
- `uncertain_needs_check` emits `answer=null` and `status=uncertain`;
- `unanswerable_or_underspecified` emits the reviewed explanation without promoting a factual answer;
- every deep adjudication record retains evidence URLs;
- blind labels are removed only after reviews are frozen.

- [ ] **Step 2: Implement finalization validation**

The script fails closed unless:

- sample, all three challenger caches, and Codex reviews contain the same 400 unique IDs;
- the Perplexity source hash matches the manifest;
- system-map labels resolve exactly once;
- every `combine` has nonempty synthesis and rationale;
- no selected answer comes from an unrecognized system;
- all inputs and outputs receive byte size and SHA-256 entries using `manifest_entries()` and `manifest_aggregate_sha256()`.

- [ ] **Step 3: Build summary metrics without prevalence claims**

Report counts—not population estimates—for outcomes, uncertainty, deep adjudication, question strata, Perplexity unresolved cases, provider failures, response lengths, elapsed time, memory, and cost. Clearly label the sample `purposive_stratified_rough_pass`.

- [ ] **Step 4: Run finalization and tests**

Run:

```bash
uv run pytest -q tests/test_reference_adjudication.py tests/test_reference_data.py
uv run python scripts/finalize_reference_validation.py \
  --artifact-dir artifacts/reference_answer_validation
```

Expected: 400 selected-reference records or explicit uncertain/unanswerable records, zero missing IDs, and a valid aggregate manifest hash.

- [ ] **Step 5: Commit finalization**

```bash
git add trec_mlm/reference_adjudication.py scripts/finalize_reference_validation.py tests/test_reference_adjudication.py
git commit -m "feat: finalize reviewed reference answers"
```

---

### Task 8: Build the progressive-disclosure report and full local explorer

**Files:**
- Create: `scripts/build_reference_validation_report.py`
- Create: `tests/test_reference_validation_report.py`
- Create after execution: `reports/trec_2026_reference_answer_validation.md`
- Create after execution: `reports/trec_2026_reference_answer_validation.html`
- Create after execution: `artifacts/reference_answer_validation/full_review_explorer.html`

**Interfaces:**
- Consumes: summary, selected references, reviews, sample manifest, and generation receipts.
- Produces: answer-first report, accessible sanitized HTML, and a private full-detail local explorer.

- [ ] **Step 1: Write failing report tests**

```python
from scripts.build_reference_validation_report import build_report, render_html


def test_report_states_rough_pass_and_dataset_limitations():
    report = build_report(FIXTURE)
    assert "purposive stratified rough pass" in report["executive_summary"].lower()
    assert "no organizer correctness judgments" in report["limitations"].lower()


def test_sanitized_html_is_accessible_and_omits_raw_payloads():
    html = render_html(FIXTURE, sanitized=True)
    assert '<meta name="viewport"' in html
    assert '<a class="skip-link" href="#main">' in html
    assert "<details" in html and "<summary" in html
    assert "GEMINI_API_KEY" not in html
    assert FIXTURE["raw_search_result_text"] not in html
```

- [ ] **Step 2: Implement answer-first report content**

Sections:

1. Decision and recommended provider roles.
2. What was compared and why this is a rough pass.
3. Outcome counts and patterns by stratum.
4. Quality/throughput/cost comparison.
5. Perplexity unresolved-case results.
6. Deep-adjudication examples.
7. What Codex reviewed across all 400.
8. Implications for the later pointwise/pairwise judge.
9. Limitations, provenance, hashes, and reproducibility commands.

Do not claim corpus-wide error rates or qrel validation.

- [ ] **Step 3: Implement progressive disclosure**

The sanitized report shows summary cards and aggregate tables first, then `<details>` for methodology, representative disagreements, and provenance. It includes responsive layout, keyboard-visible focus, semantic tables, a skip link, reduced-motion handling, light/dark contrast, and no external JavaScript. The full explorer may include all 400 questions and answers but remains only under ignored `artifacts/` and is never copied to the rendered portal.

- [ ] **Step 4: Run report tests and build outputs**

Run:

```bash
uv run pytest -q tests/test_reference_validation_report.py
uv run python scripts/build_reference_validation_report.py \
  --artifact-dir artifacts/reference_answer_validation \
  --markdown reports/trec_2026_reference_answer_validation.md \
  --html reports/trec_2026_reference_answer_validation.html \
  --full-explorer artifacts/reference_answer_validation/full_review_explorer.html
```

Expected: tests pass; all three outputs build deterministically.

- [ ] **Step 5: Commit code and canonical sanitized report**

```bash
git add scripts/build_reference_validation_report.py tests/test_reference_validation_report.py reports/trec_2026_reference_answer_validation.md reports/trec_2026_reference_answer_validation.html
git commit -m "docs: report reference answer validation"
```

---

### Task 9: End-to-end verification and private rendered handoff

**Files:**
- Modify derived copy: `/home/npatta01/codex-rendered/plans/trec-2026-reference-answer-validation/index.html`
- Modify derived index: `/home/npatta01/codex-rendered/index.html`

**Interfaces:**
- Consumes: completed code, tests, artifact manifest, and sanitized canonical HTML.
- Produces: verified repository evidence and tailnet-only report link.

- [ ] **Step 1: Run formatting, focused tests, then the full suite**

```bash
uv run ruff check trec_mlm/reference_*.py scripts/*reference*.py tests/test_reference_*.py
uv run pytest -q tests/test_reference_data.py tests/test_reference_sampling.py tests/test_reference_generation.py tests/test_reference_adjudication.py tests/test_reference_validation_report.py
uv run pytest -q
```

Expected: Ruff passes; focused and full suites pass.

- [ ] **Step 2: Rebuild in a clean temporary artifact directory**

Run the sample and all post-generation derivations from frozen caches into a second temporary directory. Compare sample IDs, selected-reference bytes, summary bytes, and aggregate hashes. Runtime-only fields must be isolated from deterministic artifacts.

- [ ] **Step 3: Audit completeness and leakage**

Verify:

- exactly 400 sampled IDs;
- exactly 400 records in every challenger cache or an explicit provider failure subsequently resolved;
- exactly 400 Codex reviews;
- approximately 30–50 deep-adjudication flags, unless the report explains expansion;
- no candidate responses or qrels in generation inputs;
- no API keys, authorization headers, environment dumps, raw logs, or full raw dataset embedded in tracked or rendered outputs;
- every uncertain/unanswerable case remains explicit.

- [ ] **Step 4: Copy only the sanitized report to the authorized portal**

Use `install -D -m 0644` or an equivalent non-public copy operation to place the sanitized HTML under `/home/npatta01/codex-rendered/plans/trec-2026-reference-answer-validation/index.html`. Update the existing rendered index without exposing the full explorer, caches, or raw review packet.

- [ ] **Step 5: Verify the live tailnet-only artifact**

Confirm:

```bash
curl --fail --silent --show-error \
  https://npatta01-framework.tail481212.ts.net/plans/trec-2026-reference-answer-validation/ \
  | rg -q "TREC 2026 reference-answer validation"
tailscale serve status
```

Expected: HTTPS returns the intended report; Serve remains tailnet-only; no Funnel/public listener is enabled.

- [ ] **Step 6: Record final status and commit any verification receipts**

Update the canonical design/plan status with actual commands, hashes, costs, runtimes, review count, and portal URL. Commit only tracked source, tests, sanitized reports, and receipts; do not commit raw caches or the full explorer.

- [ ] **Step 7: Request code review before integration**

Invoke `superpowers:requesting-code-review` against the implementation range. Address only verified issues, rerun the relevant tests, and then use `superpowers:finishing-a-development-branch` to decide whether to merge, push, or retain the worktree.

---

## Execution checkpoints

1. **After Task 2:** Review the actual 400-question sample composition before any model work.
2. **After Task 4:** Obtain explicit authorization for the exact large downloads and Gemini cost cap.
3. **After each 20-question model smoke:** Reject incompatible runtimes or visibly broken response behavior before scaling.
4. **After Task 6 automatic triage:** Begin Codex's all-400 review; do not finalize references from automation alone.
5. **Before portal copy:** Verify the sanitized report contains no raw dataset, keys, or logs.

## Completion definition

The implementation is complete only when the reproducible sample, three challenger caches, blind comparisons, 400 Codex reviews, deeper adjudications, selected references, manifests, sanitized report, full local explorer, tests, and tailnet-only handoff have all been verified. A partial model run or automatic comparison without the all-400 Codex review is not completion.

## Grounding experiment status (2026-07-21)

- [x] Added opt-in Google Search grounding with a distinct provider/cache
  identity and safe count-only grounding metadata.
- [x] Added and passed focused tests for request construction, metadata
  extraction, and cache sanitization.
- [x] Ran all 28 originally flagged Gemini serious-error cases: 28/28 complete,
  26 Search-using requests, 53 queries, 116 chunks, and 110 supports.
- [x] Manually reviewed all 28 paired answers and recorded 12 clear rescues,
  six partial rescues, six no-rescues, one no-Search regeneration rescue, and
  three original-review false positives.
- [x] Built and browser-verified the sanitized progressive-disclosure report.
- [ ] Run the same grounded comparison on the 33 non-failure controls selected
  by the proposed disagreement route before adopting that route at corpus
  scale.
- [ ] Calibrate a local comparison judge against all 400 locked
  Gemini–Perplexity labels; the manual trigger result is an upper bound, not an
  operational classifier result.

## Full grounded-reference run (2026-07-21)

Objective: generate a grounded Gemini 3.6 Flash reference for every one of the
7,759 discovery questions while retaining auditable provenance privately.

Approved data contract:

- The model begins every response with one of `ANSWERED`, `AMBIGUOUS`,
  `INSUFFICIENT_EVIDENCE`, or `CONFLICTING_EVIDENCE`, followed by a short
  `UNCERTAINTY_REASON`.
- The public generation cache stores the normalized status and reason separately
  from the clean answer.
- A separate owner-readable (`0600`) evidence ledger stores returned source URLs
  and titles, supported answer passages, Search queries and invocation state,
  model, prompt hash, UTC timestamp, finish reason, and token usage.
- Raw evidence is excluded from sanitized reports and the rendered portal.

Pre-spend gate:

- [x] Built a deterministic 50-question canary and a 7,759-question full manifest.
- [x] Canary completed 50/50 with zero failures, malformed statuses, empty answers,
  truncations, abnormal finishes, or missing source URLs on Search-using responses.
- [x] Search ran on 39/50. The 11 non-search responses were inspected; six broad
  keyword flags were documented as false positives in a separate review ledger.
- [x] Conservative full-run projection: **$73.40**, below the hard **$150** cap.
- [x] Complete the full grounded run and verify exactly 7,759 unique IDs with zero
  missing records.
- [x] Publish a sanitized progressive-disclosure summary listing failures,
  refusals, non-search responses, and uncertain cases; retain raw provenance in
  the private reference-corpus archive.

Execution receipt at the 2026-07-21 quota boundary:

- 1,906 valid records and matching private evidence records are checkpointed.
- Raising concurrency to 64 workers exposed a provider quota boundary. Google returned
  `RESOURCE_EXHAUSTED` without naming the quota metric; 454 HTTP 429 receipts are
  resumable and do not replace valid records.
- The likely exhausted bucket is the 5,000/day Search-grounding quota visible in AI
  Studio. This is an inference: concurrent requests plus four automatic retries
  amplified request attempts, but Google's error did not include a quota identifier.
- Recovery: after the daily reset, resume at the previously stable 12 workers and 120
  RPM using a capped 4,500-question pending tranche. Finish the remaining tranche after
  the next reset, then run the exact-ID/provenance/cost audit and build the sanitized
  report.

Follow-up diagnosis on 2026-07-22:

- The reset tranche admitted 1,740 additional valid grounded responses, reaching 3,646,
  before sustained 429s resumed.
- One isolated request succeeded after the tranche, but a controlled 20-RPM run produced
  no sustained progress. Three subsequent requests spaced at 1 RPM all returned 429.
- Therefore lowering RPM does not reopen throughput after the active quota/capacity
  window is exhausted. The precise limiting dimension remains unidentified because the
  provider omits the quota metric from its `RESOURCE_EXHAUSTED` response.
- Fixed local retry amplification: HTTP 429 is now checkpointed after one attempt rather
  than retried four times outside the rate limiter. HTTP 503 and transport retries remain
  bounded as before.
- Google documents that only requests-per-day (RPD) resets at midnight Pacific;
  RPM/TPM use minute windows, and the separate spend-rate limit uses a rolling
  ten-minute window. Because the observed 429 omits its quota metric, it must not
  automatically be classified as an RPD exhaustion.
- Ravi's final independent receipt is frozen at a conservative maximum of
  **$40.90945**. The run then switched to Tanwir's working project with separate
  credential-fingerprint accounting.
- Tanwir authorized an independent maximum of **$25**. The durable runner enforces
  that limit against the conservative total that counts every Search query as
  billable. At the cap it checkpoints and stops; the owner's key may be retried
  after the next midnight-Pacific RPD boundary.
