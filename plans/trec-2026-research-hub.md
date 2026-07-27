# TREC 2026 Research Hub Implementation Plan

**Lifecycle status:** Completed on 2026-07-25 by
[PR #35](https://github.com/npatta01/trec_mlm_2026/pull/35) at merge commit
[`45872ef`](https://github.com/npatta01/trec_mlm_2026/commit/45872efe6b2b9553876af80991437b1f1c045df2).
This implementation plan is superseded for ongoing repository navigation by
the [newcomer research-map plan](trec-2026-newcomer-research-map.md). Its
historical checklist is preserved below without retroactive rewriting.

The living Hub is verified from the repository root with:

```bash
uv run python scripts/build_research_hub.py --check
uv run --extra dev pytest tests/research_hub -q
quarto render research/hub
uv run python scripts/audit_research_hub_site.py \
  --root . \
  --output-dir research/hub/_site
```

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-backed Quarto Research Hub that teaches the TREC task, catalogs data and prior work, explains research questions and studies, tracks ideas and next tests, and preserves drill-down access to the existing 98-system evidence registry.

**Architecture:** Canonical QMD records with typed YAML front matter hold human-authored research context. A focused Python package validates those records against `experiments/ledger.json`, generates deterministic QMD/JSON views, and leaves Quarto 1.9.38 to render the responsive static site. Humans read Quarto HTML; agents use `research/README.md`, canonical QMD, and the generated `research/catalog.json`.

**Tech Stack:** Python 3.11+, PyYAML 6.x, existing `trec_mlm.experiment_ledger`, Quarto 1.9.38, Pandoc 3.8.3, Bootstrap/SCSS supplied by Quarto, small dependency-free browser JavaScript, pytest, Ruff.

## Global Constraints

- Use Quarto `1.9.38`; fail clearly when another version attempts a checked build.
- Keep `experiments/ledger.json` and frozen receipts canonical for run identities, metrics, evidence status, artifact state, caveats, and provenance.
- Do not change any historical score, evaluation scope, promotion decision, source digest, or existing generated ledger output.
- Canonical research narrative lives in QMD with structured YAML front matter; generated QMD, JSON, Markdown, and HTML are never edited directly.
- Research questions are primary navigation; technical approaches are many-to-many tags and secondary navigation.
- Every research question requires a technical title, `plain_english`, current answer, confidence boundary, evidence references, and explicit open/parked status.
- Backfilled rationale is exactly `contemporaneous`, `reconstructed`, or `unknown`; never present reconstructed history as preregistered.
- Studies may have many runs, one run, or no ranking runs. Every existing run must eventually link to at least one study.
- A test may enter `ready` only with hypothesis, rationale, expected signal, decision rule, dependencies, priority, evaluation boundary, artifact plan, and next action.
- Only systems in the same evaluation scope may share a numerical comparison.
- Diagnostic, restricted, invalid, aggregate-only, and unscored evidence must remain visibly distinct from adopted evidence.
- The summary is a decision brief; it must not become a complete leaderboard, caveat dump, or history page.
- The site is static and must retain essential conclusions without JavaScript. Do not add Shiny, R, a server runtime, an external CDN, remote fonts, analytics, or a third-party comment system.
- Rendered samples must not expose secrets, raw private logs, restricted datasets, unnecessary personal data, or direct identifiers.
- All generated repository paths are relative, deterministic, UTF-8, newline-terminated, and atomically replaced only after complete in-memory validation.
- Do not publish publicly, change sharing permissions, or enable Tailscale Funnel. Private portal delivery is a separate verified handoff after implementation.
- Normal build commands are:

```bash
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run python scripts/build_research_hub.py --check
```

---

## File and Module Map

### Python package

- `trec_mlm/research_hub/__init__.py` — public hub interfaces.
- `trec_mlm/research_hub/model.py` — immutable record and build dataclasses plus controlled vocabularies.
- `trec_mlm/research_hub/load.py` — safe QMD front-matter parsing and source discovery.
- `trec_mlm/research_hub/validate.py` — entity, relationship, path, evidence, and readiness validation.
- `trec_mlm/research_hub/render.py` — deterministic catalog, Markdown, QMD fragment, and run-page rendering.
- `trec_mlm/research_hub/build.py` — orchestration, atomic writes, and current-output checks.
- `trec_mlm/research_hub/audit.py` — rendered HTML privacy, offline-resource, semantic-structure, and link audit.

### Command adapters

- `scripts/build_research_hub.py` — build/check generated source views.
- `scripts/audit_research_hub.py` — audit a rendered Quarto site.

### Canonical research sources

- `research/README.md` — concise human/agent repository entry point.
- `research/hub/_quarto.yml` — static website configuration and navigation.
- `research/hub/index.qmd` — generated-summary host.
- `research/hub/start-here/*.qmd` — task, worked example, evaluation, and glossary.
- `research/hub/background/*.qmd` — prior art, competitions, literature, and methods landscape.
- `research/hub/data/*.qmd` — structured data-catalog records.
- `research/hub/questions/*.qmd` — research-question records.
- `research/hub/studies/*.qmd` — coherent-study records.
- `research/hub/ideas/*.qmd` — idea/ready/running/parked records.
- `research/hub/approaches/*.qmd` — technical approach records.
- `research/hub/protocols/index.qmd` — indexed links to canonical protocols.
- `research/hub/evidence/*.qmd` — coverage, history, and run-index hosts.
- `research/hub/assets/theme.scss` — responsive visual system.
- `research/hub/assets/ledger.js` — progressive client-side filters.
- `research/hub/assets/ledger-script.html` — local script include.

### Generated and tracked

- `research/catalog.json` — machine-readable agent catalog.
- `reports/trec_2026_research_hub_summary.md` — concise terminal/GitHub decision brief.
- `research/hub/generated/*.qmd` — summary, coverage, history, and runs fragments.
- `research/hub/generated/runs/*.qmd` — one generated technical page per run.

### Tests

- `tests/research_hub/test_load.py`
- `tests/research_hub/test_validate.py`
- `tests/research_hub/test_render.py`
- `tests/research_hub/test_build.py`
- `tests/research_hub/test_audit.py`
- `tests/research_hub/test_real_sources.py`
- `tests/research_hub/test_quarto_smoke.py`

---

### Task 1: Define the record model and parse canonical QMD safely

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `trec_mlm/research_hub/__init__.py`
- Create: `trec_mlm/research_hub/model.py`
- Create: `trec_mlm/research_hub/load.py`
- Create: `tests/research_hub/conftest.py`
- Create: `tests/research_hub/test_load.py`

**Interfaces:**
- Consumes: repository-relative QMD files with YAML front matter bounded by `---`.
- Produces:
  - `HubRecord(kind: str, record_id: str, title: str, source_path: str, metadata: Mapping[str, Any], body: str)`
  - `HubSources(questions, studies, tests, approaches, datasets, background)`
  - `parse_qmd_record(root: Path, path: Path, kind: str) -> HubRecord`
  - `load_hub_sources(root: Path, source_dir: Path = Path("research/hub")) -> HubSources`

- [ ] **Step 1: Add the direct YAML dependency and failing parser tests**

Add this core dependency in `pyproject.toml`:

```toml
"PyYAML>=6.0.3,<7",
```

Create `tests/research_hub/test_load.py`:

```python
from pathlib import Path

import pytest

from trec_mlm.research_hub.load import load_hub_sources, parse_qmd_record


def test_parse_qmd_record_preserves_metadata_and_markdown(tmp_path: Path) -> None:
    path = tmp_path / "research/hub/questions/candidate-evidence.qmd"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\n"
        "question_id: candidate-evidence\n"
        "title: Can candidate evidence help?\n"
        "status: open\n"
        "---\n\n"
        "## In plain English\n\nDoes model information help ranking?\n",
        encoding="utf-8",
    )

    record = parse_qmd_record(tmp_path, path, "question")

    assert record.kind == "question"
    assert record.record_id == "candidate-evidence"
    assert record.title == "Can candidate evidence help?"
    assert record.source_path == "research/hub/questions/candidate-evidence.qmd"
    assert record.body.startswith("## In plain English")


def test_parse_qmd_record_rejects_unsafe_yaml_tag(tmp_path: Path) -> None:
    path = tmp_path / "bad.qmd"
    path.write_text(
        "---\nquestion_id: !!python/object/apply:os.system [echo unsafe]\n"
        "title: Unsafe\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid YAML front matter"):
        parse_qmd_record(tmp_path, path, "question")


def test_load_hub_sources_sorts_records_by_id(tmp_path: Path) -> None:
    directory = tmp_path / "research/hub/questions"
    directory.mkdir(parents=True)
    for record_id in ("zeta", "alpha"):
        (directory / f"{record_id}.qmd").write_text(
            f"---\nquestion_id: {record_id}\ntitle: {record_id.title()}\n---\n",
            encoding="utf-8",
        )

    sources = load_hub_sources(tmp_path)

    assert [record.record_id for record in sources.questions] == ["alpha", "zeta"]
```

- [ ] **Step 2: Lock dependencies and verify RED**

Run:

```bash
uv lock
uv run --extra dev pytest tests/research_hub/test_load.py -q
```

Expected: collection fails because `trec_mlm.research_hub` does not exist.

- [ ] **Step 3: Implement immutable model types**

Create `trec_mlm/research_hub/model.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RECORD_KINDS = frozenset(
    {"question", "study", "test", "approach", "dataset", "background"}
)


@dataclass(frozen=True)
class HubRecord:
    kind: str
    record_id: str
    title: str
    source_path: str
    metadata: Mapping[str, Any]
    body: str


@dataclass(frozen=True)
class HubSources:
    questions: tuple[HubRecord, ...] = ()
    studies: tuple[HubRecord, ...] = ()
    tests: tuple[HubRecord, ...] = ()
    approaches: tuple[HubRecord, ...] = ()
    datasets: tuple[HubRecord, ...] = ()
    background: tuple[HubRecord, ...] = ()

    def all_records(self) -> tuple[HubRecord, ...]:
        return (
            self.questions
            + self.studies
            + self.tests
            + self.approaches
            + self.datasets
            + self.background
        )
```

- [ ] **Step 4: Implement safe parsing and deterministic discovery**

Create `trec_mlm/research_hub/load.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .model import HubRecord, HubSources, RECORD_KINDS

DIRECTORIES = {
    "question": "questions",
    "study": "studies",
    "test": "ideas",
    "approach": "approaches",
    "dataset": "data",
    "background": "background",
}
ID_FIELDS = {
    "question": "question_id",
    "study": "study_id",
    "test": "test_id",
    "approach": "approach_id",
    "dataset": "dataset_id",
    "background": "resource_id",
}


def _relative_source(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"hub source escapes repository root: {path}") from exc


def parse_qmd_record(root: Path, path: Path, kind: str) -> HubRecord:
    if kind not in RECORD_KINDS:
        raise ValueError(f"unknown hub record kind: {kind}")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"{path}: QMD record must start with YAML front matter")
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path}: unterminated YAML front matter") from exc
    try:
        metadata: Any = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML front matter") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: YAML front matter must be an object")
    id_field = ID_FIELDS[kind]
    record_id = metadata.get(id_field)
    title = metadata.get("title")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError(f"{path}: {id_field} must be a nonempty string")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{path}: title must be a nonempty string")
    body = "\n".join(lines[closing + 1 :]).strip()
    return HubRecord(
        kind=kind,
        record_id=record_id,
        title=title,
        source_path=_relative_source(root, path),
        metadata=metadata,
        body=body,
    )


def load_hub_sources(
    root: Path,
    source_dir: Path = Path("research/hub"),
) -> HubSources:
    grouped: dict[str, tuple[HubRecord, ...]] = {}
    for kind, directory in DIRECTORIES.items():
        paths = (
            path
            for path in (root / source_dir / directory).glob("*.qmd")
            if path.name != "index.qmd"
        )
        records = [parse_qmd_record(root, path, kind) for path in paths]
        grouped[kind] = tuple(sorted(records, key=lambda row: row.record_id))
    return HubSources(
        questions=grouped["question"],
        studies=grouped["study"],
        tests=grouped["test"],
        approaches=grouped["approach"],
        datasets=grouped["dataset"],
        background=grouped["background"],
    )
```

Export the interfaces from `trec_mlm/research_hub/__init__.py`.

- [ ] **Step 5: Verify GREEN and lint**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_load.py -q
uv run --extra dev ruff check trec_mlm/research_hub tests/research_hub/test_load.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit the parser foundation**

```bash
git add pyproject.toml uv.lock trec_mlm/research_hub tests/research_hub/test_load.py
git commit -m "feat: parse research hub records"
```

---

### Task 2: Validate schemas, relationships, readiness, and evidence scope

**Files:**
- Create: `trec_mlm/research_hub/validate.py`
- Create: `tests/research_hub/test_validate.py`
- Modify: `trec_mlm/research_hub/__init__.py`

**Interfaces:**
- Consumes: `HubSources` and the existing `LedgerBuild` returned by `build_ledger(root)`.
- Produces:
  - `ValidationSummary(record_counts: Mapping[str, int], linked_system_count: int)`
  - `validate_hub_sources(sources: HubSources, ledger: LedgerBuild, root: Path) -> ValidationSummary`

- [ ] **Step 1: Write failing validation-contract tests**

Create shared fixtures in `tests/research_hub/conftest.py`. `ledger_build` calls
`build_ledger(Path("."))`; `minimal_sources` contains one valid record per
entity kind and uses the real `dense_nested` system ID plus tracked evidence
paths. Include:

```python
def test_validation_requires_plain_english_question(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    question = replace(
        minimal_sources.questions[0],
        metadata={
            **minimal_sources.questions[0].metadata,
            "plain_english": "",
        },
    )
    with pytest.raises(ValueError, match="plain_english"):
        validate_hub_sources(
            replace(minimal_sources, questions=(question,)),
            ledger_build,
            tmp_path,
        )


def test_validation_rejects_ready_test_without_decision_rule(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    ready = replace(
        minimal_sources.tests[0],
        metadata={
            **minimal_sources.tests[0].metadata,
            "stage": "ready",
            "decision_rule": "",
        },
    )
    with pytest.raises(ValueError, match="decision_rule"):
        validate_hub_sources(
            replace(minimal_sources, tests=(ready,)),
            ledger_build,
            tmp_path,
        )


def test_validation_rejects_unknown_study_system(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    study = replace(
        minimal_sources.studies[0],
        metadata={
            **minimal_sources.studies[0].metadata,
            "system_ids": ["not-a-real-system"],
        },
    )
    with pytest.raises(ValueError, match="not-a-real-system"):
        validate_hub_sources(
            replace(minimal_sources, studies=(study,)),
            ledger_build,
            tmp_path,
        )


def test_validation_requires_reconstructed_rationale_evidence(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    study = replace(
        minimal_sources.studies[0],
        metadata={
            **minimal_sources.studies[0].metadata,
            "rationale_provenance": "reconstructed",
            "evidence_refs": [],
        },
    )
    with pytest.raises(ValueError, match="reconstructed.*evidence_refs"):
        validate_hub_sources(
            replace(minimal_sources, studies=(study,)),
            ledger_build,
            tmp_path,
        )
```

The fixture must contain one valid record of every entity kind, one real ledger
system ID, and repository-relative evidence files under `tmp_path`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_validate.py -q
```

Expected: collection fails because `trec_mlm.research_hub.validate` does not exist.

- [ ] **Step 3: Define controlled vocabularies and validation summary**

Create `trec_mlm/research_hub/validate.py` with:

```python
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

from trec_mlm.experiment_ledger import LedgerBuild

from .model import HubRecord, HubSources

QUESTION_STATUSES = frozenset({"open", "partially_answered", "answered", "parked"})
CONFIDENCE_LEVELS = frozenset({"low", "moderate", "high"})
STUDY_TYPES = frozenset(
    {
        "ranking_experiment",
        "audit",
        "data_quality",
        "reference_review",
        "feasibility",
        "evidence_synthesis",
    }
)
STUDY_STATUSES = frozenset({"planned", "running", "concluded", "blocked"})
STUDY_OUTCOMES = frozenset(
    {"positive", "negative", "mixed", "inconclusive", "invalid", "unscored", "pending"}
)
DECISIONS = frozenset(
    {
        "adopt",
        "retain_control",
        "reject",
        "stop",
        "iterate",
        "revisit_on_trigger",
        "informational",
        "pending",
    }
)
TEST_STAGES = frozenset({"idea", "ready", "running", "completed", "parked"})
RATIONALE_PROVENANCE = frozenset({"contemporaneous", "reconstructed", "unknown"})
DATA_SENSITIVITY = frozenset({"public", "project_internal", "restricted", "secret"})
DATA_SHARING = frozenset(
    {"repository", "local_only", "external_reference", "metadata_only"}
)

READY_FIELDS = (
    "hypothesis",
    "rationale",
    "expected_signal",
    "decision_rule",
    "dependencies",
    "priority",
    "evaluation_boundary",
    "artifact_plan",
    "next_action",
)


@dataclass(frozen=True)
class ValidationSummary:
    record_counts: Mapping[str, int]
    linked_system_count: int
```

- [ ] **Step 4: Implement strict field and relationship validation**

Implement the strict scalar helpers exactly:

```python
def require_string(record: HubRecord, field: str) -> str:
    value = record.metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record.source_path}: {field} must be a nonempty string")
    return value.strip()


def require_string_list(record: HubRecord, field: str) -> tuple[str, ...]:
    value = record.metadata.get(field)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{record.source_path}: {field} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{record.source_path}: {field} contains duplicates")
    return tuple(value)


def require_choice(
    record: HubRecord,
    field: str,
    allowed: frozenset[str],
) -> str:
    value = require_string(record, field)
    if value not in allowed:
        raise ValueError(
            f"{record.source_path}: {field} must be one of {sorted(allowed)}"
        )
    return value


def validate_repo_reference(root: Path, record: HubRecord, value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{record.source_path}: evidence path must be repository-relative: {value}"
        )
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{record.source_path}: evidence path escapes repository: {value}"
        ) from exc
    if not resolved.exists():
        raise ValueError(
            f"{record.source_path}: referenced evidence does not exist: {value}"
        )
```

Implement `validate_question`, `validate_study`, `validate_test`,
`validate_dataset`, and `validate_background` by calling these helpers for every
required field and controlled vocabulary declared in the design specification.
`validate_hub_sources` starts with:

```python
records = sources.all_records()
owners: dict[tuple[str, str], str] = {}
for record in records:
    key = (record.kind, record.record_id)
    if key in owners:
        raise ValueError(
            f"duplicate {record.kind} id {record.record_id!r}: "
            f"{owners[key]} and {record.source_path}"
        )
    owners[key] = record.source_path

question_ids = {row.record_id for row in sources.questions}
study_ids = {row.record_id for row in sources.studies}
test_ids = {row.record_id for row in sources.tests}
approach_ids = {row.record_id for row in sources.approaches}
dataset_ids = {row.record_id for row in sources.datasets}
system_ids = {row.system_id for row in ledger.systems}
```

It then validates every cross-reference, requires each representative system to
also occur in its study's `system_ids`, rejects completed tests without a
`resulting_study_id`, and rejects a study comparison whose declared
`comparison_system_ids` spans more than one ledger evaluation scope.

- [ ] **Step 5: Run validation tests, existing ledger tests, and Ruff**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_validate.py tests/test_experiment_ledger_integration.py -q
uv run --extra dev ruff check trec_mlm/research_hub tests/research_hub
```

Expected: all tests pass; existing ledger invariants remain 98 systems, 3,542
per-query rows, 1,650 relevant-candidate slices, and 33 density slices.

- [ ] **Step 6: Commit validation**

```bash
git add trec_mlm/research_hub tests/research_hub/test_validate.py
git commit -m "feat: validate research hub relationships"
```

---

### Task 3: Generate deterministic agent, summary, coverage, history, and run views

**Files:**
- Create: `trec_mlm/research_hub/render.py`
- Create: `trec_mlm/research_hub/build.py`
- Create: `scripts/build_research_hub.py`
- Create: `tests/research_hub/test_render.py`
- Create: `tests/research_hub/test_build.py`
- Modify: `trec_mlm/research_hub/__init__.py`

**Interfaces:**
- Consumes: validated `HubSources` and existing `LedgerBuild`.
- Produces:
  - `HubBuild(sources, ledger, validation)`
  - `build_research_hub(root: Path) -> HubBuild`
  - `render_hub_outputs(build: HubBuild) -> Mapping[Path, str]`
  - `write_hub_outputs(build: HubBuild, root: Path) -> tuple[Path, ...]`
  - `assert_hub_outputs_current(build: HubBuild, root: Path) -> None`

- [ ] **Step 1: Write failing deterministic-render tests**

Create tests proving:

```python
def test_render_catalog_is_deterministic(sample_build: HubBuild) -> None:
    first = render_catalog(sample_build)
    second = render_catalog(sample_build)
    assert first == second
    payload = json.loads(first)
    assert payload["resources"][0]["canonical_path"].startswith("research/hub/")
    assert "generated_at" not in payload


def test_summary_leads_with_decision_not_system_count(sample_build: HubBuild) -> None:
    summary = render_summary_markdown(sample_build)
    assert summary.index("## Current position") < summary.index("## Open questions")
    assert "98 systems tested" not in summary


def test_run_pages_preserve_evaluation_scope(sample_build: HubBuild) -> None:
    pages = render_run_pages(sample_build)
    page = pages[Path("research/hub/generated/runs/dense_profile_equal.qmd")]
    assert "dev_322x150" in page
    assert "promoted_development" in page


def test_write_is_atomic_on_renderer_failure(
    sample_build: HubBuild,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "research/catalog.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("preserve me\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "render_hub_outputs",
        lambda build: (_ for _ in ()).throw(RuntimeError("render failed")),
    )
    with pytest.raises(RuntimeError, match="render failed"):
        write_hub_outputs(sample_build, tmp_path)
    assert existing.read_text(encoding="utf-8") == "preserve me\n"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_render.py tests/research_hub/test_build.py -q
```

Expected: collection fails because the render/build modules do not exist.

- [ ] **Step 3: Implement build and rendering dataclasses**

In `build.py`:

```python
@dataclass(frozen=True)
class HubBuild:
    sources: HubSources
    ledger: LedgerBuild
    validation: ValidationSummary


def build_research_hub(root: Path) -> HubBuild:
    sources = load_hub_sources(root)
    ledger = build_ledger(root)
    validation = validate_hub_sources(sources, ledger, root)
    return HubBuild(sources=sources, ledger=ledger, validation=validation)
```

In `render.py`, implement catalog rendering in full:

```python
def _related_ids(record: HubRecord) -> list[str]:
    fields = (
        "question_ids",
        "study_ids",
        "next_test_ids",
        "approach_ids",
        "dataset_ids",
    )
    return sorted(
        {
            value
            for field in fields
            for value in record.metadata.get(field, [])
            if isinstance(value, str)
        }
    )


def _rendered_path(record: HubRecord) -> str:
    directory = {
        "question": "questions",
        "study": "studies",
        "test": "ideas",
        "approach": "approaches",
        "dataset": "data",
        "background": "background",
    }[record.kind]
    return f"{directory}/{record.record_id}.html"


def render_catalog(build: HubBuild) -> str:
    resources = [
        {
            "id": record.record_id,
            "type": record.kind,
            "title": record.title,
            "summary": str(
                record.metadata.get("plain_english")
                or record.metadata.get("summary")
                or record.metadata.get("objective")
                or ""
            ),
            "tags": sorted(record.metadata.get("tags", [])),
            "canonical_path": record.source_path,
            "rendered_path": _rendered_path(record),
            "related_ids": _related_ids(record),
        }
        for record in build.sources.all_records()
    ]
    for system in build.ledger.systems:
        study_ids = sorted(
            study.record_id
            for study in build.sources.studies
            if system.system_id in study.metadata.get("system_ids", [])
        )
        resources.append(
            {
                "id": system.system_id,
                "type": "run",
                "title": system.system_id,
                "summary": system.method_summary,
                "tags": sorted(
                    {
                        system.experiment_family,
                        system.evaluation_scope,
                        system.promotion_status,
                        system.evidence_status,
                    }
                ),
                "canonical_path": "experiments/ledger.json",
                "rendered_path":
                    f"generated/runs/{system.system_id}.html",
                "related_ids": study_ids,
            }
        )
    resources.sort(key=lambda row: (row["type"], row["id"]))
    return json.dumps(
        {"version": 1, "resources": resources},
        indent=2,
        sort_keys=True,
    ) + "\n"
```

Implement the remaining functions with these exact signatures and order:

```python
def render_summary_markdown(build: HubBuild) -> str:
    # Sections: title, evidence boundary, Current position, Open questions,
    # Next tests, Navigation. Questions sort by status priority then ID.
    lines = ["# TREC 2026 Research Hub", "", "Development evidence only."]
    lines.extend(_summary_current_position(build))
    lines.extend(_summary_open_questions(build))
    lines.extend(_summary_next_tests(build))
    lines.extend(_summary_navigation())
    return "\n".join(lines).rstrip() + "\n"


def render_summary_qmd(build: HubBuild) -> str:
    return _render_summary_cards(build).rstrip() + "\n"


def render_coverage_qmd(build: HubBuild) -> str:
    return _render_coverage_matrix(build).rstrip() + "\n"


def render_history_qmd(build: HubBuild) -> str:
    return _render_decision_history(build).rstrip() + "\n"


def render_runs_qmd(build: HubBuild) -> str:
    return _render_run_index(build).rstrip() + "\n"


def render_run_pages(build: HubBuild) -> dict[Path, str]:
    return {
        Path(f"research/hub/generated/runs/{system.system_id}.qmd"):
            _render_run_page(build, system).rstrip() + "\n"
        for system in sorted(build.ledger.systems, key=lambda row: row.system_id)
    }
```

The private helpers named above are implemented in the same focused module and
are directly tested for ordering, scope guards, and escaping. The catalog shape
is:

```json
{
  "version": 1,
  "resources": [
    {
      "id": "candidate-evidence",
      "type": "question",
      "title": "Can candidate-side semantic evidence improve query-only retrieval?",
      "summary": "Plain-English summary.",
      "tags": ["candidate-evidence"],
      "canonical_path": "research/hub/questions/candidate-evidence.qmd",
      "rendered_path": "questions/candidate-evidence.html",
      "related_ids": ["candidate-semantic-profiles"]
    }
  ]
}
```

Sort resources by `(type, id)`, use `json.dumps` with `indent=2` and
`sort_keys=True`,
and terminate every output with one newline.

- [ ] **Step 4: Implement atomic output orchestration**

`render_hub_outputs` must return exactly:

```python
{
    Path("research/catalog.json"): render_catalog(build),
    Path("reports/trec_2026_research_hub_summary.md"):
        render_summary_markdown(build),
    Path("research/hub/generated/summary.qmd"): render_summary_qmd(build),
    Path("research/hub/generated/coverage.qmd"): render_coverage_qmd(build),
    Path("research/hub/generated/history.qmd"): render_history_qmd(build),
    Path("research/hub/generated/runs.qmd"): render_runs_qmd(build),
    **render_run_pages(build),
}
```

Write every output into one temporary directory under the repository, verify all
content there, then call `Path.replace()` for each destination. On failure,
remove the temporary directory and leave every destination unchanged.

`assert_hub_outputs_current` compares expected bytes to tracked files and reports
missing, unexpected, or different paths in one `ValueError`.

- [ ] **Step 5: Add the thin CLI**

Create `scripts/build_research_hub.py` following the existing ledger CLI:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Build the TREC 2026 Research Hub.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build = build_research_hub(args.root)
    if args.check:
        assert_hub_outputs_current(build, args.root)
    else:
        write_hub_outputs(build, args.root)
    return 0
```

- [ ] **Step 6: Verify unit tests and atomicity**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_render.py tests/research_hub/test_build.py -q
uv run --extra dev ruff check trec_mlm/research_hub scripts/build_research_hub.py tests/research_hub
```

Expected: all tests pass and Ruff is clean.

- [ ] **Step 7: Commit deterministic generation**

```bash
git add trec_mlm/research_hub scripts/build_research_hub.py tests/research_hub
git commit -m "feat: generate research hub views"
```

---

### Task 4: Create the accessible Quarto website shell

**Files:**
- Modify: `.gitignore`
- Create: `research/README.md`
- Create: `research/hub/_quarto.yml`
- Create: `research/hub/index.qmd`
- Create: `research/hub/start-here/index.qmd`
- Create: `research/hub/background/index.qmd`
- Create: `research/hub/data/index.qmd`
- Create: `research/hub/questions/index.qmd`
- Create: `research/hub/studies/index.qmd`
- Create: `research/hub/ideas/index.qmd`
- Create: `research/hub/approaches/index.qmd`
- Create: `research/hub/protocols/index.qmd`
- Create: `research/hub/evidence/coverage.qmd`
- Create: `research/hub/evidence/history.qmd`
- Create: `research/hub/evidence/runs.qmd`
- Create: `research/hub/assets/theme.scss`
- Create: `research/hub/assets/ledger.js`
- Create: `research/hub/assets/ledger-script.html`
- Create: `tests/research_hub/test_quarto_smoke.py`

**Interfaces:**
- Consumes: generated QMD fragments from Task 3.
- Produces: a static Quarto site at `research/hub/_site/` with local assets only.

- [ ] **Step 1: Write the failing Quarto configuration test**

Create:

```python
def test_quarto_configuration_is_static_and_pinned() -> None:
    config = yaml.safe_load(Path("research/hub/_quarto.yml").read_text())
    assert config["project"]["type"] == "website"
    assert config["project"]["output-dir"] == "_site"
    assert config["quarto-required"] == "1.9.38"
    assert config["execute"]["enabled"] is False
    assert "server" not in config


def test_quarto_render_smoke(tmp_path: Path) -> None:
    result = subprocess.run(
        ["quarto", "render", "research/hub", "--output-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "questions/index.html").is_file()
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_quarto_smoke.py -q
```

Expected: FAIL because `_quarto.yml` and the site do not exist.

- [ ] **Step 3: Add Quarto configuration and navigation**

Create `_quarto.yml`:

```yaml
quarto-required: "1.9.38"
project:
  type: website
  output-dir: _site
  resources:
    - assets/ledger.js
website:
  title: "TREC 2026 Research Hub"
  search: true
  navbar:
    left:
      - text: "Start Here"
        href: start-here/index.qmd
      - text: "Background"
        href: background/index.qmd
      - text: "Data"
        href: data/index.qmd
      - text: "Research"
        menu:
          - text: "Questions"
            href: questions/index.qmd
          - text: "Studies"
            href: studies/index.qmd
          - text: "Ideas & Tests"
            href: ideas/index.qmd
          - text: "Approaches"
            href: approaches/index.qmd
      - text: "Protocols"
        href: protocols/index.qmd
      - text: "Evidence"
        menu:
          - text: "Coverage"
            href: evidence/coverage.qmd
          - text: "History"
            href: evidence/history.qmd
          - text: "Runs"
            href: evidence/runs.qmd
format:
  html:
    theme:
      light: [cosmo, assets/theme.scss]
      dark: [darkly, assets/theme.scss]
    toc: true
    anchor-sections: true
    smooth-scroll: true
    link-external-icon: true
execute:
  enabled: false
```

Add `_site/` and `.quarto/` under `research/hub/` to `.gitignore`.

- [ ] **Step 4: Add host pages and generated includes**

Use this exact pattern:

```markdown
---
title: "Research summary"
description: "Current conclusions, decisions, open questions, and next tests."
page-layout: full
---

{{< include generated/summary.qmd >}}
```

The root `index.qmd` includes `generated/summary.qmd`. From the `evidence/`
directory, `coverage.qmd`, `history.qmd`, and `runs.qmd` include
`../generated/coverage.qmd`, `../generated/history.qmd`, and
`../generated/runs.qmd` respectively. Index pages use Quarto `listing` over
their canonical directories and provide one paragraph explaining the entity
type in plain English.

- [ ] **Step 5: Add a sparse responsive theme and no-JavaScript baseline**

`theme.scss` must define:

```scss
/*-- scss:defaults --*/
$font-size-root: 17px;
$content-padding-top: 2rem;

/*-- scss:rules --*/
.hub-hero {
  padding: clamp(1.5rem, 4vw, 3rem);
  border: 1px solid var(--bs-border-color);
  border-radius: 1rem;
  background: color-mix(in srgb, var(--bs-primary) 8%, transparent);
}
.hub-section { margin-block: clamp(2.5rem, 6vw, 5rem); }
.hub-grid { display: grid; gap: 1rem; grid-template-columns: repeat(3, 1fr); }
.hub-card { padding: 1.25rem; border: 1px solid var(--bs-border-color); border-radius: .85rem; }
@media (max-width: 768px) {
  .hub-grid { grid-template-columns: 1fr; }
}
```

`ledger-script.html` contains:

```html
<script src="../assets/ledger.js" defer></script>
```

Only one-directory-deep listing hosts that need enhanced filters include this
file. `ledger.js` enhances elements carrying `data-hub-filter` but does nothing
when those elements are absent. All records remain visible in the source HTML
before JavaScript runs.

- [ ] **Step 6: Verify Quarto render and mobile-safe structure**

Run:

```bash
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run --extra dev pytest tests/research_hub/test_quarto_smoke.py -q
```

Expected: Quarto renders every navigation target successfully.

- [ ] **Step 7: Commit the Quarto shell**

```bash
git add .gitignore research/README.md research/hub tests/research_hub/test_quarto_smoke.py
git commit -m "feat: scaffold Quarto research hub"
```

---

### Task 5: Write Start Here, glossary, and the source-backed Data Catalog

**Files:**
- Create: `research/hub/start-here/problem.qmd`
- Create: `research/hub/start-here/worked-example.qmd`
- Create: `research/hub/start-here/evaluation.qmd`
- Create: `research/hub/start-here/glossary.qmd`
- Create: `research/hub/data/dev-queries.qmd`
- Create: `research/hub/data/dev-qrels.qmd`
- Create: `research/hub/data/candidate-inventory.qmd`
- Create: `research/hub/data/discovery-responses.qmd`
- Create: `research/hub/data/generated-embeddings.qmd`
- Create: `research/hub/data/run-artifacts.qmd`
- Create: `tests/research_hub/test_real_sources.py`

**Interfaces:**
- Consumes: `data/dev/README.md`, `data/dev/file_manifest.tsv`,
  `data/dev/llm_dev_data.tsv`, `data/dev/llm_dev_qrels.txt`,
  `data/llm_discovery/README.md`, `data/llm_discovery/manifest.json`, and the
  existing ledger's artifact states.
- Produces: newcomer-facing task/data pages and six validated dataset records.

- [ ] **Step 1: Add failing real-content tests**

```python
def test_start_here_contains_required_plain_english_sections() -> None:
    text = Path("research/hub/start-here/problem.qmd").read_text(encoding="utf-8")
    assert "## The problem in one sentence" in text
    assert "150 candidate LLMs" in text
    assert "hidden test" in text.lower()


def test_data_catalog_distinguishes_provided_collected_and_generated() -> None:
    sources = load_hub_sources(Path("."))
    categories = {row.metadata["category"] for row in sources.datasets}
    assert {"provided", "collected", "generated"} <= categories


def test_no_dataset_marks_qrels_as_construction_data() -> None:
    sources = load_hub_sources(Path("."))
    qrels = next(row for row in sources.datasets if row.record_id == "dev-qrels")
    assert qrels.metadata["permitted_use"] == ["development_evaluation"]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_real_sources.py -q
```

Expected: FAIL because the Start Here and dataset records do not exist.

- [ ] **Step 3: Write the plain-English task guide**

The problem page must state:

```markdown
## The problem in one sentence

Given a user's request, rank 150 candidate LLMs so that the models most likely
to answer that request well appear first.

## What information do we have?

For development, we have historical requests and relevance labels indicating
which candidate LLMs were suitable. We also have candidate identifiers and
separately collected or generated evidence described in the Data Catalog.

## What are we trying to learn?

We test whether query similarity, candidate descriptions, model responses, or
combinations of those signals improve the ranking without using a held-out
query's relevance labels to construct its answer.

## The evidence boundary

All effectiveness results in this hub are development evidence. The hidden test
labels are unavailable, so a development improvement is not a hidden-test claim.
```

The worked example uses a fabricated request and fabricated candidate names; it
must be labeled illustrative and must not reproduce private collected data.

- [ ] **Step 4: Create dataset records with sanitized samples**

Every dataset QMD front matter must contain:

```yaml
dataset_id: dev-queries
title: "Development queries"
plain_english: "Historical user requests used for development."
category: provided
source_refs:
  - data/dev/llm_dev_data.tsv
sensitivity: project_internal
sharing: repository
permitted_use:
  - development_construction
  - development_evaluation
study_ids: []
availability: tracked_complete
reproducibility: organizer_provided
tags: [queries, development]
```

Use `metadata_only` or `local_only` for discovery responses and missing artifact
classes. Do not copy raw discovery response content into QMD.

- [ ] **Step 5: Verify content, schema, and existing data hashes**

Run:

```bash
uv run python scripts/build_research_hub.py
uv run --extra dev pytest tests/research_hub/test_real_sources.py tests/test_experiment_ledger_integration.py -q
quarto render research/hub
```

Expected: all tests pass; existing ledger sources and hashes validate unchanged.

- [ ] **Step 6: Commit Start Here and Data Catalog**

```bash
git add research/hub/start-here research/hub/data research/catalog.json \
  research/hub/generated reports/trec_2026_research_hub_summary.md \
  tests/research_hub/test_real_sources.py
git commit -m "docs: add research hub task and data guide"
```

---

### Task 6: Index literature, related competitions, methods, and protocols

**Files:**
- Create: `research/hub/background/trec-2025-literature.qmd`
- Create: `research/hub/background/prior-participant-approaches.qmd`
- Create: `research/hub/background/related-competitions.qmd`
- Create: `research/hub/background/llm-routing-literature.qmd`
- Create: `research/hub/background/methods-landscape.qmd`
- Modify: `research/hub/protocols/index.qmd`
- Modify: `tests/research_hub/test_real_sources.py`

**Interfaces:**
- Consumes:
  - `reports/trec_2025_mllm_literature_review.md`
  - `reports/trec_2026_approaches_guide_source_notes.md`
  - `reports/trec_2026_reference_model_landscape_2026-07-21.md`
  - tracked Markdown protocols under `protocols/`
- Produces: five background records and a source-backed protocol index.

- [ ] **Step 1: Write failing index-coverage tests**

```python
def test_background_catalog_indexes_controlling_reviews() -> None:
    sources = load_hub_sources(Path("."))
    paths = {
        path
        for row in sources.background
        for path in row.metadata["source_refs"]
    }
    assert {
        "reports/trec_2025_mllm_literature_review.md",
        "reports/trec_2026_approaches_guide_source_notes.md",
        "reports/trec_2026_reference_model_landscape_2026-07-21.md",
    } <= paths


def test_protocol_index_links_every_tracked_markdown_protocol() -> None:
    index = Path("research/hub/protocols/index.qmd").read_text(encoding="utf-8")
    for path in sorted(Path("protocols").glob("*.md")):
        assert path.as_posix() in index
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_real_sources.py -q
```

Expected: the background and protocol coverage assertions fail.

- [ ] **Step 3: Write background records without duplicating full reports**

Use this record pattern:

```yaml
resource_id: trec-2025-literature
title: "What the 2025 Million LLMs track taught us"
resource_type: literature_review
summary: "A source-backed synthesis of the prior track and its participant approaches."
source_refs:
  - reports/trec_2025_mllm_literature_review.md
tags: [trec-2025, prior-art, llm-routing]
question_ids: []
reviewed_on: 2026-07-24
```

Each body contains:

- a five-sentence maximum plain-English summary;
- "Why this matters for 2026";
- "What this does not prove";
- links to canonical source notes; and
- citations already preserved by the controlling review.

The related-competitions page distinguishes task similarity from direct
comparability and does not import scores into the TREC development leaderboard.

- [ ] **Step 4: Render and verify links**

Run:

```bash
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run --extra dev pytest tests/research_hub/test_real_sources.py -q
```

Expected: background sources and every Markdown protocol are indexed.

- [ ] **Step 5: Commit background and protocols**

```bash
git add research/hub/background research/hub/protocols research/catalog.json \
  research/hub/generated reports/trec_2026_research_hub_summary.md \
  tests/research_hub/test_real_sources.py
git commit -m "docs: index research background and protocols"
```

---

### Task 7: Backfill research questions and ranking studies for all 98 systems

**Files:**
- Create: `research/hub/questions/query-only-routing.qmd`
- Create: `research/hub/questions/candidate-evidence.qmd`
- Create: `research/hub/questions/response-evidence.qmd`
- Create: `research/hub/questions/reference-reliability.qmd`
- Create: `research/hub/questions/failure-conditions.qmd`
- Create: `research/hub/approaches/dense-query-retrieval.qmd`
- Create: `research/hub/approaches/candidate-profiles.qmd`
- Create: `research/hub/approaches/response-evidence.qmd`
- Create: `research/hub/approaches/reference-generation.qmd`
- Create: `research/hub/approaches/reranking.qmd`
- Create: `research/hub/studies/query-controls.qmd`
- Create: `research/hub/studies/dense-query-knn.qmd`
- Create: `research/hub/studies/candidate-label-matching.qmd`
- Create: `research/hub/studies/label-ranker-pilots.qmd`
- Create: `research/hub/studies/response-retrieval.qmd`
- Create: `research/hub/studies/response-volume-normalization.qmd`
- Create: `research/hub/studies/source-query-first.qmd`
- Create: `research/hub/studies/behavior-profile.qmd`
- Create: `research/hub/studies/semantic-profiles.qmd`
- Create: `research/hub/studies/multiresolution-profiles.qmd`
- Create: `research/hub/studies/query-cluster-aggregation.qmd`
- Modify: `tests/research_hub/test_real_sources.py`

**Interfaces:**
- Consumes: all experiments and systems in `experiments/ledger.json`, their
  caveats, controlling sources, and promotion/evidence statuses.
- Produces: five question records, five approach records, and eleven
  ranking-study records whose union covers every existing system ID.

- [ ] **Step 1: Write failing coverage and provenance tests**

```python
def test_every_existing_system_belongs_to_a_study() -> None:
    sources = load_hub_sources(Path("."))
    ledger = build_ledger(Path("."))
    linked = {
        system_id
        for study in sources.studies
        for system_id in study.metadata["system_ids"]
    }
    assert linked == {system.system_id for system in ledger.systems}


def test_representative_systems_are_a_small_default_view() -> None:
    sources = load_hub_sources(Path("."))
    for study in sources.studies:
        representative = study.metadata["representative_system_ids"]
        assert set(representative) <= set(study.metadata["system_ids"])
        assert len(representative) <= 8


def test_backfilled_rationale_never_defaults_to_contemporaneous() -> None:
    sources = load_hub_sources(Path("."))
    allowed = {"contemporaneous", "reconstructed", "unknown"}
    assert {row.metadata["rationale_provenance"] for row in sources.studies} <= allowed
    for row in sources.studies:
        if row.metadata["rationale_provenance"] == "reconstructed":
            assert row.metadata["evidence_refs"]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_real_sources.py -q
```

Expected: FAIL because no ranking-study mapping exists.

- [ ] **Step 3: Create question records**

Every question record uses this complete metadata shape:

```yaml
question_id: candidate-evidence
title: "Can candidate-side semantic evidence improve query-only retrieval?"
plain_english: "When choosing an LLM, does it help to consider what each candidate model is good at, not only compare the request with past requests?"
status: partially_answered
current_answer: "Yes on development evidence, subject to the evidence limits below."
confidence: moderate
confidence_boundary: "Development-only evidence; the hidden test is unavailable."
study_ids:
  - candidate-label-matching
  - behavior-profile
  - semantic-profiles
  - multiresolution-profiles
open_subquestions:
  - "Which candidate evidence generalizes beyond development?"
next_test_ids: []
evidence_refs:
  - reports/trec_2026_experiment_ledger.md
updated_on: 2026-07-24
tags: [candidate-evidence, profiles]
```

The body starts with `## In plain English`, defines project terms, gives a
concrete comparison, and then states evidence limits before study details.

- [ ] **Step 4: Create the eleven ranking-study records**

First create the five approach records with stable IDs and plain-English
definitions so every study's `approach_ids` resolves during validation. Their
initial `study_ids` contain the ranking studies created in this task; Task 8
adds the non-ranking studies.

For each existing `experiment_id`, copy the exact canonical system IDs from:

```bash
jq -r '.experiments[] | [.experiment_id, ([.systems[].system_id] | join(","))] | @tsv' \
  experiments/ledger.json
```

Expected: 11 rows and 98 distinct system IDs.

Each study QMD must contain the complete enumerated `system_ids` list, at most
eight decision-relevant `representative_system_ids`, exact `evidence_refs`, and
this controlled metadata:

```yaml
study_id: semantic-profiles
title: "Candidate semantic profiles"
study_type: ranking_experiment
question_ids: [candidate-evidence]
approach_ids: [candidate-profiles, dense-query-retrieval]
status: concluded
outcome: mixed
objective: "Evaluate whether candidate semantic profiles improve ranking."
hypothesis: "Qrel-free candidate profiles add complementary evidence to the frozen dense control."
rationale: "The tracked protocol and result receipt define this bounded comparison."
rationale_provenance: contemporaneous
protocol_summary: "Construct profiles without qrels, freeze rankings, then evaluate."
decision: adopt
decision_summary: "Promote equal-weight dense/profile fusion; retain profile-only as a control."
system_ids:
  - profile_only
  - dense_profile_equal
  - dense_profile_dense_heavy
  - dense_profile_profile_heavy
  - dense_e5_equal
  - dense_profile_e5_equal
representative_system_ids:
  - dense_profile_equal
  - dense_profile_profile_heavy
  - profile_only
comparison_system_ids:
  - dense_profile_equal
  - profile_only
evidence_refs:
  - protocols/trec_2026_candidate_semantic_profiles.md
  - reports/trec_2026_candidate_semantic_profiles_results.json
dataset_ids: [dev-queries, dev-qrels, candidate-inventory, generated-embeddings]
started_on: unknown
concluded_on: 2026-07-16
supersedes: []
superseded_by: []
limitations:
  - "Development-only evidence."
tags: [candidate-evidence, profiles, fusion]
```

Do not infer a date, hypothesis, or rationale not supported by a controlling
source. Use `unknown` and explain the missing history in the body.

- [ ] **Step 5: Validate exact coverage and unchanged metrics**

Run:

```bash
uv run python scripts/build_research_hub.py
uv run --extra dev pytest tests/research_hub/test_real_sources.py \
  tests/test_experiment_ledger_integration.py -q
jq '[.resources[] | select(.type == "study")] | length' research/catalog.json
```

Expected: all tests pass; the catalog contains at least 11 studies; ledger
metrics and counts remain unchanged.

- [ ] **Step 6: Commit ranking-history backfill**

```bash
git add research/hub/questions research/hub/approaches research/hub/studies \
  research/catalog.json \
  research/hub/generated reports/trec_2026_research_hub_summary.md \
  tests/research_hub/test_real_sources.py
git commit -m "docs: backfill ranking research history"
```

---

### Task 8: Add non-ranking studies, approaches, and the prospective test funnel

**Files:**
- Create: `research/hub/studies/four-model-reference-review.qmd`
- Create: `research/hub/studies/gemini-grounding-failure-review.qmd`
- Create: `research/hub/studies/grounded-data-quality-audit.qmd`
- Create: `research/hub/studies/e5-raw-response-refinement.qmd`
- Create: `research/hub/ideas/failure-aware-fusion.qmd`
- Create: `research/hub/ideas/semantic-failure-clusters.qmd`
- Create: `research/hub/ideas/reference-review-trigger.qmd`
- Modify: `research/hub/questions/response-evidence.qmd`
- Modify: `research/hub/questions/reference-reliability.qmd`
- Modify: `research/hub/questions/failure-conditions.qmd`
- Modify: `research/hub/approaches/response-evidence.qmd`
- Modify: `research/hub/approaches/reference-generation.qmd`
- Modify: `research/hub/approaches/reranking.qmd`
- Modify: `tests/research_hub/test_real_sources.py`

**Interfaces:**
- Consumes: tracked July 2026 reference-review, grounding-failure, data-quality,
  and E5 refinement reports.
- Produces: four first-class non-registry studies, three prospective test
  records, and updated question/approach relationships.

- [ ] **Step 1: Write failing non-ranking and readiness tests**

```python
def test_non_ranking_studies_are_first_class_records() -> None:
    sources = load_hub_sources(Path("."))
    by_id = {row.record_id: row for row in sources.studies}
    assert by_id["four-model-reference-review"].metadata["system_ids"] == []
    assert by_id["grounded-data-quality-audit"].metadata["study_type"] == "data_quality"


def test_ready_queue_contains_only_complete_tests() -> None:
    sources = load_hub_sources(Path("."))
    for row in sources.tests:
        if row.metadata["stage"] == "ready":
            for field in READY_FIELDS:
                assert row.metadata[field]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_real_sources.py -q
```

Expected: FAIL because non-ranking studies and test records are absent.

- [ ] **Step 3: Add evidence-backed non-ranking studies**

Use exact source mappings:

```yaml
four-model-reference-review:
  - reports/trec_2026_four_model_reference_review_2026-07-21.md
  - reports/trec_2026_four_model_per_question_rankings.tsv
gemini-grounding-failure-review:
  - reports/trec_2026_gemini_grounding_failure_review_2026-07-21.md
  - reports/trec_2026_gemini_grounding_failure_review.tsv
grounded-data-quality-audit:
  - reports/trec_2026_grounded_data_quality_audit_2026-07-23.md
e5-raw-response-refinement:
  - reports/trec_2026_e5_raw_response_refinement_results.md
  - reports/trec_2026_e5_raw_response_refinement_results.json
```

These studies use `system_ids: []` unless the existing run registry already owns
the exact system identity. They link decisions and limitations from their own
receipts rather than manufacturing comparable leaderboard rows.

Update the linked question `study_ids` and the linked approach `study_ids` in
the same change. Update `next_test_ids` on affected questions when adding the
idea/test records in Step 4.

- [ ] **Step 4: Add ideas and ready-test gating**

Early ideas use:

```yaml
test_id: failure-aware-fusion
title: "Failure-aware evidence fusion"
stage: idea
summary: "Select evidence sources using a query-level reliability signal."
linked_question_ids: [failure-conditions, response-evidence]
missing_readiness_fields: [decision_rule, dependencies, artifact_plan]
tags: [fusion, reliability]
```

Do not label E5 refinement `ready` if its tracked report proves it is already
completed. A completed test links `resulting_study_id: e5-raw-response-refinement`.

- [ ] **Step 5: Add approach synthesis pages**

Each approach page lists linked questions and studies through IDs, preserves
mixed outcomes, and contains:

```yaml
approach_id: response-evidence
title: "Response evidence"
plain_english: "Use model responses or response-derived features as evidence for ranking candidate LLMs."
question_ids: [response-evidence, failure-conditions]
study_ids: [response-retrieval, response-volume-normalization, e5-raw-response-refinement]
current_assessment: "Direct response retrieval underperformed; bounded refinements have separate evidence."
tags: [responses, retrieval]
```

- [ ] **Step 6: Build, validate, and commit**

Run:

```bash
uv run python scripts/build_research_hub.py
uv run --extra dev pytest tests/research_hub -q
quarto render research/hub
```

Expected: all research-hub tests pass and every non-ranking study renders.

```bash
git add research/hub/studies research/hub/ideas research/hub/approaches \
  research/catalog.json research/hub/generated \
  reports/trec_2026_research_hub_summary.md tests/research_hub
git commit -m "docs: add research audits and next-test funnel"
```

---

### Task 9: Add progressive filtering, coverage, history, and run drill-down

**Files:**
- Modify: `trec_mlm/research_hub/render.py`
- Modify: `research/hub/assets/ledger.js`
- Modify: `research/hub/assets/theme.scss`
- Modify: `tests/research_hub/test_render.py`
- Modify: `tests/research_hub/test_quarto_smoke.py`

**Interfaces:**
- Consumes: complete validated hub build.
- Produces: filterable question/study/approach/run indexes, mixed-state coverage
  cells, decision-centered history, and generated run detail pages.

- [ ] **Step 1: Write failing progressive-view tests**

```python
def test_coverage_cell_preserves_mixed_outcomes(real_build: HubBuild) -> None:
    qmd = render_coverage_qmd(real_build)
    assert "mixed" in qmd
    assert "studies" in qmd
    assert "success" not in qmd.lower()


def test_history_orders_known_dates_and_labels_unknown_dates(
    real_build: HubBuild,
) -> None:
    qmd = render_history_qmd(real_build)
    assert "Date unknown" in qmd
    assert qmd.index("2026-07-16") < qmd.index("2026-07-23")


def test_javascript_is_enhancement_only() -> None:
    html = Path("research/hub/_site/evidence/runs.html").read_text(encoding="utf-8")
    assert "dense_profile_equal" in html
    assert "data-hub-filter" in html
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_render.py \
  tests/research_hub/test_quarto_smoke.py -q
```

Expected: the coverage/history behavior or progressive filter assertions fail.

- [ ] **Step 3: Render filterable elements with complete fallback content**

Every filterable row uses:

```html
<article
  class="hub-card hub-filter-row"
  data-hub-filter-row
  data-status="diagnostic_only"
  data-scope="dev_322x150"
  data-search="dense_profile_profile_heavy semantic profiles">
  <h3><a href="../generated/runs/dense_profile_profile_heavy.html">
    dense_profile_profile_heavy
  </a></h3>
  <p>Diagnostic-only · dev_322x150 · stored per-query evidence</p>
</article>
```

Add a local search input with `data-hub-filter`, status/scope selects, and a
screen-reader live region. Server-render all rows before JavaScript runs.

- [ ] **Step 4: Implement dependency-free filtering**

`ledger.js` must:

```javascript
document.querySelectorAll("[data-hub-filter-group]").forEach((group) => {
  const rows = [...group.querySelectorAll("[data-hub-filter-row]")];
  const search = group.querySelector("[data-hub-filter]");
  const status = group.querySelector("[data-hub-status]");
  const scope = group.querySelector("[data-hub-scope]");
  const live = group.querySelector("[data-hub-filter-live]");

  const apply = () => {
    const needle = (search?.value || "").trim().toLowerCase();
    let visible = 0;
    rows.forEach((row) => {
      const matchesSearch = !needle || row.dataset.search.includes(needle);
      const matchesStatus = !status?.value || row.dataset.status === status.value;
      const matchesScope = !scope?.value || row.dataset.scope === scope.value;
      row.hidden = !(matchesSearch && matchesStatus && matchesScope);
      if (!row.hidden) visible += 1;
    });
    if (live) live.textContent = `${visible} results`;
  };

  [search, status, scope].filter(Boolean).forEach((control) => {
    control.addEventListener("input", apply);
    control.addEventListener("change", apply);
  });
  apply();
});
```

- [ ] **Step 5: Render evidence-safe coverage and history**

Coverage groups studies by `(question_id, approach_id)` and displays:

- `uncovered` when no studies link the pair;
- a single outcome plus count when all concluded studies agree; or
- `mixed · N studies` when outcomes differ or include diagnostic/invalid evidence.

History sorts known `concluded_on` dates ascending and places `unknown` dates in
a labeled section. Each event states question, study, conclusion, decision, and
why the next linked study followed only when a source explicitly supports that
relationship.

- [ ] **Step 6: Verify browser behavior and commit**

Run:

```bash
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run --extra dev pytest tests/research_hub -q
uv run --extra dev ruff check trec_mlm/research_hub scripts tests/research_hub
```

Expected: all tests and Ruff pass.

```bash
git add trec_mlm/research_hub research/hub/assets research/hub/generated \
  research/catalog.json reports/trec_2026_research_hub_summary.md \
  tests/research_hub
git commit -m "feat: add progressive research evidence views"
```

---

### Task 10: Audit privacy, offline behavior, links, accessibility, and portability

**Files:**
- Create: `trec_mlm/research_hub/audit.py`
- Create: `scripts/audit_research_hub.py`
- Create: `tests/research_hub/test_audit.py`
- Modify: `tests/research_hub/test_quarto_smoke.py`

**Interfaces:**
- Consumes: a rendered Quarto output directory.
- Produces:
  - `AuditFinding(code: str, path: str, detail: str)`
  - `audit_rendered_hub(site_dir: Path) -> tuple[AuditFinding, ...]`
  - CLI exit `0` for clean output and `1` with all findings otherwise.

- [ ] **Step 1: Write failing audit tests**

```python
def test_audit_rejects_remote_executable_resources(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(
        '<html lang="en"><body><main><script src="https://cdn.example/x.js">'
        "</script></main></body></html>",
        encoding="utf-8",
    )
    findings = audit_rendered_hub(tmp_path)
    assert {row.code for row in findings} == {"remote-executable-resource"}


def test_audit_rejects_secret_markers(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(
        '<html lang="en"><body><main>AKIAABCDEFGHIJKLMNOP</main></body></html>',
        encoding="utf-8",
    )
    findings = audit_rendered_hub(tmp_path)
    assert {row.code for row in findings} == {"possible-secret"}


def test_audit_accepts_external_citation_links(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(
        '<html lang="en"><body><main><a href="https://trec.nist.gov/">'
        "TREC</a></main></body></html>",
        encoding="utf-8",
    )
    assert audit_rendered_hub(tmp_path) == ()
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_audit.py -q
```

Expected: collection fails because the audit module does not exist.

- [ ] **Step 3: Implement a standard-library HTML audit**

Define the immutable finding type:

```python
@dataclass(frozen=True)
class AuditFinding:
    code: str
    path: str
    detail: str
```

Subclass `html.parser.HTMLParser` to collect:

- remote `script[src]`, stylesheet `link[href]`, image `img[src]`, audio/video
  sources, and iframe sources;
- missing `html[lang]`;
- missing `main`;
- images without nonempty `alt`;
- local links whose targets do not exist;
- `localhost`, `127.0.0.1`, or `file://` URLs;
- AWS-style access keys, PEM private-key headers, GitHub tokens, and common
  bearer-token assignments; and
- accidentally rendered paths under `artifacts/`, home directories, or `/tmp`.

Do not flag ordinary external `<a href>` citations.

- [ ] **Step 4: Add CLI and real rendered-site audit**

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Audit rendered Research Hub HTML.")
    parser.add_argument("site_dir", type=Path)
    args = parser.parse_args()
    findings = audit_rendered_hub(args.site_dir)
    for finding in findings:
        print(f"{finding.path}: {finding.code}: {finding.detail}")
    return 1 if findings else 0
```

Add a smoke test that renders to `tmp_path`, audits the result, and asserts no
findings.

- [ ] **Step 5: Run automated and representative visual checks**

Run:

```bash
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run python scripts/audit_research_hub.py research/hub/_site
google-chrome --headless --disable-gpu --hide-scrollbars \
  --window-size=390,844 \
  --screenshot=/tmp/trec-research-hub-mobile.png \
  "file://$PWD/research/hub/_site/index.html"
```

Expected: audit exits `0`; the screenshot shows no clipped headline, overlapping
navigation, or horizontal page scroll. Inspect the screenshot with the local
image viewer before proceeding.

- [ ] **Step 6: Verify keyboard and no-JavaScript reading paths**

Run Chrome once with JavaScript disabled:

```bash
google-chrome --headless --disable-gpu --disable-javascript \
  --dump-dom "file://$PWD/research/hub/_site/evidence/runs.html" \
  > /tmp/trec-research-hub-no-js.html
rg -n "dense_profile_equal|Evaluation scope|Evidence status" \
  /tmp/trec-research-hub-no-js.html
```

Expected: all three strings are present. Manually tab through the primary
navigation and filter controls in the local rendered page; visible focus must
remain present and every control must have a label.

- [ ] **Step 7: Commit audit tooling**

```bash
git add trec_mlm/research_hub/audit.py scripts/audit_research_hub.py \
  tests/research_hub/test_audit.py tests/research_hub/test_quarto_smoke.py
git commit -m "test: audit research hub output"
```

---

### Task 11: Integrate the build, document maintenance, and run final verification

**Files:**
- Modify: `README.md`
- Modify: `research/README.md`
- Modify: `tests/research_hub/test_real_sources.py`
- Modify: `tests/research_hub/test_quarto_smoke.py`
- Modify: `plans/trec-2026-research-hub.md`

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: a documented, checked, reproducible Research Hub with no required
  user action beyond normal repository commands.

- [ ] **Step 1: Add a failing checked-output integration test**

```python
def test_checked_in_research_hub_sources_are_current() -> None:
    root = Path(".")
    build = build_research_hub(root)
    assert build.validation.linked_system_count == 98
    assert_hub_outputs_current(build, root)


def test_real_hub_has_required_sections() -> None:
    sources = load_hub_sources(Path("."))
    assert len(sources.questions) >= 5
    assert len(sources.studies) >= 15
    assert len(sources.approaches) >= 5
    assert len(sources.datasets) >= 6
    assert len(sources.background) >= 5
```

- [ ] **Step 2: Verify RED if outputs drift**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_real_sources.py -q
```

Expected: FAIL if any generated source has not been refreshed; otherwise the new
current-output assertion initially fails until the final build is written.

- [ ] **Step 3: Document build and authoring workflows**

Add this README navigation:

````markdown
## Research Hub

The canonical human-and-agent index for the task, data, prior work, research
questions, studies, decisions, open questions, next tests, and run evidence is
[`research/README.md`](research/README.md).

Build and validate the static Quarto view:

```bash
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run python scripts/build_research_hub.py --check
uv run python scripts/audit_research_hub.py research/hub/_site
```
````

`research/README.md` must explain:

- which files are canonical;
- how an agent should use `research/catalog.json`;
- how to add a question, study, dataset, background resource, or idea;
- readiness requirements;
- how to mark rationale provenance;
- how to build/check/render/audit; and
- that HTML output is derived and private by default.

- [ ] **Step 4: Run the complete cheapest-first verification sequence**

Run:

```bash
uv run python scripts/build_research_hub.py --check
uv run --extra dev ruff check trec_mlm scripts tests
uv run --extra dev pytest tests/research_hub -q
uv run --extra dev pytest tests/test_experiment_ledger.py \
  tests/test_experiment_ledger_outputs.py \
  tests/test_experiment_ledger_integration.py -q
quarto render research/hub
uv run python scripts/audit_research_hub.py research/hub/_site
git diff --check
git status --short
```

Expected:

- generated hub sources are current;
- Ruff reports no errors;
- research-hub and experiment-ledger tests pass;
- Quarto renders successfully;
- rendered audit returns no findings;
- `git diff --check` is clean; and
- only intentional source/generated changes are listed.

- [ ] **Step 5: Update the canonical plan with verification evidence**

In this plan's status appendix, record:

- final commit IDs by task;
- exact test counts;
- Quarto version;
- rendered audit result;
- mobile screenshot path and visual-review result;
- known content explicitly labeled `unknown`; and
- the next recommended backfill or experiment-ledger maintenance action.

- [ ] **Step 6: Commit integration and documentation**

```bash
git add README.md research plans/trec-2026-research-hub.md \
  reports/trec_2026_research_hub_summary.md tests/research_hub
git commit -m "docs: integrate TREC research hub"
```

- [ ] **Step 7: Request independent final review**

Use `superpowers:requesting-code-review` against the merge base. The review must
check:

- specification coverage;
- evidence and evaluation-scope integrity;
- backfilled-rationale provenance;
- privacy and leakage boundaries;
- human/agent source consistency;
- static/offline behavior;
- accessibility and phone layout; and
- absence of unrelated changes.

Address accepted findings with `superpowers:receiving-code-review`, rerun Step 4,
and commit fixes separately.

---

## Historical status snapshot (obsolete after PR #35)

The fields below preserve the pre-implementation handoff as historical context;
they do not describe the repository's current state.

**Status at the time:** Plan written; implementation not started.

**Branch:** `codex/trec-2026-research-hub`

**Design specification:** `docs/superpowers/specs/2026-07-24-trec-2026-research-hub-design.md`

**Next action at the time:** Execute Task 1 with subagent-driven development.
