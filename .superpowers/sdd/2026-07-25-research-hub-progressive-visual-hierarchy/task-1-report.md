# Task 1 report: validated causal-comparison visuals

## Outcome

Implemented an optional, strict `visual` metadata schema and semantic renderer
for the initial `causal_comparison` model. The approved failure-aware-fusion
record now supplies the canonical visual data; records without `visual` render
no placeholder diagram.

## Changes

- `validate_visual(record)` validates every canonical record before its
  kind-specific validation. It rejects non-mappings, unsupported types, unknown
  fields, incomplete values, empty item/route lists, and malformed routes.
- `_render_record_visual(record)` emits semantic `figure`, `section`, route
  definition-list, and `figcaption` markup using only authored visual content
  and `html.escape` for source values.
- Idea mechanism steps retain semantic list markup without the generic
  `hub-process` placeholder class.
- Added the approved causal-comparison mapping to `failure-aware-fusion.qmd`
  and an optional commented example to the test template.
- Regenerated the tracked idea-detail outputs.

## Test-first evidence

- Validation test red: unsupported `visual.type` was accepted before schema
  validation.
- Renderer tests red: causal visual markup was absent and a record without a
  visual still emitted the generic `hub-process` class.
- Route-schema test red: complete causal metadata exposed an incorrect nested
  route lookup; the implementation was corrected before final verification.

## Verification

- `uv run --extra dev pytest tests/research_hub/test_validate.py tests/research_hub/test_render.py tests/research_hub/test_site_integration.py -q` — 88 passed.
- `uv run python scripts/build_research_hub.py` — exit 0.
- `uv run python scripts/build_research_hub.py --check` — exit 0.
- `uv run ruff check trec_mlm/research_hub/validate.py trec_mlm/research_hub/render.py tests/research_hub/conftest.py tests/research_hub/test_validate.py tests/research_hub/test_render.py tests/research_hub/test_site_integration.py` — all checks passed.
- `git diff --check` — exit 0.

## Concerns

The semantic classes deliberately have no new CSS in this task; the planned
page-layout work owns visual styling. The emitted headings, lists, and caption
remain meaningful without CSS.
