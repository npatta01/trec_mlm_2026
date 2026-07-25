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

## Fix round 1

Addressed every item in `task-1-review.md`:

- Moved the curated visual into the idea hero and emitted exactly one
  `hub-next-action` element there, before mechanism and evidence content. This
  establishes the binding first-viewport hierarchy while leaving disclosure
  and broader layout work to the next task.
- Removed renderer-authored `Proposed change` and `Decision test:` copy. The
  proposed section heading and figure caption now contain only canonical source
  values; the empty, accessibility-hidden change divider remains a CSS hook.
- Required `visual.type` to be a nonempty string before supported-type lookup.
- Rejected non-string keys in both the visual mapping and route mappings before
  unknown-key set/sort operations.
- Added regressions for empty/list/mapping visual types, mixed-type visual and
  route keys, visual/next-action ordering, single next-action emission, and the
  absence of renderer-authored visual copy.

### Test-first evidence

- `uv run --extra dev pytest tests/research_hub/test_validate.py -k 'visual_type or visual_keys or route_keys' -q` initially produced five expected failures: unhashable `visual.type` values raised `TypeError`, the empty type used the wrong error, and mixed-type keys crashed while sorting.
- `uv run --extra dev pytest tests/research_hub/test_render.py -k causal_visual -q` initially failed because no `hub-next-action` element existed in the hero.
- After the fixes, those commands passed with `6 passed` and `1 passed`, respectively.

### Verification

- `uv run --extra dev pytest tests/research_hub/test_validate.py tests/research_hub/test_render.py tests/research_hub/test_site_integration.py -q` — 93 passed in 2.62s.
- `uv run python scripts/build_research_hub.py` — exit 0.
- `uv run python scripts/build_research_hub.py --check` — exit 0.
- `uv run ruff check trec_mlm/research_hub/validate.py trec_mlm/research_hub/render.py tests/research_hub/test_validate.py tests/research_hub/test_render.py tests/research_hub/test_site_integration.py` — all checks passed.
- `git diff --check` — exit 0.

### Remaining concern

The new hero ordering and semantic hooks are in place, but their visual styling
and disclosure behavior remain intentionally owned by the later layout task.
