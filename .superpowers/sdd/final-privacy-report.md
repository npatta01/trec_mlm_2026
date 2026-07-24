# Final resource-staging privacy report

## Scope

This change hardens only the allowlisted Research Hub HTML staging boundary in
`scripts/stage_research_hub_resources.py`.

The prior scanner rejected one machine-specific home prefix,
`/home/npatta01`, but allowed equivalent local paths from other Unix, macOS,
and Windows hosts. The replacement detector rejects:

- Unix home paths rooted at `/home/<user>/...`;
- macOS home paths rooted at `/Users/<user>/...`;
- Unix temporary paths rooted at `/tmp/...` (and `/var/tmp/...`);
- Windows absolute drive paths such as `C:\Users\alice\...` and
  `D:\private\...`; and
- Windows UNC paths such as `\\fileserver\research\...`.

Boundary checks prevent an absolute-path match from starting inside an HTTP
URL path. Regression coverage also confirms that ordinary relative resources,
root-relative `/api/...` and Research Hub paths, and non-executable web links
whose URL paths contain `/home/...` or `/tmp/...` remain stageable.
The scanner evaluates both literal text and one layer of JSON/JavaScript slash
escaping, so escaped local paths are rejected without treating an escaped web
URL as local.

The existing exact allowlist, output-root and symlink confinement, atomic
writes, private-key and `file:///` checks, external executable-asset checks,
external `<base>` rejection, and recursive `iframe[srcdoc]` inspection are
unchanged.

## TDD evidence

RED, after adding only the portable-path and accepted-path tests:

```bash
uv run --extra dev pytest tests/research_hub/test_resources.py -q
```

```text
6 failed, 13 passed
```

All six new local-path cases failed because staging did not raise: Linux home,
macOS home, Unix temporary, Windows user, a second Windows drive, and UNC.
The accepted relative/site-path test already passed.

The second RED cycle covered encoded content typical of generated HTML:

```text
2 failed, 18 passed, 2 deselected
```

The failures were an escaped Unix home path and an escaped UNC path.

GREEN, after adding the boundary-aware detector:

```bash
uv run --extra dev pytest tests/research_hub/test_resources.py \
  -q -k stage_resources
```

```text
20 passed, 2 deselected
```

Those 20 focused cases include the pre-existing allowlist, output confinement,
symlink, `file:///`, private-key, external executable, external `<base>`, and
recursive `srcdoc` checks.

## Verification

Changed-file and full Research Hub Ruff:

```bash
uv run --extra dev ruff check \
  trec_mlm/research_hub \
  scripts/stage_research_hub_resources.py \
  tests/research_hub
```

```text
All checks passed!
```

Direct staging of the real allowlisted reports and byte comparisons:

```bash
uv run python scripts/stage_research_hub_resources.py \
  --root . --output-dir research/hub/_site
cmp reports/trec_2025_mllm_interactive_report.html \
  research/hub/_site/library/trec-2025-interactive.html
cmp reports/trec_2026_approaches_guide.html \
  research/hub/_site/library/approaches-guide.html
```

All three commands exited 0.

The requested whole-suite command was also run:

```bash
uv run --extra dev pytest tests/research_hub -q
```

```text
108 passed in 2.98s
```

## Scope guard

This privacy commit includes only:

- `scripts/stage_research_hub_resources.py`;
- the resource-staging test hunks in
  `tests/research_hub/test_resources.py`; and
- this report.

It does not include relationship, data, background, UI, portal, progress, or
other concurrent changes.
