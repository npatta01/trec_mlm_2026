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

## Follow-up: common Unix roots and encoded paths

A whole-feature re-review found that the first portable detector still
recognized only Unix home and temporary roots. Paths rooted at standard Unix
filesystem locations such as `/root`, `/opt`, `/var`, `/srv`, and `/mnt`
therefore remained stageable. Raw-text scanning also allowed HTML
entity-encoded local paths and the noncanonical but valid `file:/...` URL form.

The detector now covers the common Linux/FHS, Nix, and macOS root families:

```text
/Applications /Library /System /Volumes
/bin /boot /dev /etc /home /lib /lib32 /lib64 /libx32
/media /mnt /nix /opt /private /proc /root /run
/sbin /srv /sys /tmp /usr /Users /var
```

It decodes HTML character references before private marker, file URL, local
path, and external CSS import checks. A scheme-specific check rejects `file:`
URLs with one or more slash/backslash separators instead of relying on the
former exact `file:///` marker.

The existing path-start boundary is retained. Absolute local roots do not
match when they occur inside `https://...`, protocol-relative web URLs,
prefixed API routes such as `/api/v1/root/...`, or site paths such as
`/research/var/...` and `/static/mnt/...`. The accepted-path regression also
covers an entity-encoded HTTPS URL.

### Follow-up TDD evidence

RED, with tests added before production changes:

```bash
uv run --extra dev pytest tests/research_hub/test_resources.py \
  -q -k stage_resources
```

```text
25 failed, 20 passed, 2 deselected
```

The failures were the noncanonical `file:/` case, one entity-encoded Unix home
path, and 23 representative common Unix-root paths. All prior staging cases
and the expanded accepted URL/site-path case already passed.

GREEN:

```text
45 passed, 2 deselected
```

### Follow-up verification

```bash
uv run --extra dev pytest tests/research_hub/test_resources.py -q
```

```text
47 passed in 0.22s
```

```bash
uv run --extra dev pytest tests/research_hub -q
```

```text
137 passed in 3.11s
```

```bash
uv run --extra dev ruff check \
  trec_mlm/research_hub \
  scripts/stage_research_hub_resources.py \
  tests/research_hub
```

```text
All checks passed!
```

Direct staging of both real allowlisted reports and byte-for-byte comparisons
against their staged copies exited 0. `git diff --check` also exited 0.

The follow-up remains confined to the staging script, its resource tests, and
this report.

## Follow-up: categorical POSIX path detection

A further whole-feature review showed that any finite root-family list could
be bypassed with another valid absolute root, including `/workspace`,
`/data`, or an arbitrary `/custom-root`. The POSIX detector no longer uses a
root blocklist.

It now classifies any boundary-delimited absolute path with a named first
segment as local, including single named roots. Three exact first-segment
exceptions permit the Research Hub's intentional root-relative site routes:

```text
/api
/research
/static
```

The exceptions cover each route root and all descendants, but not lookalikes
such as `/apiary`. Relative paths remain permitted. Complete HTTP(S) and
protocol-relative URLs are removed from the candidate text before path
classification, and the left-boundary rule excludes HTML closing tags such as
`</script>`.

The real approaches guide also contains two legitimate slash-heavy contexts:
JavaScript regular-expression literals and gzip/base64 payloads in explicitly
declared `<template data-compression="gzip-base64">` elements. Regex literals
are ignored only when their surrounding syntax and closing flags identify a
JavaScript regex expression. Only the contents of templates explicitly marked
with that compression format are omitted from path classification. Ordinary
visible text, attributes, and script strings remain subject to categorical
path detection.

Entity decoding, `file:` URL rejection, Windows drive paths, UNC paths,
external executable/base checks, recursive `srcdoc` scanning, and symlink
confinement are unchanged.

### Categorical TDD evidence

Initial RED:

```text
3 failed, 45 passed, 2 deselected
```

The failures were `/workspace/private/results.json`,
`/data/private/results.json`, and `/custom-root/project/file`. Exact
`/api`, `/research`, and `/static` descendant routes plus arbitrary external
URL paths already passed.

A second RED proved a named single-root path also needed classification:

```text
1 failed, 48 passed, 2 deselected
```

The failing case was `/arbitrary-root`; exact `/api`, `/research`, and
`/static` roots remained accepted.

The first whole-suite run then found one real-report false positive:

```text
1 failed, 140 passed
```

The copied-repository Quarto render identified slash-heavy JavaScript regex
and compressed-template content in the allowlisted approaches guide. A
focused accepted regression reproduced that context before its narrow
contextual handling was added.

Final focused GREEN:

```text
49 passed, 2 deselected
```

### Categorical verification

```bash
uv run --extra dev pytest tests/research_hub/test_resources.py -q
```

```text
51 passed in 0.29s
```

```bash
uv run --extra dev pytest tests/research_hub -q
```

```text
141 passed in 3.04s
```

Full Research Hub Ruff and `git diff --check` exited 0. Direct staging of the
two real allowlisted reports and both byte-for-byte comparisons also exited 0.

This categorical follow-up remains confined to the staging script, its
resource tests, and this report.
