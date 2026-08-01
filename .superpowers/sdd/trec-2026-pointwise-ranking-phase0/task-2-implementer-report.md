# Task 2 implementer report

## Outcome

Implemented the qrel-free Phase 0 pointwise evidence loader, ranking
construction, exact quartile assignment, and frozen zero-call pair selector.
No organizer qrels, API clients, embedding inference, pairwise calls, prompts,
or private source text enter the implementation outputs.

## Red/green evidence

- Loader red: after API scaffolding, `uv run pytest -q
  tests/test_pointwise_ranking.py` produced five expected failures at the
  unimplemented loader for label hash, record-ID identity, and
  supported/verdict consistency.
- Loader green: the same five tests passed after streamed hash/schema/count
  validation and the record-ID-only join were implemented.
- Construction/selector red: the next focused run had five expected failures
  at unimplemented construction, quartile, and selector functions while the
  first five loader tests remained green.
- Construction/selector green: the focused suite finished with 12 passed,
  including Beta(1,1), exact ten-label means, residuals, deterministic RRF
  ties, frozen quartiles, canonical selector hashing, dynamic use balancing,
  first-pass source diversity, tier priority, and cap escalation.
- Real-input performance red/green: the first selector preflight was
  interrupted after exact construction because a naive full scan of all
  eligible pairs per choice was too slow. Grouping the immutable
  `(tier, prior_distance)` prefix preserved the full dynamic order while the
  rerun completed the frozen 500-pair selection.
- Review-fix red: exact source/development query-ID overlap and normalized-text
  overlap tests both failed with `DID NOT RAISE`; missing and inconsistent
  source-text tests failed for the same reason.
- Review-fix green: source records now preserve only source IDs plus normalized
  SHA-256 fingerprints, using NFKC, casefolding, then collapsed Unicode
  whitespace. Construction validates the frozen development-query hash and
  inventory and rejects either overlap before returning predictions. The four
  new fail-close tests pass.
- Selector oracle: a deterministic seed-31 randomized test over 50 small
  inventories compares every selected pair, cap, and use count against an
  independent naive implementation of the complete dynamic key.

## Files

- `trec_mlm/pointwise_ranking.py`
- `tests/test_pointwise_ranking.py`

## Verification

- Focused plus adjacent regression tests: 42 passed in 0.68 seconds.
- Ruff on both changed files: passed.
- Staged diff whitespace check: passed.
- Tracked-file scan found no local data path, raw label filename, private
  question/response text, API-key marker, or organizer-qrel reference.

## Qrel-free real-input preflight

- Frozen labels SHA-256:
  `7cde37d0d913b97ede43c4791b2f808e6d1b0c863bdcac729efde78480113c48`.
- Frozen record inventory SHA-256:
  `e25237be81ffd86726fb4b3ccd1a2a8bdea7213197f84341b6bf50c23274ae92`.
- 182,764 unique joined records; 6,816 source questions; 6,816 normalized-text
  fingerprints; 150 candidates; 132,077 supported labels. Zero source overlap
  with the 322 frozen development queries passed by both ID and fingerprint.
- Complete 322-by-150 local/residual tensors and 322 complete rankings.
- Query quartile counts: 81, 80, 81, 80. Both candidate quartile families:
  38, 37, 38, 37.
- Selector: 3,165,622 eligible pairs across 5,000 source questions; selected
  500 pairs across 500 questions; cap remained 20; maximum candidate use 20.

## Commit and concerns

- Implementation commit: `cf27ae8` (`feat: construct qrel-free pointwise rankings`).
- Review-fix commit: `c66b32f` (`fix: fail closed on pointwise query overlap`).
- No correctness or leakage concerns remain. The selector materializes its
  IDs-only eligible-pair inventory in memory to support exact dynamic ordering;
  the real-input preflight completed successfully with the frozen data.
