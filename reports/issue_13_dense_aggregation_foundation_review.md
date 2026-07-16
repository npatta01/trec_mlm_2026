# Issue 13: Dense Aggregation Foundation Review

## Executive summary

This project ranks 150 candidate LLM responses for each of 322 development
queries. The current MPNet nearest-neighbor baseline reaches 0.3599 mean
nDCG@10 and 0.4041 mean reciprocal rank. It selected k=50 in 320 of 322 folds,
but 116 queries still have zero nDCG@10.

Issue 13 tests whether combining local-neighbor evidence, semantic-domain
evidence, and a global prior can improve that ranking without leakage. This PR
builds and validates the qrel-safe infrastructure needed to run that experiment.
It does not yet contain an effectiveness result.

The scientific protocol is frozen, the scoring and clustering building blocks
are implemented, and completed fits now go directly into one validated HDF5
file. The qrel-free feasibility software is also implemented. However, the
official feasibility gate, authoritative construction, and ranking evaluation
have not run. Passing the infrastructure tests below shows that the implemented
contracts behave as specified on test fixtures; it does not show that rankings
improve.

This remains one pull request, [PR #20](../../../pull/20), rather than a series
of six stacked pull requests. Merging it would make the experiment foundation
reviewable and ready for the next authorization decision. It would not mean
that the proposed method wins.

## The problem, from first principles

For each development query, the system must order the same 150 candidate LLMs
from most to least useful. Historical relevance judgments, called **qrels**,
say how useful each candidate was for each development query. They are needed
to evaluate a frozen ranking, but careless use during model selection can leak
the answer into the system being tested.

The current baseline represents each query with **MPNet**, a pretrained text
encoder that turns text into a numeric vector. A **nearest-neighbor** method
finds earlier queries whose vectors are most similar to the target query and
transfers evidence from them. Here, `k` is the number of neighbors used.

The experiment adds two broader sources of evidence to that local signal:

- A **semantic domain** is a cluster of training queries that discuss similar
  subjects. Domain evidence averages candidate relevance within the target
  query's cluster.
- A **global prior** is the average training-fold relevance of each candidate,
  without regard to the target query. It provides a conservative fallback when
  local evidence is sparse.

The comparator is the current single 50-neighbor similarity-weighted mean: it
takes the 50 most similar allowed queries and produces one mean relevance grade
for each candidate. The central hypothesis is that explicitly separating
local-neighbor evidence, semantic-domain evidence, and a global prior—and then
combining those three signals under a locked formula—will rank candidates more
robustly than relying on that single 50-neighbor weighted mean, especially for
sparse queries.

### Terms used in this report

| Term | Plain-language meaning |
| --- | --- |
| qrels | Human or benchmark relevance judgments that grade how useful each candidate is for a query. |
| nDCG@10 | A score of how well the ranking places highly relevant candidates in its first 10 positions; higher is better, and early positions matter most. |
| MRR | Mean reciprocal rank: the average of `1 / position` for the first relevant candidate, so putting a useful candidate first scores best. |
| fold | One held-out-query evaluation split. The held-out query is evaluated while the remaining queries supply selection and ranking evidence. |
| leakage | Letting held-out qrels, directly or indirectly, influence the ranking or configuration being evaluated. This can create an unrealistically strong result. |
| MPNet | The frozen pretrained encoder used to convert query text into vectors for similarity comparisons. |
| nearest neighbor | A previously judged query whose MPNet vector is close to the target query's vector. |
| semantic domain | A cluster formed only from allowed training-query vectors, used to summarize broader topic-level evidence. |
| global prior | A candidate's average qrel grade within the allowed training fold, used as query-independent fallback evidence. |

## Where we started

The primary comparison is the leakage-safe, nested dense query-neighbor result
already frozen in the [scientific protocol](../protocols/trec_2026_dense_aggregation.md).
The displayed values below are rounded for readability; the protocol preserves
their full precision.

| Baseline observation | Frozen result |
| --- | ---: |
| Mean nDCG@10 | **0.3599** |
| Mean reciprocal rank (MRR) | **0.4041** |
| MPNet `k=50` selected | **320/322 folds** |
| Queries with zero nDCG@10 | **116** |

The 116 zero-nDCG queries motivate testing broader evidence, but they do not
authorize tuning after results are observed. The protocol locks the candidate
grid, selection procedure, comparison, diagnostics, and decision thresholds in
advance.

## What the experiment is testing

For every held-out outer query, the experiment selects among 16 predefined
local/domain/prior configurations using only permitted inner-fold evidence.
An inner prediction excludes both the outer query and the inner target from
the evidence used to fit clusters and compute scores. The final outer ranking
excludes the outer query. These exclusions are what make the comparison
leakage-safe.

The method succeeds only if all five locked criteria hold at full precision:

1. Paired nDCG confidence-interval lower bound above zero.
2. MRR delta at least `-0.005`.
3. Fewer than `116` zero-nDCG queries.
4. Positive gain on low-density queries.
5. All integrity checks pass.

“Low-density” means queries with at most 20 positively relevant candidates,
as defined by the frozen protocol. The paired confidence interval measures
candidate-minus-baseline nDCG differences for the same 322 queries; requiring
its lower bound to exceed zero is stronger than requiring only a positive
average point estimate.

### The actual scale

“About 180,000 responses” describes the broader source corpus, not the
immediate unit of this effectiveness test. Ranking quality is measured on one
locked **322 × 150** task: 322 query rankings, each containing 150 candidates.

Producing those rankings is nevertheless computationally expensive. Exact
nested construction requires 51,681 leave-two-query-out support sets and 322
leave-one-query-out support sets, each at two cluster counts. That is
**104,006 clustering fits** and, with 10 initializations per fit,
**1,040,060 clustering starts** for each authoritative construction run.

## What this pull request implements

| Area | Status |
| --- | --- |
| Scientific protocol | Frozen |
| Scoring/clustering primitives | Implemented and unit tested |
| Direct HDF5 fit artifact | Implemented and unit tested |
| Qrel-free feasibility software | Implemented and unit tested |
| Official feasibility gate | Not run |
| Authoritative construction | Not run |
| Ranking evaluation | Not run |
| Effectiveness finding | None yet |

The implementation currently provides:

- deterministic scoring, ranking, clustering, fold-exclusion, and hashing
  primitives;
- a fixed HDF5 layout that writes each scheduled fit once, reconstructs a
  bounded row, validates the complete file independently, and publishes it
  without replacing an existing artifact;
- qrel-free feasibility sampling, two-repeat coordination, timing and memory
  evidence, provenance checks, canonical configuration bytes, and stop-gate
  validation;
- immutable artifact and manifest checks, exact experiment schemas, and an
  isolated fixed MPNet `k=50` legacy-control lock;
- command-line preflight boundaries that keep the official qrel leaf outside
  the feasibility command path.

Fits are stored in one directly accessed HDF5 file with a fixed final schema
and fixed datasets. Each scheduled fit occupies a predetermined slot; readers
load bounded rows, full-file validation checks the completed artifact, and
publication atomically installs the validated file without replacing an
existing destination. The final code supports only this architecture. Storage
mechanics do not change the frozen folds, candidate set, scoring formula,
clustering contract, metrics, or promotion criteria. The exact dataset layout,
validation rules, and artifact lifecycle are in the
[HDF5 fit-store design](../docs/superpowers/specs/2026-07-15-dense-aggregation-hdf5-fit-store-design.md).

## What this pull request does not test

The test suite does not establish any of the following:

- that the official two-repeat feasibility gate meets its time, memory,
  repeatability, and integrity thresholds on the authoritative host;
- that a full 104,006-fit construction completes successfully;
- that authoritative construction artifacts pass their final-scale validation;
- that the proposed rankings improve nDCG@10, preserve MRR, rescue zero-nDCG
  queries, or help low-density queries;
- that two authoritative construction and evaluation runs agree;
- that the locked success criteria are met.

No official feasibility receipt, construction, qrel-using evaluation,
repeat-comparison receipt, or effectiveness decision is reported here.

## Verification evidence

The following task-scoped checks were recorded while implementing and reviewing
the direct-HDF5 architecture. Dates are the dates of their associated
implementation commits. These commands overlap in coverage, so their passing
counts must not be added together.

### 2026-07-15: HDF5 fit file and artifact validation

```bash
python -m pytest -q tests/test_dense_aggregation_fit_file.py
# 34 passed in 0.77s

python -m pytest -q \
  tests/test_dense_aggregation_fit_file.py \
  tests/test_dense_aggregation_artifacts.py
# 100 passed in 0.90s
```

These checks cover the final direct-HDF5 fit-file contract and its integration
with artifact validation. They use controlled fixtures, not the authoritative
construction workload.

### 2026-07-16: feasibility and command-line boundaries

```bash
python -m pytest -q \
  tests/test_dense_aggregation_feasibility.py \
  tests/test_dense_aggregation_cli.py
# 101 passed in 32.56s

python -m pytest -q tests/test_dense_aggregation_feasibility.py -k \
  'fifo or inode_swap or qrel or drift or fresh_passing_retry or canonical'
# 33 passed, 29 deselected in 5.49s
```

The 101-test result covers the feasibility and CLI suites. The 33-test
adversarial selection exercises named pipe, inode-swap, qrel-embargo, drift,
fresh-retry, and canonical-byte failure paths. Passing these simulated and
fixture-based checks does not mean the official feasibility gate has run.

### 2026-07-16: affected infrastructure suites

```bash
PYTHONPATH=.venv/lib/python3.13/site-packages python -m pytest -q \
  tests/test_dense_aggregation_fit_file.py \
  tests/test_dense_aggregation_contract.py \
  tests/test_legacy_dense_control.py \
  tests/test_dense_aggregation_feasibility.py \
  tests/test_dense_aggregation_cli.py
# 259 passed in 46.87s
```

This affected-suite run covers the fit-file contract, experiment contract,
legacy control, feasibility software, and CLI. It includes a reviewed fix that
preserves the legacy lock hash binding. It is still infrastructure evidence,
not ranking-quality evidence.

### 2026-07-16: final whole-branch verification

Fresh verification was run at commit
`be258b99c209359be7625cace12edbbcfa726aee` over the branch range
`origin/main...be258b99c209359be7625cace12edbbcfa726aee`. The shell Anaconda
Python provides pytest, while the repository `.venv` provides the pinned
scientific packages, so the selected scientific suites used this exact prefix:
`PYTHONPATH=.venv/lib/python3.13/site-packages python -m pytest`.

The focused final-HDF5 path passed:

```bash
PYTHONPATH=.venv/lib/python3.13/site-packages python -m pytest -q \
  tests/test_dense_aggregation_fit_file.py \
  tests/test_dense_aggregation_artifacts.py \
  tests/test_dense_aggregation_feasibility.py \
  tests/test_dense_aggregation_cli.py \
  tests/test_dense_aggregation_contract.py \
  tests/test_legacy_dense_control.py
# 325 passed in 47.05s
```

The complete dense-aggregation/qrel-safe suite passed:

```bash
PYTHONPATH=.venv/lib/python3.13/site-packages python -m pytest -q \
  tests/test_dense_aggregation.py \
  tests/test_dense_aggregation_artifacts.py \
  tests/test_dense_aggregation_cli.py \
  tests/test_dense_aggregation_clustering.py \
  tests/test_dense_aggregation_contract.py \
  tests/test_dense_aggregation_experiment.py \
  tests/test_dense_aggregation_feasibility.py \
  tests/test_dense_aggregation_fit_file.py \
  tests/test_dense_aggregation_hashing.py \
  tests/test_dense_aggregation_runtime.py \
  tests/test_legacy_dense_control.py
# 616 passed in 49.36s
```

The required bare full-repository attempt did not collect the suite:

```bash
python -m pytest -q
# ERROR tests/test_label_inference.py
# ModuleNotFoundError: No module named 'torch'
# 1 error in 0.60s; exit 2
```

Torch is unavailable in the shell Anaconda environment. It was not installed
and dependencies were not changed for this optional collection blocker. The
passing 616-test command above is the complete selected qrel-safe scientific
suite, not a substitute claim that the full repository suite passed.

Static and safety checks found no whitespace errors, compilation failures,
HDF5 fit-file qrel-path literals, or credential-like assignments in the design,
cleanup plan, or this report. The legacy-name sweep returned only the deliberate
contract assertion that `trec_mlm/dense_aggregation_cache.py` is absent from the
implementation inventory. `git status --short` was empty before this report
update. Across `origin/main...HEAD`, the branch contains 35 changed files with
26,104 insertions and 1 deletion. That whole-branch growth includes the complete
new experiment foundation. The direct-HDF5 cleanup range `36db726..HEAD`,
restricted to `trec_mlm` and `tests`, has 3,958 insertions and 8,826 deletions:
a net reduction of 4,868 lines, including deletion of the custom cache module
and its two cache-specific test files.

## Safety and scientific boundaries

The design separates construction from evaluation deliberately:

- The feasibility gate has no qrel argument and must finish twice in fresh
  processes before the experiment can proceed.
- Qrels remain embargoed until both the qrel-free feasibility gate and the
  subsequent legacy-control lock pass.
- Training evidence for an evaluated query is restricted by the frozen fold
  rules; tests exercise mutations intended to catch excluded-query leakage.
- Candidate configuration selection cannot use outer-fold nDCG, MRR, zero
  counts, density slices, or the legacy control result.
- Artifacts fail closed on missing, extra, reordered, malformed, mutated, or
  inconsistent content.

These are implementation and protocol safeguards. Their passing tests increase
confidence that an authorized run will follow the preregistered experiment;
they cannot predict whether the resulting ranking will be better.

## Limitations

- All passing evidence in this report is fixture-based infrastructure evidence;
  the bare full-repository suite remains uncollected because Torch is absent
  from the shell Anaconda environment.
- Official host runtime and memory are unknown until the two-repeat qrel-free
  feasibility run is authorized and completed.
- Construction-scale I/O and validation have not been observed on the full fit
  inventory.
- No official qrels have been used for this experiment's ranking evaluation,
  so there is no candidate ranking result to compare with the baseline.
- Even a future passing development result would be a preregistered development
  finding, not confirmation of generalization to unseen TREC 2026 test qrels.

## Next decision

After merge, the next decision is whether to authorize the official two-repeat
qrel-free feasibility run on the authoritative host. That run must use fresh
artifact roots and satisfy the locked runtime, memory, repeatability,
provenance, and integrity conditions. A stop result ends the experiment before
qrel access and requires an advisor-approved amendment discussion.

If the feasibility gate passes, the fixed legacy MPNet control is exported and
locked in a separate qrel-using step. Only after both checkpoints pass may the
authoritative construction and read-only ranking evaluation begin. The five
locked success criteria then determine `promote`, `reject`, or
`reject-invalid`; infrastructure test counts do not enter that decision.

## Review map

Suggested order for a newcomer:

1. [Frozen scientific protocol](../protocols/trec_2026_dense_aggregation.md) —
   research question, folds, qrel embargo, workload, evaluation, and locked
   decision rule.
2. [HDF5 fit-store design](../docs/superpowers/specs/2026-07-15-dense-aggregation-hdf5-fit-store-design.md) —
   fixed datasets, bounded reads, complete-file validation, and atomic
   publication.
3. [Original implementation plan](../docs/superpowers/plans/2026-07-14-dense-aggregation.md) —
   the complete preregistered experiment sequence.
4. [Scoring primitives](../trec_mlm/dense_aggregation.py) and
   [clustering primitives](../trec_mlm/dense_aggregation_clustering.py) — the
   numerical core.
5. [Direct HDF5 fit file](../trec_mlm/dense_aggregation_fit_file.py) and
   [artifact validation](../trec_mlm/dense_aggregation_artifacts.py) — fit
   persistence and immutable publication.
6. [Qrel-free feasibility software](../trec_mlm/dense_aggregation_feasibility.py),
   [CLI](../trec_mlm/dense_aggregation_cli.py), and
   [entry point](../trec_mlm/dense_aggregation_entrypoint.py) — preflight,
   sampled execution, repeat evidence, and command boundaries.
7. [Experiment contracts](../trec_mlm/dense_aggregation_experiment.py) and
   [legacy control](../trec_mlm/legacy_dense_control.py) — frozen construction
   schemas and the isolated MPNet comparison lock.
8. [PR #20](../../../pull/20) — the single review and merge discussion for this
   foundation.
