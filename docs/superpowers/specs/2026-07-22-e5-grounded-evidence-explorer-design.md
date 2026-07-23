# E5 grounded-evidence failure explorer design

## Objective

Replace the current profile-only failure explorer with a private, self-contained
HTML report that makes the frozen E5 raw-response ranking evidence inspectable.
For every development query, a reviewer must be able to see which candidates
ranked highly, which candidates were development-positive, what each component
ranked, and which grounded historical question-response records most strongly
matched the target.

The report is diagnostic. It must not alter rankings, tune the experiment, or
describe historical behavior as verified expertise.

## Inputs and boundaries

The builder reads only already-frozen artifacts:

- the 322 development query texts and complete development qrels;
- the five sealed per-query rankings in the E5 evaluation;
- the exact query and candidate inventories from refinement run 1;
- `selected_record_ids.json` and `top_record_similarities.npy` from refinement
  run 1;
- the validated historical record inventory containing candidate ID, source
  query ID, question, response, and record ID; and
- the existing candidate behavior profiles.

The qrels annotate the frozen ordering and define post-run failure cohorts. They
must not influence candidate ordering, evidence selection, or similarity. The
selected record IDs and similarities must be consumed exactly as persisted; the
explorer must not recompute nearest neighbors.

Raw historical answers may appear only in the private rendered report and its
private build inputs. They must not be committed to Git, copied into the portal
index, or exposed through a public listener. The final HTML remains under the
existing tailnet-only portal.

## Approaches considered

### Selected: self-contained evidence drawers

Extend the static explorer with an optional, validated evidence bundle. Embed
only the three strongest historical records for the final top-ten candidates
and every development-positive candidate. This keeps the report portable while
providing enough evidence for useful spot checks.

### Rejected: separate page per query

Per-query pages would reduce individual page size but fragment filtering,
comparison, and navigation across 322 files.

### Rejected: server-backed lazy loading

A live backend could serve every historical record on demand, but it would add
an operational service and weaken the existing self-contained, static,
tailnet-only delivery model.

## Architecture

### Evidence-bundle builder

Add a task-specific builder that validates and joins the frozen E5 artifacts
into a renderer-neutral evidence bundle. It must:

1. require exact equality of query and candidate inventories across artifacts;
2. verify every selected record belongs to the displayed candidate;
3. verify similarities are finite and non-increasing in persisted evidence
   order;
4. retain exactly the first three persisted records for each included
   query-candidate pair;
5. attach component ranks without comparing raw scores across systems;
6. derive post-run cohort and mechanism tags from frozen ranks and qrels; and
7. emit no response text to stdout or logs.

The evidence schema is optional from the generic explorer's perspective. A
normal run without E5 evidence must continue to produce the existing explorer.

### Generic renderer extension

Extend the current query explorer renderer to accept the validated optional
bundle. Rendering remains entirely client-side after page load; the HTML must
make no network requests and require no adjacent JSON files.

The renderer must escape all query, profile, question, and response text. It
must label cosine values as `E5 cosine similarity`, component positions as
`rank`, and the exported TREC value as `System score`. It must not call any of
these values probability, confidence, correctness, or expertise.

### Published artifact

Generate one replacement report at the existing private explorer location and
keep the portal link stable. The portal description must disclose that full
historical answers are embedded and that the page is private and potentially
large.

## User interface

### Query list and filters

Keep weakest-first navigation and add filters for:

- zero fused nDCG@10;
- persistent all-system zero;
- raw-success/fusion-suppressed;
- new zero regression;
- rescued prior zero; and
- possible missing-context wording.

The missing-context flag is a deterministic surface heuristic for pronouns or
deictic wording without an explicit antecedent. It must be labeled `context-risk
flag`, never a proven failure cause.

### Query summary

Each selected query shows:

- query ID and text;
- fused nDCG@10 and first relevant rank;
- cohort tags;
- the best development-positive candidate's ranks in dense, profile, prior
  fusion, raw hierarchical, and new fusion; and
- a concise mechanism tag.

Mechanism tags are assigned in this order:

1. `fusion suppression` when a positive candidate is top ten in raw
   hierarchical ranking but no positive survives the new fused top ten;
2. `aggregation suppression` when a positive candidate is top ten by its best
   persisted raw-record cosine across candidates but is outside the raw
   hierarchical top ten;
3. `evidence/routing gap` when no positive candidate reaches the top ten even by
   best-record cosine; and
4. `relevant candidate retrieved` when the new fused top ten contains a
   positive candidate.

These are deterministic descriptions of ranks, not assessor explanations.

### Candidate table

Continue to show the final top 20 plus all development-positive candidates.
For every row show:

- final rank and development grade;
- dense, profile, prior-fusion, raw-hierarchical, and new-fusion ranks;
- best selected-record cosine and mean of the persisted top ten; and
- a visible marker when evidence details are available.

Candidate rows must remain ordered by the frozen final ranking. Qrels can add
rows for positives below rank 20 but cannot reorder existing rows.

### Grounded historical evidence drawer

Provide an expandable drawer for every final top-ten candidate and every
development-positive candidate. It contains exactly three records in persisted
similarity order. Each record shows:

- E5 cosine similarity;
- historical source query ID and full question;
- full historical answer; and
- a neutral behavior tag: `substantive response`, `exact refusal`, or `empty`.

The drawer heading must say `Nearest historical behavior`, with an inline note
that similarity and substantive output do not establish correctness.

Candidate behavior profiles remain available but secondary to the exact
evidence drawer.

## Error handling

Fail closed before writing HTML if inventories differ, selected records are
missing or assigned to the wrong candidate, array shapes differ, similarities
are non-finite or out of order, component rankings are incomplete, or qrels do
not cover the exact 322-by-150 inventory.

Write the final report atomically. A failed build must leave the previous live
report untouched. Do not silently fall back to lexical response matching or to
profile-only rendering when the E5 evidence bundle was requested.

## Testing and verification

Focused fixture tests must cover:

- exact top-three selection from persisted order;
- record-to-candidate membership validation;
- component-rank joins and candidate union rules;
- each mechanism and cohort tag;
- context-risk wording without claiming causality;
- HTML escaping of query and response text;
- preservation of frozen candidate ordering;
- refusal, empty, and substantive behavior tags;
- failure atomicity; and
- unchanged generic explorer behavior without an evidence bundle.

Full verification must additionally confirm:

- all 322 queries render;
- the final top 20 and all positives appear for every query;
- evidence exists for every final top-ten and positive candidate;
- no more than three full historical records appear per candidate;
- the displayed aggregate metrics match the sealed evaluation;
- the generated HTML makes no external requests;
- no credentials or unrelated raw data are embedded;
- the live HTTPS artifact returns the expected title and metadata; and
- `tailscale serve status` remains `tailnet only`.

## Success criterion

A reviewer can open any failed query and answer, without consulting raw arrays:

1. which candidate should have been retrieved according to development labels;
2. where that candidate ranked in each frozen component;
3. which historical examples supported the predicted and relevant candidates;
4. whether fusion or aggregation suppressed an otherwise top-ten signal; and
5. whether the released query text carries an obvious context-risk flag.
