# Grounded-reference judging policy

## Decision

Use each grounded response as an evidence packet, not as a canonical string that
candidate answers must reproduce. Route the question before judging candidate
answers:

- `SINGLE`: judge against the supported central answer.
- `MULTI`: accept a coherent answer to any supported interpretation.
- `DEBATABLE`: accept either conclusion when the factual premises, recognized
  framework, and reasoning support it.
- `SKIP`: exclude the question for every candidate because essential context is
  missing.
- `MANUAL`: hold the question out until its reference contract is repaired.

The 1,485-question Gemini Search retry currently provides 1,352 usable
references, 119 global skips, and 14 manual holds after review of all 206
uncertainty-labeled responses.

## Candidate verdicts

- `correct` (2): substantively answers at least one accepted interpretation;
  the central claims are supported and there is no material contradiction.
- `partial` (1): contains a useful supported core plus an important factual
  flaw, materially misleading omission, or contradictory additional claim.
- `incorrect` (0): the central responsive answer is contradicted by the
  evidence.
- `nonanswer` (0): refuses, answers the wrong subject, or only says the question
  is ambiguous without making a substantive supported commitment.
- `not_established` (pending): offers a plausible lane that the reference did
  not investigate sufficiently.

Do not penalize brevity, different wording, missing citations, or failure to
enumerate every interpretation unless the original question explicitly asks
for those features. Judge the whole response: listing one correct answer among
several material false guesses does not earn full credit.

## Symmetric handling of missing reference coverage

`not_established` is not automatically incorrect and must not disappear only
from the candidate that triggered it. Resolve it at the question level:

1. If the candidate exposed a valid interpretation, add that branch and rejudge
   every candidate for the question.
2. If the lane is contradicted or outside the question's scope, assign the
   appropriate `incorrect` or `nonanswer` verdict.
3. If the question or evidence packet is defective, move the question globally
   to `MANUAL` or `SKIP`.

Until resolution, report the candidate's point as pending and compute official
comparisons only over a common resolved question set. This prevents both false
negatives from incomplete references and denominator gaming.

## Production boundary

The current pointwise judge did not receive explicit interpretation branches
and mapped `not_established` to unsupported. Existing judgments therefore
cannot be mechanically relabeled as if this policy had already been applied.
Persist the routing and branch contract, validate it on a small blinded
calibration set, and then rejudge the affected subset before mixing old and new
scores.
