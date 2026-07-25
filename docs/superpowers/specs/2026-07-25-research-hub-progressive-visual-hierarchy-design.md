# Research Hub Progressive Visual Hierarchy

## Objective

Reduce the Research Hub's first-view text density and make the meaning of a
specific research record visually obvious.

The first screen of a theme, question, idea, plan, or study must present:

1. one takeaway;
2. one meaningful visual when the source record defines one; and
3. one next action.

Evidence, readiness gaps, related work, provenance, and technical details remain
available through progressive disclosure.

## Approved visual direction

Use the approved warm editorial treatment:

- warm off-white page background;
- near-black primary text;
- rust for current problems, risks, and unresolved status;
- teal for proposed changes, selected paths, and positive action;
- stronger body contrast with less gray text;
- Inter Display for large headlines and Inter for body copy;
- open layouts and rules instead of a grid of cards.

Color reinforces labels but never replaces them. Every risk, proposal, state,
and action remains explicitly named.

## Information hierarchy

### First view

The first viewport contains only:

- a small record type, theme, and status line;
- a large plain-English takeaway;
- one supporting sentence;
- a source-defined visual;
- a clearly labeled decision test when applicable; and
- one next action.

Long explanations, metadata, and evidence lists must not compete with the
takeaway.

### Details

The following sections start collapsed:

- evidence and current knowledge;
- readiness gaps or evidence limits;
- related work and relationships;
- provenance, paths, run identifiers, and technical details.

The page remains understandable if every collapsed section stays closed.
Expanding a section does not navigate away or lose the reader's position.

## Meaning-first visuals

Do not generate a generic step diagram merely because a record has several
fields.

A visual is rendered only when its source record declares a supported visual
model with explicit content. Supported models are:

- **Causal comparison:** current problem versus proposed change, followed by a
  decision test.
- **Evidence argument:** claim connected to supporting and limiting evidence,
  followed by the bounded conclusion.
- **Option comparison:** alternatives compared on named criteria, followed by
  the selected or unresolved choice.
- **Narrative timeline:** consequential changes over time, followed by the
  current position.

The first implementation uses a causal comparison for
`failure-aware-fusion`:

- **Today:** one fixed mixture combines historical queries, candidate profiles,
  responses, and generated references even when some signals are unreliable.
- **Proposed:** a reliability gate chooses which evidence sources should
  influence the request.
- **Decision test:** determine whether the adaptive mixture beats the frozen
  dense/profile control on development evidence.
- **Next action:** define and freeze the reliability rule before evaluation.

Every visual has the same information in nearby semantic text. Its meaning must
survive grayscale, forced colors, CSS removal, and a narrow viewport.

## Source model

Canonical Markdown front matter may define:

```yaml
visual:
  type: causal_comparison
  current_label: "Today"
  current_summary: "One fixed evidence mixture creates uneven failures."
  current_items:
    - "Past requests"
    - "Candidate profiles"
    - "Responses"
    - "Generated reference"
  change_label: "Proposed change"
  proposed_label: "Proposed"
  proposed_summary: "A reliability gate chooses the evidence mixture."
  proposed_routes:
    - label: "Known request pattern"
      action: "Use similar historical requests."
    - label: "Stable candidate strength"
      action: "Use semantic profiles."
    - label: "Response evidence passes checks"
      action: "Use responses or references."
  decision_test: "Does the adaptive mixture beat the frozen dense/profile control?"
```

The Python loader validates the declared model and its required fields. The
renderer selects a deterministic semantic template by `visual.type`. It never
infers or invents visual content during a build.

Records without a curated visual model show the takeaway and next action
without a placeholder diagram.

## Index views

Indexes display a short, expandable list rather than miniature detail pages.
Each initial row shows:

- status;
- title;
- one-sentence takeaway; and
- next action.

Theme, stage, and readiness remain available as grouping or filter labels.
Long summaries and relationship lists appear only after expansion or on the
detail page.

## Responsive behavior

- Desktop causal comparisons use a clearly labeled side-by-side layout.
- Mobile layouts stack `Today`, `Proposed`, and `Decision test` vertically.
- Headlines use a large but bounded responsive scale.
- No first-view paragraph exceeds roughly 45–55 characters per line.
- Collapsed controls have visible focus, state, and minimum touch size.
- Animation is optional and disabled under reduced-motion preferences.

## Verification

The implementation is complete when:

- tests reject incomplete or unsupported visual models;
- tests prove records without a visual model do not receive a generic diagram;
- tests prove the causal comparison uses only explicit source text;
- semantic text retains all visual meaning with CSS removed;
- representative index and detail pages fit 390-pixel and desktop viewports;
- true light, dark, grayscale, accent-disabled, and forced-colors captures are
  inspected;
- collapsed sections work with keyboard input and preserve heading order;
- the Quarto generator, site audit, catalog, and private tailnet delivery gates
  remain green.

## Out of scope

- inventing diagrams from unreviewed prose;
- adding a JavaScript visualization framework;
- hiding evidence from agents or the machine-readable catalog;
- changing research conclusions;
- public deployment.
