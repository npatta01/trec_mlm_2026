# Research Hub Visual Language and Idea Explainers

## Objective

Make the research hub easier to read, remember, and explain without changing
its Quarto/Markdown source-of-truth architecture.

The redesign must help a reader understand a specific theme, question, idea,
plan, or study. It must not imply that the project has only a small fixed set of
ideas, plans, or themes.

## Approved direction

The approved visual direction uses:

- Inter for headings and body text, bundled locally so the portal has no remote
  font dependency;
- a near-black, off-white, and deliberate neutral-gray palette;
- strong contrast and generous line height for long-form reading;
- limited accent color that is never the only carrier of meaning;
- status words, borders, shapes, patterns, and line styles that remain
  understandable in grayscale;
- open sections and dividers instead of dense card grids.

The site continues to support light and dark modes. Dark mode is the primary
design reference, while both modes must meet the same readability and
grayscale requirements.

## Information model

The existing records remain the source of truth:

- themes describe durable areas of inquiry;
- questions state what the project is trying to learn;
- ideas and plans describe possible or intended future work;
- studies capture decision-relevant bodies of evidence;
- runs remain technical datapoints inside studies.

The redesign does not impose a fixed count on any record type. Indexes and
relationship views must continue to expand automatically as Markdown records
are added.

## Record explainer pattern

Every human-facing detail page should answer these questions in this order:

1. **What does this mean?** A plain-English headline and short explanation.
2. **Why does it matter?** The research or operational motivation.
3. **How does it work?** A small visual appropriate to that record.
4. **What do we know?** Current evidence, conclusion, or uncertainty.
5. **What is missing?** Readiness gaps or evidence limits.
6. **What happens next?** A concrete next action or linked plan.
7. **How does it connect?** Related themes, questions, ideas, studies, and
   evidence.

Technical metadata remains available, but it follows the explanation instead
of leading it.

## Visual grammar

Visuals are generated from structured record relationships and fields. They
are not hand-authored HTML fragments.

The generator may select among a small set of accessible visual patterns:

- **Process:** numbered steps connected by directional lines.
- **Comparison:** a baseline and alternative shown on shared dimensions.
- **Evidence path:** claim → supporting studies → decision → next test.
- **Relationship map:** the current record connected to related records.
- **State path:** open idea → readiness gaps → planned test → evidence.

Every visual includes a text equivalent in the document. Meaning must survive
CSS removal, grayscale printing, narrow screens, and dark mode.

## Summary and Ideas views

The summary retains a compact cross-theme overview. It shows current
conclusions, open questions, and a concise sample of next work without trying
to display the full ledger.

The Ideas and Tests view is the complete expandable index. It groups any number
of ideas and plans by theme and stage, and exposes a plain-English explanation,
readiness, priority, and next action. Selecting an item opens its dedicated
explainer page.

## Plain-language rules

- Define specialist terms when first used.
- Prefer a concrete verb over an abstract research noun.
- Explain the comparison before naming the method.
- Expand abbreviations in human-facing copy.
- Label development evidence as development evidence.
- Distinguish observed results, interpretations, proposals, and unknowns.
- Keep exact metric names, run IDs, and artifact paths in the technical
  evidence section.

Existing source text remains authoritative. Generated plain-English copy must
come from explicit source fields or deterministic templates, never from
unreviewed inference during the site build.

## Responsive and accessible behavior

- Use a readable measure of roughly 65–75 characters for prose.
- Use a base font size of at least 17 pixels and a body line height near 1.6.
- Collapse multi-step visuals into a vertical sequence on narrow screens.
- Preserve visible keyboard focus and semantic heading order.
- Meet WCAG AA contrast for text and interactive states.
- Respect reduced-motion and forced-colors preferences.
- Do not require hover, color vision, or a wide viewport to recover meaning.

## Source and build behavior

Quarto and Markdown remain canonical. The Python generator:

1. loads and validates source records;
2. generates indexes, relationship summaries, and record explainer fragments;
3. emits semantic HTML through Quarto;
4. keeps the machine-readable catalog synchronized.

Adding a valid record must automatically update its index, owning theme,
related pages, visual relationships, and catalog entry. Contributors should
not edit generated HTML.

## Verification

The implementation is complete when:

- existing research-hub tests pass;
- new tests cover deterministic explainer output and expanding record counts;
- the Quarto site renders without stale generated files;
- internal links and the machine-readable catalog validate;
- representative summary, theme, idea, question, and study pages are visually
  inspected in desktop dark mode and a narrow mobile viewport;
- representative visuals remain understandable in grayscale and with color
  accents removed;
- the private tailnet portal serves the intended sanitized output.

## Out of scope

- changing research conclusions or inventing unrecorded ideas;
- replacing the ledger model;
- adding a JavaScript visualization framework;
- publishing the hub publicly;
- exposing raw datasets, logs, or self-contained interactive HTML bundles.
