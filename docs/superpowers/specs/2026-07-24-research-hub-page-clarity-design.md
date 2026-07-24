# Research Hub Page-Clarity Redesign

## Objective

Make the task explanation, research-question index, and data catalog immediately
understandable without changing the research model or adding decorative density.

## Task explanation

Replace the four similar text sections with:

1. a plain-English statement of the ranking task;
2. a responsive visual flow showing `new request → available evidence → ranking
   method → ordered list of 150 candidate LLMs`;
3. one concrete miniature example that distinguishes the user's request,
   candidate evidence, and the produced ranking; and
4. a visually distinct evidence-boundary callout explaining that development
   labels can evaluate historical queries but hidden-test labels are unavailable.

The flow is semantic HTML and CSS, not a bitmap, so it remains readable,
accessible, theme-aware, and editable in Quarto.

## Research-question index

Replace Quarto's dense key/value grid with wide stacked cards. Each card shows:

- the research question as the primary link;
- its plain-English interpretation;
- the current bounded answer;
- status and confidence as compact badges.

Cards use one column at all widths so explanatory text never competes for a
narrow column.

## Data catalog

Replace the three-column key/value grid with two-column horizontal cards,
collapsing to one column on narrow screens. Group records under
organizer-provided, collected, and generated/evidence headings. Each card
prioritizes the dataset title and plain-English description; category and
availability become compact badges.

## Content and architecture

The source remains Quarto Markdown. Index-card markup may be maintained directly
in the small index pages, but facts must continue to match the canonical record
front matter. Shared visual rules belong in `research/hub/assets/theme.scss`.
No external scripts, fonts, images, or public hosting are introduced.

## Verification

- Render the complete Quarto site without warnings.
- Check the three revised pages at desktop and phone widths.
- Confirm links and catalog resources still resolve.
- Run the focused research-hub integration/render tests and Ruff.
- Copy only the sanitized `_site` output to the existing tailnet-only portal.
