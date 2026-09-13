# Muse ambiguity-handling pilot

- Model: `muse-spark-1.2-contributor`
- Prompt variant: `reference_generic_uncertainty`
- Endpoint: Meta native Responses API with `web_search`
- Policy tested: two calls per item; no application Search or output-length cap
- Questions: 4 (1 ambiguous, 2 clear, 1 known-failing)
- Repeats per question: 1
- Expected records: 4
- Completed records: 4
- Hard failures: 0
- Stage-2 finals: 4
- Cache-only rerun items skipped without API calls: 0
- Search queries recorded: 58
- URL citations recorded: 33

The signals below are mechanical indicators for review; they are not a semantic quality judgment.

## Output signals

| Repeat | ID | Category | Status signal | Branch headings | Acceptance set | Citations |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | `57202972-sample` | ambiguous | ANSWERED | 0 | no | 16 |
| 1 | `5ab9e3d25542994dbf019889` | known-failing | ANSWERED | 0 | no | 5 |
| 1 | `5adce8c15542992c1e3a249d` | clear | ANSWERED | 0 | no | 8 |
| 1 | `sample-hamlet` | clear | ANSWERED | 0 | no | 4 |

## Answers for manual review

### Repeat 1 — ambiguous — `57202972-sample`

Question: What could be considered a trait of Rome's military campaigns?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "ANSWERED"}`

Reference status: ANSWERED
Uncertainty reason: none

1. Canonical answer: Rome's military campaigns can be characterized as highly disciplined and organized, professional/standing heavy-infantry legions, expansionist and offensive, exhibiting obstinate persistence after catastrophic losses, with a preference for direct engagement in open battle and, where necessary, overcoming fortified positions via superior military engineering, logistics, road/bridge building and systematic fortified camp (castra) construction. Other consistently attested traits include versatility/strategic innovation, strict organization and rigorous training, and serving the triple purpose of securing borders, exploiting peripheral areas via tribute, and maintaining internal order.

2. Acceptable variants:
- Well-trained / highly trained / rigorous training and strict / brutal discipline including summary execution / unit cohesion
- Highly organized / structured / systematic / strict organisation
- Professional / standing army / long-term service (16-25 years) of heavy-infantry legions
- Expansionist / territorial expansionist campaign normally begun as counter-offensive that grew Rome from a small town to ~55 million in early empire
- Offensive / aggressive, showing little reluctance to engage in direct military engagement against numerous adversaries
- Preference for direct engagement in open battle rather than awaiting battle in camp
- Military engineering prowess / siege warfare / roads, bridges, camps and field fortifications / siegecraft
- Systematic fortified camp (castra) construction with palisade and ditch at end of each day's march, persisting over long period with no other ancient army doing so
- Obstinate persistence overcoming appalling losses / dourly raising legion after legion
- Triple purpose: securing borders, exploiting peripheral areas through imposing tribute, and maintaining internal order
- Versatile / adaptable / strategic innovation / logistical planning / tactical handbook with limited variations
Any single accurate trait or combination thereof should be accepted; the question is open-ended ("could be considered").

3. Necessary caveats:
- The question asks for a trait that "could be considered" — it does not require an exhaustive list; any one verifiable trait from the consensus suffices for a correct answer.
- Chronology matters only to avoid outright error: early Republic used annually levied citizen militia, later transformed (Marian/Augustan reforms) into a professional standing army; describing the mature system as professional is correct for late Republic/Empire.
- Do not require naming of specific formations (e.g., testudo, manipular system, quincunx) unless given; general descriptions of discipline/organization/engineering are sufficient.

4. Evidence:
- Rome's military served the triple purpose of securing its borders, exploiting peripheral areas through measures such as imposing tribute on conquered peoples, and maintaining internal order
- The majority of campaigns were characterised by one of two types; the first is the territorial expansionist campaign, normally begun as a counter-offensive, which allowed Rome to grow from a small town to a population of 55 million in the early empire when expansion was halted
- History of Rome's campaigning is a history of obstinate persistence overcoming appalling losses; Romans produced incompetents leading to catastrophic defeats yet even greatest enemies won battle but lost war; also summarized as history of obstinate persistence overcoming appalling losses
- Roman army was extremely disciplined, kept standing between conflicts, made mostly of heavy infantry
- Soldiers were well-trained and disciplined; combination of intense training, improved weapons, knowledge of enemies and strict organisation were reasons for success
- Earlier in its history, in Republic and early Empire Rome showed little reluctance to become engaged in direct military engagement, prosecuting offensive operations against numerous adversaries
- Preponderance of campaigns exhibit preference for direct engagement in open battle and, where necessary, overcoming fortified positions via military engineering; adept at building fortified camps but reluctant to sit in camp awaiting battle and history of seeking open battle
- At end of day's march Romans typically established strong field camp called castra, complete with palisade and deep ditch; no other ancient army persisted over such long period in systematic camp construction even if rested for single day
- Roman legions excelled in military engineering and almost always included skilled military engineers to build roads and bridges, to construct camps and field fortifications, and to create siegecraft; larger machines used only for sieges, smaller ballistae/catapults on battlefield
- By time of Jesus service had become professional, with soldiers serving 16 to 25 years before discharge with pension and grants; originally citizen-soldiers serving as needed before returning to private life

5. Grading guidance:
- Status is ANSWERED. Accept any answer that states at least one trait listed above with accurate description or synonym (e.g., disciplined, organized, professional, expansionist/imperialist, aggressive/offensive, engineering/siege, fortified camps/roads, seeks pitched battle, persistent/resilient, secures borders/exploits periphery/maintains order).
- Do not penalize for providing only one trait, nor for adding extra correct traits. Reject only if answer asserts opposite (e.g., "avoided open battle," "primarily naval," "always defensive, never expansionist," "disorganized/untrained throughout").
- For ruthlessness/brutality descriptions, accept if phrased as "ruthless/disciplined through brutal punishments" where supported, but do not require it; do not fail an answer solely for omitting it.
- Minor imprecisions in dates, numbers, or formation names should not cause failure if core trait is correct.

6. Confidence: high, with a brief reason. Traits converge across primary synthesis (Campaign history), dedicated strategy article, and secondary overviews (BYU Studies, Twinkl summary) on discipline, organization, expansionist/offensive orientation, preference for open battle, engineering and systematic camps, and persistence.

### Repeat 1 — known-failing — `5ab9e3d25542994dbf019889`

Question: What type of vegetation does Lythrum and Erythrina have in common?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "ANSWERED"}`

Reference status: ANSWERED
Uncertainty reason: none

1. Canonical answer: Both genera have a **herbaceous / herb (forb) vegetation type** in common — specifically **herbaceous perennials / perennial herbs**. *Lythrum* is exclusively herbaceous and *Erythrina*, though predominantly woody, includes herbaceous perennial species (e.g., *Erythrina herbacea*, *Erythrina leptorhiza*), so the shared type is herbs / herbaceous plants.

2. Acceptable variants: herb; herbaceous plant; forb; herbaceous perennial; perennial herb; herbaceous annual or perennial; shrublike herbaceous perennial; non-woody herbaceous vegetation; perennial herbaceous shrub/herb. Any answer expressing that both genera include herbaceous (non-woody) members is equivalent. "Perennial" alone is less precise but reflects the same shared perennial-herbaceous habit and should be considered partially correct; full credit requires the herbaceous/herb/forb concept.

3. Necessary caveats: *Erythrina* as a whole is not exclusively herbaceous — most species are woody trees or shrubs with only a few herbaceous ones. The commonality is at the genus level that both genera contain herbaceous life-forms; it does not mean all *Erythrina* species are herbs. *Lythrum* is consistently described as herbaceous across its species, not shrubby or woody.

4. Evidence:
- *Lythrum* is a genus of herbaceous annuals or perennials
- Annual or perennial herbs. Leaves alternate or whorled...
- *Lythrum* can be annuals or herbaceous perennials, with simple leaves in opposite pairs and small star-shaped flowers in leafy racemes
- **Herbs perennial,** slender, 5–12 dm, green, glabrous [*Lythrum virgatum*]
- **Species Description** - Stout, erect perennial herb 5–15 dm [*Lythrum salicaria*]
- The Erythrina genus comprises 108 species of orange or red-flowered trees, shrubs, and herbaceous plants
- The Erythrina genus of the family Fabaceae includes trees, shrubs, and herbaceous plants
- Most species of the genus Erythrina are trees and shrubs, with few herbaceous ones
- Flora of New Zealand | Taxon Profile | Erythrina Shrubs, trees, or very rarely herbs
- Unlike most *Erythrina* species, which usually are woody trees or shrubs, *Erythrina leptorhiza* is a perennial which is herbaceous
- **Intro:** Shrublike herbaceous perennial found in maritime forests, dry sandy woodlands and sandhills [*Erythrina herbacea*]
- Erythrina herbacea, Linn. Perennial: sts. several and herbaceous, from a very thick root, 2—4 ft. high
- Scientific name: Erythrina herbacea | ... | Plant type: perennial; herbaceous
- **FORM:** multiple-stemmed herbaceous shrub [*Erythrina herbacea*]

5. Grading guidance: Under ANSWERED status, accept any answer that states the shared type is herbaceous / herb / forb / herbaceous perennial / perennial herb. Accept synonyms: forb = herb, herbaceous = non-woody herb. Require the herbaceous concept for full credit; "perennial" alone without herb/herbaceous is incomplete — give partial credit at grader discretion but do not treat as fully equivalent to the canonical answer. Reject answers claiming exclusive woody types only (e.g., both are only trees, only shrubs, only woody) or habitat types not supported as common (e.g., both are only wetland/only tropical trees) unless the response also includes the herbaceous component.

6. Confidence: high, with a brief reason. Multiple authoritative floras and taxonomic databases consistently describe *Lythrum* as herbaceous perennial herbs and document that *Erythrina* includes trees, shrubs, and herbaceous plants with explicitly herbaceous species (*E. herbacea*, *E. leptorhiza*), directly supporting herb/herbaceous perennial as the common vegetation type, with no credible source conflict.

### Repeat 1 — clear — `5adce8c15542992c1e3a249d`

Question: Are Thyme and Cortaderia both aromatic plants?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "ANSWERED"}`

Reference status: ANSWERED
Uncertainty reason: none

1. Canonical answer:
No — Thyme and *Cortaderia* are not both aromatic plants. Thyme (*Thymus*, especially *Thymus vulgaris*) is an aromatic plant/herb, while *Cortaderia* (pampas grass) is not; it is an ornamental grass in the family Poaceae that is characterized as non-fragrant / not known for fragrance.

2. Acceptable variants:
- No / False — only Thyme is aromatic; *Cortaderia* is not aromatic.
- Thyme = yes, aromatic; *Cortaderia* = no, non-aromatic / non-fragrant / not known for fragrance / ornamental grass without notable aroma.
- Rephrasings that convey the same distinction: “Thyme is a fragrant aromatic herb, *Cortaderia* is a non-fragrant ornamental grass” are acceptable.
- Correct identification of Thyme as *Thymus vulgaris* / genus *Thymus* and *Cortaderia* as *Cortaderia selloana* / pampas grass with the same yes/no judgment is acceptable.

3. Necessary caveats:
- “Aromatic plant” is used here in the standard horticultural/botanical sense of a plant with pronounced fragrance/aroma due to essential oils, not in the chemistry sense of “aromatic ring.”
- Thyme unambiguously meets this definition across sources; *Cortaderia* is consistently described without fragrance. One low-quality aggregator page among search results contains a stray line “Medium green leaves are aromatic when crushed” with unclear attribution among many concatenated plant entries; it is outweighed by multiple consistent horticultural and commercial sources stating *Cortaderia* is non-fragrant / not known for fragrance. An answer claiming “both are aromatic” based solely on that isolated line should be considered incorrect.
- A study of essential oil content found *Cortaderia selloana* had no essential oils in leaves and did not have essential oil in leaves, supporting its non-aromatic classification.

4. Evidence:
- Thyme is defined as aromatic herbs or shrubby plants of the mint family.
- The genus *Thymus* comprises a number of fragrant aromatic undershrubs, and is a genus of about 350 species grown primarily for their aromatic foliage.
- *Thymus vulgaris* is described as a woody, evergreen subshrub with small and highly aromatic grey-green leaves, and the leaves are highly aromatic and frequently used as seasoning; leaves are evergreen in most species and usually aromatic.
- Thyme (*Thymus vulgaris*) is known for the aroma and flavour of its dried leaves.
- *Cortaderia* is a genus of plants in the Poaceae or grass family.
- In the genus *Cortaderia* are a couple of dozen species of grasses from New Zealand, New Guinea, and South America.
- *Cortaderia* species are perennial grasses, characterised by robust culms and feathery inflorescent plumes and described as robust evergreen grasses forming large, compact tufts of narrowly linear, rough-edged leaves.
- *Cortaderia selloana* is listed as Type: Ornamental grass, Family: Poaceae.
- Direct statement on fragrance: Are members of the *Cortaderia* plant genus fragrant? *Cortaderia* species are not known for their fragrance, but their visual impact in a landscape is significant.
- Commercial horticultural product data lists Fragrance | Non-Fragrant for *Cortaderia selloana*.
- Essential oil study: *Cortaderia selloana*, and other species had no essential oils and specifically *Cortaderia selloana* did not have essential oil in their leaves.

5. Grading guidance:
- With status ANSWERED, the correct binary judgment is “No, not both.”
- Full credit: Answer states Thyme is aromatic and *Cortaderia* is not aromatic / non-fragrant / not known for fragrance / ornamental grass without aroma, or otherwise clearly indicates the answer to “both aromatic?” is no with correct attribution.
- Partial credit (if rubric allows): Answer correctly identifies Thyme as aromatic but is vague or incomplete on *Cortaderia* yet does not affirm *Cortaderia* is aromatic; still demonstrates core understanding.
- No credit: Answer states “Yes, both are aromatic,” or claims *Cortaderia* is an aromatic herb / fragrant aromatic plant comparable to Thyme, or reverses the attribution.
- Do not penalize for using synonyms: pampas grass = *Cortaderia selloana*; thyme = *Thymus* / *Thymus vulgaris* / common thyme. Do not require Latin nomenclature if intent is clear.

6. Confidence: high, with brief reason: Multiple reputable encyclopedic, botanical, and horticultural sources consistently classify *Thymus*/thyme as a fragrant aromatic herb with highly aromatic leaves, while consistently classifying *Cortaderia* as a Poaceae ornamental grass that is non-fragrant/not known for fragrance, with essential-oil analysis showing no essential oils in *Cortaderia selloana* leaves; no credible conflicting source establishes *Cortaderia* as an aromatic plant in the relevant sense.

### Repeat 1 — clear — `sample-hamlet`

Question: Who wrote Hamlet?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "ANSWERED"}`

Reference status: ANSWERED
Uncertainty reason: none

1. Canonical answer: **William Shakespeare** — the English playwright and poet (1564-1616) — wrote *The Tragedy of Hamlet, Prince of Denmark*, commonly shortened to *Hamlet*.

2. Acceptable variants:
- "William Shakespeare" is canonical.
- Equivalent acceptable answers: "Shakespeare", "W. Shakespeare", "William Shakspere / Shakspeare" (contemporary spelling variants), or "the Bard" / "the Bard of Avon" when clearly referring to William Shakespeare of Stratford-upon-Avon.
- The play's title variant "The Tragedy of Hamlet, Prince of Denmark" does not change the required author; adding correct contextual detail (e.g., date c. 1599-1601) is acceptable if author is William Shakespeare.

3. Necessary caveats: None that affect grading of the core authorship attribution. Mainstream scholarship universally attributes *Hamlet* to William Shakespeare, written c. 1599-1601 / 1599-1602 depending on source precision. Sources for the story (legend of Amleth, Saxo Grammaticus, Belleforest, Ur-Hamlet) are influences, not authors of Shakespeare's play. Fringe authorship theories that claim Bacon, Marlowe, Edward de Vere, or others wrote Shakespeare's works are rejected by authoritative and academic sources and should not be credited.

4. Evidence:
- Wikipedia infobox lists Hamlet as: Written by | William Shakespeare
- Wikipedia lead: "***The Tragedy of Hamlet, Prince of Denmark***, often shortened to ***Hamlet*** (), is a tragedy written by William Shakespeare sometime between 1599 and 1601."
- Consolidated search summary repeats: "***The Tragedy of Hamlet, Prince of Denmark***, often shortened to ***Hamlet*** (), is a tragedy written by William Shakespeare sometime between 1599 and 1601."
- Google Books publisher description: "The Tragedy of Hamlet, Prince of Denmark, often shortened to Hamlet is a tragedy written by William Shakespeare at an uncertain date between 1599 and 1602."
- Amazon edition description: "The Tragedy of Hamlet, Prince of Denmark is a tragedy written by William Shakespeare between 1599 and 1602."
- Folger Shakespeare Library authoritative edition lists: "Hamlet ... by William Shakespeare (Author)" and describes it as "The authoritative edition of *Hamlet* from The Folger Shakespeare Library"
- Library catalogue record: "## by William Shakespeare" and "Hamlet is Shakespeare's famous tragedy of a prince sworn to avenge his father's death."

5. Grading guidance: Grade as CORRECT any answer that unambiguously identifies William Shakespeare (or shorthand "Shakespeare") as the author/playwright of *Hamlet*. Do not penalize for omission of first name, for archaic spelling variants, for use of "Bard"/"Bard of Avon" if clearly linked to Shakespeare, or for additional true context such as dates or genre. Grade as INCORRECT any answer that names only a different person as the author (e.g., Thomas Kyd, Christopher Marlowe, Francis Bacon, Edward de Vere) or that claims anonymous/unknown authorship, or that lists Shakespeare as co-equal author with another writer. Mentioning sources/influences (Saxo Grammaticus, Belleforest, Ur-Hamlet, The Spanish Tragedy) alongside Shakespeare is acceptable only if Shakespeare is clearly given as the author of the Shakespearean play *Hamlet*.

6. Confidence: high, with a brief reason. Multiple independent reputable sources — encyclopedic (Wikipedia), publisher/Folger, library catalogues, and commercial bibliographic records — converge without credible conflict on William Shakespeare as author; authoritative attribution is unanimous and central to literary history.

## Pilot boundary

This is a 4-question, 1-repeat pilot for prompt behavior. No full-dataset run was started.

Private cache: `artifacts/muse_uncertainty_sample_2026-08-22.json`
