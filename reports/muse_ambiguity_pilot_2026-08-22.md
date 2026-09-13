# Muse ambiguity-handling pilot

- Model: `muse-spark-1.2-contributor`
- Prompt variant: `reference_generic_branches`
- Endpoint: Meta native Responses API with `web_search`
- Policy tested: two calls per item; no application Search or output-length cap
- Questions: 9 (3 ambiguous, 3 broad-but-single, 3 clear)
- Repeats per question: 3
- Expected records: 27
- Completed records: 27
- Hard failures: 0
- Stage-2 finals: 27
- Cache-only rerun items skipped without API calls: 27
- Search queries recorded: 282
- URL citations recorded: 240

The signals below are mechanical indicators for review; they are not a semantic quality judgment.

## Output signals

| Repeat | ID | Category | Status signal | Branch headings | Acceptance set | Citations |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | `56e2419acd28a01900c67c2a` | broad-but-single | SINGLE | 0 | yes | 9 |
| 1 | `56f71740711bf01900a44932` | broad-but-single | SINGLE | 0 | yes | 17 |
| 1 | `57260b4389a1e219009ac188` | ambiguous | SINGLE | 0 | yes | 0 |
| 1 | `57302972b2c2fd140056899e` | ambiguous | SINGLE | 0 | yes | 12 |
| 1 | `5ab9e3d25542994dbf019889` | ambiguous | SINGLE | 0 | yes | 6 |
| 1 | `5ac4e4595542996feb3fe968` | clear | SINGLE | 0 | yes | 11 |
| 1 | `5adce8c15542992c1e3a249d` | clear | SINGLE | 0 | yes | 8 |
| 1 | `5ae336ef5542991a06ce995d` | broad-but-single | SINGLE | 0 | yes | 6 |
| 1 | `5ae6168f5542996de7b71b47` | clear | SINGLE | 0 | yes | 14 |
| 2 | `56e2419acd28a01900c67c2a` | broad-but-single | SINGLE | 0 | yes | 14 |
| 2 | `56f71740711bf01900a44932` | broad-but-single | SINGLE | 0 | yes | 16 |
| 2 | `57260b4389a1e219009ac188` | ambiguous | SINGLE | 0 | yes | 0 |
| 2 | `57302972b2c2fd140056899e` | ambiguous | SINGLE | 0 | yes | 5 |
| 2 | `5ab9e3d25542994dbf019889` | ambiguous | SINGLE | 0 | yes | 4 |
| 2 | `5ac4e4595542996feb3fe968` | clear | SINGLE | 0 | yes | 13 |
| 2 | `5adce8c15542992c1e3a249d` | clear | SINGLE | 0 | yes | 4 |
| 2 | `5ae336ef5542991a06ce995d` | broad-but-single | SINGLE | 0 | no | 0 |
| 2 | `5ae6168f5542996de7b71b47` | clear | SINGLE | 0 | no | 6 |
| 3 | `56e2419acd28a01900c67c2a` | broad-but-single | SINGLE | 0 | yes | 7 |
| 3 | `56f71740711bf01900a44932` | broad-but-single | SINGLE | 0 | yes | 15 |
| 3 | `57260b4389a1e219009ac188` | ambiguous | SINGLE | 0 | yes | 18 |
| 3 | `57302972b2c2fd140056899e` | ambiguous | SINGLE | 0 | yes | 14 |
| 3 | `5ab9e3d25542994dbf019889` | ambiguous | SINGLE | 0 | yes | 12 |
| 3 | `5ac4e4595542996feb3fe968` | clear | SINGLE | 0 | no | 12 |
| 3 | `5adce8c15542992c1e3a249d` | clear | SINGLE | 0 | yes | 4 |
| 3 | `5ae336ef5542991a06ce995d` | broad-but-single | SINGLE | 0 | no | 0 |
| 3 | `5ae6168f5542996de7b71b47` | clear | SINGLE | 0 | yes | 13 |

## Answers for manual review

### Repeat 1 — broad-but-single — `56e2419acd28a01900c67c2a`

Question: What structures can be viewed as both works of art and symbols of certain cultures?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR - Question has one clear reading

Question wording inspection: "What structures can be viewed as both works of art and symbols of certain cultures?" The term "structures" is constrained by both predicates: must be capable of being a work of art AND a symbol of a culture. That limits the domain to built/physical structures (buildings, monuments, landmarks). No alternative reading (e.g., social structures, chemical structures, abstract structures) satisfies both constraints simultaneously while honoring the explicit "works of art" and "symbols of certain cultures" language. Scope/granularity variations (architecture vs. monument vs. landmark vs. specific example) are nested levels of detail, not materially different interpretations that change grading outcome. Therefore collapse to single canonical answer.

### Canonical Answer
**Architectural works / buildings / monuments / landmarks / iconic structures - in their material form as buildings and other physical structures - can be viewed as both works of art and symbols of certain cultures.**

Definitional core:
* "Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements."
* Architecture "is considered a visual art" and is the study and practice of designing structures, especially habitable ones, including buildings or other structures
* "Monuments are timeless and universal. They hold meaning as historical symbols that help societies remember their past and create a cultural identity. They can be world-renowned structures such as the Acropolis and the Sydney Opera House"
* "Public monuments are structures created to commemorate significant events, individuals, or cultural values within society. These works of art serve not only as symbols of remembrance but also as expressions of civic pride and identity"

Exemplars that satisfy both conditions are the globally recognized iconic structures routinely grouped together as famous world buildings / monuments:
Eiffel Tower (France), Taj Mahal (India), Great Pyramids of Giza / Pyramids (Egypt), Statue of Liberty (USA), Great Wall of China, Colosseum (Rome/Italy), Sydney Opera House (Australia), Acropolis (Greece), Big Ben / Elizabeth Tower (UK), St. Basil's Cathedral, Christ the Redeemer, Machu Picchu, etc., which are "renowned for its architectural significance and cultural impact" and listed as "Famous buildings tourist places ... Paris Eiffel tower, Statue of Liberty ... Giza Pyramids ... Rome Coliseum, Taj Mahal"

### Acceptable Variants / Acceptance Set
Grader should accept any answer that identifies the **category** OR provides **valid exemplars** with correct justification. All refer to same class:

* Category-level answers: architecture, architectural works, buildings, structures, monuments, public monuments, memorials, cultural monuments, heritage structures, landmarks, iconic structures / iconic buildings, world landmarks, historic structures
* Synonymous phrasing: "built works of art," "artistic structures," "works of architecture," "cultural landmarks in built form"
* Example-level answers: Any one or more of: Acropolis, Sydney Opera House, Eiffel Tower, Taj Mahal, Pyramids of Giza, Great Wall of China, Statue of Liberty, Colosseum, Big Ben, St. Basil's Cathedral, Machu Picchu, Petra, Angkor Wat, Sagrada Familia, Leaning Tower of Pisa, etc., provided the answer states or implies they are both artistic/architectural and culturally symbolic. Listing a subset is sufficient; exhaustive list not required.
* Acceptable if answer notes that historical civilizations are identified with their surviving architectural achievements

Not acceptable: natural features (mountains, rivers), purely two-dimensional artworks (paintings) without structural embodiment, utilitarian structures with no recognized artistic/symbolic value, or answers that mention only one of the two required attributes.

### Necessary Caveats
* Not every building automatically qualifies; perception as both art and cultural symbol is socially/historically constructed and depends on recognition, aesthetic evaluation, and cultural adoption over time.
* Cultural symbolism is context-dependent; a structure may symbolize one culture/nation/period strongly and be less salient for another.
* Terminology overlaps (architecture / monument / landmark / iconic building) - treat as same referent class for grading; granularity differences do not create distinct answers.
* Example lists are open-ended; no closed canonical set exists. Sources illustrate but do not exhaust the class.
* Evidence conflicts do not create ambiguity: different sources emphasize different terms but converge on same meaning (built works = art + cultural symbol).

### Evidence
* **Architecture = visual art + structures** - "Architecture is the study and practice of designing structures, especially habitable ones. It utilizes civil engineering techniques, but is considered a visual art." and "is both the process and the product of sketching, conceiving, planning, designing, and constructing buildings or other structures."
* **Architectural works as cultural symbols and works of art** - "Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements." Repeated in other source: same phrasing
* **Monuments as historical symbols creating cultural identity - with examples** - "Monuments are timeless and universal. They hold meaning as historical symbols that help societies remember their past and create a cultural identity. They can be world-renowned structures such as the Acropolis and the Sydney Opera House"
* **Monuments as works of art and symbols** - "Public monuments are structures created to commemorate significant events, individuals, or cultural values within society. These works of art serve not only as symbols of remembrance but also as expressions of civic pride and identity"
* **Famous buildings recognized for architectural significance and cultural impact** - "Ten famous buildings around the world include the Eiffel Tower in Paris, the Great Wall of China, the Taj Mahal in India, the Colosseum in Rome... Each of these structures is renowned for its architectural significance and cultural impact."
* **Grouping of world landmarks as cultural symbols** - "Famous buildings tourist places icon set. ... Paris Eiffel tower, Statue of Liberty, Big Ban clock ... Giza Pyramids, China Pagoda, Rome Coliseum, Taj Mahal"

### Confidence
HIGH for the canonical category (architectural works / monuments / landmarks viewed as both works of art and cultural symbols) - directly supported by multiple independent definitions. MEDIUM-HIGH for specific exemplar lists, because sources provide many interchangeable examples and no source claims exclusivity.

### Grading Guidance
* Accept any response that (a) names the class (architecture / buildings / monuments / landmarks / iconic structures) and affirms both attributes (art + cultural symbol), OR (b) lists one or more valid exemplars (e.g., Eiffel Tower, Taj Mahal, Pyramids, Statue of Liberty, Colosseum, Great Wall, Sydney Opera House, Acropolis) with indication that they are considered artistic and symbolic.
* Do not require exhaustive list; one valid category term or one valid example with correct dual explanation earns full credit.
* Do not penalize granularity: "architecture" = "monuments" = "Eiffel Tower" are all correct levels of the same answer if justification links art + culture.
* Reject answers that cite only non-structural art, only natural landmarks, or fail to assert both art and cultural symbolism.

### Repeat 1 — broad-but-single — `56f71740711bf01900a44932`

Question: Bilateral treaties are concluded between how many states or entities?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR

### Canonical Answer
**Two** — Bilateral treaties are concluded between **two** states or entities / two parties / two sides / two subjects of public international law.

Supporting formulations from sources:
* Treaties may be bilateral (between two countries)
* Bilateral treaties are concluded between two states or entities
* A bilateral treaty is strictly between two subjects of public international law, generally either sovereign states or international organisations established by treaty. It is an agreement made by negotiations between two parties
* These two parties can be two nations, or two international organizations, or one nation and one international organization
* Treaties made between two independent governments are called bilateral treaties
* Formal US Senate definition: Bilateral treaty: An international agreement concluded between two states

### Interpretation Analysis
Fresh grounding pass inspected exact wording, entities, relationship, requested attribute, scope, and answer type:

* Wording: "Bilateral treaties are concluded between how many states or entities?" requests the defining number of sides/parties for the treaty type "bilateral."
* Relationship: "concluded between" = number of contracting sides.
* Answer type: cardinal number / count.

Considered whether a second materially different reading exists (e.g., "how many formal signatory parties can appear on the instrument"). This does not satisfy the criteria for a branching interpretation:

* It does not honor the explicit constraint "bilateral" as defined in sources (strictly two subjects/sides).
* It is supported only as a caveat/edge-case description, not as an alternative definition: "It is possible for a bilateral treaty to have more than two parties ... The parties are divided into two groups, the Swiss ("on the one part") and the EU and its member states ("on the other part")" and "It is possible for a bilateral treaty to involve more than two parties; for example, each of the bilateral treaties between Switzerland and the European Union (EU) has seventeen parties. The parties are divided into two groups".
* The distinction is scope/granularity (counting sides vs. counting formal signatories within sides), not interpretation ambiguity. Per guidance, granularity and evidence-conflict caveats must be collapsed into one canonical answer with caveats, not retained as separate interpretation-answer pairs. No genuine interpretation ambiguity remains that would change grading outcome.

Ranked by fit to question wording, the single clear reading is: two.

### Acceptable Variants / Acceptance Set
Grader may accept any phrasing that conveys the canonical number **two** for the number of sides/parties:

* "Two" / "2" / "2 states" / "2 entities"
* "Two states or entities"
* "Two parties" / "Two sides" / "Two groups"
* "Two subjects of public international law"
* "Two sovereign states or international organisations" / "Two nations" / "Two governments" / "One state and one international organization"

Accept with qualification: "Two sides/groups, though formally more than two signatories may appear (e.g., 17 divided into two groups) but treaty remains bilateral" — acceptable because it preserves core "two sides."

Do not accept as standalone correct: "One," "Three," "Many," "More than two" without two-side qualification, "multilateral," or "unilateral."

### Necessary Caveats
* Formal-party vs. side caveat: While definition is two sides, it is possible for a bilateral treaty to have more than two formal parties but remain bilateral because parties are divided into two groups — e.g., each Switzerland-EU bilateral treaty has seventeen parties: Swiss ("on the one part") and EU and its member states ("on the other part"); similarly described as "It is possible for a bilateral treaty to involve more than two parties ... has seventeen parties ... divided into two groups". The treaty establishes rights and obligations between the two sides, not amongst members within a side. This does not change the canonical answer; it is a classification nuance.
* No conflicting evidence on core definition found; sources converge on two.

### Evidence (what each citation supports)
*  — Supports that treaties may be bilateral (between two countries) vs multilateral (more than two).
*  — Supports canonical phrasing "Bilateral treaties are concluded between two states or entities" and introduces the >2 parties caveat.
*  — Supports strict definition as between two subjects of public international law, generally sovereign states or international organisations, negotiated between two parties.
*  — Supports elaboration that two parties can be two nations, two international organizations, or one nation and one organization.
*  — Supports "Treaties made between two independent governments are called bilateral treaties."
*  — Supports official US Senate definition "An international agreement concluded between two states."
*  and  — Support caveat that bilateral treaty can technically have >2 formal parties (e.g., 17) but divided into two groups, still bilateral.

### Confidence
High. Multiple independent, reputable sources (Wikipedia Treaty and Bilateral treaty articles grounded in treaty law literature, US Senate Committee Print via govinfo.gov, and consistent instructional sources) converge unanimously on the definitional number two. Caveat is well-documented and does not undermine definition.

### Grading Guidance
* **Accept:** Any answer stating two / two states / two entities / two parties / two sides / two subjects of international law, including variants listing examples (two sovereign states, two international organizations, or one of each).
* **Accept with explanation:** Answer stating two sides/groups and noting the Switzerland-EU 17-party example as still bilateral due to two-group structure.
* **Reject:** Answers stating only "more than two," "several," "many," "17" without qualification that treaty remains bilateral via two sides; answers stating one or three as defining number; or conflating bilateral with multilateral.

### Repeat 1 — ambiguous — `57260b4389a1e219009ac188`

Question: What is an example of something that serves a function other than protection?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: DIRECT — Single clear reading (not materially ambiguous)

Question wording inspection: "What is an example of something that serves a function other than protection?"
- Entities: "something" — unconstrained, any object, structure, organ, organism, or artefact
- Relationship: "serves a function"
- Requested attribute: "other than protection" — function is not protective/barrier/defense
- Scope: open-domain, no explicit biological, anatomical, or technical restriction
- Answer type: example (any single valid instance satisfies)

No materially different reading honors all explicit constraints and changes grading outcome. Variations by domain (biology vs artefact vs everyday object) or by specific function (pumping vs photosynthesis vs sensation) are scope/granularity variants, not interpretation ambiguities. No branch has distinct supporting evidence that would require a different canonical answer.

### Canonical Answer
An example is **the heart — its function is to pump blood / circulate blood to deliver oxygen and nutrients throughout the body**, not protection.

> The function of the heart is to pump blood through blood vessels of the cardiovascular system
> The heart is made of muscle and pumps blood through the body
> Through the action of UV, the skin synthesizes vitamin D ... The skin also contains sensory receptors — used as contrast to show pumping is distinct from protection

This satisfies "other than protection" because pumping/circulation is a mechanical/transport function, not a defensive barrier function.

### Acceptable Variants / Acceptance Set
Grader should accept **any specific, factually accurate entity with a correctly described non-protective function**. The canonical answer is one instance; the following are all flatten-accepted variants:

* **Heart / cardiovascular system** — pumping blood / circulation / delivering oxygen and nutrients and removing wastes
* **Chloroplast** — formation and storage of carbohydrates from photosynthesis; principal site for photosynthesis — harnesses light energy to synthesize carbohydrates; The main function of chloroplasts in a plant cell is to perform photosynthesis
* **Skin beyond protection** — sensing stimuli, thermoregulation, and vitamin D synthesis; The main functions of the skin are protection, temperature regulation, sensation, excretion, and vitamin D synthesis; It plays critical roles in protection, sensation, thermoregulation, immune defense, and vitamin D synthesis; e.g., sensation, thermoregulation, vitamin D synthesis
* **Other biological non-protective functions**: lung/gill gas exchange, kidney filtration/excretion, root water/nutrient absorption or starch storage, neuron signal transmission, stomach/intestine digestion/absorption, mitochondria cellular respiration/ATP production, leaf photosynthesis, feather flight/display
* **Artefacts/everyday objects**: chair (seating), pen (writing), telephone (communication), vehicle (transport), etc., provided function described is not protection

Accept any synonym or paraphrase that accurately conveys the same function (e.g., "pumps blood around body," "circulates blood," "photosynthesis," "thermoregulation via sweating," "sensory perception").

### Necessary Caveats
* This is scope granularity, not interpretation ambiguity. Do not penalize for domain choice.
* Dual-function entities that have both protective and non-protective functions are valid if the answer correctly identifies the non-protective component (e.g., skin also senses/thermoregulates; feathers also enable flight/thermoregulation). An answer that claims the entity *only* serves protection fails the constraint.
* Vague answers that name an entity without specifying a function, or that describe only a protective function (e.g., "helmet protects head," "shell protects snail" without a non-protective function), do not satisfy.
* No evidence conflict creates ambiguity; multiple sources converge on same non-protective functions.

### Evidence
* **Heart pumping function — supports canonical answer:**
  * "The function of the heart is to pump blood through blood vessels of the cardiovascular system. The continuous flow of blood through the system is necessary to provide all the cells of the body with oxygen and nutrients"
  * "The heart is made of muscle and pumps blood through the body"
* **Chloroplast non-protective function — supports acceptance set:**
  * "Chloroplasts are large organelles and their function is the formation and storage of carbohydrates from photosynthesis"
  * "serves as the principal site for photosynthesis-a process that harnesses light energy ... to synthesize carbohydrates"
  * "The main function of chloroplasts in a plant cell is to perform photosynthesis"
* **Skin non-protective functions distinct from protection — supports acceptance set:**
  * "The skin plays important roles in protection, sensing stimuli, thermoregulation, and vitamin D synthesis"
  * "It plays critical roles in protection, sensation, thermoregulation, immune defense, and vitamin D synthesis"
  * "thermregulation, - sensation, - vitamin D synthesis"
  * "The main functions of the skin are protection, temperature regulation, sensation, excretion, and vitamin D synthesis"

Each URL supports the claim cited inline: 69062762154058141 supports heart pumping/circulation; 6219721614595916677 supports chloroplast photosynthesis; 8311876731945904156 supports skin sensing/thermoregulation/vitamin D synthesis as functions other than protection.

### Confidence
High — heart pumping and chloroplast photosynthesis are primary, textbook-defined non-protective functions with consistent, reputable source convergence. High that the question is unambiguous and acceptance set should be broad.

### Grading Guidance
* **Accept** any answer that (1) names a concrete entity and (2) attributes a verifiable non-protective function (mechanical, metabolic, transport, sensory, thermoregulatory, reproductive, communicative, etc.) supported by common knowledge or evidence.
* **Accept** dual-function examples if non-protective function is correctly specified.
* **Reject** answers that describe only protection, that misattribute the function to the entity, that are tautological ("something functions"), or that are factually false.
* Do not require exact phrasing; grade on conceptual match to canonical answer or acceptance set.

### Repeat 1 — ambiguous — `57302972b2c2fd140056899e`

Question: What could be considered a trait of Rome's military campaigns?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR - Question has one clear reading

Question wording inspection: "What could be considered a trait of Rome's military campaigns?" asks for any characteristic attribute of campaigns conducted by Ancient Rome (city-state / Republic / Empire, 8th century BC to 5th century AD). Entity: Rome = Ancient Rome. Relationship: trait/characteristic of its military campaigns. Requested attribute: descriptive property, not a specific date/person. Scope: open-ended - any verifiably attested trait qualifies ("could be considered" is disjunctive/permissive). Answer type: exemplar trait, not exhaustive list.

No materially different interpretation honors all explicit constraints and would change grading outcome. Variations between specific traits (expansionist vs disciplined vs engineering) are scope/granularity variants of the same reading, not competing interpretations. No branching into "Rome, Italy modern" or "trait of a single campaign" is supported by wording or evidence. Therefore collapse to single canonical answer.

### Canonical Answer

Rome's military campaigns over the roughly thirteen centuries from city-state to empire can be validly described by **any** of the following interrelated traits:

**1. Typology of campaigns:** The majority were characterised by one of two types - (a) territorial expansionist campaign, normally begun as a counter-offensive, in which each victory brought subjugation of large areas of territory and allowed Rome to grow from a small town to a population of 55 million in the early empire when expansion was halted, and (b) civil war, which plagued Rome from its foundation to its eventual demise.

**2. Operational preference:** The preponderance exhibit a preference for direct engagement in open battle and, where necessary, the overcoming of fortified positions via military engineering. The army was adept at building fortified camps for protection but history shows a reluctance to sit in the camp awaiting battle and a history of seeking open battle.

**3. Systematic engineering and logistics:** At the end of a day's march Romans would typically establish a strong field camp called a castra, complete with palisade and a deep ditch, with streets, designated places and guarded gates - construction taking 2-5 hours. No other ancient army persisted over such a long period in systematic camp construction like the Romans, even if the army rested for only a single day. More broadly the Roman military was highly disciplined, well-trained, and benefited from advanced engineering skills, enabling construction of roads, forts, and siege equipment, and siege warfare was a core technique to achieve victory where pitched battles alone were insufficient.

**4. Discipline and institutional persistence:** Brutal discipline including summary execution for dereliction of duty and highest standards of unit cohesion and combat drills; and a history of obstinate persistence overcoming appalling losses - generally the fate of even Rome's greatest enemies such as Pyrrhus and Hannibal was to win the battle but lose the war, and Roman armies were not invincible and produced incompetent commanders leading to catastrophic defeats but persisted.

Any one of these correctly satisfies the question; a complete answer may list several.

### Acceptable Variants / Acceptance Set

Grader should accept **any one or more** of the following paraphrases as correct, even if phrasing differs:

- Expansionist / imperial / territorial conquest / growth from town to empire of ~55 million; normally begun as counter-offensive
- Characterised by / plagued by civil wars
- Preference for pitched/open battle / direct engagement over avoidance/stratagem
- Military engineering / siegecraft / ability to destroy, bypass or scale walls / taking fortified positions via engineering
- Systematic daily fortified marching camps (castra) with ditch and palisade
- Highly disciplined / brutal discipline / rigid drill / thoroughly regimented / decimation for cowardice/mutiny / professional standing legions
- Persistence / resilience / obstinacy after catastrophic defeats / absorbing losses and continuing
- Triple strategic purpose framing (securing borders, exploiting peripheral areas via tribute, maintaining internal order) if tied to campaigns
- Reliance on heavy infantry in organized formations (three lines, quincunx/checkered pattern) with cavalry on wings, light infantry screen, pila and scutum, supported by artillery (ballistas/onagers)

Synonyms and narrower exemplars at same detail level (e.g., "built circumvallation at Alesia," "used testudo," "built ramp at Masada" as instance of siege engineering) are acceptable.

### Necessary Caveats

- Temporal variation: Early Republic/early Empire showed little reluctance for direct offensive engagement; later Empire (Vegetius, 4th century AD) argued for limited direct engagement and preference for stratagem, so preference for open battle describes preponderance not universality.
- Land-centric: Following the First Punic War naval battles were less significant than land battles due to Mediterranean dominance; answer about land campaigns remains valid.
- Question is permissive ("could be considered") - enumerating one valid trait is sufficient; do not penalize omission of others.
- Evidence conflicts are about emphasis, not contradiction; do not treat as ambiguity requiring branches.

### Evidence

- Campaign history page - two-type typology and expansion to 55 million plus civil war, and persistence claim: "the majority of Rome's campaigns were characterised by one of two types. The first is the territorial expansionist campaign, normally begun as a counter-offensive... allowed Rome to grow from a small town to a population of 55 million... The second is the civil war, which plagued Rome" and "The history of Rome's campaigning is, if nothing else, a history of obstinate persistence overcoming appalling losses"

- Strategy page - operational preference for open battle and engineering: "the preponderance of Roman campaigns exhibit a preference for direct engagement in open battle and, where necessary, the overcoming of fortified positions via military engineering" and "The Roman army was adept at building fortified camps for protection from enemy attack, but history shows a reluctance to sit in the camp awaiting battle and a history of seeking open battle"

- Strategy page - systematic camp construction: "At the end of a day's march, the Romans would typically establish a strong field camp called a castra, complete with palisade and a deep ditch... No other ancient army persisted over such a long period in systematic camp construction like the Romans, even if the army rested for only a single day"

- Highly disciplined with engineering support: "The Roman military was highly disciplined, well-trained, and benefited from advanced engineering skills, enabling the construction of roads, forts, and siege equipment"

- Two-type restatement across mirrors confirms stability of claim: "the majority of Rome's campaigns were characterised by one of two types. The first is the territorial expansionist campaign, normally begun as a counter-offensive... allowed Rome to grow from a small town to a population of 55 million... The second is the civil war, which plagued Rome"

Each URL supports the specific factual claim noted above; Wikipedia entries synthesize primary histories (Livy, Vegetius) and secondary scholarship (Trigger, Luttwak, Goldsworthy).

### Confidence

High. Core traits (expansionist/civil-war typology, preference for open battle + engineering/siege capability, systematic camps, discipline and persistence) are consistently attested across Campaign history of the Roman military and Strategy of the Roman military and corroborated by tertiary summaries. No reliable source contradicts these as non-traits.

### Grading Guidance

- **Accept** if response contains any one trait from Acceptance Set with approximately accurate description, even if phrased at different granularity (e.g., "very disciplined army," "built a fort every night," "loved pitched battles," "excellent at sieges," "constantly expanding," "many civil wars," "never gave up even after big defeats").
- **Reject** if trait is factually wrong for Rome (e.g., "avoided open battle as rule," "avoided sieges," "undisciplined tribal levy with no engineering," "primarily naval campaigns") or describes a different polity without linking to Rome.
- **Do not require** exhaustive list; "could be considered" means one valid exemplar earns full credit.
- **Do not penalize** extra correct detail or narrower correct example that nests within a broader trait (e.g., testudo, pila, ballista as part of disciplined heavy infantry/engineering).
- **Do not treat** granularity differences as interpretation differences.

### Repeat 1 — ambiguous — `5ab9e3d25542994dbf019889`

Question: What type of vegetation does Lythrum and Erythrina have in common?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: UNAMBIGUOUS

### Canonical Answer
**Herb / herbaceous perennial (perennial herb)** - also expressible as herbaceous perennial herb, perennial herb or subshrub.

* *Lythrum* is a genus of herbaceous perennials (herbs, annual or perennial, or subshrubs).
* *Erythrina* is mostly woody trees or shrubs but includes species that are perennial herbs, so the genera overlap in the herbaceous perennial / subshrub habit.

### Acceptable Variants / Acceptance Set
Grader should accept any equivalent phrasing of the same life-form category at different levels of granularity:

* herb
* herbaceous perennial
* perennial herb / perennial herbaceous plant
* herbaceous annuals or perennials (if perennial element included)
* subshrub / perennial herb or subshrub
* shrublike herbaceous perennial
* forb / herbaceous flowering plant

Do not accept answers that limit the common type to woody vegetation only (e.g., tree only, shrub only, timber tree without herbaceous component) or that claim no common vegetation type.

### Necessary Caveats
At genus level the overlap is partial and not the predominant habit for *Erythrina*:

* *Lythrum* is consistently described as herbaceous/herbs and as perennial herbs or subshrubs, and as annuals or herbaceous perennials, and as a genus of herbaceous annuals or perennials.
* *Erythrina* is generally described as deciduous or evergreen subshrubs, shrubs or trees and mostly woody trees or shrubs. The herbaceous overlap is provided by species such as *Erythrina herbacea* described as perennial herb or shrub or small tree and *Erythrina leptorhiza* described as perennial herb and as a perennial which is herbaceous unlike most woody Erythrina species.

Therefore answers that specify "herbaceous perennial" and answers that specify "subshrub" are both technically supported by the sources, with herb/herbaceous perennial being the most commonly cited common type. Habitat/vegetation association (e.g., wetland/marsh for *Lythrum* vs dry woodland/maritime forest for *Erythrina*) is not common and should not be credited as the shared vegetation type.

Evidence of habitat difference for context - *Lythrum salicaria* habitat is wet areas, marshes, ditches, swamps, shores - is present but not the commonity requested.

### Evidence
* **Lythrum habit as herbs/herbaceous perennials or subshrubs:**
  * *Lythrum* - FNA: *Herbs,* annual or perennial,or subshrubs, terrestrial or amphibious
  * *Lythrum salicaria* - FNA: *Herbs* perennial, or subshrubs, robust
  * *Lythrum salicaria* - WorldFloraOnline: Perennial herbs or subshrubs
  * RHS genus description: Lythrum can be annuals or herbaceous perennials
  * Campus Flora: *Lythrum* is a genus of herbaceous annuals or perennials

* **Erythrina habit as predominantly woody but including perennial herbs/subshrubs:**
  * RHS genus description: Erythrina can be deciduous or evergreen subshrubs, shrubs or trees
  * WorldFloraOnline *Erythrina herbacea*: Perennial herb or shrub or small tree
  * WorldFloraOnline *Erythrina leptorhiza*: Perennial herb, leafy at anthesis
  * Wikipedia *Erythrina leptorhiza*: Unlike most *Erythrina* species, which usually are woody trees or shrubs, *Erythrina leptorhiza* is a perennial which is herbaceous and duplicated

Each URL supports the habit claim cited; no source was found that documents a shared ecological vegetation/ habitat type between the genera.

### Confidence: High
High confidence that the intended "type of vegetation" is life-form habit (herbaceous/perennial) rather than ecological formation. The herbaceous perennial overlap is directly supported by multiple primary floras for both genera. Low residual uncertainty only about taxonomic scope (genus-wide vs species-specific expression), which does not change the canonical answer.

### Grading Guidance
* **Credit:** Any answer containing herb / herbaceous / perennial herb / herbaceous perennial / perennial herb or subshrub / forb, even with qualifiers like deciduous, clump-forming, shrublike.
* **Credit subshrub:** "subshrub" alone should be credited because both genera are explicitly described as subshrubs.
* **Do not credit:** tree only, shrub only, woody tree only, aquatic/hydrophyte only, wetland only, or "no common type".
* **Insufficient alone:** "perennial" without herbaceous/herb component is incomplete; require herbaceous element.
* **Granularity:** Do not penalize for additional correct descriptors (deciduous, erect, etc.) if core herbaceous perennial concept is present.

### Repeat 1 — clear — `5ac4e4595542996feb3fe968`

Question: Which cola drink is the leader in India, Thums Up or Ricqlès?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR

### Canonical Answer

**Thums Up** is the cola leader in India. Thums Up is an Indian cola; Ricqlès is not a cola and is not an Indian cola leader.

*Question framing check:* Wording = "Which cola drink is the leader in India, Thums Up or Ricqlès?" Entities = Thums Up vs Ricqlès. Relationship = market leadership. Requested attribute = cola leadership. Scope = India, cola category. Answer type = forced choice between the two named drinks.

Inspection for ambiguity: No materially different reading honors all explicit constraints and could change grading outcome. Potential re-readings considered and rejected:
- "Leader globally" - fails explicit scope "in India"
- "Leader in overall soft drinks (including non-cola)" - fails explicit "cola drink" constraint; evidence conflict about Sprite topping overall carbonated market is a granularity issue, not a different interpretation of the question
- "Historical vs current leader" - both timeframes still point to Thums Up for cola, no evidence for Ricqlès in either

Hence one canonical answer is appropriate. Branches for evidence conflicts or scope granularity (cola vs all carbonated) are treated as caveats, not interpretations.

**Answer: Thums Up**

### Acceptable Variants / Acceptance Set

Grader should accept any response that identifies **Thums Up** as the leader:

* Thums Up, Thumbs Up, THUMS UP, ThumsUp
* "Thums Up is the largest / No.1 / leading cola (brand) in India" - any phrasing expressing leadership in Indian cola market

Not acceptable:
* Ricqlès / Ricqles / any variant claiming Ricqlès is the Indian cola leader
* "Neither" / "Tie" / "Both" / "Unknown"
* Claim that Ricqlès is a cola that competes in India

### Necessary Caveats

* **Category scope:** Thums Up leads the *cola* segment specifically. In the broader carbonated soft-drink market, Sprite (also Coca-Cola) is cited as topping overall share: "Thums Up is the largest cola brand, but Sprite, another brand owned by Coca-Cola, tops the carbonated beverage market in India" and "today Thums Up rules the cola scene in India, only topped in the carbonated beverage arena by Sprite". This does not contradict cola leadership.
* **Ricqlès category:** Ricqlès is consistently classified as a French mint/peppermint drink, not a cola: listed as "Ricqlès – mint-flavoured soft drink" under France and described as "a peppermint spirit and a French soft drink. The liquor Alcool de Menthe de Ricqlès was created in 1838 by Heyman de Ricqlès in Lyon, France". It has no cola market presence in India to evaluate.
* **Market-share variation by year/metric is not ambiguity:** Euromonitor cited 15.4% share of carbonated beverages by retail value in year ending 2016, 42% cola share in other reporting, and 85% share in India at time of 1993 sale - all support same directional answer (Thums Up = leader).

### Evidence

* **Thums Up is a cola:** Type Cola; "**Thums Up** is a brand of **cola**. It was introduced in 1977 to offset the withdrawal of The Coca-Cola Company from India"

* **Origin India:** Origin India

* **Largest cola brand in India / dominates cola market:** "Thums Up, the carbonated beverage brand owned by Coca-Cola Co.’s Indian unit, is the largest cola brand in India with Rs5,000 crore in annual sales"; "that has dominated India's cola market in the past four decades"

* **Corroborating leadership descriptions:** "Thums Up... is the largest cola brand in India with Rs5,000 crore in annual sales"; "today Thums Up rules the cola scene in India, only topped in the carbonated beverage arena by Sprite"

* **Historical dominance:** "When these were sold to Coca-Cola, Thums Up had a market share of 85 percent in India"

* **Ricqlès is French mint, not cola:** "Ricqlès – mint-flavoured soft drink"; "**Ricqlès** is a peppermint spirit and a French soft drink. The liquor **Alcool de Menthe de Ricqlès** was created in 1838 by Heyman de Ricqlès in Lyon, France"

### Confidence

High. Multiple independent sources (Wikipedia infobox + article, Livemint/Euromonitor business reporting) unanimously converge on Thums Up as India's leading cola. Ricqlès classification as French mint drink is unanimous across country-list and brand pages. No credible source classifies Ricqlès as a cola or as competing in Indian cola ranking.

### Grading Guidance

* **Correct:** Any answer selecting Thums Up (including spelling variant Thumbs Up) as the Indian cola leader. Mentioning Sprite as overall carbonated leader is acceptable as additional detail if Thums Up is still identified as cola leader.
* **Incorrect:** Selecting Ricqlès, claiming Ricqlès is a cola leader in India, claiming tie / no leader, or selecting a third brand (Coca-Cola, Pepsi, Campa Cola, Sprite) as the answer to this forced choice.
* **Evidence conflicts to ignore for grading:** Differences in exact percentage (15.4% vs 40-42% vs 85%) reflect year/metric (value vs volume, cola vs all carbonated, 1993 vs 2016-2024) and do not affect leader identification.

### Repeat 1 — clear — `5adce8c15542992c1e3a249d`

Question: Are Thyme and Cortaderia both aromatic plants?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: UNAMBIGUOUS - Single clear reading

Question wording has one materially different reading that honors all constraints. The phrase "aromatic plants" in botanical/horticultural usage means plants with a pleasant distinctive smell due to essential oils / volatile aroma compounds, not the chemical sense of containing benzene-ring compounds. Alternative chemical reading (containing aromatic hydrocarbons) does not satisfy the explicit constraint "aromatic plants" as a plant category and would misapply the term; it is therefore excluded as a branch. No genuine interpretation ambiguity remains.

### Canonical Answer
**No - They are not both aromatic plants. Only Thyme is aromatic; Cortaderia is not.**

- **Thyme (*Thymus*)** is classified as aromatic: *The genus Thymus contains about 350 species of aromatic perennial herbaceous plants and subshrubs in the family Lamiaceae*, with *leaves ... usually aromatic*. *Thymus vulgaris* is specifically described as *forming a woody, evergreen subshrub with small and highly aromatic grey-green leaves*, cultivated for *its fragrant leaves*.
- **Cortaderia (*Cortaderia selloana* - pampas grass)** is classified as a non-aromatic ornamental grass in Poaceae: *Type: Ornamental grass Family: Poaceae* and *In the genus Cortaderia are a couple of dozen species of grasses*. Horticultural sources state: *Cortaderia species are not known for their fragrance, but their visual impact in a landscape is significant*, with product data listing *Fragrance | Non-Fragrant*, *Fragrance: Non-fragrant*, and *Fragrance Non-fragrant*.

“Aromatic Plant” in this context equals *plant species having a pleasant distinctive smell ... high content of aromatic volatile organic substances collectively known as essential oil* and *having a strong smell*, which Thyme satisfies and Cortaderia does not.

Answer type: Boolean - No/False.

### Acceptable Variants / Acceptance Set
Grader must accept any paraphrase that entails the canonical answer:
- No / False / Not both
- Only Thyme is aromatic; Cortaderia is not aromatic / is non-fragrant / odorless
- Thyme yes, Cortaderia no
- Thyme is an aromatic herb (Lamiaceae, fragrant/essential-oil plant), Cortaderia is a non-aromatic ornamental grass (Poaceae, pampas grass)
- Thyme aromatic, Cortaderia not classified as aromatic plant

Grader must reject:
- Yes / Both are aromatic
- Cortaderia is (generally) aromatic / fragrant / an aromatic herb

### Necessary Caveats
- One commercial nursery line describes Cortaderia leaves as "aromatic when crushed" and another marketing listing calls it for "aromatic plant enthusiasts." These isolated vendor claims conflict with the consistent classification in botanical and horticultural authorities as Non-Fragrant / not known for fragrance. They do not establish Cortaderia as an aromatic plant in the standard essential-oil/fragrance sense and should be treated as evidence conflict, not as a basis for a second interpretation branch.
- Chemical sense of "aromatic" (containing benzene-ring hydrocarbons) is not the intended meaning. While pyrolysis of Cortaderia biomass can yield aromatic hydrocarbons, that does not make it an "aromatic plant" as a botanical category. Do not accept chemical-reading arguments for "both are aromatic."

### Evidence
What each citation supports:
- Thyme genus aromatic Lamiaceae: *contains about 350 species of aromatic perennial herbaceous plants and subshrubs in the family Lamiaceae*
- Thyme leaves usually aromatic
- *Thymus vulgaris* highly aromatic grey-green leaves
- Thyme cultivated for fragrant leaves
- Cortaderia is ornamental grass Poaceae
- Cortaderia is genus of grasses
- Cortaderia not known for fragrance
- Cortaderia listed Non-Fragrant
- Cortaderia Fragrance: Non-fragrant
- Cortaderia Fragrance Non-fragrant (second source)
- Definition aromatic plant = pleasant distinctive smell + essential oil
- Definition aromatic = having a strong smell

### Confidence
High. Convergent evidence from taxonomic (Wikipedia/Flora), horticultural (RHS, Missouri Botanical Garden, GardenersHQ, commercial nursery fragrance fields) and definitional sources supports the dichotomy. The counter-claims are isolated marketing text and do not lower confidence in the botanical classification.

### Grading Guidance
- **Accept** if answer is No/False with correct attribution that Thyme is aromatic and Cortaderia is not (or is non-fragrant, odorless, not an aromatic plant). Accept equivalents that name Thyme as Lamiaceae fragrant herb and Cortaderia as Poaceae ornamental pampas grass not aromatic.
- **Reject** if answer is Yes/Both are aromatic, or asserts Cortaderia generally qualifies as an aromatic plant/fragrant herb without qualification.
- **Partial (if used):** Award partial if correctly identifies Thyme as aromatic but gives vague/hedged answer on Cortaderia without clear "not aromatic" statement. Do not award for answers that rely solely on the minority "aromatic when crushed" vendor claim to justify "both aromatic."
- Do not create or credit alternative interpretation branches based on chemical aromaticity, incomplete evidence, or granularity differences.

### Repeat 1 — broad-but-single — `5ae336ef5542991a06ce995d`

Question: What profession do both Phil Alden Robinson and David Nixon share?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: UNAMBIGUOUS

Canonical Answer: **Film director** (also described as **director** / **filmmaker** / **movie director**). Both Phil Alden Robinson and David Nixon (the American film director) are film directors.

### Acceptable Variants / Acceptance Set
Grader should accept any phrasing that denotes the same profession at same or slightly broader/narrower granularity that includes directing films:

- film director
- director
- movie director
- film director and screenwriter / film director and producer (acceptable if director/filmmaker is included; the shared core is director)
- filmmaker

Do NOT accept alone (without director): screenwriter, producer, magician, television presenter, choreographer, footballer, entertainer, or "writer" alone.

### Necessary Caveats
Referential disambiguation: The name "David Nixon" refers to multiple persons. The question presupposes a shared profession exists with Phil Alden Robinson.

- Phil Alden Robinson's listed occupation is `Film director, screenwriter` and he is described as `an American film director and screenwriter` ; also summarized as `He is a writer and director`
- David Nixon (director) is described as `an American film director and film producer` /
- David Nixon (magician) has `Occupations | Magician and television presenter` which does **not** overlap with Phil Alden Robinson and therefore fails the explicit constraint "both ... share" a profession. It is not a valid alternative interpretation, only a disambiguation note.

No conflicting evidence about the director occupation was found. Granularity differences (director vs. film director vs. filmmaker) do not constitute separate interpretations and do not change the grading outcome.

### Evidence
**Phil Alden Robinson - is a film director:**
- Occupation infobox: `Occupation | Film director, screenwriter`
- Lead description: `**Phil Alden Robinson** (born March 1, 1950) is an American film director and screenwriter whose films include *Field of Dreams*`
- Same description in search snippet: `Phil Alden Robinson ... is an American film director and screenwriter whose films include *Field of Dreams*, *Sneakers*, and *The Sum of All Fears*.`
- IMDb summary: `He is a writer and director, known for Field of Dreams (1989)`

**David Nixon (director) - is a film director:**
- Lead description: `**David Nixon** is an American film director and film producer. He is the director and producer of *Letters to God*`
- Search result duplicate: `**David Nixon** is an American film director and film producer. He is the director and producer of *Letters to God* and a co-producer and assistant director of *Fireproof*`
- Supporting duplicate: `David Nixon is an American film director and film producer. He is the director and producer of Letters to God`

**David Nixon ambiguity - alternative referent does not satisfy shared-profession constraint:**
- Magician entry: `Occupations | Magician and television presenter`

### Confidence
High. Both professions are directly and consistently attested in Wikipedia infobox and lead paragraphs and replicated across mirrors. The only source of name collision (David Nixon magician/ choreographer/ footballer) is explicitly disambiguated and does not provide a shared profession, leaving one clear reading that satisfies all constraints.

### Grading Guidance
- **Accept** answer if it states the shared profession is film director / director / movie director / filmmaker. Inclusion of an additional correct role (e.g., "film director and producer" or "director/screenwriter") is still acceptable if director/filmmaker is explicitly included.
- **Reject** answer if it states only magician, television presenter, entertainer, choreographer, footballer, producer alone, or screenwriter alone, or claims no shared profession.
- **Do not penalize** granularity: "director" and "film director" are equivalent for this question.
- **Do not treat** the existence of other David Nixons (magician, etc.) as a valid alternative answer branch; those readings violate the question's explicit "both ... share" presupposition.

### Repeat 1 — clear — `5ae6168f5542996de7b71b47`

Question: Adrian the Barbarian (later Olaff the Madlander) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR - Single canonical reading

The question wording has one clear meaning that honors all explicit constraints: the year the strip *initially published as Adrian the Barbarian* (the original name, with *Olaff the Madlander* noted only as the later reprint title) was first introduced *in the comic The Beezer* [and then continued in the merged Beezer and Topper].

A potential alternative reading - the year it was first introduced *as Olaff the Madlander* - does not satisfy the constraint "in the comic The Beezer, and later the merged Beezer and Topper" (the Olaff name was first used for reprints in *The Beano*), has no supporting evidence in the Beezer context, and would not produce a materially different grading outcome that honors the question wording. It is therefore an evidence/related-fact issue, not a genuine interpretation ambiguity.

No interpretation branches are required.

### Canonical answer
**1989** - *Adrian the Barbarian* (later reprinted as *Olaff the Madlander*) was first introduced in *The Beezer* in **1989**, specifically in **No 1727 dated 18th Feb 1989**.

### Acceptable variants / Acceptance set
- Canonical: `1989`
- Accept as correct if answer gives `1989` by any clear phrasing:
  - `1989`
  - `February 1989`
  - `18 February 1989` / `18th Feb 1989`
  - `Beezer No 1727 1989`
- All of the above flatten to the same grading outcome: **1989**.

Do NOT accept:
- `1990` alone
- `2008` / `July 2008` (year of Beano reprint as Olaff)
- `1993`, `2003`, or other years

### Necessary caveats
- **Evidence conflict - 1990 vs 1989:** The Wikipedia *List of Beezer comic strips* table is organized by year of release date and lists the strip under the heading `## 1989` with the column value `1990` being the **End Date** column (`| Strip Title | Artist(s) | End Date |`). Some mirrors misread this `1990` as a start date. The explicit start/end table on UK Comics Wiki resolves it as `| Adrian the Barbarian | Sid Burgon | 1989 | 1990 |`, and the article text confirms `first introduced in 1989`. This is a data-presentation conflict, not an interpretation ambiguity.
- **Scope/granularity - 2008 reprint:** In July 2008 the strip was reprinted in *The Beano* under the new title *Olaff the Madlander*. This is a reprint/re-titling, not the first introduction in *The Beezer* requested by the question.
- **Granularity:** Year-level (`1989`) is the requested attribute. A more precise date (`18 Feb 1989`) is compatible detail at a finer granularity, not a different interpretation.

### Evidence
- Definition matching the exact question wording and year: "***Olaff the Madlander*** (initially published as ***Adrian the Barbarian***) was a comic strip in the comic The Beezer, and later the merged Beezer and [Topper], first introduced in 1989."
- Detailed first-issue evidence: "Adrian the Barbarian aka Olaff the Badlander first appeared in the Beezer in No 1727 dated 18th Feb 1989 and ran on through to the comics final issue No 1809 (15/9/90) and then went on to appear in the combined Beezer & Topper"
- Chronological placement: Table is organized by year of release date and under `## 1989` lists `| *Adrian the Barbarian* | Sid Burgon | 1990 | | Was the comic's cover star. Continued in the *Beezer and Topper*, and was reprinted as Olaff the Madlander in *The Beano* |`, confirming 1990 is End Date.
- Explicit start/end pair: `| Adrian the Barbarian | Sid Burgon | 1989 | 1990 | Comic strip | Continued in the Beezer and TopperReprinted as Olaff the Madlander in The Beano |`
- Artist history corroboration: "Burgon began to draw for DC Thomson in 1989 drawing a revival of Biffo the Bear in The Beano and Adrian the Barbarian for The Beezer"
- Reprint distinction: "In July 2008, as of issue 3441, *Adrian the Barbarian* was reprinted in The Beano under the new title of *Olaff the Madlander*."

### Confidence
**HIGH** - Direct article statement, corroborated by specific issue number/date from a knowledgeable comics forum, consistent placement in two independent chronological tables, and artist biography. The only counter-signal (`1990`) is demonstrably the End Date column, not a competing start-date claim.

### Grading guidance
- **CORRECT:** Any answer that clearly states the first introduction in *The Beezer* was in **1989** (including more precise forms like `18 Feb 1989`, `February 1989`, `Beezer No. 1727 1989`). Mentioning the 2008 Beano reprint alongside 1989 is acceptable only if 1989 is identified as the original Beezer first appearance.
- **INCORRECT:** Answer that gives only `1990`, only `2008`, or any year other than 1989 as the first introduction in *The Beezer*. Answer that describes 1990 as the start date because of misreading the End Date column is incorrect.
- **Granularity:** Do not penalize extra correct detail (exact day/issue). Do not require exact day; year alone is sufficient.

### Repeat 2 — broad-but-single — `56e2419acd28a01900c67c2a`

Question: What structures can be viewed as both works of art and symbols of certain cultures?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR - Single clear reading; no materially distinct interpretation satisfying all constraints requires branching.

Canonical Answer:
**Architectural works, in the material form of buildings (also described as buildings / structures / monuments / architectural monuments)** — which are **often perceived as cultural symbols and as works of art**.

In elaborated form: **Architecture is frequently considered a fine art** and **Architectural works are perceived as cultural and political symbols and works of art**. **Historical civilizations are often identified with their surviving architectural achievements**.

Concrete exemplars that satisfy both predicates are the same class: **the pyramids of Egypt and the Roman Colosseum are cultural symbols** and are explicitly cited as examples of architectural works that are cultural symbols and works of art; similarly **Architecture is the study and practice of designing structures, especially habitable ones. It utilizes civil engineering techniques, but is considered a visual art**.

Acceptable Variants / Acceptance Set:
Grader should accept any answer that conveys the general class and/or valid instances of it:

* General class phrasing: `architectural works`, `buildings`, `structures` (in the physical built-environment sense), `monuments`, `architectural monuments`, `historic buildings`, `iconic architecture` - when described as perceived as both cultural symbols and works of art.
* Specific instance phrasing: any single or list of iconic structures that sources describe as both - including but not limited to: Pyramids of Egypt / Great Pyramid of Giza, Roman Colosseum, Parthenon/Acropolis, Eiffel Tower, Taj Mahal, Statue of Liberty, Sydney Opera House. A single correct example counts as correct; exhaustive list not required.
* Synonymous duality phrasing: `cultural and political symbols and works of art` / `cultural symbols and as works of art` accepted as same proposition.

Do not require both level of generality and examples; either level satisfies. Different levels of granularity (category name vs. example) are scope variants, not different interpretations.

Necessary Caveats:
* The claim is qualified as **often perceived**, not universally - not every utilitarian structure qualifies; only works with aesthetic/design quality and cultural significance.
* `Structures` means physical built structures (buildings and other physical structures; the art and science of designing buildings and (some) nonbuilding structures), not social, grammatical, or organizational structures.
* The symbolic function is culture-specific and contingent; the same definition does not assert that a given building is a symbol of *all* cultures.
* Evidence conflict does not exist for this definition; sources converge. Wikipedia and liquisearch independently reproduce the same definitional sentence. Disagreement about which examples are most iconic is evidence variation, not interpretation ambiguity.

Evidence:
* Central definitional sentence directly answering question: `Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements.` - Supports canonical answer and the culture-symbol + work-of-art duality.
* Duplicate statement across sources: `Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art.` - Supports reproducibility of definition.
* Architecture as art: `Architecture is the study and practice of designing structures...but is considered a visual art.` - Supports work-of-art predicate.
* Extended definition with political dimension: `Architectural works are perceived as cultural and political symbols and works of art.` - Supports cultural/political symbol predicate.
* Exemplar linkage: `Such buildings as the pyramids of Egypt and the Roman Colosseum are cultural symbols, and are important links in public consciousness` and `Such buildings as the pyramids of Egypt and the Roman Colosseum are cultural symbols` - Supports that pyramids/Colosseum exemplify the class.
* Scope of architecture: `A general term to describe buildings and other physical structures.` - Supports that `structures` in question = physical buildings.

Confidence: HIGH - Central definitional sentence is reproduced verbatim in multiple independent mirrors of Wikipedia Architecture and in secondary synthesis (liquisearch) with identical wording and with consistent exemplars (pyramids, Colosseum). No contradictory source found for the general proposition.

Grading Guidance:
* Award full credit for any response stating the general class (architectural works / buildings / monuments / architectural monuments / structures in material form of buildings) qualified as cultural symbols and works of art, with or without examples.
* Award full credit for responses listing one or more correct iconic examples (pyramids, Colosseum, Parthenon, etc.) if they are implicitly or explicitly presented as both art works and cultural symbols.
* Do not penalize for granularity: category-only answer and example-only answer are equally correct.
* Do not award credit for non-physical structures (e.g., social structures, chemical structures) or for answers mentioning only one of the two predicates (only art without symbol, or only symbol without art).
* Do not require mention of `often` qualifier, but if answer claims *all* structures are both art and symbols, treat as overgeneralization - downgrade to partial credit.

### Repeat 2 — broad-but-single — `56f71740711bf01900a44932`

Question: Bilateral treaties are concluded between how many states or entities?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR

Question has one clear reading. Inspection of exact wording:

* Entities: `Bilateral treaties`
* Relationship: `concluded between`
* Requested attribute: `how many states or entities` - a count
* Scope: generic definitional category in international law
* Answer type: numeric cardinality

No materially different reading satisfies all explicit constraints. A potential grouping distinction (two *sides* vs more than two *formal signatory instruments* in e.g. Switzerland-EU treaties) was considered and rejected as an interpretation branch because it is a granularity / caveat about formal party count, not a different interpretation of the interrogative phrase `how many states or entities` which in treaty typology always asks for the number of sides to the agreement. It does not create a competing valid answer that honors the wording; it is an evidence nuance to be noted as a caveat. It would not produce a materially different grading outcome where `two` remains correct.

### Canonical Answer

**Two.** Bilateral treaties are concluded between **two** states or entities — i.e., between two parties / two sides / two subjects of public international law.

Definitional support:

* Treaties may be bilateral (between two countries) or multilateral (involving more than two countries)
* Treaties are typically divided into two main categories, multilateral and bilateral. A multilateral treaty is a treaty involving more than two parties, while a bilateral treaty involves an agreement between two parties
* A bilateral treaty (also called a bipartite treaty) is a treaty strictly between two state entities ... Any agreement with more than two parties is a multilateral treaty
* These two parties can be two nations, or two international organizations, or one nation and one international organization

### Acceptable Variants / Acceptance Set

Grader should accept any equivalent expressing the canonical count of 2:

* `2`, `two`
* `two states`, `two states or entities`, `two countries`, `two nations`, `two governments`
* `two parties`, `two sides`, `two subjects of international law`, `two state entities`, `two sovereign states / international organizations`

All specify quantity = 2 on two sides. Verbose forms that include the same quantity with synonyms for state/entity are acceptable.

Do NOT accept: `1`, `one`, `three`, `more than two`, `several`, `many`, `multilateral`, `plurilateral` as the defining number for bilateral.

### Necessary Caveats

* **Two sides vs formal signatory count:** It is possible for a bilateral treaty to have more than two *formal* parties and still be classified as bilateral when parties are divided into two groups. For example, each of the bilateral treaties between Switzerland and the European Union has seventeen parties: parties are divided into two groups, the Swiss ("on the one part") and the EU and its member states ("on the other part"). The treaty establishes rights and obligations between the Swiss and the EU and the member states severally — it does not establish rights and obligations amongst the EU and its member states. The same nuance is expressed as: It is possible for a bilateral treaty to involve more than two parties ... parties are divided into two groups. This does not change the canonical answer; the *type* remains bilateral = two sides.

* **Entity type does not change count:** The two entities may be sovereign states or international organisations established by treaty, or one of each. The count remains two.

No evidence conflict affects the core definition; all reputable sources agree bilateral = two.

### Evidence

What each URL supports:

*  Supports core definition: `Bilateral treaties are concluded between two states or entities.`
*  Supports bilateral = between two countries, multilateral = involving more than two countries.
*  Supports Library of Congress definition: bilateral = agreement between two parties, multilateral = more than two parties.
*  Supports bipartite definition: strictly between two state entities, established by negotiations between two parties, and any agreement with more than two parties is multilateral.
*  Supports composition of the two parties: two nations, two international organizations, or one nation and one international organization.
*  Supports caveat that bilateral can formally have >2 parties but remains bilateral due to two-group structure (Switzerland-EU 17 parties example).
*  Supports same caveat in bilateral-treaty article.

### Confidence

High. The definition `bilateral = two` is uniformly attested across Wikipedia Treaty article, the dedicated Bilateral treaty article, Library of Congress Research Guide (primary institutional source), and Oxford/university libguides, with no contradictory reputable definition. The only nuance (17 formal parties still bilateral) is consistently explained as two sides, not as an alternative count.

### Grading Guidance

* Accept any answer that states the number is **2 / two** and associates it with states/entities/parties/sides/subjects/countries/nations/governments/organizations, with or without additional correct elaboration.
* Treat elaboration citing the Switzerland-EU 17-party example as correct **if** it clarifies that despite 17 formal parties the treaty is still classified as bilateral because there are only two sides/groups. Do not penalize for adding this nuance.
* Reject answers that give as the *defining* count: 1, 3, >2, several, many, unlimited, or that conflate bilateral with multilateral/plurilateral.
* Reject answers that answer a different attribute (e.g., describing treaty content, examples like Camp David Accords) without giving the count.
* Accept numeric `2` alone, word `two` alone, or full phrase `two states or entities`.

### Repeat 2 — ambiguous — `57260b4389a1e219009ac188`

Question: What is an example of something that serves a function other than protection?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

**Status: UNAMBIGUOUS** - One clear reading. After inspecting wording, entities, relationship, requested attribute, scope, and answer type, no materially different reading satisfies all constraints with supporting evidence and a different grading outcome.

*   **Wording:** "What is an example of something that serves a function other than protection?" - interrogative requesting one instance.
*   **Entities:** indefinite "something" - no domain restriction (biological, artifact, institution, etc.)
*   **Relationship:** "serves a function other than protection" - predicate requires at least one documented function distinct from protection.
*   **Requested attribute:** example
*   **Scope:** unrestricted / open-domain
*   **Answer type:** entity + non-protective function description

No genuine interpretation ambiguity. Hypothesized branches such as "entity that does not protect at all" vs "entity that protects plus has other functions" both honor the explicit constraint "other than protection" and produce the same grading outcome: any verifiable non-protective function is sufficient. Breadth of possible domains (skin, bone, membrane, clothing) is **scope/granularity**, not interpretation ambiguity. No competing evidence on core definition. Therefore collapse to one canonical answer with broad Acceptance set. Do not create branches merely for level of detail or domain choice.

### Canonical Answer

**An example is the human skin / integumentary system, which serves non-protective functions including thermoregulation, sensation, and vitamin D synthesis.**

*   The skin plays important roles in "protection, sensing stimuli, thermoregulation, and vitamin D synthesis"
*   Specifically, "The skin has multifarious functions: ... - sensation: temperature, pain, touch, proprioception, vibration are all indicated by cutaneous receptors"
*   And "Sensation, thermoregulation, vitamin D synthesis, and insulation are just a few of the functions of the skin" in addition to protection as a physical and chemical barrier

Any one of thermoregulation, sensation, or vitamin D synthesis satisfies "other than protection."

### Acceptable Variants / Acceptance Set

Grader must accept **any real entity** with a correctly stated, verifiable function that is not protection. The entity and function need not match the canonical example. Non-exhaustive acceptance set:

1.  **Skeletal system / bones:** support, facilitation of movement, mineral/fat storage, blood cell formation / hematopoiesis
    *   "The five important functions of the skeletal system are support, protection, movement, mineral storage, and blood cell formation"
    *   "The major functions of the skeletal system are body support, facilitation of movement, protection of internal organs, storage of minerals and fat, and blood cell formation"
    *   Explicitly "The two functions of the skeletal system other than protection are: They provide structural support to our body. It makes blood cells, helps to provide shape to the body, allows movement, and stores minerals"

2.  **Cell / plasma membrane:** selective transport, cell signaling / reception, cell adhesion, structural support
    *   "The cell membrane controls the movement of substances in and out of the cells and organelles. It is selectively permeable... It plays a role in cell adhesion, cell signaling"
    *   "Among the most sophisticated functions of the plasma membrane is the ability to transmit signals by means of complex, integral proteins known as receptors"
    *   "These membrane proteins are responsible for many specialized functions; some act as receptors that allow the cell to respond to external signals, some are responsible for the selective transport of molecules across the membrane"

3.  **Clothing, government, mucus/saliva, cell wall, etc.** if a correct non-protective function is given (e.g., clothing = adornment/modesty, government = order/conflict resolution, saliva = lubrication/binding food into bolus, digestion, taste, mucus = pH control). Variant phrasings accepted: thermoregulation = temperature regulation / sweating / vasodilation; sensation = touch/pain/pressure perception; support = structural framework; movement = leverage for muscles; blood cell formation = hematopoiesis/hemopoiesis.

### Necessary Caveats

*   The stated function must be **distinct from protection**. Synonyms of protection alone (defense, barrier against pathogens/injury, shielding, waterproofing as protection) do not satisfy. Must name a different functional category.
*   Protections need not be denied - most valid examples (skin, bone, membrane) have **both** protective and non-protective functions. An answer is valid if it correctly identifies at least one non-protective function, regardless of whether it also mentions protection.
*   The open domain is intentional scope. Grader must not penalize for choosing a different domain than the canonical skin example.
*   Function must be factually verifiable. Incorrect functions (e.g., "skin performs photosynthesis") must be rejected.
*   No evidence conflict requires branching; sources agree on multiple non-protective functions across domains. Disagreement on emphasis is granularity, not ambiguity.

### Evidence

| URL | Supports |
|-----|----------|
|  | Skin roles include protection, sensing stimuli, thermoregulation, vitamin D synthesis |
|  | Skin sensation function: temperature, pain, touch via cutaneous receptors |
|  | Sensation, thermoregulation, vitamin D synthesis listed as functions in addition to protection |
|  | Five skeletal functions: support, protection, movement, mineral storage, blood cell formation |
|  | Major skeletal functions: body support, facilitation of movement, protection, storage, blood cell formation |
|  | Two functions other than protection explicitly: structural support, blood cell production, movement, mineral storage |
|  | Cell membrane: controls movement/selective permeability, roles in adhesion and signaling |
|  | Plasma membrane sophisticated function: signal transmission via receptors |
|  | Membrane proteins: receptor signaling and selective transport |

### Confidence

**High.** Direct evidence for canonical skin functions and for multiple alternative domains (skeletal, membrane) is consistent across textbooks and reputable sources. No conflicting evidence undermines the classification of these as non-protective functions. Confidence that open-scope handling via Acceptance set is correct per instruction: High.

### Grading Guidance

*   **Accept** any response that: (a) names a specific entity/structure/object/institution, and (b) accurately describes at least one of its functions that is not protection. Exact wording not required.
*   **Accept** both biological and non-biological examples and any level of detail (e.g., "bones store calcium" = mineral storage; "skin sweats to cool body" = thermoregulation).
*   **Reject** if: only protective function given, function is factually incorrect, entity is fictive or not verifiable, or answer is incoherent/empty.
*   For borderline cases, verify that the claimed function is documented in reputable sources as distinct from protection (e.g., excretion via sweat is acceptable as separate from barrier protection).

### Repeat 2 — ambiguous — `57302972b2c2fd140056899e`

Question: What could be considered a trait of Rome's military campaigns?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: UNAMBIGUOUS

Canonical answer:

Any one trait that is reliably attributed to Rome's (ancient Roman) military campaigns is correct. Central, well-supported traits include:

* **High organization, strict discipline / *disciplina militaris* and unit cohesion**
* **Strategic/ tactical versatility and innovation**
* **Engineering and construction (roads, fortifications, fortified marching camps / *castra*)**
* **Preference for direct open-battle engagement and, where necessary, overcoming fortified positions via siege engineering**
* **Expansionist / territorial character, serving the triple state purposes of securing borders, exploiting peripheral areas via tribute, and maintaining internal order**

The question phrasing "What could be considered a trait" is open-ended and non-exhaustive; providing any one of the above (or a synonymous paraphrase) satisfies the question. No second materially different interpretation is required.

Evidence for canonical traits:

* Roman warfare was characterized by its organization, discipline, and strategic innovation
* Roman warfare emphasis included engineering, constructing roads and fortifications to support campaigns
* Versatility and discipline characterized their approach
* Roman soldiers were subject to brutal discipline, up to and including summary execution if judged derelict and were held to the highest standards of unit cohesion, and their combat drills meant they were constantly ready for battle
* The preponderance of Roman campaigns exhibit a preference for direct engagement in open battle and, where necessary, the overcoming of fortified positions via military engineering
* The majority of Rome's campaigns were characterised by one of two types. The first is the territorial expansionist campaign, also stated as the majority ... were characterised by one of two types. The first is the territorial expansionist campaign
* Rome's military served the triple purpose of securing its borders, exploiting peripheral areas through measures such as imposing tribute on conquered peoples, and maintaining internal order

Acceptable variants / Acceptance set:

Grader should accept **any one or more** of the following, including synonymous phrasing:

* disciplined / strictly disciplined / brutal discipline / *disciplina militaris* / cohesive / well-drilled / constantly ready for battle
* highly organized / structured / systematic / legion/cohort/century organization
* versatile / adaptable / innovative / strategic innovation / tactical flexibility (testudo, maniple, quincunx, double envelopment)
* engineering-focused: building roads, bridges, forts, fortified camps (*castra* with palisade and ditch), siege warfare expertise
* logistics / supply-chain capable / able to march 20+ miles per day with heavy pack and operate for long periods in hostile territory
* expansionist / territorial / imperialist / aggressive / offensive / counter-offensive that subjugates large areas
* preference for direct / open / pitched battle / seeking decisive engagement; overcoming fortified positions by engineering
* triple-purpose function: border security, exploiting periphery via tribute, maintaining internal order
* rigorous training / professional / long-service career soldiers
* obstinate persistence / resilience after losses (history of overcoming appalling losses)

Paraphrases such as "well-trained," "professional standing army," "fortified marching camps each night," "use of formations like testudo," "logistically sustained via road networks" are acceptable.

Necessary caveats:

* Scope is granularity, not ambiguity. Traits varied across time (Kingdom 753-509 BC, Republic 509-27 BC, Empire). Early Rome raised armies annually; from Augustan period a standing professional army with legions predominated. This is a difference in level of detail, not a separate interpretation.
* The preponderance claim (preference for open battle) describes a tendency, not universality. Later Empire strategy (described by Vegetius, 4th c. AD) advocated more limited direct engagement and reliance on stratagem, contrasting with Republic/early Empire offensiveness. Evidence conflicts on degree of aggressiveness are caveats, not interpretation branches.
* After the First Punic War naval battles were less significant than land battles to Roman military history due to Mediterranean dominance; "naval power" should not be credited as a primary campaign trait for most periods.
* "Could be considered" permits one valid example; answer need not be exhaustive.

Evidence:

* [380369234786810282] - supports organization/discipline/strategic innovation, engineering (roads/fortifications), versatility/discipline characterization, and preference for open battle via engineering
* [1107388844473832428] - supports brutal discipline and unit cohesion/constant readiness, and standing professional army context
* [4551117213693965326] / [1863377875719471220] - supports territorial expansionist campaign as dominant campaign type and triple purpose (borders/tribute/internal order)

Each URL supports the specific claim cited inline above; no additional distinct interpretations satisfy all explicit constraints (entity = ancient Rome, relationship = campaigns exhibit trait, requested attribute = characteristic feature, scope = any valid trait, answer type = descriptive trait).

Confidence: HIGH that the listed traits are historically recognized and citation-supported; HIGH that the question has one clear reading; MODERATE that any single item from the Acceptance set will match an automated grader's expected answer due to open wording, but all items listed are defensible.

Grading guidance:

* **Accept** if answer provides any one trait from Acceptance set, even if phrased differently (e.g., "discipline," "highly organized legions," "great engineers who built camps and roads," "logistical excellence," "expansionist," "sought pitched battles").
* **Accept** a single correct trait alone; do not require comprehensive list.
* **Reject** if answer asserts a trait contradicted by evidence (e.g., "relied primarily on mercenaries/undisciplined hordes," "avoided open battle as a rule throughout history," "primarily a naval power after 241 BC," "lacked organization").
* Do not penalize additional correct traits alongside a correct central trait. Incorrect additional claims do not negate a correct central trait unless the entire response is fundamentally inaccurate.
* Vague but directionally correct statements (e.g., "good organization") should be accepted as partial/variant if context shows intent toward a supported trait.

### Repeat 2 — ambiguous — `5ab9e3d25542994dbf019889`

Question: What type of vegetation does Lythrum and Erythrina have in common?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CANONICAL - Single clear reading

The question asks for a shared **botanical habit / life-form vegetation type** between the two genera. `Type of vegetation` in standard botanical usage means growth form and duration - herb / shrub / tree, herbaceous / woody, annual / perennial, deciduous / evergreen - not flower colour or name etymology. The collateral etymological reading (both names mean `blood/red`) describes flower colour, not vegetation type, and fails the explicit constraint.

### Canonical Answer

**Both are perennial herbs / herbaceous perennials** - more broadly **perennial vegetation** and at the coarsest level **flowering perennial dicot herbs**. The most precise shared type supported at species level is **herbaceous perennial (perennial herb / perennial forb)**, exemplified by *Lythrum salicaria* and *Erythrina herbacea*.

* *Lythrum* is consistently described as a perennial herb / herbaceous perennial.
* *Erythrina* as a genus is predominantly woody but is explicitly described as including perennial herbs, specifically *Erythrina herbacea* as a perennial herb; more generally both genera are perennial.

### Acceptable Variants / Acceptance Set

Grader should accept any phrasing at equivalent or coarser granularity that denotes the same habit:

* herb / forb / herbaceous plant
* perennial herb / herbaceous perennial / perennial forb / perennial herbaceous plant
* perennial (with or without `herbaceous`)
* deciduous perennial herb (deciduous is additional correct foliage trait but not required)
* flowering herb / angiosperm herb / dicot herb - accept as partially correct but prefer `perennial herb` if specificity is required

Do **not** accept as canonical:
* `tree` or `shrub` alone for both (incorrect for *Lythrum*)
* `annual` alone (incorrect for both as perennial type)
* `red / blood-red flowered` alone as a vegetation *type* - this is etymology/flower colour, not vegetation type (see Caveats)
* `aquatic`, `succulent`, `woody` alone, `grass`

### Necessary Caveats

* **Genus-level scope conflict:** *Lythrum* is uniformly herbaceous perennial, while *Erythrina* is predominantly woody. Flora of New Zealand: `Shrubs, trees, or very rarely herbs`; Flora of China: `Trees or shrubs [or perennial herbs]`; RHS genus description: `Erythrina can be deciduous or evergreen subshrubs, shrubs or trees`. The shared `herb / herbaceous` trait is therefore strictly true at the species level (*E. herbacea* and similar) and as a possible habit for the genus, not as the dominant habit for all 100+ *Erythrina* species.
* **Granularity:** `perennial` alone is universally true for both genera; `herbaceous perennial` is more precise but requires invoking the herbaceous *Erythrina* species. Both levels should be accepted; do not penalize for coarser (`perennial`) vs finer (`herbaceous perennial`) within this reading.
* **Etymology is not vegetation type:** Both generic names do derive from Greek for blood/red referring to flower colour - *Lythrum* `from the Greek word lythron, which means blood`, *Lythrum* `is from the Greek lythron blood referring to the colour of the flowers`, *Erythrina* `generic name is derived from the Greek word erythros, meaning red, referring to the flower color`, `The name of the genus comes from the Greek erythros = red, with reference to the prevailing colour of the flowers`, `It is derived from the Greek word erythros, which means red or flame-colored`. This is a related etymological fact, not a `type of vegetation` as habit, and does not create a second valid interpretation under the question wording.
* **No evidence of other shared types** e.g., both wetland, both nitrogen-fixing, both invasive - not supported as common vegetation type.

### Evidence

*Lythrum as perennial herb / herbaceous perennial:*
* `Herbaceous Perennial Lythrum salicaria 'Feuerkerze' purple loosestrife 'Feuerkerze' Herbaceous perennial forming a compact clump`
* `Lythrum salicaria 'Zigeunerblut' Herbaceous Perennial A compact, clump-forming perennial`
* `Lythrum virgatum ... Herbs perennial, slender, 5–12 dm`
* `An erect perennial herb in the loosestrife family`
* `Hairy, perennial herb; stems erect, ± quadrangular, 1–2 m tall`

*Erythrina as perennial and as herb/perennial herb:*
* `Erythrina herbacea | Cardinal Spear, Redcardinal | Perennial | 1.0`
* `Plant Growth Form | Tree (Small), Shrub | Lifespan | Perennial` for *E. herbacea* (shows perennial lifespan despite tree/shrub form in Singapore)
* `Morphology General Habit Perennial, Not climbing, Herb/Shrub/Tree` for *E. herbacea* at Kew
* Genus-level habit range: `Erythrina can be deciduous or evergreen subshrubs, shrubs or trees`
* `Shrubs, trees, or very rarely herbs`
* `Trees or shrubs [or perennial herbs]`

*Etymology (for caveat only):*
* `Lyth'rum comes from the Greek word lythron, which means blood`
* `Lythrum is from the Greek lythron blood referring to the colour of the flowers`
* `The generic name is derived from the Greek word erythros, meaning red, referring to the flower color`
* `The name of the genus comes from the Greek erythros = red, with reference to the prevailing colour of the flowers`

### Confidence

* **High** that `perennial` / `perennial herb` is the intended canonical vegetation type.
* **Medium-High** for the precise phrase `herbaceous perennial` as universally applicable to both genera - well supported for *Lythrum* and for *E. herbacea* specifically, but at genus level *Erythrina* is predominantly woody and only `very rarely herbs` / `[or perennial herbs]`.

### Grading Guidance

* **Award full credit** for: `herb`, `forb`, `herbaceous perennial`, `perennial herb`, `perennial forb`, `perennial herbaceous plant`, or simply `perennial` (and optionally `deciduous perennial herb`).
* **Do not require** both words `herbaceous` and `perennial` together - either `herb` or `perennial` captures the core shared type; `herbaceous perennial` is the ideal complete answer.
* **Mark incorrect** if answer states shared type is `tree`, `shrub`, `annual`, `aquatic`, `woody`, `succulent`, or `grass` without qualification as perennial herb.
* **Treat `red / blood-red flowered` as incorrect** for this question's `type of vegetation` wording, even though etymologically true. If an answer gives `red perennial herbs`, accept because it includes the correct habit.
* **Evidence conflicts** about woody vs herbaceous for *Erythrina* should not be used to mark `perennial herb` wrong - the genus explicitly includes perennial herbs per Flora of China and Kew, and *E. herbacea* is the type species bearing the shared herbaceous trait.

### Repeat 2 — clear — `5ac4e4595542996feb3fe968`

Question: Which cola drink is the leader in India, Thums Up or Ricqlès?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CANONICAL - Single clear reading; no genuine interpretation ambiguity requiring branches

### Canonical Answer
**Thums Up** is the cola drink that is the leader in India.

- Thums Up is a brand of cola with Type Cola and Origin India
- Thums Up is the largest cola brand in India with Rs5,000 crore in annual sales and has dominated India's cola market in the past four decades
- Thums Up quickly became India's biggest cola brand, trumping competitors such as Campa Cola and Double Seven
- Today Thums Up rules the cola scene in India, only topped in the carbonated beverage arena by Sprite

Ricqlès is not an Indian cola leader:

- Ricqlès – mint-flavoured soft drink
- France: Ricqlès - Originally a medical soda, Ricqlès is a mint-flavored soda
- Ricqlès is a peppermint spirit and a French soft drink. The liquor Alcool de Menthe de Ricqlès was created in 1838 by Heyman de Ricqlès in Lyon, France

Therefore between the two named options, Thums Up satisfies all explicit constraints (cola drink + leader in India); Ricqlès fails the cola and India-leader constraints.

### Acceptable Variants / Acceptance Set
Grader should accept any response that unambiguously selects Thums Up as the answer:

- "Thums Up" (canonical orthography without 'b')
- Common variants: "Thumbs Up", "Thums-Up", "THUMS UP", "Thums up"
- Explanatory variants: "Thums Up is India's largest/No.1 cola brand; Ricqlès is a French mint drink, not an Indian cola leader"

Flattened Acceptance set: { Thums Up }

Grader should reject:
- Ricqlès / Ricqles / any spelling of Ricqlès as leader
- "Both are leaders" / "Neither" / "Tie"
- "Ricqlès is the cola leader in India" or "Ricqlès is the cola leader in France therefore leader in India"

### Necessary Caveats
- Question premise presupposes both Thums Up and Ricqlès are cola drinks. Evidence shows Ricqlès is not a cola but a mint-flavoured soft drink / peppermint spirit from France. This is a premise inaccuracy, not a grading ambiguity; the binary choice still resolves to Thums Up as the only cola that meets the constraints.
- Scope is cola segment in India. Some sources note that within the broader carbonated beverages market, Sprite tops the market by retail value, while Thums Up remains the largest cola brand. This granularity distinction does not change the cola-specific answer.
- No evidence supports Ricqlès having significant market share or leadership in India; all leadership evidence points to Thums Up historically (85% share at acquisition, monopoly in 1980s) and currently (largest cola brand).

### Evidence
- Thums Up is a brand of cola — supports Thums Up is a cola drink
- Type Cola — supports classification
- Origin India — supports Indian origin
- Largest cola brand in India with Rs5,000 crore sales — supports leadership claim
- Has dominated India's cola market in the past four decades — supports sustained leadership
- Quickly became India's biggest cola brand — supports historical leadership from 1977
- Today Thums Up rules the cola scene in India, only topped in carbonated beverage arena by Sprite — supports current cola leadership with nuance about overall carbonated market
- Ricqlès – mint-flavoured soft drink — supports Ricqlès is not a cola
- France: Ricqlès — supports Ricqlès associated with France, not India
- Ricqlès is a peppermint spirit and French soft drink created 1838 in Lyon, France — supports French origin and mint category, not Indian cola
- Originally a medical soda, Ricqlès is a mint-flavored soda — supports mint soda description

### Confidence
High. Multiple independent reputable sources (Livemint/Euromonitor reporting, Wikipedia with cited sources, Atlas Obscura) converge on Thums Up as India's largest / ruling cola brand; multiple sources consistently classify Ricqlès as a French mint soft drink, not a cola, with zero evidence of Indian cola leadership.

### Grading Guidance
- **Interpretation ambiguity check:** The question was examined for materially different readings (e.g., leader = market share vs brand perception vs global leader sold in India; leader = current vs historical; leader = among all soft drinks). All honor the explicit constraints "cola drink" + "in India" + choice between the two names, and all evidence converges on Thums Up. No alternate reading has supporting evidence that would make Ricqlès the correct choice and produce a different grading outcome. Therefore collapse to one canonical answer per instruction; do not create branches for related facts (e.g., Sprite overall carbonated leader), evidence incompleteness, or detail level.
- **Full credit:** Answer selects Thums Up in any recognizable spelling, even if it additionally notes the premise error (Ricqlès is not a cola) or mentions Sprite's overall carbonated beverage lead.
- **No credit:** Answer selects Ricqlès, selects both/null, claims Ricqlès leads in India, or denies Thums Up is a cola/Indian brand.
- **Citaiton requirement:** Full-credit answer should be groundable to evidence that Thums Up is an Indian cola and India's largest cola brand; reference to Ricqlès's mint/French nature strengthens but is not required to score the binary choice if Thums Up is correctly chosen.

### Repeat 2 — clear — `5adce8c15542992c1e3a249d`

Question: Are Thyme and Cortaderia both aromatic plants?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: UNAMBIGUOUS - Single clear reading

Question wording inspection: "Are Thyme and Cortaderia both aromatic plants?"
- Entities: Thyme = common name for genus *Thymus* (Lamiaceae); Cortaderia = genus of pampas grasses (Poaceae)
- Relationship: conjunction "both ... are" requiring each entity satisfies attribute
- Requested attribute: aromatic plant = fragrant / possessing aroma / essential-oil producing herb in horticultural sense
- Scope: botanical/horticultural classification of living plants
- Answer type: binary Yes/No (with justification)

Ambiguity check performed: Alternative reading of "aromatic" as chemical sense (containing aromatic benzene-ring compounds) was considered. It was rejected as a structured branch because (a) it does not fit the plain horticultural wording "aromatic plant" as ordinarily applied to herbs, (b) no evidence describes living Cortaderia as aromatic in either sense, (c) it would not produce a materially different grading outcome (both readings return No for Cortaderia). No other reading honors all explicit constraints, has supporting evidence, and changes grading. Evidence conflicts about pyrolysis products are scope/granularity issues, not interpretation ambiguity. Ranked by fit to wording, the horticultural fragrant-plant reading is primary.

### Canonical Answer

**No.** Thyme and Cortaderia are not both aromatic plants.

* **Thyme (*Thymus*) IS aromatic:** described as aromatic herbs / shrubs, highly aromatic leaves, usually aromatic leaves, cultivated for fragrant leaves.
* **Cortaderia (pampas grass, *Cortaderia selloana* and allies) IS NOT aromatic / IS NOT fragrant:** explicitly listed as Non-Fragrant / Fragrant: No / not known for fragrance.

### Acceptable Variants / Acceptance Set

A grader should accept any semantic equivalent of the canonical No that correctly distinguishes the two genera:

* No / No, not both / Only one is
* Thyme is aromatic but Cortaderia is not (non-aromatic / non-fragrant / odorless / not known for fragrance)
* *Thymus* = aromatic herb (Lamiaceae), *Cortaderia* = ornamental grass (Poaceae) valued for plumes, not aroma
* Naming variants acceptable: common thyme / *Thymus vulgaris* / thyme for Thyme; pampas grass / *Cortaderia selloana* / *Cortaderia* spp. for Cortaderia, as long as aromatic distinction is preserved

A grader should NOT accept:
* Yes, both are aromatic
* Cortaderia is aromatic / fragrant / aromatic plant
* Both are herbs / both are fragrant plants

### Necessary Caveats

* References to Cortaderia biomass pyrolysis producing "aromatic hydrocarbons" or "polycyclic aromatic hydrocarbons" describe chemical products of thermal decomposition, not the living plant being fragrant/aromatic in the horticultural sense, and do not make Cortaderia an aromatic plant.
* Infraspecific variation in Thyme scent strength (e.g., "strength of scent varies according to season and habitat") does not negate the genus-level classification as aromatic.
* Cortaderia is consistently described by visual, not olfactory, impact when not fragrant, which is not a conflict but confirms the classification.

### Evidence

**Thyme is aromatic:**

* thyme, any species of the genus Thymus, aromatic herbs or shrubby plants of the family Labiatae (mint family)
* The genus *Thymus* (nat. ord. Labiatæ) comprises a number of fragrant aromatic undershrubs
* The genus *Thymus* contains about 350 species of aromatic perennial herbaceous plants and subshrubs in the family Lamiaceae
* The leaves are evergreen in most species, arranged in opposite pairs, oval, entire, and small, 4-20 millimetres long, and usually aromatic
* *Thymus* is cultivated for its fragrant leaves and used as a culinary herb
* *Thymus vulgaris* ... forming a woody, evergreen subshrub with small and highly aromatic grey-green leaves
* *Thymus vulgaris* (thyme) is an aromatic shrub from the Mediterranean

**Cortaderia is not aromatic / not fragrant:**

* In the genus *Cortaderia* are a couple of dozen species of grasses - establishes entity
* Are members of the Cortaderia plant genus fragrant? *Cortaderia* species are not known for their fragrance, but their visual impact in a landscape is significant
* Cortaderia selloana | Fragrance | Non-Fragrant
* Dwarf Pampas Grass | Cortaderia selloana 'Pumila' ... Fragrance: Non-fragrant
* Cortaderia selloana ... Fragrant | No
* Fragrant: No for Cortaderia selloana 'Pumila'
* Fragrant: No for Cortaderia selloana 'Pink Feather'

Each URL support:
* 4622017329586534810 supports Thyme aromatic definition
* 3477650383603913364 supports genus Thymus aromatic classification
* 522839166327572725 supports Cortaderia not known for fragrance
* 2873601223178016643 supports Cortaderia taxonomy and Non-Fragrant listing
* 2733980965377451887 supports repeated Fragrant: No listings for Cortaderia cultivars

### Confidence

High. Multiple concordant primary, institutional (RHS, Missouri Botanical Garden), commercial horticultural databases, and encyclopedic sources unanimously classify Thymus as aromatic/fragrant and Cortaderia as Non-Fragrant / Fragrant: No / not known for fragrance. No credible source classifies living Cortaderia as an aromatic plant.

### Grading Guidance

* **Full credit:** Answer is No (or equivalent) and correctly attributes aromatic to Thyme only, and non-aromatic/non-fragrant/odorless to Cortaderia.
* **Partial credit (if scale allows):** Bare "No" without elaboration is acceptable for binary question, but preferred answer includes the distinction. An answer stating "No, only Thyme is aromatic" without naming Cortaderia's grass family is still full credit if fragrance distinction is correct.
* **No credit:** Any answer stating Yes, both are aromatic, or stating Cortaderia is aromatic/fragrant.
* Do not penalize synonymous phrasing: fragrant, highly aromatic, aromatic herb, aromatic shrub, aromatic leaves for Thyme; non-fragrant, non-aromatic, odorless, not fragrant, not known for fragrance, no fragrance for Cortaderia. Do not penalize use of scientific synonyms (*Thymus*, *Thymus vulgaris*, *Cortaderia selloana*, pampas grass).

### Repeat 2 — broad-but-single — `5ae336ef5542991a06ce995d`

Question: What profession do both Phil Alden Robinson and David Nixon share?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR

Canonical answer: **Film director** — Both Phil Alden Robinson and David Nixon (the American film director) share the profession of film director / filmmaker [also described as film director and film producer / director].

* Phil Alden Robinson occupation is Film director, screenwriter and described as an American film director and screenwriter; also described as a writer and director
* David Nixon (director) is an American film director and film producer

Acceptable variants:
* director
* film director / movie director
* filmmaker / film-maker
* film director and film producer / director-producer
All denote directing films. "Producer" without "director/filmmaker" is a granularity variant of the same film-production profession but is not the most direct answer; acceptance depends on grading leniency - see Grading guidance. "Screenwriter/writer" alone is not acceptable for the canonical disambiguation because David Nixon (director) is not primarily described as a screenwriter.

Necessary caveats:
* **Name collision - not interpretation ambiguity:** "David Nixon" is ambiguous on its own. Wikipedia disambiguation lists: David Nixon (magician) (1919–1978), English magician and television personality - David Nixon (choreographer) - David Nixon (footballer) - David Nixon (director), American film director and producer - David Nixon (American football). The added constraint "share a profession with Phil Alden Robinson" disambiguates to the director, because only that David Nixon has a primary occupation matching Robinson's.
* **Conflicting secondary evidence - not a separate interpretation:** David Nixon (magician) has infobox Occupations | Magician and television presenter, not film director. Secondary sources list that same David Porter Nixon was a writer and actor and is credited as Writer on David Nixon's Magic Box. This creates a potential evidence conflict that could be misread as "both are writers," but it relies on a secondary writing credit rather than primary profession field and does not satisfy the "shared primary profession" constraint as well as the director match.
* **Scope/granularity - not separate interpretations:** "Director" vs "film director" vs "film director and producer" are nested levels of detail of the same profession, not materially different answers. Phil Alden Robinson's and David Nixon's shared film-production roles include producer credits, but director is the most specific common primary occupation supported by infobox/lead definitions.

Evidence:
* Phil Alden Robinson primary occupation = Film director, screenwriter - supports canonical answer attribute for Robinson
* Phil Alden Robinson is an American film director and screenwriter whose films include Field of Dreams - supports Robinson is a film director
* Phil Alden Robinson is a writer and director - supports writer/director as profession
* David Nixon (director) is an American film director and film producer - supports canonical shared profession = film director (and producer as granularity variant)
* Disambiguation shows multiple David Nixons exist - supports why name appears ambiguous but shared-profession wording resolves it
* David Nixon (magician) occupations are Magician and television presenter - supports magician does NOT share film director profession as primary occupation
* David Nixon (magician) was a writer and actor; listed as Writer on his show - supports existence of secondary writer credit that could be confused as shared profession but is evidence-conflict caveat, not a valid primary-profession interpretation

Confidence: HIGH for canonical answer "film director" under the constraint-satisfying disambiguation to David Nixon (director). MEDIUM-HIGH that the question setter intended that disambiguation; if a grader strictly requires explicit disambiguation, confidence that "writer" should be rejected as correct is HIGH.

Grading guidance:
* Accept any phrasing that asserts the shared profession is directing films: "director," "film director," "movie director," "filmmaker," "film director and producer." Do not require both director and producer.
* Reject answers that state only: magician, television presenter, choreographer, footballer/linebacker, or other non-film-directing professions, as they indicate selection of wrong David Nixon.
* Reject "writer" / "screenwriter" alone as the shared profession. Only the magician has a secondary writer credit; the American film director - the only David Nixon who shares Robinson's primary profession - is defined as film director/producer, not writer. "Writer" should not be counted correct unless the response explicitly justifies the magician disambiguation with evidence and the grader intentionally allows secondary-credit answers (then mark as non-canonical).
* Accepting "producer" alone: Strict grading should require "director" (producer is a granularity variant). Lenient grading may accept "producer" or "film producer" because both have producer credits (Robinson is listed with Profession Writer, Director, Producer in secondary source and Nixon is a film producer), but canonical remains director.

### Repeat 2 — clear — `5ae6168f5542996de7b71b47`

Question: Adrian the Barbarian (later Olaff the Madlander) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: UNAMBIGUOUS

Canonical Answer: **1989**

Adrian the Barbarian (later reprinted as Olaff the Madlander) was first introduced in the comic *The Beezer* in 1989 - specifically *Beezer* No. 1727 dated 18 February 1989. It later continued through the final standalone *Beezer* (No. 1809, 15 September 1990) and into the merged *Beezer and Topper*.

### Interpretation Check
The question wording was inspected for materially different readings:

* **Requested entity:** *Adrian the Barbarian (later Olaff the Madlander)* - parenthetical clarifies identity, not a separate strip
* **Requested attribute:** year *first introduced*
* **Requested scope/location:** *in the comic The Beezer, and later the merged Beezer and Topper*
* **Answer type:** calendar year

Considered alternative readings and rejected as not honoring all explicit constraints:

1. **Year introduced as *Olaff the Madlander* in *The Beano* (2008)** - fails location constraint. The question explicitly locates the strip in *The Beezer* and then the merged *Beezer and Topper*. The *Olaff* title was only used for reprints in *The Beano* from July 2008, not in *The Beezer*. This is a related fact about reprints, not a valid reading of the question.

2. **Year first introduced in the merged *Beezer and Topper* (1990)** - fails *first introduced* constraint. The strip existed in *The Beezer* before the merger in 1990. This reading isolates only part of the location description and would change the grading outcome, but it does not honor the full phrase *The Beezer, and later the merged Beezer and Topper*.

Neither alternative both honors all constraints and has supporting evidence for a different year for the *Beezer* introduction. The question has one clear reading. Distinction is interpretation ambiguity vs. evidence conflict/scope granularity: the 1990 and 2008 dates are evidence conflicts / different scopes (end date, merger date, reprint date), not genuine alternative interpretations.

### Acceptable Variants
All forms that clearly assert **1989** as the first introduction year in *The Beezer*:
* 1989
* 1989 (Beezer No. 1727)
* 18 February 1989 / 18th Feb 1989 / February 1989

Variations in specificity that include 1989 (e.g., "1989-1990, debut 1989, continued to 1990/1993") are acceptable if 1989 is given as the start.

Not acceptable:
* 1990 alone (end date in *The Beezer* / start of *Beezer and Topper* run)
* 1990-1993 (period of merged *Beezer and Topper*)
* 2008 (first reprint as *Olaff the Madlander* in *The Beano* issue 3441)
* 1988

### Necessary Caveats
* The table in *List of Beezer comic strips* shows `| *Adrian the Barbarian* | Sid Burgon | 1990 |` where 1990 is the **End Date** column, not the start date. The entry is grouped under the `## 1989` section heading - this is a presentation detail, not conflicting evidence for a 1990 introduction.
* The strip ran through *Beezer* No. 1809 (15/9/90) and then for most of the 153 issues of the combined *Beezer & Topper*. The 1990 date marks continuation/merger, not first introduction.
* The *Olaff the Madlander* reprint in *The Beano* began in July 2008, issue 3441. This is a reprint date, not the original introduction.
* Sid Burgon is consistently dated to beginning work for DC Thomson in 1989 drawing *Adrian the Barbarian*, corroborating the 1989 start.

### Evidence
* Central definition - first introduced year: "***Olaff the Madlander*** (initially published as ***Adrian the Barbarian***) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989."
* Same statement replicated: "**Adrian the Barbarian** (later **Olaff the Madlander**) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989."
* Precise issue and date - specialist forum citing issue numbers: "Adrian the Barbarian aka Olaff the Badlander first appeared in the Beezer in No 1727 dated 18th Feb 1989 and ran on through to the comics final issue No 1809 (15/9/90) and then went on to appear in the combined Beezer & Topper for most of its 153 issues."
* Artist chronology corroboration - Burgon began drawing the strip in 1989: "Burgon began to draw for DC Thomson in 1989 drawing a revival of Biffo the Bear in The Beano and Adrian the Barbarian for The Beezer"
* Reprint vs original distinction: "In July 2008, as of issue 3441, *Adrian the Barbarian* was reprinted in The Beano under the new title of *Olaff the Madlander*."

### Confidence
**HIGH** - Multiple independent sources converge on the same year (general reference works, artist biography, and issue-specific specialist citation with exact number and cover date). No credible source supports a different introduction year for the original *Beezer* appearance. Conflicting 1990/2008 dates are clearly attributable to different scopes (end/merger date, reprint date).

### Grading Guidance
* **Award credit** for any answer where the year 1989 is clearly identified as the first introduction in *The Beezer* (including more precise dates containing 1989 such as 18 Feb 1989, Feb 1989, Beezer 1727).
* **Do not award credit** for answers giving only 1990, 1993, or 2008 without stating 1989. Treat 1990 as incorrect if presented as the introduction year.
* **Range answers:** If answer states "1989-1990" or "1989, continued to 1990/1993" with 1989 as start year, mark correct. If answer states "Introduced in 1990" and lists 1989 only as ancillary, mark incorrect.
* Minor spelling variations (*Adrian/Olaff*, *Beezer/Topper*, *Madlander/Mad Landers*) do not affect correctness if the year is 1989.

### Repeat 3 — broad-but-single — `56e2419acd28a01900c67c2a`

Question: What structures can be viewed as both works of art and symbols of certain cultures?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: UNAMBIGUOUS

### Canonical Answer
**Architectural works / buildings and other built structures - in their material form - specifically monuments and iconic buildings.**

Architecture is defined as the study and practice of designing structures, especially habitable ones, utilizing civil engineering techniques but considered a visual art  and as both the process and product of sketching, conceiving, planning, designing, and constructing buildings or other structures .

The central definitional sentence that directly answers the question is:

> *Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements.* /

Supporting characterization:

> *Architecture utilizes civil engineering techniques, but is considered a visual art.*

> *The practice of architecture itself, which began in the prehistoric era, has been used as a way of expressing culture by civilizations on all seven continents. For this reason, architecture is considered to be a form of art.*

For that reason architecture / built structures satisfy both constraints in the question simultaneously:
* **works of art** - architecture is a visual art / form of art
* **symbols of certain cultures** - architectural works are perceived as cultural symbols and means of cultural expression; historical civilizations are identified by surviving architecture

### Acceptable Variants / Acceptance Set
Question requests a **category** of structures; answers at category-level or at correct example-level are acceptable. Grader should accept any of:

**Category-level (canonical):**
* architecture / architectural works / buildings / built structures / the built environment
* monuments / memorials / temples / palaces / religious buildings / civic buildings / iconic buildings / physical structures designed and constructed by humans

**Example-level (acceptable as instantiation of category - single example sufficient, exhaustive list not required):**
* Pyramids of Giza / Parthenon / Colosseum / Pantheon / Taj Mahal / Eiffel Tower / Statue of Liberty / Great Wall of China / Sydney Opera House / Angkor Wat / St. Peter's Basilica / Machu Picchu, etc. - any specific famous building/monument that is documented as both an artistic work and a symbol of its culture/civilization.

Acceptance set = {architecture, architectural works, buildings (in material form), monuments, iconic buildings/structures} ∪ {any specific instance thereof that is a human-made physical structure recognized as art and cultural symbol}

### Necessary Caveats
* **Scope/granularity, not ambiguity:** The question is open-ended in scope ("certain cultures" is generic). Providing the general category vs. providing one or more correct examples are nested levels of detail, not materially different interpretations. Do not penalize for choosing either level.
* **No closed list:** The definition is categorical ("often perceived as"). There is no single exhaustive list of structures; any valid member of the category fulfills the question.
* **Physical structures only:** "Structures" here must be human-made physical structures (buildings/other structures). Abstract structures (social structures, organizational structures, narrative structures, syntactic structures) fail both explicit constraints - they are not described in sources as being "considered a visual art" and "perceived as cultural symbols and as works of art" in this definitional sense.
* **Wording match:** Sources phrase as "often perceived as" — the question wording "can be viewed as" is the hedged equivalent and should be treated as identical.

### Evidence
* **Definitional sentence linking architectural works to both cultural symbols and works of art:**
  * https://en.wikipedia.org/wiki/Architectural supports: "Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements."
  * Same on Wikipedia main page: "Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art."
* **Architecture is designing structures / buildings or other structures:**
  * https://en.wikipedia.org/wiki/Architectural supports: "Architecture is the study and practice of designing structures, especially habitable ones." and "It is both the process and the product of sketching, conceiving, planning, designing, and constructing buildings or other structures."
* **Architecture as visual art / form of art and cultural expression:**
  * https://en.wikipedia.org/wiki/Architectural supports: "It utilizes civil engineering techniques, but is considered a visual art."
  * https://en.wikipedia.org/wiki/Architecture via search snippet supports: "The practice of architecture itself, which began in the prehistoric era, has been used as a way of expressing culture by civilizations on all seven continents. For this reason, architecture is considered to be a form of art."

### Confidence
High. The definitional sentence is replicated verbatim across Wikipedia and its licensed mirrors (nanowerk.com, darxstar, etc.) and is consistent with humanities treatment of architecture as culturally significant art. No conflicting source was found for this definitional claim.

### Grading Guidance
* **Accept** if candidate states or entails: buildings / architectural works / architecture / built structures / monuments / temples / palaces / iconic structures etc., described as being both art/artistic works and cultural symbols/representatives of a culture/civilization.
  * Accept paraphrases: "buildings designed as art that symbolize a culture," "structures that are artistic achievements and cultural icons," etc.
  * Accept single or multiple correct examples (Eiffel Tower → France, Pyramids → Egypt, Colosseum → Rome) even without stating the general category, provided example is correctly linked to both art and cultural symbol properties.
* **Do not require** exact phrasing "often perceived as cultural symbols and as works of art" — semantic equivalence is sufficient.
* **Do not require** exhaustive enumeration of cultures or structures; one valid category or one valid example is sufficient for full credit.
* **Reject** only if answer identifies non-physical or non-artistic structures with no connection to architecture/art (e.g., "social hierarchies," "protein structures," "grammatical structures") and does not explain how they are viewed as works of art and cultural symbols.
* **Partial credit not applicable** — categorical question is binary, but accept either category-level or example-level as fully correct per Acceptance set above.

### Repeat 3 — broad-but-single — `56f71740711bf01900a44932`

Question: Bilateral treaties are concluded between how many states or entities?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CANONICAL - Single Clear Reading (Not Ambiguous)

Fresh wording inspection: Question "Bilateral treaties are concluded between how many states or entities?" asks for cardinal number of treaty-making subjects on which bilateral status depends. Entities, relationship, attribute, and answer type are explicit: quantity of states/entities that conclude a bilateral treaty.

Interpretation-ambiguity check performed:
- Considered whether "how many" could mean (a) number of nominal signatory parties vs (b) number of sides/groups. Both readings honor all explicit constraints but converge on same grading outcome - bilateral = 2 sides/entities, with the Switzerland-EU 17-party example still classified as bilateral because parties are divided into two groups. This is scope/granularity nuance, not a materially different answer requiring separate branches.
- Considered whether "states or entities" could include international organizations. Evidence confirms subjects can be sovereign states or international organizations, but count remains two.
- No second reading satisfies all constraints, has supporting evidence, AND would produce a materially different grading outcome. Conflicting evidence / detail-level variations are not interpretation ambiguity.

Ranked by fit to wording: "two states or entities / two parties / two sides / two subjects" is strongest fit; no competing interpretation warrants a structured branch. Collapsed to one canonical answer per instructions.

Canonical Answer: **Two (2)** - Bilateral treaties are concluded between two states or entities (two parties / two sides / two subjects of public international law).

Acceptable Variants / Acceptance Set (flattened - grader may accept any):
- "2" or "two"
- "two states" / "two entities" / "two sovereign states" / "two countries" / "two governments"
- "two parties" / "two sides" / "two groups" / "two subjects of (public) international law"
- "two state entities" / "two nations" / "two international organizations or one state and one international organization" as long as count = 2
- Answers adding correct nuance: "two sides, even if formally more than two signatories grouped into two sides (e.g., Switzerland-EU 17 parties divided into two groups)"
- Do not accept: "1", "3 or more", "several", "many", "multiple without specifying two", "multilateral = three or more" as answer to bilateral

Necessary Caveats (evidence conflicts as caveats, not separate interpretations):
- Formal party-count nuance: A bilateral treaty can list more than two formal parties but remains bilateral because parties are divided into two groups. Example: each bilateral treaty between Switzerland and the EU has seventeen parties divided into Swiss ("on the one part") and EU and its member states ("on the other part"); similarly described.
- Contrast category: Multilateral treaties involve more than two parties / several countries and any agreement with more than two parties is otherwise a multilateral treaty, confirming bilateral = two.
- Entity-type scope: Entities can be sovereign states or international organizations established by treaty; these two parties can be two nations, or two international organizations, or one nation and one international organization.

Evidence (what each citation supports):
- Bilateral treaties are concluded between two states or entities - Supports core count = two states/entities.
- Treaties may be bilateral (between two countries) or multilateral (involving more than two countries) - Supports definition contrast and count = two vs >2.
- A bilateral treaty (also called a bipartite treaty) is a treaty strictly between two subjects of public international law, generally either sovereign states or international organisations... Any agreement with more than two parties is a multilateral treaty - Supports strict two-subject definition and multilateral boundary.
- These two parties can be two nations, or two international organizations, or one nation and one international organization - Supports acceptable entity types while maintaining count = two.
- Bilateral treaties are concluded between two states or entities. It is possible however for a bilateral treaty to have more than two parties... The parties are divided into two groups, the Swiss ("on the one part") and the EU... and parallel detail - Supports caveat about two-sided nature despite multi-party formal list.

Confidence: High - Convergent primary definitional sources (encyclopedic treaty article, dedicated bilateral treaty article, treaty typology pages) unanimously define bilateral as two states/entities/parties/sides, with consistent contrast to multilateral (>2) and no contradictory evidence on the core count.

Grading Guidance:
- Award full credit for any response stating "two" / "2" with reference to states, entities, parties, sides, subjects, countries, nations, sovereign states, or international organizations.
- Accept numeric "2", word "two", or phrasing "a pair of" if clearly denotes two treaty-making sides.
- Do not penalize elaborations that correctly note entities may be states or international organizations or that formal signatory count may exceed two but is grouped into two sides, provided the answer retains "bilateral = two sides".
- Reject answers giving any number other than two, describing bilateral as unilateral (1) or multilateral (3+, several, many), or leaving count unspecified.

### Repeat 3 — ambiguous — `57260b4389a1e219009ac188`

Question: What is an example of something that serves a function other than protection?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR

Question wording has one clear reading. Inspection of exact wording before searching:

* **Wording:** "What is an example of something that serves a function other than protection?"
* **Entities:** underspecified `something` - no domain restriction
* **Relationship:** `serves a function`
* **Requested attribute:** function `other than protection` - i.e., a verifiable function that is not classified as protection
* **Scope/answer type:** open-ended example - any concrete object, organ, structure, system, or artifact with at least one non-protective function

Considered whether materially different readings exist that each satisfy all explicit constraints, have supporting evidence, and would change grading outcome. Two candidate readings were evaluated:

1. `something that also protects but additionally serves another function` (e.g., skin protects but also thermoregulates)
2. `something whose function is solely non-protective` (e.g., eyeglasses correct vision)

Both satisfy the literal constraint `exists function served and function ≠ protection`. They are not distinct interpretations of the wording, but nested levels of detail / scope within the same interpretation. Creating branches for biological vs artifactual vs social domains would be scope/granularity, not interpretation ambiguity. No reading honors a different explicit constraint, has independent supporting evidence, and would produce a materially different accept/reject decision. Evidence conflicts (e.g., whether eyeglasses are described as protective in some contexts) are evidence granularity, not wording ambiguity.

Therefore collapse to one canonical answer.

### Canonical Answer

**Any specific example with a verifiable non-protective function satisfies the question. Canonical examples:**

* **Skin / integumentary system** - serves **thermoregulation, sensation (sensing stimuli/touch/pain/heat), and synthesis of vitamin D**, in addition to protection, also summarized as `The skin plays important roles in protection, sensing stimuli, thermoregulation, and vitamin D synthesis`
* **Clothing** - serves **social and cultural functions other than protection, including adornment, modesty, social status, gender/cultural/religious differentiation and personal style/fashion** and `Clothing performs a range of social and cultural functions, such as individual, occupational, gender differentiation, and social status. In many societies, norms about clothing reflect standards of modesty, religion, gender, and social status. Clothing may also function as an adornment`
* **Eyeglasses / corrective lenses** - textbook example of `those that serve a function other than protection (eyeglasses)`, defined as `lenses that are intended to fully correct visual acuity or eliminate refractive error` - i.e., vision correction, not protection

The minimal sufficient answer is: **eyeglasses (vision correction), or skin (thermoregulation/sensation/vitamin D synthesis), or clothing (adornment/modesty/status).**

### Acceptable Variants / Acceptance Set

Grader must accept **any** concrete entity + function pair where the function is verifiably not protection, even if domain differs from canonical. Flattened acceptance set includes but is not limited to:

* Biological: skin - thermoregulation via sweating/vasodilation, sensation via Meissner/Pacinian corpuscles and sensory nerves detecting touch, pressure, pain, temperature, vitamin D synthesis in epidermis when exposed to UV, preventing dehydration/electrolyte balance, hair - sensory/thermoregulation, sweat glands - excretion, cell membrane - selective transport/gating nutrients/waste, feathers/fur - keeping warm, wings - flight, large ears - hearing
* Artifactual/cultural: eyeglasses - vision correction, jewelry - adornment, clothing - adornment/modesty/status/style, government - order/conflict resolution
* Phrasing variants accepted: "sweating to cool body," "sensing touch/pain/heat," "making vitamin D," "decorative/adornment/fashion," "modesty covering," "correcting refractive error/nearsightedness," "selective transport" all acceptable for the respective entity.

Reject only if response describes **only** a protective function (e.g., "skin protects from germs" alone, "clothing keeps out cold" alone) with no non-protective function, or describes a property (e.g., color) that is not a function, or invents a function without evidence.

### Necessary Caveats

* Question is highly underspecified and domain-open; no course-specific context is provided in the wording. If intended context (e.g., integumentary-system quiz) was omitted, grading must not restrict to that domain - any domain-valid example counts. Previous hypothesis record's multiple domain examples are not separate interpretations; they are acceptable variants under the single reading.
* Eyeglasses nuance: one source lists eyeglasses as the example of `function other than protection`, while a mirrored passage notes corrective eyeglasses have protective functions in some classifications. This reflects classification granularity (vision correction is primary function; UV/impact protection is secondary for some eyewear), not wording ambiguity. Vision correction remains a valid non-protective function per definition.
* Protection vs thermoregulation/sensation/vitamin D are treated as distinct functional categories in anatomy sources; do not generalize beyond evidence.

### Evidence

* **Skin serves non-protective functions:** `The skin and accessory structures perform a variety of essential functions, such as protecting the body from invasion by microorganisms, chemicals, and other environmental factors; preventing dehydration; acting as a sensory organ; modulating body temperature and electrolyte balance; and synthesizing vitamin D`; restated as `The skin plays important roles in protection, sensing stimuli, thermoregulation, and vitamin D synthesis`; same in open-education source `The skin plays important roles in protection, sensing stimuli, thermoregulation, and vitamin D synthesis`
* **Thermoregulation detail:** `The integumentary system helps regulate body temperature... sweat glands... secrete water, salt, and other substances to cool the body when it becomes warm`
* **Sensation detail:** `The skin acts as a sense organ because the epidermis, dermis, and the hypodermis contain specialized sensory nerve structures that detect touch, surface temperature, and pain`
* **Vitamin D detail:** `The epidermal layer of human skin synthesizes the inactive form of vitamin D when exposed to UV radiation`
* **Direct textbook example of "function other than protection":** `those that serve a function other than protection (eyeglasses), are normally considered accessories rather than clothing`
* **Eyeglasses defined as vision correction:** `the term "ordinary eyeglasses or contact lenses" means lenses that are intended to fully correct visual acuity or eliminate refractive error`
* **Clothing non-protective functions:** `Clothing serves protective, physiological, social, and cultural functions by providing comfort, thermal regulation, protection against environmental hazards... Clothing also expresses social status, gender, cultural identity, religious affiliation, modesty, and personal style, and plays a central role in fashion`; `Clothing performs a range of social and cultural functions, such as individual, occupational, gender differentiation, and social status. In many societies, norms about clothing reflect standards of modesty, religion, gender, and social status. Clothing may also function as an adornment`

What each URL supports:
* 352081050813872231 - skin's non-protective functions (sensory, thermoregulation, vitamin D) and that they are distinct from protection
* 437681845138697272 - independent corroboration of same skin functions
* 2220080684882181609 - canonical phrasing "function other than protection (eyeglasses)"
* 3636499966696582141 - eyeglasses function = vision correction / eliminating refractive error
* 7801404648426482053 - clothing adornment/modesty/status as non-protective functions

### Confidence

**High** that the canonical examples (eyeglasses - vision correction; skin - thermoregulation/sensation/vitamin D; clothing - adornment/modesty) correctly satisfy the single clear reading and are strongly supported by primary academic/institutional sources. Medium that the acceptance set is exhaustive - the open-ended wording permits an unbounded number of correct examples, so any verifiable non-protective function not listed should still be accepted.

### Grading Guidance

* **Accept** if response names a concrete entity and states a verifiable non-protective function (see Acceptance Set for synonyms). Do not require exact string match to "eyeglasses" or "skin" - any valid pair counts.
* **Accept** even if response also lists a protective function alongside the non-protective one.
* **Reject** only if response fails to provide a non-protective function, describes protection alone, or invents an unsupported function.
* **Do not** penalize for domain choice (biology, clothing, tools, etc.) or level of detail (e.g., "thermoregulation" vs "sweating and vasodilation to cool body" both acceptable).
* No branching required - treat all domain variants as same interpretation for scoring.

### Repeat 3 — ambiguous — `57302972b2c2fd140056899e`

Question: What could be considered a trait of Rome's military campaigns?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR - Single Interpretation

The wording `What could be considered a trait of Rome's military campaigns?` has one clear reading under exact wording/entity/relationship/attribute/scope/answer-type checks:

* **Entity:** Ancient Rome (Roman Republic / Roman Empire) - not a modern toponym; context military campaigns disambiguates.
* **Relationship:** trait/characteristic `of` (attributive)
* **Requested attribute:** a distinguishing feature of the conduct, organization or purpose of campaigns
* **Scope:** indefinite / existential (`could be considered` / `a trait` = one instance suffices)
* **Answer type:** descriptive noun phrase / short statement

There is no second reading that honors all explicit constraints and would produce a materially different grading outcome. Different correct traits (e.g., expansionist vs disciplined vs engineering) are **scope/granularity variants within the same interpretation**, not separate interpretations. No branches for evidence conflicts or levels of detail are warranted.

### Canonical Answer

Any one well-supported trait of Rome's / Roman military campaigns is correct. Central, well-evidenced traits include:

**Rome's military campaigns were characteristically territorial-expansionist (and, secondly, civil wars), offensive and persistent, preferring direct engagement in open battle and overcoming fortified positions via superior military engineering, with systematic fortified-camp construction, and were executed by a highly disciplined, well-organized, well-trained standing heavy-infantry force.**

* **Purpose/type:** Served triple purpose of securing borders, exploiting peripheral areas via tribute, maintaining internal order, and the majority were characterised by one of two types - the first is the territorial **expansionist campaign**, normally begun as counter-offensive, which allowed growth from a small town to 55 million; the second is civil war.
* **Operational preference:** Preponderance exhibit preference for **direct engagement in open battle** and, where necessary, overcoming fortified positions **via military engineering**. Adept at building fortified camps for protection but reluctant to sit in camp awaiting battle and a history of seeking open battle.
* **Engineering / logistics:** At end of day's march typically established strong field camp **castra with palisade and deep ditch** in 2-5 hours; **no other ancient army persisted so long in systematic camp construction**. Rome had **no equal in siege warfare** due to engineering ability to destroy, bypass or scale walls; dedicated corps of engineers (*immunes*) with specialized technical skills; characterized by **discipline, organization, and adaptability** and **engineering prowess** and strategic innovation and discipline.
* **Discipline / organization:** **Extremely disciplined, kept standing between conflicts, made mostly of heavy infantry**; also described as **well-trained and disciplined** and **noted for rigor of its training and discipline**.
* **Persistence:** History of campaigning is a history of **obstinate persistence overcoming appalling losses**, despite not being invincible and producing incompetents who led to catastrophic defeats.

Any single trait from this set satisfies `a trait`.

### Acceptable Variants / Acceptance Set

Grader should accept **any one** paraphrase / synonym containing the core meaning below (case-insensitive, no need to list all):

* Expansionist / imperialist / territorial expansion / aggressive expansion overseas / counter-offensive that subjugates large territory
* Civil war as characteristic campaign type
* Securing borders / exploiting periphery via tribute / maintaining internal order
* Disciplined / strictly disciplined / brutal discipline / disciplina militaris / rigorous training
* Well-organized / highly organized / systematic / structured / legion-cohort-century system / self-sufficient legions / logistically proficient
* Engineering-focused / military engineering / siege warfare / fortification / camp (castra) building / bridges/roads/siege engines (ballistae/onagers)
* Preference for direct / open / pitched battle / seeking battle rather than awaiting in camp
* Offensive / proactive / aggressive campaigning (Republic and early Empire)
* Persistent / resilient / obstinate / ability to absorb catastrophic losses and continue
* Professional / standing army / well-trained / drilled

**Unacceptable:** opposite claims (e.g., isolationist, pacifist, undisciplined, unorganized, avoiding open battle in Republic/early Empire, lacking engineering/siege capability).

### Necessary Caveats

* **Period variation is not interpretation ambiguity.** Early Rome raised troops annually reactively without standing army; post-Augustan reforms created standing professional force. Preference for offensive open battle preponderantly describes Republic and early Empire; late Empire shifted toward limited direct engagement and stratagem. Both are still within same trait framework.
* **Typical, not absolute.** Sources emphasize preponderance/majority, not universality; Roman armies were not invincible and produced incompetent leaders leading to catastrophic defeats.
* **Granularity:** Answers at different specificity levels (e.g., "disciplined" vs "built fortified camps in 2-5 hours" vs "no equal in siege warfare") are all valid single-trait answers, not separate interpretations.

### Evidence

| Claim | Evidence |
|-------|----------|
| Triple purpose: securing borders, exploiting periphery via tribute, maintaining internal order |  |
| Majority characterised by territorial expansionist campaign (counter-offensive) + growth to 55m |  |
| Second type is civil war |  |
| Preponderance prefer direct open battle, overcoming forts via military engineering |  |
| Adept at fortified camps, reluctance to sit awaiting battle, seeking open battle |  |
| Daily castra with palisade/ditch in 2-5 hrs; no other army persisted so long in systematic camp construction |  |
| No equal in siege warfare due to engineering |  |
| Dedicated engineers, technical skills, immunes |  |
| Characterized by discipline, organization, adaptability, engineering prowess |  |
| Strategic innovation and discipline, emphasis on engineering/logistics, siege engines at Alesia/Masada |  |
| Extremely disciplined, standing between conflicts, heavy infantry |  |
| Well-trained and disciplined |  |
| Noted for rigor of training and discipline |  |
| Obstinate persistence overcoming appalling losses |  |

### Confidence

**High.** Convergence across high-reputation sources (Campaign History of the Roman Military, Strategy of the Roman Military, Gallic Wars, secondary analyses) on same trait cluster with strong direct phrasing. Minor chronological nuance noted above does not lower confidence in core traits.

### Grading Guidance

* **Accept** any answer that correctly states **one or more** traits in the Acceptance Set, regardless of phrasing, brevity, or level of detail (e.g., "disciplined", "good at building camps", "expansionist", "loved open battles and sieges", "never gave up even after huge losses" all correct).
* **Do not require** enumeration of both campaign types; expansionist alone is sufficient, civil war alone is sufficient.
* **Do not require** citation or period qualification; accept trait without specifying Republic vs Empire.
* **Reject** only if answer asserts opposite/contradictory trait as its sole claim or describes Rome's campaigns as characteristically lacking that trait (e.g., "Rome avoided warfare and was undisciplined").
* **Mixed answers:** If answer lists multiple traits and at least one is valid, mark correct; penalize only if valid trait is explicitly retracted or negated.
* **No branching penalty:** Do not treat differently worded valid traits as different interpretations requiring separate keys.

### Repeat 3 — ambiguous — `5ab9e3d25542994dbf019889`

Question: What type of vegetation does Lythrum and Erythrina have in common?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR

### Canonical Answer

Both genera share **hydric / wetland vegetation** — classified as **hygrophytic / hydrophytic / helophytic / wetland / swamp - marsh - riverine vegetation**.

In precise terms they are **hygro-hydrophytes / emergent wetland helophytes**: plants of **swamps, marshes, other wet places and poorly-drained littoral / riverine sites**.

* ***Lythrum***: **Habitat: Swamps, marshes, other wet places**; **Invades many wetland and marsh habitats including freshwater wet meadows, tidal and non-tidal marshes, river and stream banks, pond edges**; described as an **Eurasian perennial hydrophyte that has become naturalized in wetlands** and listed in **Examples of genera containing hygrophytes - *Lythrum***.

* ***Erythrina***: Overlap via wetland-adapted members, most clearly *Erythrina fusca*:
  - Vernacular **swamp immortelle**
  - **It seems to prefer littoral locations with badly drained soils like swamps and stream banks and upland riverine marshes. In lowlying freshwater swamps *Erythrina fusca* attains huge dimensions and sometimes develops almost pure stands**
  - At genus level **Erythrina has been recorded from primary and secondary habitats including lowland rain forest, Eucalyptus savannah, coastal and swamp areas, from sea level to 2000 m**
  - *Erythrina speciosa* **is a Neotropical tree that grows mainly in moist habitats**

A hygrophyte is a plant that **inhabits moist areas** growing on **wet soils** while hydrophytes live within aquatic environments; both are adapted to **low-oxygen (anaerobic) environments where there is extended periods of water saturation or flooding** — the defining condition of hydrophytic / hygrophytic / helophytic wetland vegetation.

### Acceptable Variants / Acceptance Set

All convey the same hydric formation at different granularity; accept any:

* hydrophytic vegetation / hydrophytes / aquatic-emergent vegetation
* hygrophytic vegetation / hygrophytes
* helophytic / helophyte / emergent marsh vegetation
* wetland vegetation / wetland plants / obligate wetland (OBL) vegetation / freshwater wetlands
* swamp vegetation / swamp forest / swamp immortelle woods / low-lying freshwater swamps
* marsh / tidal and non-tidal marsh / wet meadows / riparian - riverbank - stream-bank - littoral badly-drained / hydric-soil / waterlogged / moist-habitat vegetation

Accept combinations (e.g., "wetland helophytes", "hygro-hydrophytic swamp-marsh vegetation").

Do **not** accept:
* xerophytic / dry / desert / upland / purely mesophytic alone
* purely taxonomic (both dicots, both Fabaceae/Lythraceae) or purely life-form without hydric qualifier (e.g., "both are herbs" — *Erythrina* is woody tree/shrub)
* ornamental/cultivated alone

### Why No Ambiguity Branch

Wording has one clear reading: ecological habitat/formation type shared by members of both genera. Potential second readings (Raunkiaer life-form, geographic range, use) either fail to honour the explicit constraint *have in common* with supporting evidence, or collapse to scope/granularity variants of the same hydric vegetation (hygrophyte vs hydrophyte vs helophyte vs swamp vs marsh). No second materially different interpretation is both evidence-supported and changes grading outcome.

### Necessary Caveats

* **Genus breadth - evidence conflict retained as caveat, not separate interpretation:** *Lythrum* is consistently OBL / wetland. *Erythrina* (c. 120 spp. throughout tropics/subtropics) is ecologically broad — many species occur in dry forest, savanna, and upland woodland — the commonality is via wetland species, principally *E. fusca*, *E. speciosa*, *E. crista-galli* and coastal/swamp records. The question asks for the type they *have in common*, not the exclusive habitat of either genus.
* **Terminological granularity:** hydrophyte, hygrophyte, helophyte, wetland, swamp, marsh, riparian are nested descriptors for saturated/periodic-anaerobic substrates, not competing interpretations. Hygrophyte (wet soils) and hydrophyte (grows in water) are used interchangeably for *Lythrum* in literature; swamp immortelle terminology reflects helophytic emergent habit.
* **Wetland indicator status** is regional; OBL status is well-documented for *Lythrum salicaria* and not universally assigned at genus level for *Erythrina*.

### Evidence

| Claim | Citation |
|---|---|
| Lythrum habitat swamps, marshes, other wet places |  |
| Lythrum invades many wetland/marsh habitats incl. wet meadows, tidal/non-tidal marshes, river/stream banks |  |
| Lythrum salicaria = Eurasian perennial hydrophyte naturalized in wetlands |  |
| Lythrum listed as hygrophyte genus |  |
| Hygrophyte definition inhabits moist areas, least resistant to drought |  |
| Hygrophyte vs hydrophyte both adapted to low-oxygen saturated soils |  |
| Erythrina fusca = swamp immortelle |  |
| Erythrina fusca prefers littoral badly drained soils like swamps/stream banks, attains huge dimensions in low-lying freshwater swamps |  |
| Genus Erythrina recorded from coastal and swamp areas |  |
| Erythrina speciosa mainly in moist habitats |  |

### Confidence

**High** for the wetland / hydric / hygrophytic-hydrophytic / helophytic answer. Habitat documentation from floras, PROSEA and wetland literature is convergent for *Lythrum* and for wetland *Erythrina* species. Residual uncertainty limited to genus-level heterogeneity in *Erythrina*, which does not undermine the existence of a shared hydric vegetation type.

### Grading Guidance

* Accept any hydric synonym above; do not penalize specificity (swamp vs marsh vs wetland vs hydrophyte vs hygrophyte vs helophyte) — all represent the same formation.
* Require mention of wet/hydric/aquatic/wetland/moist/swamp/marsh/riverine character; generic "both are perennials/trees" without hydric qualifier is insufficient.
* Species-level instantiation (e.g., "*Erythrina fusca* swamp forests and *Lythrum salicaria* marshes — both wetland helophytes") is acceptable as evidence of genus-level commonality.
* Do not create a second interpretation branch for mesophytic, xerophytic, or purely morphological answers; treat those as incorrect under the single canonical answer.

### Repeat 3 — clear — `5ac4e4595542996feb3fe968`

Question: Which cola drink is the leader in India, Thums Up or Ricqlès?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR - UNAMBIGUOUS

### Canonical Answer
**Thums Up** is the leader in India.

Thums Up is the largest cola brand in India, described as the carbonated beverage brand owned by Coca-Cola Co.'s Indian unit with Rs5,000 crore in annual sales, that has dominated India's cola market in the past four decades and that today "rules the cola scene in India", noted as the iconic Cola brand of the country and the largest cola brand by far.

Ricqlès is not a cola leader in India. Ricqlès is a peppermint spirit and a French soft drink created in 1838 in Lyon, France, originally a medicinal mint spirit, categorized as a mint-flavoured soft drink - not a cola, and with no evidence of cola-market leadership in India.

Question wording review: The phrasing "Which cola drink is the leader in India, Thums Up or Ricqlès?" presupposes both are cola drinks. Only Thums Up satisfies the explicit constraint "cola drink" with supporting evidence. No alternative interpretation honors all explicit constraints ("cola drink" + "in India" + choice between the two named entities) while also having supporting evidence and producing a different grading outcome. Potential distinctions such as global vs. India scope or historical vs. current leadership do not create ambiguity: the question explicitly scopes to India, and Thums Up is documented as both the long-term dominant cola and the current largest cola brand. Changing phrasing level of detail (e.g., "largest by sales" vs "largest by retail value" vs "rules the cola scene") is granularity, not interpretation ambiguity.

### Acceptable Variants
Accept any answer that clearly selects **Thums Up** as the answer, including:
- "Thums Up" with variant spellings "Thums Up" / "Thumbs Up" [original name was Thumbs Up] when unambiguously referring to the Indian cola brand manufactured by The Coca-Cola Company
- Phrasing equivalents: Thums Up is India's No. 1 / largest / dominant / leading cola brand; Thums Up leads the Indian cola market by sales/volume/share
- Answers that additionally correct the presupposition: "Ricqlès is not a cola (it is a French mint/peppermint soft drink/spirit), so Thums Up is the only cola and is the leader" are fully correct and equivalent to selecting Thums Up

Not acceptable: Selecting Ricqlès as the cola leader in India, stating both are co-leaders, stating neither is leader, or stating Ricqlès is a cola.

### Necessary Caveats
- **False presupposition in question:** Ricqlès is not a cola drink. Evidence consistently classifies it as a peppermint spirit and French soft drink and as a mint-flavoured soft drink. It should not be described as a cola.
- **Category distinction:** Within the broader carbonated beverages market in India, sources note Sprite tops the overall carbonated beverage market while Thums Up remains the largest cola brand specifically. This does not conflict with Thums Up being the cola leader; it clarifies "cola" vs. "carbonated beverage" categories.
- **No evidence gap creates ambiguity:** Absence of evidence for Ricqlès cola leadership in India is consistent with its French mint-drink identity; this is not conflicting evidence requiring a branch, but a factual absence to be noted as a caveat.

### Evidence
- Thums Up is a cola/Cola brand: "the iconic Cola brand of the country"; 40-year-old cola brand
- Thums Up is the largest cola brand in India with Rs5,000 crore in annual sales and aiming for $1 billion; repeated in detailed article
- Thums Up has dominated India's cola market in the past four decades and today rules the cola scene in India
- Thums Up is the largest cola brand by far and the largest cola brand according to Euromonitor, while Sprite tops the broader carbonated beverage market
- Ricqlès identity: peppermint spirit and a French soft drink, liquor Alcool de Menthe de Ricqlès created in 1838 in Lyon, originally a medicinal mint spirit; also described as peppermint spirit and French soft drink with 1954 soft drink introduction
- Ricqlès category: listed as mint-flavoured soft drink, not cola

### Confidence
High. Multiple independent reputable sources (Livemint reporting on Coca-Cola India and Euromonitor data, Atlas Obscura summary, brand history sources, Wikipedia-derived encyclopedic entries for both drinks) converge on Thums Up as India's largest/dominant cola brand and on Ricqlès as a French mint/peppermint drink with no cola classification or Indian cola leadership.

### Grading Guidance
- **Full credit:** Answer selects Thums Up as the leader/largest cola in India. Correction of the presupposition (stating Ricqlès is a French mint/peppermint soft drink, not a cola) is correct and should be rewarded but is not required if Thums Up is clearly chosen.
- **Full credit equivalents:** Any phrasing that conveys Thums Up exceeds Coca-Cola and Pepsi in the Indian cola segment, rules the cola scene, or is India's No. 1 cola brand.
- **No credit:** Selecting Ricqlès as the cola leader in India; stating Ricqlès is a cola; claiming both are equal cola leaders; or denying Thums Up's cola leadership.
- **Partial credit not applicable:** This is a binary forced-choice factual question with one clear answer; do not award partial credit for selecting Ricqlès but adding context.
- **Evidence conflicts vs. ambiguity:** Do not treat the Sprite-overall-carbonated-leadership fact as a reason to mark Thums Up incorrect for cola leadership. That distinction is a scope/granularity note, not a conflicting answer.

**What each URL supports:**
-  and  support Thums Up is the largest cola brand in India
-  supports dominance over four decades
-  supports Thums Up rules the cola scene in India
-  supports Thums Up is an iconic Cola brand and largest cola brand by far
-  and  support Ricqlès is a peppermint/French soft drink originating in France, not a cola
-  supports Ricqlès is classified as mint-flavoured soft drink

### Repeat 3 — clear — `5adce8c15542992c1e3a249d`

Question: Are Thyme and Cortaderia both aromatic plants?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: UNAMBIGUOUS - Single Canonical Answer

### Interpretation Ambiguity Check
Question wording: "Are Thyme and Cortaderia both aromatic plants?" was inspected for multiple materially different readings.

* Entities: Thyme = common name for genus *Thymus* (best-known species *Thymus vulgaris*); *Cortaderia* = genus *Cortaderia* (best-known species *Cortaderia selloana*, pampas grass). Both genus-level and species-level readings refer to same taxa.
* Relationship / attribute: "aromatic plants" = plants characterized by fragrance/essential oils and cultivated/used as aromatic herbs in horticultural/botanical sense.
* Scope: "both" = logical AND - answer Yes only if both taxa satisfy attribute.
* Answer type: Boolean Yes/No.

Potential alternative reading (chemical sense of "aromatic" = containing benzene-ring compounds, e.g., pyrolysis bio-oils containing aromatic hydrocarbons) was considered and rejected. It does not honor the explicit constraint "aromatic plants" as used in botany/horticulture, lacks supporting evidence that *Cortaderia* is classified as an aromatic plant in that sense, and would not produce a materially different defensible grading outcome. No other reading honors all constraints, has supporting evidence, and changes outcome.

Result: **One clear reading.** No interpretation branches required. Evidence conflicts are treated as caveats, not ambiguities.

### Canonical Answer
**No.** Thyme and *Cortaderia* are not both aromatic plants.

* Thyme (*Thymus* spp., especially *Thymus vulgaris*) **is** an aromatic plant.
* *Cortaderia* spp. (*Cortaderia selloana* - pampas grass) **is not** classified as an aromatic plant; it is a large ornamental perennial grass in family Poaceae.

### Acceptable Variants / Acceptance Set
Grader should accept any phrasing that expresses the conjunction false with correct attribution:

* No / False / Not both
* Only Thyme is aromatic; *Cortaderia* is not
* *Thymus* is aromatic, *Cortaderia* is not aromatic
* *Thymus vulgaris* is highly aromatic; *Cortaderia selloana* is a non-aromatic ornamental/invasive grass
* Thyme = aromatic herb in Lamiaceae (mint family); pampas grass = Poaceae grass, not listed as aromatic

**Unacceptable:**
* Yes / True / Both are aromatic
* *Cortaderia* is aromatic / is an aromatic herb
* Neither is aromatic / Thyme is not aromatic

### Necessary Caveats
* "Thyme" refers to genus *Thymus* containing about 350 species, native to Mediterranean/Europe, North Africa and Turkey, in family Lamiaceae. All members are generally aromatic, though strength of scent varies by species/cultivar, season and habitat. The reference species *Thymus vulgaris* is canonically "highly aromatic."
* "Cortaderia" refers to genus *Cortaderia* (about two dozen species from South America, New Zealand, New Guinea), most commonly *Cortaderia selloana* (pampas grass), family Poaceae. Standard floras and horticultural descriptions characterize it as robust evergreen grasses forming large tufts with narrow rough-edged leaves and silvery/pink plumes, noted as ornamental and invasive, without any aromatic classification. Absence of "aromatic" descriptor in authoritative descriptions is evidence of non-aromatic status. Incidental vegetative smell when crushed does not make it an aromatic plant in culinary/medicinal/horticultural usage.
* Evidence conflict caveat: Pyrolysis studies note *Cortaderia selloana* biomass can yield bio-oils containing monoaromatic and polycyclic aromatic hydrocarbons upon thermal decomposition, but this is a chemical analysis of combustion products, not a botanical classification of the living plant as an aromatic plant.

### Evidence

**Thyme is an aromatic plant:**

* The genus *Thymus* contains about 350 species of aromatic perennial herbaceous plants and subshrubs in the family Lamiaceae
* *Thymus vulgaris* (common thyme) forming a woody, evergreen subshrub with small and highly aromatic grey-green leaves
* **THYME.** The genus *Thymus* (nat. ord. *Labiatæ*) comprises a number of fragrant aromatic undershrubs
* thyme, any species of the genus *Thymus*, aromatic herbs or shrubby plants of the family Labiatae (mint family)
* Thyme is a genus of about 350 species of aromatic perennial herbs in the family Lamiaceae
* Thyme belongs to mint family Lamiaceae and is known for aroma and flavour of its dried leaves and flowering tops
* Thyme is genus of about 350 species of aromatic perennial herbs and sub-shrubs to 40 cm tall, in Family Lamiaceae (via search result)
* Thyme known for aroma and flavour

**Cortaderia is not an aromatic plant; it is a Poaceae grass:**

* *Cortaderia selloana* is a species of flowering plant in the family Poaceae. It is referred to by the common name pampas grass
* Cortaderia selloana Family: Poaceae - Clump forming grass with arching leaves
* Cortaderia selloana 'Pumila' Type: Ornamental grass Family: Poaceae
* [Cal-IPC] Invasive: Cortaderia selloana (pampasgrass) is a large perennial grass (family Poaceae)
* Cortaderia are robust evergreen grasses forming large, compact tufts of narrowly linear, rough-edged leaves, with erect stems bearing dense silvery or pink-tinged flower panicles which are excellent for drying - standard genus description with no aromatic descriptor

### Confidence
**High.** *Thymus* aromatic status is uniformly attested across encyclopedic, taxonomic, and horticultural sources. *Cortaderia* is uniformly described as a large perennial ornamental/invasive grass in Poaceae with no reputable source classifying it as an aromatic plant. The distinction is taxonomic (Lamiaceae aromatic herbs vs Poaceae grasses) and descriptive.

### Grading Guidance
For prompt "Are Thyme and Cortaderia both aromatic plants?" grade as:

* **CORRECT** if answer is No/False and/or correctly attributes aromatic status only to Thyme (*Thymus*), explicitly stating *Cortaderia* (pampas grass) is not aromatic / is an ornamental grass in Poaceae.
* **INCORRECT** if answer is Yes/True, claims both are aromatic, claims *Cortaderia* is aromatic, or claims Thyme is not aromatic.
* Accept minor phrasing variations and inclusion of binomials (*Thymus vulgaris*, *Cortaderia selloana*) if correct.
* Do not require mention of pyrolysis chemistry; answers focusing on botanical fragrance classification are sufficient.
* No interpretation branches need to be considered for grading.

### Repeat 3 — broad-but-single — `5ae336ef5542991a06ce995d`

Question: What profession do both Phil Alden Robinson and David Nixon share?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: CLEAR

### Canonical answer
Both **Phil Alden Robinson and David Nixon (director) share the profession film director** - also described as director / filmmaker.

Phil Alden Robinson is identified with Occupation: Film director, screenwriterand as an American film director and screenwriter whose films include *Field of Dreams*.

David Nixon (director) is identified as an American film director and film producer. He is the director and producer of *Letters to God*.

### Acceptable variants
For grading, accept any semantically equivalent formulation of the shared core profession at this granularity:
* director
* film director / movie director
* filmmaker / film-maker
* film director and producer / director-producer (includes the shared director element)

Do not require both "screenwriter" and "director" - director alone is sufficient. Generic "director" is acceptable when context is film/television directing.

### Necessary caveats
* Name ambiguity exists for "David Nixon." Wikipedia lists: David Nixon (magician), David Nixon (choreographer), David Nixon (footballer), David Nixon (director), David Nixon (American football).
* Only David Nixon (director) satisfies the explicit constraint "profession do *both* ... share" with Phil Alden Robinson. The other persons do not share a profession with Robinson:
  - David Nixon (magician) Occupations: Magician and television presenter- no overlap with Robinson.
  - Choreographer, footballers likewise have no shared profession.
* Scope/granularity note: "film director," "director," and "filmmaker" are nested levels of detail for the same profession, not materially different interpretations. "Screenwriter" or "producer" alone are not shared: Robinson is director/screenwriter, Nixon is director/producer; the only common element is directing.

### Evidence
* **Phil Alden Robinson - Occupation film director, screenwriter:** Infobox line Occupation | Film director, screenwriter | Years active- supports Robinson's profession is film director.
* **Phil Alden Robinson - description as American film director and screenwriter:** Text **Phil Alden Robinson (born March 1, 1950) is an American film director and screenwriter**- supports Robinson profession.
* **David Nixon (director) - American film director and film producer:** Lead sentence **David Nixon is an American film director and film producer. He is the director and producer of *Letters to God***- supports Nixon profession and shared overlap (director).
* **David Nixon ambiguity - disambiguation list:** **David Nixon may refer to: - David Nixon (magician)... - David Nixon (director), American film director and producer**- supports that name has multiple referents and identifies which referent shares profession.
* **David Nixon (magician) occupation:** **Occupations | Magician and television presenter**- supports evidence conflict is not an interpretation branch (magician does not share profession).

### Confidence
High. Both professions are directly stated in lead/infobox of Wikipedia pages consistent across sources. Name ambiguity is explicitly documented but does not create a second valid shared-profession reading.

### Grading guidance
* **Accept:** Any answer stating directing films is the shared profession (film director, director, movie director, filmmaker). Accept "director and producer" if director is included.
* **Reject:** Answers stating magician / television personality / choreographer / footballer / American football player, or "screenwriter" alone, or "producer" alone - no evidence of shared profession on those terms. Do not credit an answer that identifies David Nixon as the magician to claim a shared profession.
* **Granularity:** Do not penalize specificity variation between "director" and "film director."

### Repeat 3 — clear — `5ae6168f5542996de7b71b47`

Question: Adrian the Barbarian (later Olaff the Madlander) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Status: UNAMBIGUOUS - Single Clear Reading

Question has one materially coherent reading under fresh inspection. No branching required.

### Exact Wording Check

* **Entities:** Strip = *Adrian the Barbarian* (later reprint title *Olaff the Madlander*); Publication = *The Beezer* (1956-1990) and later the merged *Beezer and Topper* (1990-1993)
* **Relationship:** "was a comic strip in the comic X, and later the merged Y" - describes continuity of the same strip across two titles
* **Requested Attribute:** "first introduced in which year?" - year of original debut
* **Scope:** Constrained explicitly to *The Beezer* / *Beezer and Topper* lineage
* **Answer Type:** Year (4-digit; more precise date is narrower granularity of same answer, not a different interpretation)

Interpretation branches considered and **removed**:

* **2008 = first as *Olaff the Madlander* in *The Beano*:** Fails all explicit constraints - wrong publication (*The Beano* not *The Beezer*), wrong scope (reprint vs original), and contradicts "first introduced" + "(later Olaff)" which marks Olaff as secondary. Evidence for 2008 exists but answers a different question.
* **1990 = first in merged *Beezer and Topper*:** Fails "first introduced" - strip already running in *The Beezer* before the merger in 1990; 1990 is the *End Date* in *The Beezer* before continuation, not the debut.
* **1990 from table cell:** This is a conflicting-evidence / granularity issue (misreading End Date column as Start Year), not an interpretation ambiguity. Retained as caveat only.

Ranking by fit to wording, 1989 is the only interpretation honoring all explicit constraints.

### Canonical Answer

**1989**

*Adrian the Barbarian* (initially published as *Adrian the Barbarian*, later reprinted as *Olaff the Madlander*) was first introduced in *The Beezer* in **1989**. The precise debut was *Beezer* No. 1727 dated **18 February 1989**, after which it continued through the final *Beezer* issue No.1809 (15 Sept 1990) and then into the combined *Beezer and Topper*.

### Acceptable Variants / Acceptance Set

**Accept:**
* 1989
* February 1989 / Feb 1989 / Feb 18 1989
* 18 February 1989 / 18th Feb 1989 / 18/02/1989 / No.1727 1989

**Reject:**
* 1990 (end date in *The Beezer* / start of *Beezer and Topper* merger year, not debut)
* 2008 (reprint as *Olaff the Madlander* in *The Beano*)
* 1988 (date of unsuccessful Nigel Parkinson trial, never published), 1993, or any year range that does not identify 1989 as the first introduction

Granularity note: Day/month specificity is a more precise nested version of the same 1989 answer and is fully acceptable; graders should not require day-level precision nor penalize it.

### Necessary Caveats

* **Evidence conflict is scope/column misreading, not ambiguity:** The chronological *List of Beezer comic strips* is organized by *year of release date* and in *chronological order*. The entry for *Adrian the Barbarian* appears under the heading **## 1989** with the next column showing **1990** — that column is **End Date** (1990 = final *Beezer* year before the 1990 merger). Mirror/scraper tables that list only `Sid Burgon | 1990` without the 1989 heading cause apparent conflict.
* **Merger vs debut:** *The Beezer* was renamed into *The Beezer and Topper* on merging with *The Topper* in **1990**. Entries showing *Adrian the Barbarian | Sid Burgon | 1 | 153 | The Beezer | Reprinted in The Beano in 2008...* refer to its run in the merged title (issues 1-153), not a 1990 debut.
* **Reprint year is distinct scope:** *Adrian the Barbarian* was reprinted in *The Beano* under the new title *Olaff the Madlander* in **July 2008, as of issue 3441**. This is not "first introduced in *The Beezer*" and does not satisfy the question constraints.

### Evidence

* **Primary definition - first introduced in 1989 in *The Beezer* (later merged *Beezer and Topper*):** *Olaff the Madlander* (initially published as *Adrian the Barbarian*) was a comic strip in the comic *The Beezer*, and later the merged Beezer and Topper, first introduced in 1989 — *Olaff the Madlander Wikipedia page* supports central year, publication, and later-name relationship.
* **Chronological placement under 1989:** The *List of Beezer comic strips* groups strips under **## 1989** with *Adrian the Barbarian | Sid Burgon | 1990 | Was the comic's cover star. Continued in the *Beezer and Topper*...* — supports 1989 as release year and explains 1990 as End Date.
* **Precise debut issue/date:** *Adrian the Barbarian aka Olaff... first appeared in the Beezer in No 1727 dated 18th Feb 1989 and ran on through to the comics final issue No 1809 (15/9/90) and then went on to appear in the combined Beezer & Topper...* — *ComicsUK forum (Kashgar post, 18 Jul 2008)* supports exact day-month-year and continuation, corroborated by the list's 1990 End Date.
* **Corroborating artist chronology:** *began to draw for DC Thomson in 1989 drawing a revival of Biffo the Bear in The Beano and Adrian the Barbarian for The Beezer* — *Sid Burgon context* supports 1989 start.
* **Merger date disambiguation:** *It was, then, renamed into *The Beezer and Topper* on merging with *The Topper* in 1990* — supports why 1990 is not the debut year.
* **Reprint scope disambiguation:** *In July 2008, as of issue 3441, *Adrian the Barbarian* was reprinted in *The Beano* under the new title of *Olaff the Madlander** — supports exclusion of 2008.

### Confidence

**High** - Convergent primary and secondary sources agree on 1989 (exact 18 Feb 1989). The only apparent conflict (1990) is a table-column misinterpretation resolvable by heading structure.

### Grading Guidance

* Award **full credit** for any response that unambiguously states **1989** as the first appearance year, including more precise forms (Feb 1989, 18 Feb 1989, No.1727 1989).
* Treat **1990, 1990-1993, or 2008** as incorrect; they answer End Date / merger-run / reprint scope, not "first introduced in *The Beezer*."
* Do not penalize additional correct detail (artist Sid Burgon, issue numbers, cover-star note, continuation in *Beezer and Topper*).
* A range answer (e.g., "1989-1990") is incorrect unless it explicitly identifies **1989** as the debut year; require 1989 to be clearly asserted.

## Pilot boundary

This is a nine-question, three-repeat pilot for prompt behavior. No full-dataset run was started.

Private cache: `artifacts/muse_ambiguity_pilot_2026-08-22.json`

## Manual review and recommendation

### What held up

- All 27 two-call runs completed at stage 2; there were no hard failures or follow-up fallbacks.
- The three clear controls stayed single-answer records. Adrian's debut year was consistently 1989, Thums Up was consistently selected, and Thyme/Cortaderia was consistently answered as not both aromatic.
- The three broad-but-single controls also stayed single-reading records: architectural works/monuments, two parties for bilateral treaties, and film director.
- The open-ended questions were handled sensibly as one reading with a broad acceptance set. The model did not need to manufacture multiple interpretations just because many answers could be valid.

### Gap found

The prompt did not reliably produce multiple interpretation branches when they were needed. None of the nine ambiguous-category outputs emitted two structured branches. Q10 was the decisive failure mode:

1. Repeat 1 selected the physical-form/life-form reading: herbaceous perennial/herb.
2. Repeat 2 selected the same general life-form reading with different wording.
3. Repeat 3 selected the habitat reading: wetland/hydric vegetation.

This is not simply harmless wording variation. Independent references support both semantic dimensions: Kew describes *Lythrum* as annual or perennial herbs and *Erythrina* as usually trees/shrubs but rarely perennial herbs ([Kew Lythrum](https://powo.science.kew.org/taxon/urn%3Alsid%3Aipni.org%3Anames%3A30002863-2/general-information), [Kew Erythrina](https://powo.science.kew.org/taxon/urn%3Alsid%3Aipni.org%3Anames%3A30007957-2/general-information)); other authoritative sources support wetland habitat for *Lythrum* and wet/swamp habitat for *Erythrina* ([USGS Lythrum vegetation](https://www1.usgs.gov/csas/nvcs/unitDetails/899229), [Kew Erythrina fusca](https://powo.science.kew.org/taxon/urn%3Alsid%3Aipni.org%3Anames%3A494431-1/general-information), [PROSEA Erythrina fusca](https://prosea.prota4u.org/view.aspx?id=5382)). The correct reference behavior is therefore to preserve both supported readings, explain the dimension difference, and provide a flattened acceptance set—not silently choose one.

### Other quality signal

The branch prompt recorded 240 URL annotations, but 4/27 final answers had zero parsed URL citations despite having Search queries. The answers were still readable, but citation enforcement is less reliable in this variant than in the prior citation-focused canary. The pilot also produced deliberately long reference records; that is acceptable under the current requirement, but it increases the cost and makes stable structure more important.

### Go/no-go

**Do not run the full dataset yet.** Keep the two-call design, but strengthen the generic ambiguity instruction so that a fresh pass cannot silently replace one supported semantic dimension with another. Then rerun Q10 plus clear and broad controls before scaling. The current pilot is useful evidence, not a safe reference-generation prompt for the full set.
