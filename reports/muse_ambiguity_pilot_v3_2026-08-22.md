# Muse ambiguity-handling pilot

- Model: `muse-spark-1.2-contributor`
- Prompt variant: `reference_generic_branches_v3`
- Endpoint: Meta native Responses API with `web_search`
- Policy tested: two calls per item; no application Search or output-length cap
- Questions: 9 (3 ambiguous, 3 broad-but-single, 3 clear)
- Repeats per question: 3
- Expected records: 27
- Completed records: 27
- Hard failures: 0
- Stage-2 finals: 27
- Cache-only rerun items skipped without API calls: 0
- Search queries recorded: 364
- URL citations recorded: 182

The signals below are mechanical indicators for review; they are not a semantic quality judgment.

## Output signals

| Repeat | ID | Category | Status signal | Branch headings | Acceptance set | Citations |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | `56e2419acd28a01900c67c2a` | broad-but-single | SINGLE | 0 | no | 2 |
| 1 | `56f71740711bf01900a44932` | broad-but-single | SINGLE | 0 | no | 8 |
| 1 | `57260b4389a1e219009ac188` | ambiguous | SINGLE | 0 | no | 0 |
| 1 | `57302972b2c2fd140056899e` | ambiguous | SINGLE | 0 | yes | 8 |
| 1 | `5ab9e3d25542994dbf019889` | ambiguous | MULTIPLE | 2 | yes | 14 |
| 1 | `5ac4e4595542996feb3fe968` | clear | SINGLE | 0 | no | 16 |
| 1 | `5adce8c15542992c1e3a249d` | clear | SINGLE | 0 | no | 4 |
| 1 | `5ae336ef5542991a06ce995d` | broad-but-single | SINGLE | 0 | no | 5 |
| 1 | `5ae6168f5542996de7b71b47` | clear | SINGLE | 0 | no | 8 |
| 2 | `56e2419acd28a01900c67c2a` | broad-but-single | SINGLE | 0 | no | 4 |
| 2 | `56f71740711bf01900a44932` | broad-but-single | SINGLE | 0 | no | 13 |
| 2 | `57260b4389a1e219009ac188` | ambiguous | SINGLE | 0 | yes | 0 |
| 2 | `57302972b2c2fd140056899e` | ambiguous | SINGLE | 0 | yes | 19 |
| 2 | `5ab9e3d25542994dbf019889` | ambiguous | SINGLE | 0 | no | 4 |
| 2 | `5ac4e4595542996feb3fe968` | clear | SINGLE | 0 | no | 6 |
| 2 | `5adce8c15542992c1e3a249d` | clear | SINGLE | 0 | no | 11 |
| 2 | `5ae336ef5542991a06ce995d` | broad-but-single | SINGLE | 0 | no | 0 |
| 2 | `5ae6168f5542996de7b71b47` | clear | SINGLE | 0 | no | 1 |
| 3 | `56e2419acd28a01900c67c2a` | broad-but-single | SINGLE | 0 | yes | 8 |
| 3 | `56f71740711bf01900a44932` | broad-but-single | SINGLE | 0 | yes | 12 |
| 3 | `57260b4389a1e219009ac188` | ambiguous | SINGLE | 0 | yes | 0 |
| 3 | `57302972b2c2fd140056899e` | ambiguous | SINGLE | 0 | no | 11 |
| 3 | `5ab9e3d25542994dbf019889` | ambiguous | MULTIPLE | 0 | yes | 4 |
| 3 | `5ac4e4595542996feb3fe968` | clear | SINGLE | 0 | no | 5 |
| 3 | `5adce8c15542992c1e3a249d` | clear | SINGLE | 0 | no | 5 |
| 3 | `5ae336ef5542991a06ce995d` | broad-but-single | SINGLE | 0 | no | 7 |
| 3 | `5ae6168f5542996de7b71b47` | clear | SINGLE | 0 | no | 7 |

## Answers for manual review

### Repeat 1 — broad-but-single — `56e2419acd28a01900c67c2a`

Question: What structures can be viewed as both works of art and symbols of certain cultures?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: Before choosing SINGLE, every plausible semantic dimension of "structures" was tested with focused grounding. Dimension 1 = physical built structures (buildings, monuments, temples, etc.). Dimension 2 = abstract social/institutional structures. Dimension 3 = chemical/molecular structures. Only Dimension 1 satisfies all explicit constraints in the question — "can be viewed as both works of art and symbols of certain cultures" — with direct definitional evidence. Dimensions 2 and 3 returned no evidence that such structures are defined or consistently described as both works of art and cultural symbols in the sense queried; they would fail the "works of art + cultural symbol" conjunction. No second dimension produced a materially different answer that honors all constraints and is evidence-supported.

Status: SINGLE

Canonical answer: Architectural works — buildings and other physical structures, especially monuments and landmark architecture. Architecture is the study and practice of designing structures, especially habitable ones, and is considered a visual art / form of art, while architectural works in their material form are often perceived as both cultural symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements.

Representative instantiations include temples, churches, mosques, stupas, synagogues, palaces, forts, cathedrals, pagodas, pyramids, etc., and well-known examples such as the Parthenon, Egyptian Pyramids, Great Wall, Taj Mahal, Statue of Liberty, Eiffel Tower and similar iconic monuments that function as international cultural statements / symbols of a people's values, devotion, patriotism, power and civilization.

Acceptable variants:
- Category-level: "architecture," "architectural works," "buildings," "built structures," "monuments," "landmarks," "historic / sacred architecture," "physical structures" when explicitly tied to built form.
- Example-level: Any specific famous building/monument that is documented as both an artwork and a cultural symbol is acceptable as an instantiation, e.g., Parthenon, Pyramids of Giza, Great Wall of China, Taj Mahal, Statue of Liberty, Eiffel Tower, Angkor Wat, St. Basil's Cathedral, Sydney Opera House, etc., or generic types such as temples, churches, mosques, pagodas, palaces, forts.
- Phrasing variants: "works of architecture," "cultural heritage buildings," "public art monuments" acceptable if they denote material built form with both artistic and symbolic status.

Necessary caveats:
- Not every built structure qualifies automatically; the status requires both aesthetic/artistic recognition and cultural-symbolic recognition. Purely utilitarian construction without artistic or symbolic valorization does not meet the conjunction.
- Symbolic status is often retrospective: civilizations are identified with surviving achievements, and monuments accrue symbolic meaning over time as cultural statements.
- The question does not extend to abstract structures (social structures, organizational structures, sentence / narrative structures) or natural/chemical structures. Focused searches for "social structures works of art" and "chemical structures works of art" returned no definitional support for both criteria jointly; e.g., "social sculpture" is defined as human activity that strives to structure and shape society, not as sociological structures being works of art and cultural symbols, and chemical structures are described only as artistic motifs (e.g., "Chemical Still Life" weds chemical structures with objects), not as symbols of certain cultures.
- Evidence conflicts regarding which monuments are most iconic are scope/granularity differences, not interpretation ambiguity; all belong to same dimension.

Evidence:
- Architecture is the study and practice of designing structures, especially habitable ones. It utilizes civil engineering techniques, but is considered a visual art — supports that "structures" here = designed built structures and that they are art.
- Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art — directly satisfies both relations in the question.
- The practice of architecture has been used as a way of expressing culture by civilizations on all seven continents. For this reason, architecture is considered to be a form of art — supports art status and cultural-symbol function.
- Architecture can mean a general term to describe buildings and other physical structures — clarifies scope of "structures" as physical built form.
- As international cultural statements without words, symbols of a peoples values devotion, patriotism, power symbols of a civilization's grandeur, these monuments still fascinate... as does the Parthenon in Athens — supports monuments as cultural symbols.
- All cultures had some kind of sacred belief ... and they erected different types of structures, including temples, churches, mosques, or pagodas where they observed their religion. Globally, all of the designs of the buildings were an architectural inspiration, enhanced with some type of cultural art — supports breadth of types that meet both criteria.
- Sacral architecture is a religious architectural practice concerned with design and construction of places of worship or sacred or intentional space, such as churches, mosques, stupas, synagogues, and temples — supports specific exemplars.

Confidence: High. The core definitional sentence is replicated consistently across Wikipedia and multiple mirrors and directly answers the question's dual predicate. Supporting evidence for monument symbolism and sacred building typology is corroborated by encyclopedic and heritage sources.

Grading guidance:
- ACCEPT: Any answer that identifies the category as architectural / built / monumental structures with both artistic and cultural-symbolic status, with or without listing examples. Accept singular or plural, and accept specific iconic examples (Parthenon, Pyramids, Great Wall, Taj Mahal, Statue of Liberty, Eiffel Tower, etc.) or generic types (temples, churches, mosques, palaces, cathedrals, forts, bridges, towers) when accompanied by or implying the dual art + symbol claim.
- ACCEPT granularity variation: Broad category alone ("architectural works / buildings") is sufficient; listing one or more correct examples without stating the category also satisfies (instantiation is within dimension).
- REJECT: Answers limited to abstract structures without material built form — e.g., "social structures," "family structures," "chemical structures," "sentence structures" — as they lack evidence for the conjunctive definition and belong to unsearched-failed dimensions. Reject purely natural features (mountains, rivers) or portable artworks alone (paintings, novels) if no architectural/structural element is mentioned.
- No partial credit for mixed answers that conflate dimensions without identifying the physical built dimension.

### Repeat 1 — broad-but-single — `56f71740711bf01900a44932`

Question: Bilateral treaties are concluded between how many states or entities?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks "between how many states or entities" – an explicit quantitative count of parties/sides. Prior to searching, plausible semantic dimensions inventoried were: (1) number of sides/groups, (2) number of formal signatory entities in collective-party examples, (3) type of entity (state vs international organization), (4) functional form. Dimensions 3-4 do not answer "how many" and are ruled out by wording. Dimensions 1 and 2 were both tested with focused grounding searches ("bilateral treaty concluded between two states or entities definition" and "bilateral treaty seventeen parties two groups Switzerland EU"). Both dimensions converge to the same grading answer – two sides/groups – with the apparently conflicting "seventeen parties" evidence explicitly explained as still bilateral because parties are divided into two groups. No alternative dimension yields a distinct evidence-supported numeric answer that honors all explicit constraints and would change grading outcome; the >2 formal parties case is a nested caveat, not a separate interpretation.

Status: SINGLE

Canonical answer: Two – a bilateral treaty is concluded between two states or entities (two parties / two sides / two subjects of public international law).

Acceptable variants:
- "2" or "two"
- "two states" / "two entities" / "two parties" / "two sides" / "two groups" / "two subjects of public international law"
- "two states or entities" / "two countries" / "two sovereign states or international organizations"
- "two nations, or two international organizations, or one nation and one international organization" is acceptable as elaboration if the number two remains explicit
- Answers that add the nuance "two sides/groups, even though formally more than two signatories in collective-party cases" are acceptable

Necessary caveats:
- A bilateral treaty is definitionally between two sides, but may formally list more than two signatory parties when one side is a collective (e.g., EU). Sources explain this is still bilateral, not multilateral.
- Example: each bilateral treaty between Switzerland and the European Union has seventeen parties divided into two groups – the Swiss ("on the one part") and the EU and its member states ("on the other part"), and "The treaty establishes rights and obligations between the Swiss and the EU and the member states severally—it does not establish any rights and obligations amongst the EU and its member states."
- This does not support an answer of "17" or "more than two" for the general definition; "two" remains the correct count. Any answer claiming unilateral (1) or multilateral (3+ / several / many) for bilateral is incorrect.
- Entity type nuance does not change count: "These two parties can be two nations, or two international organizations, or one nation and one international organization."

Evidence:
- Core definition: "Bilateral treaties are concluded between two states or entities."
- Strict definition: "A bilateral treaty (also called a bipartite treaty) is a treaty strictly between two subjects of public international law, generally either sovereign states or international organisations established by treaty."
- Contrast with multilateral: "Treaties may be bilateral (between two countries) or multilateral (involving more than two countries)."
- Further contrast: "Treaties may be bilateral (two parties) or multilateral (between several parties)" and "A multilateral treaty is a treaty involving more than two parties, while a bilateral treaty involves an agreement between two parties."
- Rule for >2 parties: "Any agreement with more than two parties is a multilateral treaty."
- Caveat explaining the apparent exception but preserving two-sidedness: "It is possible for a bilateral treaty to have more than two parties; for example, each of the bilateral treaties between Switzerland and the European Union (EU) has seventeen parties. The parties are divided into two groups, the Swiss ("on the one part") and the EU and its member states ("on the other part")"; same in main treaty article lines

Confidence: High – definition is consistent across Wikipedia treaty articles, specialized bilateral-treaty article, Library of Congress research guide, and law guides, with explicit citations to the Vienna Convention on the Law of Treaties context. The only apparent conflict (17 parties) is consistently resolved by sources as two groups, not a competing definition.

Grading guidance:
- Grade as CORRECT any response stating the number is two, including phrasing with "states," "entities," "parties," "sides," "groups," or "subjects."
- Grade as CORRECT elaborations that note the two parties may be states or international organizations, per .
- Grade as CORRECT if answer adds the Switzerland-EU collective-party caveat but retains "two sides/groups" as the answer.
- Grade as INCORRECT any answer that states one, three or more, "several," "many," "unlimited," or equates bilateral with multilateral, or that answers "17" as the general definition without qualification that it is still two sides.
- Grade as INCORRECT answers that describe only treaty type/function (e.g., "contractual treaty") without providing the count of two.

### Repeat 1 — ambiguous — `57260b4389a1e219009ac188`

Question: What is an example of something that serves a function other than protection?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE

Status: SINGLE - One clear reading

Reason no branch: The question asks for any example of an entity that has a function not categorized as protection. Wording contains no term like type/kind/form/class/role that selects a semantic dimension (e.g., taxonomy vs habitat vs physical form). The placeholder "something" is explicitly domain-open, and the predicate "serves a function other than protection" has one stable meaning: has a documented role ≠ protection/defense. Focused grounding was performed for at least two plausible domains that could instantiate this predicate - (1) biological structures/systems and (2) material-culture artifacts/clothing taxonomy - and both returned evidence that satisfies the *same* predicate rather than creating mutually exclusive readings. The difference between e.g., skin/skeleton vs eyeglasses is scope/granularity (different valid examples under one reading), not interpretation ambiguity that would change whether an answer is correct. No wording constraint rules out one domain in favor of the other, and no evidence supports a second materially different reading that honors all constraints. Therefore the question collapses to one canonical answer type with many acceptable variants.

Canonical answer:
An example of something that serves a function other than protection is the human skin / integumentary system serving thermoregulation, sensation, or vitamin D synthesis; equivalently the skeletal system serving support, movement, mineral storage, or blood cell production; equivalently eyeglasses serving vision correction. All are documented functions distinct from protection.

* Core biological example: The skin plays important roles in protection, sensing stimuli, thermoregulation, and vitamin D synthesis. More fully, the skin and accessory structures perform a variety of essential functions, such as protecting the body from invasion by microorganisms, chemicals, and other environmental factors; preventing dehydration; acting as a sensory organ; modulating body temperature and electrolyte balance; and synthesizing vitamin D.
* The integumentary system serves many functions to include protection, thermoregulation, sensation, vitamin D synthesis.
* Skeletal system example: The skeletal system is the body system composed of bones and cartilage and performs the following critical functions: supports the body, facilitates movement, protects internal organs, produces blood cells, stores and releases minerals and fat. Also described as support and protection, facilitation of movement through muscle attachment, storage of minerals (primarily calcium and phosphorus), production of blood cells in red bone marrow.
* Artifact example satisfying same predicate: Articles ... or those that serve a function other than protection (eyeglasses), are normally considered accessories rather than clothing, repeated as those that serve a function other than protection (eyeglasses), are normally considered accessories rather than clothing.

Acceptable variants (any one suffices; list not exhaustive):
- Skin / integumentary system → thermoregulation / temperature regulation / sweating and vasodilation; sensation / sensory perception / touch, pain, pressure, temperature; vitamin D synthesis; excretion / electrolyte balance; preventing dehydration; immune surveillance
- Skeletal system / bones / bone marrow → support / structural framework; movement / leverage for muscles / facilitating movement; mineral storage (calcium, phosphorus, fat/triglycerides); hematopoiesis / blood cell production (red and white cells, platelets) in red marrow; endocrine functions
- Eyeglasses / spectacles / glasses / contact lenses → vision correction / correcting refractive error / improving visual acuity (explicitly a function other than protection)
- Other entities valid under same reading if source-supported non-protective function is given: e.g., hair → thermoregulation, sensory detection; lungs/diaphragm → gas exchange; nervous system → communication/transmission of information; government → order, conflict resolution, public services, etc. Any entity with a credible non-protective function counts.

Necessary caveats:
- The question is intentionally open-ended. It does not require enumerating all functions, only one verifiable function that is not protection.
- "Protection" includes defense against physical, chemical, biological, radiation, or mechanical harm. Answers that cite only a protective function (e.g., "skin as barrier against pathogens", "helmet protects head", "safety goggles protect eyes") do not satisfy "other than protection".
- Granularity: Accept both organ-level (skin) and system-level (integumentary system; skeletal system) and artifact-level (eyeglasses) answers; they are nested levels of detail, not separate interpretations.
- Evidence conflicts: None material. Different sources emphasize different non-protective functions (thermoregulation vs vitamin D vs support), but they are complementary, not contradictory.

Evidence:
- Skin plays important roles in protection, sensing stimuli, thermoregulation, and vitamin D synthesis — supports skin has non-protective functions distinct from protection.
- Skin and accessory structures perform functions including acting as sensory organ, modulating body temperature and electrolyte balance, synthesizing vitamin D — supports thermoregulation/sensation/vitamin D as examples.
- Integumentary system serves many functions to include protection, thermoregulation, sensation, vitamin D synthesis — supports multiple non-protective functions.
- Skeletal system performs supports the body, facilitates movement, protects internal organs, produces blood cells, stores and releases minerals and fat — supports non-protective functions for bones.
- Skeletal functions include support and protection, facilitation of movement, storage of minerals, production of blood cells in red bone marrow — supports same.
- Articles ... that serve a function other than protection (eyeglasses), are normally considered accessories rather than clothing — supports eyeglasses as canonical non-protective-function example.

Confidence: High (0.92) — canonical examples are supported by multiple textbook-level and definitional sources converging on the same predicate. Lower uncertainty only about which example a grader expects, but any of the listed variants satisfies the single reading.

Grading guidance:
- Accept any answer that names a specific entity (structure, organ, system, object, institution) and correctly identifies at least one documented function that is not protection. Require that the function be factually non-protective and not be defense/barrier/immune protection.
- Accept synonyms: thermoregulation = temperature regulation / cooling via sweat / vasodilation; sensation = sensory perception; support = structural framework; movement = locomotion / leverage; blood cell production = hematopoiesis; vision correction = correcting refractive error.
- Reject if: (1) only a protective function is listed, (2) entity does not exist or function is invented / unsupported (e.g., "skeleton digests food"), (3) answer is vague with no identifiable function ("something helps people").
- Do not penalize domain choice: biological and artefactual answers are equally valid under the single reading. Do not require multiple examples; one correct example is sufficient.
- For broad answers like "heart pumps blood" or "stomach digests food" — accept, as they are non-protective functions, even if not in the canonical list, provided they are factually correct.

### Repeat 1 — ambiguous — `57302972b2c2fd140056899e`

Question: What could be considered a trait of Rome's military campaigns?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The wording "What could be considered a trait of Rome's military campaigns?" uses the unqualified attribute term "trait" without selecting a semantic dimension. Plausible dimensions inventory was: (A) qualitative characteristic (discipline/organization), (B) typological kind (expansionist vs civil war), (C) functional purpose (securing borders/exploiting periphery/maintaining order), (D) tactical/technological method (preference for open battle, engineering/logistics). Focused grounding checks were performed for each dimension and each returned evidence-supported traits. However all dimensions satisfy the same single reading — "a characteristic feature of Rome's campaigns" — and produce answers that are nested levels of detail/scope within one interpretation rather than mutually exclusive readings that honor the constraints while yielding materially different grading outcomes. A student citing any evidence-supported trait from any tested dimension would be correct under the same question meaning. This is scope/granularity, not material interpretation ambiguity under the contract.

Status: SINGLE

Canonical answer: A trait that could be considered characteristic of Rome's military campaigns is that they were highly organized, disciplined, aggressive and expansionist, with a preference for direct engagement in open battle and the use of military engineering and logistics to sustain expansion. The majority of campaigns are described as falling into two types — territorial expansionist (normally begun as a counter-offensive in which each victory brought subjugation of large areas) and civil war — and as serving the triple purpose of securing borders, exploiting peripheral areas (e.g., imposing tribute), and maintaining internal order.

Acceptable variants / Acceptance set: Grader should accept any single evidence-supported phrasing equivalent to one or more of:
- disciplined / highly disciplined / restrained and disciplined / brutal discipline
- organized / well-structured / well-trained, well-equipped professional legions / famed for discipline, organisation, and innovation / characterized by organization, discipline, and strategic innovation
- aggressive / offensive / prosecuting offensive operations / brutal hand-to-hand combat, tactics were aggressive / fought with aggression, discipline, and courage
- territorial expansionist / expansionist / aggressive and expansionist / each victory brought subjugation of large areas, allowing growth from small town to 55 million
- civil war as a characteristic campaign type (second of two types)
- securing borders, exploiting peripheral areas through tribute, maintaining internal order (triple purpose)
- preference for direct engagement in open battle and overcoming fortified positions via military engineering; reluctance to sit in camp awaiting battle
- emphasis on engineering, logistics, technological innovation, roads, fortifications, siege warfare (ballistae/onagers, construction of forts/camps)

Necessary caveats:
- Traits are predominant tendencies across Republic and early Empire; later Empire shows more limited/defensive operational engagement. Not every campaign displayed every trait.
- Historiographic conflict on evaluation of brutality exists: scholarship up to mid-20th century saw warfare as restrained and disciplined, later work challenges that and argues violence was considerable, but evidence cannot conclusively demonstrate it was particularly brutal versus ordinary for period — this is evidence conflict, not interpretation ambiguity.
- "Could be considered" signals open set; no single trait is uniquely required.

Evidence:
- Triple purpose: "As with most ancient civilizations, Rome's military served the triple purpose of securing its borders, exploiting peripheral areas through measures such as imposing tribute on conquered peoples, and maintaining internal order."
- Two types characterising majority of campaigns: "majority of Rome's campaigns were characterised by one of two types. The first is the territorial expansionist campaign"
- Expansionist definition: "territorial expansionist campaign, normally begun as a counter-offensive, in which each victory brought subjugation of large areas of territory"
- Second type is civil war: "The second is the civil war, which plagued Rome from its foundation to its eventual demise."
- Characterized by organization, discipline, strategic innovation: "Roman warfare was characterized by its organization, discipline, and strategic innovation."
- Famed for discipline etc.: "The Roman army, famed for its discipline, organisation, and innovation in both weapons and tactics"
- Well-trained highly disciplined legions: "The Roman army, with its well-trained, well-equipped, and highly disciplined professional legions, was both feared and hated"
- Extremely disciplined: "The Roman army was extremely disciplined, kept standing between conflicts, and made mostly of heavy infantry"
- Aggressive tactics: "Ancient Roman warfare often involved brutal hand-to-hand combat. Tactics were aggressive."
- Fought with aggression etc.: "The ancient Romans fought with aggression, discipline, and courage."
- Preference for open battle via engineering: "preponderance of Roman campaigns exhibit a preference for direct engagement in open battle and, where necessary, the overcoming of fortified positions via military engineering."
- Reluctance to sit in camp: "history shows a reluctance to sit in the camp awaiting battle and a history of seeking open battle."
- Engineering/logistics supporting campaigns: "Roman emphasis on engineering and logistics supported sustained campaigns far from the city."
- Technological/logistical advantage: "Technological innovations in weaponry and superb logistics meant that Roman strategies in warfare became ever more varied"

Confidence: High that the canonical trait set (disciplined/organized, expansionist/offensive, open-battle preference, engineering/logistics, civil-war vs expansionist typology, triple purpose) correctly answers the question; Medium-high on completeness within open-ended "could be considered" scope.

Grading guidance:
- Accept any one trait from Acceptance set if accurately phrased; do not require listing all.
- Accept synonyms: structured, professional, standing army, systematic, adaptable, persistent, imperialist, militaristic, engineering-focused, road/fort building, siege capable.
- Reject answers that assert a trait not evidenced for Rome (e.g., primarily naval-avoidant of land battle as characteristic, or purely defensive isolationism) or that misattribute anachronistic concepts.
- If multiple traits listed, mark correct if at least one is in Acceptance set and none are factually false about Rome's campaigns.

### Repeat 1 — ambiguous — `5ab9e3d25542994dbf019889`

Question: What type of vegetation does Lythrum and Erythrina have in common?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 2, "grading_guidance_signal": true, "status_signal": "MULTIPLE"}`

Interpretation decision: MULTIPLE

Status: AMBIGUOUS

**Why ambiguity is material:** The unqualified phrase “type of vegetation” does not select a semantic dimension. In botanical usage it plausibly means (a) taxonomic kind (clade/class) and (b) growth-form / life-history kind (habit, woodiness, duration, foliage retention). Both dimensions honor all explicit constraints, have independent supporting evidence for Lythrum and Erythrina, and yield materially different correct answers (dicot vs perennial) that would change grading.

Interpretation 1 — Taxonomic / phylogenetic kind
Interpretation: “type of vegetation” = major taxonomic vegetation group (clade / class) to which the genus belongs.
Answer: Both are dicotyledonous flowering plants (angiosperms), specifically magnoliopsid / rosid dicot / eudicot genera. Acceptable statement: both are dicots (dicot genera, magnoliopsid genera, rosid dicot genera, eudicots, angiosperms, flowering plants).
Scope: Genus-level, applies to all species in both genera without exception.
Supporting evidence:
- Lythrum is type of dicot genus, magnoliopsid genus - genus of flowering plants having two cotyledons
- Lythrum - dicot genus, magnoliopsid genus - genus of flowering plants having two cotyledons
- Genus Erythrina is rosid dicot genus (a genus of dicotyledonous plants)
- Erythrina rosid dicot genus - genus of attractive tropical shrubs or trees
- Erythrina can be deciduous or evergreen subshrubs, shrubs or trees, often spiny, with trifoliate leaves and showy pea-like flowers - confirming Fabaceae genus placement in same angiosperm/dicot majority group
Confidence: High
Grading guidance: Accept any rank in this nested hierarchy that necessarily includes both genera: angiosperm, flowering plant (Anthophyta/Magnoliophyta), dicot / dicotyledon, magnoliopsid, eudicot, rosid, rosid dicot. These are granularity variants of one dimension, not separate interpretations. Reject monocot, gymnosperm, non-flowering, cryptogam. Do not penalize granularity (e.g., “dicot” vs “eudicot rosid angiosperm” are equivalent).

Interpretation 2 — Growth-form / life-history kind
Interpretation: “type of vegetation” = morphological life-form / duration / foliage type (herb vs shrub vs tree, herbaceous vs woody, annual vs perennial, deciduous vs evergreen).
Answer: Both include perennial (and specifically deciduous perennial) forms; both genera contain herbaceous perennial (herbaceous) representatives, and both are woody-capable / herbaceous-capable perennials with deciduous foliage. Minimal common answer: both are perennials / both are deciduous perennials; fuller answer: both can occur as herbaceous perennial / perennial herb / forb (rare for Erythrina, typical for Lythrum).
Scope: Shared trait at genus-trait level; not universal to every species. Lythrum species are annuals OR herbaceous perennials; Erythrina species are mostly trees or shrubs, rarely perennial herbs. Common overlap is: Life history perennial and Foliage retention deciduous, Growth form includes herb among other forms.
Supporting evidence:
- Lythrum can be annuals or herbaceous perennials
- Lythrum can be annuals or herbaceous perennials, with simple leaves in opposite pairs
- Foliage, Deciduous for Lythrum
- Erythrina Trees or shrubs [or perennial herbs]
- Mostly trees or shrubs, rarely perennial herbs (herbaceous resprouters), mostly deciduous
- Erythrina is a perennial deciduous herb, shrub, subshrub or tree
- Life history perennial
- Woodiness herbaceous woody
- Growth form herb shrub subshrub tree
- Foliage retention deciduous
Confidence: Moderate-High for perennial/deciduous; Moderate for herbaceous (high for Lythrum, low-frequency for Erythrina but evidenced as rare herbaceous resprouters/perennial herbs). Habit commonality is best exemplified by Lythrum salicaria as herbaceous perennial and Erythrina herbacea as perennial herb/herbaceous perennial shrub, but genus-level generalization requires qualification.
Grading guidance: Accept perennial, deciduous, deciduous perennial, herbaceous perennial, perennial herb, herb/forb, subshrub/shrub/tree form overlapping with herb. Accept qualification that both are perennials that can be herbaceous and deciduous. Reject annual only, evergreen only, strictly woody tree only with no herbaceous overlap, or monocot herb. If answer states “both are perennial herbs” without qualification, accept as sufficient evidence of correct dimension even though strictly genus-wide herbaceous is rare for Erythrina — the assessed common element is perennial/deciduous herbaceous capability.

Acceptance set
A grader may accept any ONE of the following because the prompt is materially ambiguous between dimensions; combinations are also acceptable:
- dicot / dicotyledon / magnoliopsid genus / rosid dicot genus / eudicot
- angiosperm / flowering plant / Anthophyta / Magnoliophyta
- perennial / perennial plant / life history perennial
- deciduous / deciduous perennial / foliage retention deciduous
- herbaceous perennial / perennial herb / herbaceous herb / forb / herbaceous resprouter perennial herb
- perennial deciduous herb-shrub-tree complex (any growth form listed that includes herb)
Do not accept: monocot, gymnosperm, non-vascular, annual-only, evergreen-only.

Necessary caveats
- Habitat/ecological dimension (e.g., wetland vs tropical) was tested and shows no supported common habitat vegetation type; do not treat as interpretation.
- Functional/use dimension (ornamental, medicinal) is not a vegetation type under the question.
- Taxonomic granularity differences (angiosperm > eudicot > rosid > dicot) are one interpretation, not multiple.
- Habit granularity differences (perennial vs herbaceous perennial vs deciduous perennial) are one interpretation, not multiple.
- No single source asserts the intersection; commonality is inferred by intersecting independently described genus traits.
- Genus-level habit is variable: Lythrum includes annuals; Erythrina is mostly woody trees/shrubs and only rarely herbaceous, so “exclusively herbaceous” would overgeneralize.

Evidence
- Lythrum dicot genus evidence: Lythrum type of dicot genus, magnoliopsid genus; Lythrum dicot genus definition
- Erythrina dicot evidence: Genus Erythrina rosid dicot genus; Erythrina rosid dicot genus description
- Lythrum habit evidence: Lythrum can be annuals or herbaceous perennials; Lythrum can be annuals or herbaceous perennials with deciduous foliage
- Erythrina habit evidence: Trees or shrubs [or perennial herbs]; Mostly trees or shrubs, rarely perennial herbs, mostly deciduous; perennial deciduous herb, shrub, subshrub or tree; Life history perennial; Growth form herb shrub subshrub tree; Foliage retention deciduous

Confidence
- Overall ambiguity: High
- Interpretation 1 (taxonomic): High
- Interpretation 2 (habit/duration): Moderate-High (perennial/deciduous high, herbaceous moderate due to rarity in Erythrina)

Grading guidance
- Under material ambiguity, credit either Interpretation 1 OR Interpretation 2 if supported.
- Do not treat granularity within one dimension as wrong.
- Do not create additional branches for evidence conflicts or related facts beyond the two dimensions.
- If answer mixes dimensions (e.g., “deciduous perennial dicot flowering herbs”), credit as satisfying both.

### Repeat 1 — clear — `5ac4e4595542996feb3fe968`

Question: Which cola drink is the leader in India, Thums Up or Ricqlès?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question explicitly constrains to (a) cola drink type and (b) leader in India, choosing between Thums Up or Ricqlès. Focused grounding was performed for every plausible semantic dimension: (1) drink-type dimension – whether each entity satisfies "cola drink" vs other soft-drink type, and (2) leader-dimension – cola-segment market leader vs overall carbonated-beverage leader. Thums Up is evidenced as Type Cola, a brand of colaof Origin India, while Ricqlès is evidenced as a peppermint spirit and a French soft drink, a French manufacturing products based on peppermint, listed as mint-flavoured soft drink – not a cola. Leader evidence converges on Thums Up for the cola segment. No evidence supports Ricqlès as a cola or as a leader in India in any dimension, so no second interpretation honors all explicit constraints and produces a materially different grading answer.

Status: SINGLE

Canonical answer: Thums Up – the leader among cola drinks in India (when forced to choose between Thums Up and Ricqlès).

Acceptable variants:
- Thums Up (canonical spelling without 'b')
- Thums Up cola / Thums Up cola drink
- Minor spelling variant "Thumbs Up" accepted if clearly referring to the Indian cola brand Thums Up

Necessary caveats:
- Scope is cola drinks only. For cola, Thums Up is the largest cola brand in India with Rs5,000 crore in annual salesand While Thums Up continues to be the largest cola brand in India. For the broader carbonated beverage market, Sprite tops the segment: According to market research agency Euromonitor International, Thums Up is the largest cola brand, but Sprite, another brand owned by Coca-Cola, tops the carbonated beverage market in Indiaand today Thums Up rules the cola scene in India, only topped in the carbonated beverage arena by Sprite. This does not change the answer because the question restricts to cola drink.
- Ricqlès fails the cola constraint: it is a mint/peppermint product of French origin, not a cola, and has no evidenced leadership in India's cola market. It therefore cannot satisfy the question even if "leader" is read broadly.
- Temporal consistency: historic dominance (quickly became India's biggest cola brand, trumping competitors such as Campa Cola and Double Seven, achieved a near-monopoly among cola products in India during the 1980s) and current dominance align; no temporal ambiguity creates a different answer.

Evidence:
- Thums Up is Type Cola – infobox Type | Cola
- Thums Up is a brand of cola
- Thums Up Origin Indiasupports India scope
- Thums Up is the largest cola brand in India with Rs5,000 crore in annual sales
- While Thums Up continues to be the largest cola brand in India, Sprite tops the carbonated drink segment
- Thums Up rules the cola scene in India, only topped in the carbonated beverage arena by Sprite
- Thums Up quickly became India's biggest cola brand
- Ricqlès is a peppermint spirit and a French soft drink. The liquor Alcool de Menthe de Ricqlès was created in 1838 in Lyon, France
- Ricqlès is a French manufacturing products based on peppermint
- Ricqlès – mint-flavoured soft drink (list of soft drinks by country, France section)

Confidence: High – Convergent evidence from Wikipedia, LiveMint (Coca-Cola India interview), Atlas Obscura, and Euromonitor-cited reporting unanimously identifies Thums Up as India's cola leader; Ricqlès is unanimously described as mint/peppermint French product, not cola, with no India cola-leadership evidence.

Grading guidance:
- CORRECT: Any response identifying Thums Up (including "Thums Up cola") as the leader in India among the two options.
- INCORRECT: Ricqlès, "both," "neither," "Coca-Cola/Pepsi/Sprite/Campa Cola," or any drink other than Thums Up.
- Do not penalize additional correct context (e.g., noting Sprite leads overall carbonated market but Thums Up leads cola) – still correct.
- Do not accept Ricqlès on grounds of French popularity or mint-category leadership; question explicitly requires cola drink in India, which Ricqlès does not satisfy.

### Repeat 1 — clear — `5adce8c15542992c1e3a249d`

Question: Are Thyme and Cortaderia both aromatic plants?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks whether both genera are "aromatic plants" – a term whose established horticultural/botanical meaning selects the sensory-fragrance dimension (plants with noticeable pleasant smell/essential oils), not the organic-chemistry dimension (containing benzene-ring compounds). Two semantic dimensions were inventoried and tested: (1) fragrant/essential-oil aromatic herb, (2) chemical aromatic (aromatic hydrocarbons). Dimension 1 is evidence-supported for Thyme and evidence-negative for Cortaderia, yielding a clear "No". Dimension 2 was tested with focused grounding (Cortaderia aromatic compounds/chemistry) and found no reputable classification of Cortaderia as an aromatic plant; isolated pyrolysis-oil chemistry does not make the living plant an "aromatic plant" in this sense and does not create an alternative graded answer. No other dimension (taxonomy, habitat, growth form) both honors the wording "aromatic" and produces a materially different grading outcome with supporting evidence. Therefore no material interpretation ambiguity remains.

Status: SINGLE

Canonical answer: No. Thyme and Cortaderia are not both aromatic plants. Thyme (genus Thymus, e.g., Thymus vulgaris) is an aromatic plant/herb, while Cortaderia (pampas grass, e.g., Cortaderia selloana) is not – it is consistently described as non-fragrant / odorless / not known for fragrance.

Acceptable variants:
- No – only Thyme is aromatic; Cortaderia is not.
- False. Thyme is aromatic, Cortaderia is non-aromatic/non-fragrant.
- Thyme: yes (aromatic); Cortaderia: no.
A bare "No" with correct explanation is acceptable. "Yes, both are aromatic" is unacceptable.

Necessary caveats:
- The answer refers to the conventional botanical/horticultural definition of "aromatic plant" = fragrant plant/herb, as defined in dictionaries: "having a noticeable and pleasant smell : fragrant: aromatic herbs" and "Having an aroma; fragrant or sweet-smelling: aromatic herbs" – distinct from the chemistry sense "Of, relating to, or containing one or more molecular ring structures having properties of benzene".
- Under the chemistry sense, Thyme does contain aromatic compounds (e.g., thymol provides distinctive flavor and is an "aromatic" phenol), but this does not reclassify Cortaderia as an aromatic plant. No evidence classifies living Cortaderia as an aromatic plant on chemical grounds.
- Evidence conflict: one commercial description claims Cortaderia nitida is used "for its fragrant flowers," but this is outweighed by multiple horticultural sources explicitly marking the genus/species as Non-Fragrant / Odorless / not known for fragrance. It does not support "both are aromatic" at genus level.

Evidence:
- Thyme genus definition: "The genus Thymus contains about 350 species of aromatic perennial herbaceous plants and subshrubs in the family Lamiaceae."
- Thyme leaves described: "The leaves are evergreen in most species, arranged in opposite pairs, oval, entire, and small ... and usually aromatic."
- Thyme cultivated for fragrant leaves supports aromatic status.
- Cortaderia genus fragrance status: "Are members of the Cortaderia plant genus fragrant? Cortaderia species are not known for their fragrance, but their visual impact in a landscape is significant."
- Cortaderia selloana product specifications: "| Botanical Name | Cortaderia selloana | ... | Fragrance | Non-Fragrant |" and second listing also "| Fragrance | Non-Fragrant |"
- Cortaderia selloana 'Pumila' and related cultivars listed as "Fragrance | Odorless" and "| Fragrance | Odorless |" for 'Rosea', and "Although it doesn't have a fragrance, its elegant appearance more than makes up for it." for 'Evita'.
- Dictionary support for dominant sense: aromatic as fragrant herb vs chemistry sense noted above.

Confidence: High (0.95) – multiple independent reputable sources converge that Thyme is aromatic and Cortaderia is not aromatic/fragrant; dictionary grounding confirms the intended sensory dimension.

Grading guidance:
- Award full credit for "No" or any variant correctly denying that both are aromatic and correctly attributing aromatic status to Thyme only.
- Accept phrasing such as "Thyme is aromatic, Cortaderia is non-fragrant/odorless/not aromatic" or "Only Thyme is an aromatic plant."
- Reject "Yes, both are aromatic plants" – even if justified by the single outlier C. nitida fragrant-flowers claim or by chemical-aromatic argument, it fails the question's conventional botanical meaning.
- Reject "Both are non-aromatic," "Neither is aromatic," or "Cortaderia is aromatic, Thyme is not."
- Partial identification of Thyme as aromatic without correctly denying Cortaderia's aromatic status is insufficient for full credit.
- Do not penalize mentioning thymol/aromatic chemistry as long as final classification remains: Thyme aromatic, Cortaderia not aromatic, therefore not both.

### Repeat 1 — broad-but-single — `5ae336ef5542991a06ce995d`

Question: What profession do both Phil Alden Robinson and David Nixon share?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The attribute term "profession" has no material semantic split here. Plausible dimensions (specific occupational title vs. broad industry) were grounded: (1) functional title = director/filmmaker and (2) functional title = writer/entertainer. Phil Alden Robinson is consistently an American film director and screenwriter / film director, screenwriter / writer and director. David Nixon (director) is consistently an American film director and film producer. David Nixon (magician) is consistently Magician and television presenter / English magician and television personality, not film director. The only entity that satisfies the question's presupposition of a shared profession under primary occupational sources is David Nixon (director) sharing film director/director. The alternative referent (David Nixon magician) shares no primary occupation with Robinson; its secondary IMDb credit as writer and actor is not listed as a primary Occupation and would not create an equally-supported, presupposition-satisfying reading. Producer is a nested granularity variant of the same filmmaker interpretation, not a separate dimension.

Status: SINGLE

Canonical answer: Film director — both are film directors (directors / filmmakers). Phil Alden Robinson is a film director (and screenwriter) and David Nixon (director) is a film director (and film producer).

Evidence:
- Phil Alden Robinson — Occupation is Film director, screenwriter
- Phil Alden Robinson — is an American film director and screenwriter whose films include Field of Dreams
- Phil Alden Robinson — He is a writer and director, known for Field of Dreams (1989)
- Phil Alden Robinson — Profession listed as Writer, Director, Producer, Soundtrack (shows producer as secondary scope variant)
- David Nixon (director) — David Nixon is an American film director and film producer
- David Nixon (director) — duplicate confirmation: David Nixon is an American film director and film producer. He is the director and producer of Letters to God
- David Nixon name disambiguation — David Nixon may refer to several persons including David Nixon (magician) English magician and television personality and David Nixon (director), American film director and producer
- David Nixon (magician) primary occupations — Occupations | Magician and television presenter
- David Nixon (magician) — was an English magician and television personality
- David Nixon (magician) secondary conflicting IMDb credit considered and rejected as primary profession — He was a writer and actor, known for Aladdin (1951)

Acceptable variants:
- director
- film director / movie director / cinema director
- filmmaker / film-maker
- director and producer / film director and film producer (acceptable as expanded form covering the shared director core; producer alone is not required but should not be marked wrong if combined with director)
- Case/plural variations: "both are directors", "film directing"

Necessary caveats:
- The minimal correct answer is director / film director. Broader "filmmaker" is acceptable; narrower "film director and producer" is also acceptable.
- "Screenwriter" / "writer" alone is not acceptable as the shared profession for the director David Nixon reading — primary sources list Robinson as screenwriter but Nixon (director) is not listed as screenwriter/writer.
- "Magician", "television presenter"/"television personality", "entertainer", "choreographer", "footballer" are not shared with Robinson and must be marked incorrect.
- Producer overlap exists (Robinson listed as Producer in one database, Nixon as film producer) but is granularity within the same filmmaker interpretation, not a separate answer; do not require producer.
- If answer cites David Nixon (magician) writer credit from IMDb, grade as incorrect for primary profession, but note it as evidence conflict — primary Wikipedia occupation fields outrank secondary IMDb credit.

Confidence: High

Grading guidance:
- Accept any response that unambiguously states the shared profession is directing films — e.g., director, film director, movie director, filmmaker.
- Accept combined answers that include director plus producer/writer as long as director/film director is present.
- Reject answers that only state magician, entertainer, television presenter, dancer/choreographer, athlete, or only screenwriter/writer without director.
- Do not penalize for specifying American/British or adding film titles (Field of Dreams, Letters to God, Fireproof) as long as profession is correct.

### Repeat 1 — clear — `5ae6168f5542996de7b71b47`

Question: Adrian the Barbarian (later Olaff the Madlander) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: Before choosing SINGLE, all plausible semantic dimensions of the requested attribute were tested. The only plausible dimension split was (A) year first introduced as 'Adrian the Barbarian' in The Beezer vs (B) year first appearing under the later title 'Olaff the Madlander' (reprint in The Beano). Dimension B was tested with focused grounding and is evidenced as 2008, but it fails explicit constraints in the question wording — 'in the comic The Beezer, and later the merged Beezer and Topper' — because the Olaff title was only used for Beano reprints, not in Beezer/Beezer and Topper. Dimension B therefore does not honor all constraints and does not create a valid alternative interpretation. No other dimension (physical form, habitat, taxonomy, function) applies to a 'which year' question. The remaining surface conflict (1989 vs 1990) is a table-labeling/evidence-granularity issue (Start Date vs End Date), not a separate semantic dimension yielding a different grading answer.

Status: SINGLE

Canonical answer: 1989 — Adrian the Barbarian (later reprinted as Olaff the Madlander) was first introduced in The Beezer in 1989, specifically Beezer No. 1727 dated 18 February 1989, continuing through the final Beezer No. 1809 and then into the merged Beezer and Topper.

Acceptable variants:
- 1989
- 18 February 1989
- February 1989
- Beezer No. 1727 (1989) is acceptable if year 1989 is entailed

Necessary caveats:
- A Wikipedia table under heading ## 1989 lists Adrian the Barbarian with value 1990. That table's columns are Strip Title | Artist(s) | End Date | Genre | Note(s), so 1990 is the End Date in The Beezer, not the first introduction year. The UK Comics Wiki table with Start Date | End Date confirms Start Date 1989 and End Date 1990.
- The later title 'Olaff the Madlander' debuted in July 2008 as a Beano reprint, with Beano lists showing 2008. That is the reprint debut, not the Beezer/Beezer and Topper first introduction asked for.
- The Beezer merged with The Topper in 1990, explaining why the strip continued into the merged title after its 1989 debut.

Evidence:
- Defining description matching question wording and year: "***Olaff the Madlander*** (initially published as ***Adrian the Barbarian***) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989."
- Duplicate statement: "**Adrian the Barbarian** (later **Olaff the Madlander**) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989."
- Precise issue-level dating: "Adrian the Barbarian aka Olaff the Badlander first appeared in the Beezer in No 1727 dated 18th Feb 1989 and ran on through to the comics final issue No 1809 (15/9/90) and then went on to appear in the combined Beezer & Topper for most of its 153 issues."
- Start Date table confirming 1989: "| Adrian the Barbarian | Sid Burgon | 1989 | 1990 | Comic strip | Continued in the Beezer and TopperReprinted as Olaff the Madlander in The Beano |"
- Artist chronology confirming 1989 start: "began to draw for DC Thomson in 1989 drawing a revival of Biffo the Bear in The Beano and Adrian the Barbarian for The Beezer"
- Reprint distinction: "In July 2008, as of issue 3441, *Adrian the Barbarian* was reprinted in The Beano under the new title of *Olaff the Madlander*."
- Beano reprint year 2008: "| Olaff the Madlander | Reprints from *The Beezer and Topper. * | Sid Burgon |  | 2008 | 2008 |"
- Merger context: "It was, then, renamed into *The Beezer and Topper* on merging with *The Topper* in 1990."

Confidence: HIGH — Convergent primary and secondary sources (dedicated article, UK Comics Wiki Start Date table, artist biography, and a contemporaneous forum post giving exact issue number and cover date) all agree on 1989; the sole conflicting 1990 value is explained by column semantics.

Grading guidance:
- Accept any answer that asserts year 1989, including more specific forms (e.g., 18 Feb 1989) that entail 1989.
- Reject 1990 (confuses End Date or merger year), 2008 (Olaff reprint debut in The Beano), or any other year. Do not penalize for additional correct detail (issue number, artist Sid Burgon) if year is correct.

### Repeat 2 — broad-but-single — `56e2419acd28a01900c67c2a`

Question: What structures can be viewed as both works of art and symbols of certain cultures?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question wording imposes two simultaneous constraints: (1) "structures" that can be (2) viewed as *both* works of art *and* symbols of certain cultures. A fresh grounding inventory identified three plausible semantic dimensions of "structures": (a) physical/architectural structures (buildings, monuments, other built forms), (b) sculptural/monumental structures (statues, public sculpture), and (c) abstract/social structures (social, organizational, linguistic structures). Focused searches tested each. Dimension (a) is directly and repeatedly supported by encyclopedic and academic sources using the exact phrase "often perceived as cultural symbols and as works of art." Dimension (b) is a nested subset of (a) — monumental sculpture is defined as architectural works / monuments — and does not produce a materially different answer type, only a granularity difference. Dimension (c) was tested with queries for "social structures viewed as works of art cultural symbols" and returned no evidence that social/abstract structures satisfy the "works of art" constraint; the only near-match is the avant-garde term "social sculpture" which describes human activity shaping society, not a structure class that graders could accept as answering the question. No other physical-form dimension yielded an evidence-supported, materially distinct answer that honors both constraints. The ambiguity is therefore scope/granularity within one interpretation, not material interpretation ambiguity.

Status: SINGLE

Canonical answer: **Architectural works / buildings / monuments and other built physical structures — i.e., architecture in the material form of buildings and other structures, including monumental architecture, religious buildings, palaces, civic buildings, and cultural landmarks/heritage sites.**

These are structures that are considered a visual art and are often perceived as both cultural (and political) symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements.

In exemplar form this includes: Eiffel Tower, Taj Mahal, Great Pyramid of Giza, Parthenon/Acropolis, Colosseum, Hagia Sophia, Palace of Versailles, Sanchi Stupa, Gateway of the Sun, Parthenon friezes/Gothic gargoyles, and similar iconic monuments — any such iconic built work instantiates the category.

Acceptable variants:
- Generic category wording: "architecture," "architectural works," "buildings," "monuments," "monumental architecture," "cultural buildings/landmarks," "heritage sites," "historic buildings," "religious architecture (temples/churches/mosques/pagodas/cathedrals/palaces)" — all acceptable as the same answer.
- Specific exemplars: any correct iconic built structure named individually or as a list (e.g., Eiffel Tower, Taj Mahal, Pyramids of Giza, Parthenon, Colosseum, Statue of Liberty) is acceptable as an instantiation.
- UNESCO-phrased definition: "monuments: architectural works, works of monumental sculpture and painting, elements or structures of an archaeological nature, inscriptions, cave dwellings and combinations of features" of outstanding universal value from point of view of history, art or science; and "groups of buildings" — acceptable if quoted.
- Architectural styles/movements are not required and do not by themselves answer the question unless linked to built works.

Necessary caveats:
- Does not include abstract/social/organizational/linguistic/data structures — these are not viewed as works of art and do not satisfy both constraints.
- Granularity does not create separate interpretations: generic category vs. specific exemplars vs. sub-types (e.g., temples, churches) are nested levels of detail within the same interpretation.
- Answer must imply both attributes (artistic/aesthetic value *and* cultural symbolism); unqualified "structures" alone is incomplete, but "buildings/architecture/monuments" in this context is understood to carry both.
- Infrastructure that is civil engineering (bridges, roads) is generally excluded by definition unless presented as architectural work perceived as art/symbol.

Evidence:
- Architecture is the study and practice of designing structures, especially habitable ones. It utilizes civil engineering techniques, but is considered a visual art
- Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art — identically phrased as Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art
- Historical civilizations are often identified with their surviving architectural achievements
- Architecture is frequently considered a fine art... Architectural works are perceived as cultural and political symbols and works of art
- Practice of architecture has been used as a way of expressing culture by civilizations on all seven continents. For this reason, architecture is considered to be a form of art
- Sculptural/architectural identity example supporting the same physical class: This form of art has been pivotal in defining the visual and cultural identity of structures throughout history, from the intricate friezes of the Parthenon in ancient Greece to the Gothic gargoyles adorning European cathedrals
- Tested alternative dimension — social structures — showed no supporting evidence for both constraints; the closest concept is social sculpture: As a work of art, a social sculpture includes human activity that strives to structure and shape society or the environment — a niche artistic theory, not a structure class both artworks and cultural symbols in the question's sense, and therefore not a viable branch.

Confidence: HIGH — Core definitional sentence is stable across multiple independent mirrors of Wikipedia/Outline of architecture and corroborated by academic summaries; exemplar evidence is abundant and consistent. No conflicting evidence requiring resolution.

Grading guidance:
- ACCEPT any response identifying the built/architectural category: architecture, buildings, architectural works, monuments, monumental architecture, cultural landmarks, heritage buildings/sites, or types thereof (temples, churches, mosques, pagodas, palaces, cathedrals) when linked to art and cultural symbolism.
- ACCEPT lists or single correct exemplars (Eiffel Tower, Taj Mahal, Great Pyramid of Giza, Parthenon, Colosseum, Hagia Sophia, etc.) as correct instantiations.
- ACCEPT UNESCO heritage phrasing for monuments/groups of buildings as correct.
- REJECT answers naming only abstract structures (social structures, family structures, power structures, sentence structures, data structures) or non-physical concepts without built form.
- REJECT generic "art" or "symbols" without structure/built-form reference.
- For partial credit: generic category alone = full; one correct exemplar alone = full; vague "structures that are beautiful" without cultural-symbol link = partial only.

### Repeat 2 — broad-but-single — `56f71740711bf01900a44932`

Question: Bilateral treaties are concluded between how many states or entities?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question wording "Bilateral treaties are concluded between how many states or entities?" unambiguously requests a cardinal count of contracting sides. The only semantic dimension that satisfies all explicit constraints is the number of parties/sides. Before choosing SINGLE, focused grounding searches were performed for the plausible dimensions: (a) definitional number of sides/parties, and (b) formal signatory count / entity-type composition. Both dimensions returned convergent evidence that the definitional number is two, with the 17-party Switzerland-EU example explicitly described as still bilateral because parties are divided into two groups — a counting caveat, not a distinct interpretation that yields a different grading number. No alternative reading (e.g., habitat, function, or physical form) honors the phrase "how many states or entities" and no evidence supports an answer other than two that would change grading.

Status: SINGLE

Canonical answer: Two — a bilateral treaty is concluded between two states or entities (two parties / two subjects of public international law — i.e., two sovereign states, or two international organizations, or one state and one international organization — constituting two sides/groups).

Acceptable variants:
- "Two" / "2" / "Two states" / "Two entities" / "Two states or entities" / "Two parties" / "Two subjects" / "Two sides" / "Two groups" / "Two countries/nations" / "Between two" — any paraphrase stating the number is exactly two is acceptable. Synonyms such as "two sovereign states or international organizations" or "two state parties" are acceptable variants of the same answer.

Necessary caveats:
- Bilateral = two sides, not necessarily two formal signatories. Evidence notes it is possible for a bilateral treaty to have more than two formal parties divided into two groups but remain bilateral: e.g., each bilateral treaty between Switzerland and the EU has seventeen parties divided into two groups, the Swiss ("on the one part") and the EU and its member states ("on the other part"), establishing rights and obligations between the two groups severally; similarly, "It is possible for a bilateral treaty to involve more than two parties ... has seventeen parties. The parties are divided into two groups". This is a nuance of counting method, not a separate interpretation; the definitional answer remains two.
- Multilateral treaties are the contrast class: concluded among several countries / involving more than two countries / more than two parties.
- "Entities" includes international organizations established by treaty as subjects of public international law.

Evidence:
- "Bilateral treaties are concluded between two states or entities." — supports core answer Two states/entities
- "Treaties may be bilateral (between two countries) or multilateral (involving more than two countries)." — supports two vs. more-than-two dichotomy
- "A bilateral treaty (also called a bipartite treaty) is a treaty strictly between two subjects of public international law, generally either sovereign states or international organisations established by treaty. It is an agreement made by negotiations between two parties" — supports two subjects/parties and entity types
- "These two parties can be two nations, or two international organizations, or one nation and one international organization." — supports composition of the two parties
- "A multilateral treaty is a treaty involving more than two parties, while a bilateral treaty involves an agreement between two parties." — Library of Congress distinction supporting two parties
- "Bilateral treaty: An international agreement concluded between two states." — U.S. Senate / GovInfo definitional support for two states
- Exception accounting still yielding two sides: Swiss-EU 17 parties divided into two groups example and

Confidence: High — multiple independent reputable sources (English Wikipedia Treaty and Bilateral treaty articles, Library of Congress Research Guides, U.S. Senate/GovInfo definition) converge verbatim on "two" with no substantive conflict; the only nuance (multi-signatory but two-sided) is consistently explained and does not alter the number.

Grading guidance:
- CORRECT: Any answer stating the number is 2 / two, or "two states or entities/parties/subjects/sides/groups/countries/nations." Accept "two" alone. Accept compositional clarifications (two sovereign states, two international organizations, or one of each) as correct provided number is two.
- INCORRECT: Answers stating one, three or more as the definitional number; defining bilateral as multilateral/"more than two" or "two or more" without specifying two; stating bilateral can be any number without the two-sides qualification.
- If answer adds the Swiss-EU 17-party caveat but still affirms bilateral = two sides/groups, grade as CORRECT (caveat does not invalidate). If answer states 17 as the answer without explaining two sides, grade as INCORRECT.

### Repeat 2 — ambiguous — `57260b4389a1e219009ac188`

Question: What is an example of something that serves a function other than protection?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE

Status: SINGLE
Reason no branch: The question explicitly selects the functional dimension ("serves a function other than protection"). Plausible alternative semantic dimensions for vague attribute terms (e.g., physical form, habitat/taxonomy/class) are ruled out by the wording itself. The phrase "other than protection" requires at least one non-protective function (inclusive reading); an exclusive reading ("serves *only* a non-protective function and no protective function") adds a constraint not present in the wording and does not honor all explicit constraints. Remaining variation is domain/scope (biology vs. artifact vs. institution) and granularity (which specific example), which is not material interpretation ambiguity. Focused grounding confirms multiple domains yield the same answer type - any entity with a documented non-protective function - and no second dimension produces a mutually exclusive, evidence-supported grading answer.

Canonical answer: Any specific entity that has a documented function distinct from protection satisfies the question. Canonical well-evidenced examples:

* **Biological:** Human skin / integumentary system - functions include sensation, thermoregulation, and vitamin D synthesis, in addition to protection; skin has receptors that sense touch, pressure, pain and temperature and features involved in thermoregulation; skin functions occur in thermoregulation, protection, metabolic functions and sensation. Human skeleton / bones - performs supports the body, facilitates movement, protects internal organs, produces blood cells, stores and releases minerals and fat; the major functions are body support, facilitation of movement, protection of internal organs, storage of minerals and fat, and hematopoiesis; five important functions are support, protection, movement, mineral storage, and blood cell formation.
* **Artifact:** Food containers - key function is to store and preserve food, keeping it fresh and safe - a function other than protection.

Thus the answer-type is open-ended: one valid example with one valid non-protective function is sufficient.

Acceptable variants / Acceptance set:
A grader should accept ANY single example where the stated function is verifiably not protection, even if the entity also has a coexisting protective function. The set includes but is not limited to:

* Skin/skeleton examples above: thermoregulation, sensation (touch/pressure/pain/temperature), vitamin D synthesis, homeostasis of fluids, support, movement/leverage, blood cell production/hematopoiesis, mineral/fat storage
* Other biological: heart - pumping blood/circulation; lungs - gas exchange; eyes - vision; stomach/intestine - digestion/absorption; chlorophyll - photosynthesis; muscles - contraction/locomotion; nerves - signal transmission
* Artifact/functional object: pencil - writing; chair - sitting; clock/watch - timekeeping; stove/pot - cooking/heating; vehicle - transportation; mug/cutting board - containing/consuming/cutting food
* Place/institution: library - information access/lending; park - recreation; school - education

Accept if entity is named and function is plausibly non-protective and factually correct under common knowledge or cited sources.

Necessary caveats:
* No single exhaustive answer; question is intentionally open-ended and domain-unrestricted.
* Many entities serve both protective and non-protective functions; demonstrating a non-protective function does not require proving absence of any protective function.
* Granularity does not affect correctness: "skin - thermoregulation" and "sweat glands - cooling" are the same interpretation at different levels of detail.
* Evidence conflicts about exact classification/number of functions (e.g., sources list 5 vs. 6 skeletal functions) do not create interpretation ambiguity; they are scope detail differences.

Evidence:
*  supports skeletal system supports body, facilitates movement, protects internal organs, produces blood cells, stores/releases minerals and fat.
*  supports five important skeletal functions are support, protection, movement, mineral storage, and blood cell formation.
*  supports skin plays important roles in protection, sensing stimuli, thermoregulation, and vitamin D synthesis.
*  supports skin sensation and thermoregulation definitions.
*  supports skin functions occur in thermoregulation, protection, metabolic functions and sensation.
*  supports food containers store and preserve food as key non-protective artifact function.

Confidence: High. The existence of multiple documented non-protective functions for skin and skeleton is consistent across primary anatomy/physiology sources (LibreTexts, pressbooks, histology guides). The artifact example is similarly well-supported. Low uncertainty applies only to exhaustiveness, which is not required.

Grading guidance:
* ACCEPT: Response that (1) identifies a concrete entity (organ, tissue, structure, object, place, system) and (2) states a function clearly distinct from protection (e.g., thermoregulation, sensation, support, movement, hematopoiesis, storage, photosynthesis, digestion, circulation, writing, cooking, timekeeping, transportation, storage/preservation, education, recreation). Do not require citation from examinee; plausibility per standard sources suffices.
* REJECT: Response listing only a protective function (e.g., "helmet protects head" alone, "skin protects from pathogens" alone), vague answer with no entity or no function ("something useful"), or factually incorrect function attribution.
* Do not penalize if the entity also has a protective role in addition to the cited non-protective role; the question requires *a* function other than protection, not exclusivity.

### Repeat 2 — ambiguous — `57302972b2c2fd140056899e`

Question: What could be considered a trait of Rome's military campaigns?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question uses the unqualified existential term "a trait" with no dimension selector. Before choosing SINGLE, five plausible semantic dimensions were inventoried and each received a focused grounding search: (1) organizational/disciplinary, (2) tactical/flexibility, (3) logistical/engineering, (4) strategic/expansionist-offensive, (5) persistence/resilience. All dimensions returned supporting evidence, but each yields a different *exemplar* of the same requested attribute (a characteristic of Roman campaigns), not a competing reading of the question wording that would require mutually exclusive truth conditions. The variation is therefore scope/nested detail (any valid exemplar satisfies "what could be considered a trait") rather than material interpretation ambiguity. No alternative reading honors all explicit constraints, is evidence-supported, and would produce a materially different grading outcome if accepted vs rejected.

Status: SINGLE

Canonical answer: A trait that could be considered characteristic of Rome's military campaigns is a highly disciplined, standardized, organized and adaptable war machine sustained by systematic logistics and military engineering, typically offensive and expansionist, seeking direct engagement in open battle and overcoming fortified positions via engineering, noted for systematic construction of fortified marching camps (*castra*) and for obstinate persistence in overcoming appalling losses.

* Discipline / organization / standardization as a defining system and well-organized logistics sustaining combat power
* Adaptability / flexibility in tactics and methods, including practical borrowing and copying and breaking phalanxes illustrating flexibility and disciplined, organizational depth and adaptability characterized the army
* Engineering and fortified camps: adept at building fortified camps, systematic construction even for a single day's rest, and at end of day establishing strong field camp *castra* with palisade, ditch, streets and gates, 2-5 hours construction, and carrying tools for building *castra*; also systematized castra allowed staying in field on favourable ground
* Offensive/expansionist strategy: majority of campaigns characterised by territorial expansionist campaign normally begun as counter-offensive and also by securing borders, exploiting peripheral areas via tribute and maintaining internal order; offensive warfare was hallmark of Roman military strategy; preponderance exhibit preference for direct engagement in open battle and overcoming fortified positions via military engineering, and Republic/early Empire showed little reluctance to prosecute offensive operations
* Persistence / resilience: history of campaigning is a history of obstinate persistence overcoming appalling losses and also

Acceptable variants (Acceptance set — any one suffices for credit):
- disciplined / highly organized / standardized / systematized / hierarchical (legion/cohort/century)
- adaptable / flexible / versatile / pragmatic borrowing / tactically innovative (maniple to cohort, manipular checkerboard/quincunx, testudo)
- excellent logistics / supply lines / roads / arsenals / naval convoys / ability to stay in field and resupply
- engineering prowess: nightly *castra* (palisade, ditch, gates, grid streets), roads, bridges, siege works (circumvallation/contravallation, ballista/onager/scorpio), field fortification
- offensive / expansionist / aggressive / imperialist / counter-offensive expansion + civil war as second type; preference for decisive open battle
- persistent / resilient / obstinate / able to absorb and replace massive losses / refuse to concede defeat
- professional / long-service / rigorously trained / cohesive (citizenship under arms)
- More specific correct instances (e.g., "built marching camps each night," "marched in columns with vanguard and baggage train," "used pila and gladius from behind scutum") are acceptable as specifications of the above.

Necessary caveats:
- Evolution caveat: Roman military spanned c. 753 BC to 5th century AD (Kingdom phalanx → Republic manipular → late Republic/Imperial cohort legion → Late Empire limitanei/comitatenses). Not every trait applied identically in every period, but the core traits above are cited as waxing/waning elements forming distinct basis underlying Rome's rise.
- Evidence conflict is not interpretation ambiguity: sources emphasize both offensive direct-battle preference in Republic/early Empire and later more limited/defensive posture; both fit under same "trait" question as period-specific exemplars.
- "Could be considered" is existential and non-exhaustive; listing one valid trait satisfies the question. No requirement to list all traits or to specify period unless answer fabricates period-specific falsehood.
- Generic statements with no distinctive military characteristic (e.g., "they fought battles") do not satisfy; trait must be historically attested distinctive feature.

Evidence:
- Campaign history as expansionist vs civil war and triple purpose and persistence — https://en.wikipedia.org/wiki/Campaign_history_of_the_Roman_military — supports that majority of campaigns were expansionist counter-offensives, triple purpose of securing borders/exploiting peripheries, and obstinate persistence
- Strategy of Roman military — https://en.wikipedia.org/wiki/Strategy_of_the_Roman_military — supports preference for direct engagement via engineering, offensive operations in Republic/early Empire, and systematic castra construction
- Roman infantry tactics — https://en.wikipedia.org/wiki/Roman_infantry_tactics — supports distinctive Roman way including standardization, borrowing, flexibility, discipline, ruthless persistence and systematic castra/logistics and tools for camps
- Offensive warfare hallmark — search result supports offensive warfare hallmark
- Flexibility/adaptability — supports flexibility in breaking phalanxes and characterized by adaptability
- Persistence restatement — supports same persistence phrasing

Confidence: High — multiple corroborating sources (Wikipedia campaign history, strategy, infantry tactics; secondary syntheses) converge on same trait cluster; no credible source contradicts these as traits.

Grading guidance:
- ACCEPT any answer that names one or more items from Acceptance set using synonyms or specific correct examples (e.g., "disciplined," "well-organized legions," "built fortified camps nightly," "superior engineering/logistics," "adaptable tactics," "expansionist/offensive seeking open battle," "persistent despite defeats at Allia/Cannae etc.").
- ACCEPT period-qualified answers (Republic vs Empire) if trait is accurate for that period.
- REJECT if answer claims Roman campaigns were characteristically non-disciplined, avoided engineering/camps, were solely defensive with no expansion, or describes trait of an enemy (e.g., Carthaginian) as Roman without evidence.
- REJECT vague non-traits ("they had armies," "they were ancient") or invented quantitative claims without source.
- No penalty for incompleteness; one valid trait = pass.

### Repeat 2 — ambiguous — `5ab9e3d25542994dbf019889`

Question: What type of vegetation does Lythrum and Erythrina have in common?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE

Status: SINGLE

Reason no branch: Inventory of plausible readings of unqualified "type of vegetation" identified four semantic dimensions: (1) physical form / growth habit (herb/forb vs shrub/tree/grass), (2) duration/leaf persistence (annual/perennial, deciduous/evergreen), (3) ecological habitat/formation (wetland vs dryland forest), (4) systematic/taxonomic group (family/order, dicot/mono­cot, angiosperm). Each was tested with focused grounding. (3) fails evidence: Lythrum is wetland perennial forb of wet meadows, marshes and shores while Erythrina wilwili grows primarily in tropical dryland forest — no shared habitat/formation. (4) fails fit and distinctness: the two genera are in different orders/families — Lythrum Lythraceae Myrtales vs Erythrina Fabaceae Fabales — any shared rank (angiosperm/dicot/eudicot) is a trivial related fact true of ~75% of flowering plants, not a "vegetation type" in physiognomic/ecological usage, and is a different level of detail vs habit, not a competing interpretation that honors the wording's implied physiognomic meaning. (2) is nested granularity within (1): Leaf Duration deciduous and Duration Perennial are attributes of the same habit. Only (1) honors all explicit constraints (asks for shared vegetation type of two plant genera), has direct supporting evidence for a common value, and yields a materially different, rankable answer.

Canonical answer: Both are herbaceous perennial plants — perennial herbs / perennial forbs (herb/forb growth form). In vegetation-type terminology the shared type is herb / forb / herbaceous perennial forb, not woody shrub/tree, grass, or aquatic.

Acceptable variants:
- herb
- forb
- herb/forb
- herbaceous plant / herbaceous perennial
- perennial herb / perennial forb / perennial herbaceous plant
- broad-leaved non-woody herbaceous vegetation / forb-herb vegetation

Do NOT accept as satisfying the question (factually true but not the asked vegetation type):
- dicot / eudicot / angiosperm / flowering plant alone
- wetland plant / aquatic plant alone
- shrub alone / tree alone / grass alone
- annual alone
- family/order names (Lythraceae vs Fabaceae) as common — they are different

Necessary caveats:
- Lythrum as a genus is consistently described as Annual, perennial herb, commonly a perennial forb with Growth Habit forb and Plant Habit Herb/Forb; a few species are noted as perennial forb or woody sub-shrub but herbaceous is dominant.
- Erythrina as a genus is predominantly woody shrubs/trees of tropical/subtropical regions, but the authoritative habit includes Perennial, Not climbing, Herb/Shrub/Tree — the overlap with Lythrum is the herbaceous perennial part; the woody tree/shrub expression is not shared with Lythrum. Evidence conflict on woodiness is habit variation within Erythrina, not a different interpretation branch.
- Habitat is not shared: Lythrum invades wet meadows, marshes, river banks, shores; Erythrina sandwicensis is dryland forest.
- Taxonomic commonality above family level (both angiosperms) is trivial and not a vegetation physiognomic type.

Evidence:
- Lythrum Habit Annual, perennial herb
- Lythrum salicaria is a perennial forb
- Lythrum Characteristics Growth Habit forb, Duration Perennial, Leaf Type broadleaf
- Lythrum is a wetland perennial forb
- Lythrum Plant Habit Herb/Forb, Life cycle Perennial
- Winged Loosestrife (Lythrum) Plant Habit Herb/Forb Life cycle Perennial
- Erythrina herbacea General Habit Perennial, Herb/Shrub/Tree
- Lythraceae belongs to order Myrtales and contains herbs, shrubs and trees
- Lythraceae Order Myrtales Family Lythraceae
- Erythrina dryland forest habitat

Confidence: High (0.88) for habit answer; the perennial herb/forb overlap is directly supported by multiple primary/institutional sources for both genera; alternative dimensions were explicitly tested and ruled out as non-common or mis-fitting.

Grading guidance:
- Award full credit for any response that clearly states the common type is herbaceous / herb / forb / perennial herb / perennial forb / herbaceous perennial. Synonyms like "forb-herb" or "non-woody broad-leaved herb" are equivalent.
- Partial credit if say "perennial" + "herb/forb/herbaceous" is implied but incomplete (e.g., "perennial herbaceous" without forb/herb label — accept).
- No credit for answering only "dicot / eudicot / angiosperm / flowering plant" — while factually true, it does not answer vegetation type as physiognomic form; mark incorrect for this question.
- No credit for "shrub" or "tree" or "grass" or "wetland" alone as shared type.
- No credit for stating families are the same — they are different (Myrtales/Lythraceae vs Fabales/Fabaceae).

### Repeat 2 — clear — `5ac4e4595542996feb3fe968`

Question: Which cola drink is the leader in India, Thums Up or Ricqlès?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question wording selects a single semantic dimension with explicit constraints that rule out alternatives. "Cola drink" selects beverage type/flavor category (cola vs non-cola). "Leader in India" selects geographic/market scope (India cola market/sales leadership). "Thums Up or Ricqlès" selects answer type (binary choice between two named brands). Focused grounding was performed for both dimensions on both entities. Thums Up consistently satisfies type=Cola and leader-in-India; Ricqlès consistently satisfies type=mint-flavoured soft drink / peppermint spirit of French origin and does not satisfy type=Cola nor India cola leadership. No second interpretation honors all explicit constraints, is supported by evidence, and would produce a materially different grading outcome; alternative readings (e.g., treating "leader" as non-market meaning, or re-defining "cola" beyond flavor type) are not supported by evidence under the stated constraints. Related facts (overall soft-drink ranking vs cola-segment ranking) and differing market-share numbers across years are evidence conflicts/scope granularity, not interpretation ambiguity.

Status: SINGLE

Canonical answer: **Thums Up** is the cola leader in India. Ricqlès is not a cola — it is a French mint-flavoured soft drink / peppermint spirit — and is not the cola leader in India.

Acceptable variants:
- Answer selecting Thums Up (accepted spellings: Thums Up, ThumsUp, Thumps Up variant when clearly referring to same brand) as the (cola) leader / largest / No.1 cola brand in India
- May include supplemental correct detail: owned by The Coca-Cola Company, introduced 1977 to offset Coca-Cola withdrawal, billion-dollar brand in India, ~40% cola share / 85% historic share at acquisition, largest cola brand in Coca-Cola India portfolio. Such detail is optional and does not change correctness if core selection is Thums Up.

Necessary caveats:
- Type distinction is decisive: Thums Up is typed as Cola and described as a brand of cola; Ricqlès is typed as mint-flavoured soft drink and as a peppermint spirit and a French soft drink, originally a medical mint soda, created in 1838 by Heyman de Ricqlès in Lyon, France.
- Market-leadership evidence is cola-segment specific: Thums Up is described as the largest cola brand in India, India's biggest cola brand, and the largest cola brand by far. One source notes Thums Up rules the cola scene in India, only topped in the overall carbonated beverage arena by Sprite — this does not contradict cola-segment leadership; it is a scope/granularity distinction (cola vs all carbonated).
- Historical share figures vary by date (85% at 1993 sale, ~40-42% more recently) but direction is consistent; conflicts in exact percentage do not create a different interpretation.
- No evidence found that Ricqlès is a cola or holds Indian cola market leadership; searches for Ricqlès + India cola leader return no supporting claim.

Evidence:
- Thums Up Type = Cola: infobox shows Type | Cola
- Thums Up is a brand of cola: **Thums Up** is a brand of cola
- Thums Up Flavor Cola listing: Flavor, Cola
- Thums Up largest cola brand in India statement: is the largest cola brand in India with Rs5,000 crore in annual sales, which the company said it is aiming to increase to $1 billion...has dominated India's cola market in the past four decades
- Thums Up biggest cola brand confirmation: Thums Up quickly became India's biggest cola brand, trumping competitors such as Campa Cola and Double Seven
- Thums Up is the largest cola brand by far in Indian beverages market
- Ricqlès typed as mint-flavoured soft drink in France section; also listed as Ricqlès – mint-flavoured soft drink
- Ricqlès is a peppermint spirit and a French soft drink. The liquor Alcool de Menthe de Ricqlès was created in 1838 by Heyman de Ricqlès in Lyon, France, and was originally a medicinal mint spirit
- Ricqlès described as mint-flavored soda, originally a medical soda

Confidence: High

Grading guidance:
- Award full credit for any answer that selects Thums Up as the leader / correct cola drink for India versus Ricqlès. Spelling variations of Thums Up that clearly refer to the Indian cola brand are acceptable.
- Mark incorrect: any answer selecting Ricqlès as the cola leader in India, describing Ricqlès as a cola, or stating that Ricqlès leads India's cola market.
- Answers that correctly note Ricqlès is a mint/French drink and therefore not a cola, and then select Thums Up, are fully correct.
- Do not penalize additional true statements about Sprite topping overall carbonated segment or Maaza elsewhere, provided cola-segment leadership is correctly assigned to Thums Up.
- Do not accept Ricqlès as correct under any alternative semantic dimension, because no evidence supports Ricqlès satisfying both "cola" and "leader in India" constraints simultaneously.

### Repeat 2 — clear — `5adce8c15542992c1e3a249d`

Question: Are Thyme and Cortaderia both aromatic plants?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: Two semantic dimensions were inventoried before searching: (H1) horticultural/botanical 'aromatic' = fragrant, yielding essential oils/pleasant aroma; (H2) chemistry 'aromatic' = containing benzene-ring structures. Focused grounding searches were run for both dimensions. H1 is strongly supported as the default collocation ("aromatic herbs", "having a noticeable and pleasant smell : fragrant ... aromatic herbs", "Having an aroma; fragrant or sweet-smelling: aromatic herbs"). H2 definition exists ("Chemistry Of, relating to, or containing one or more molecular ring structures having properties ... of benzene"; "(Chemistry) having an unsaturated ring containing alternating double and single bonds, esp containing a benzene ring") but focused search "aromatic compound chemistry plants thyme thymol Cortaderia" returned evidence only that thyme's thymol is "a predominantly aromatic substance" with "pleasant aromatic odor" and "a white crystalline substance with a pleasant aromatic odor" - still in fragrance sense - and no source classifying Cortaderia as an "aromatic plant" in the chemistry sense. No evidence supports a materially different true answer under H2, and wording fit strongly favors H1 (unqualified "aromatic plants" in a general plant question). Therefore H2 is ruled out by wording-plus-evidence and does not create a grading-altering ambiguity.

Status: SINGLE - One clear reading (fragrant / essential-oil-producing plants)

Canonical answer: No - Thyme and Cortaderia are not both aromatic plants. Thyme (Thymus, e.g., Thymus vulgaris) is an aromatic plant; Cortaderia (pampas grass, e.g., Cortaderia selloana) is not aromatic - it is a non-fragrant ornamental grass in Poaceae.

Acceptable variants:
- No / False / Incorrect
- Thyme yes, Cortaderia no; only thyme is aromatic
- Thyme is aromatic but Cortaderia (pampas grass) is not fragrant/aromatic / is non-fragrant
All convey that the conjunction "both are aromatic" is false.

Necessary caveats:
- "Thyme" = genus Thymus in family Lamiaceae (mint family); includes ~350 species.
- "Cortaderia" = genus Cortaderia in family Poaceae (grass family), commonly called pampas grass (C. selloana).
- "Aromatic plant" here means fragrance sense: having aroma, fragrant, producing volatile essential oils, not the organic-chemistry sense of aromatic ring compounds. The generic chemistry fact that all vascular plants contain some aromatic polymers (e.g., lignin) does not make Cortaderia an "aromatic plant" in the sense queried.

Evidence:
- Genus Thymus contains about 350 species of aromatic perennial herbaceous plants and subshrubs in the family Lamiaceae
- Thymus vulgaris forming a woody, evergreen subshrub with small and highly aromatic grey-green leaves
- Thymus is cultivated for its fragrant leaves and used as a culinary herb
- Aromatic defined as Having an aroma; fragrant or sweet-smelling: aromatic herbs
- Aromatic defined as having a noticeable and pleasant smell : fragrant ... aromatic herbs
- Also aromatic adj 1. having a distinctive, usually fragrant smell
- Cortaderia - Fine Gardening Poaceae / In the genus Cortaderia are a couple of dozen species of grasses from New Zealand, New Guinea, and South America
- Cortaderia are robust evergreen grasses forming large, compact tufts of narrowly linear, rough-edged leaves, with erect stems bearing dense silvery or pink-tinged flower panicles
- Are members of the Cortaderia plant genus fragrant? Cortaderia species are not known for their fragrance, but their visual impact in a landscape is significant
- Product specification for Cortaderia selloana: Botanical Name Cortaderia selloana | Common Name Family Pampas Grass | Fragrance | Non-Fragrant |
- Alternative chemistry sense definition exists but does not support classification as "aromatic plant": Of, relating to, or containing one or more molecular ring structures... of benzene; no source was found calling Cortaderia an aromatic plant in this sense.

Confidence: High - Multiple independent botanical/horticultural sources consistently classify Thymus as aromatic and Cortaderia as non-fragrant ornamental grasses; definition of aromatic plant is stable and evidence is convergent.

Grading guidance:
- CORRECT if answer states No / False / Not both, with or without explanation that only thyme is aromatic / fragrant / essential-oil-bearing and Cortaderia/pampas grass is non-fragrant ornamental grass (Poaceae).
- INCORRECT if answer states Yes / Both are aromatic / Both are aromatic plants / Cortaderia is aromatic in same sense as thyme.
- Do not accept re-interpretation to chemistry "all plants contain aromatic compounds therefore both are aromatic" - that reading was tested, lacks source support for labeling Cortaderia an "aromatic plant," and misapplies the unqualified phrase "aromatic plants" which in dictionaries and plant literature defaults to fragrance sense. Partial answer mentioning only thyme without addressing Cortaderia is incomplete; require at least implicit denial of "both."

### Repeat 2 — broad-but-single — `5ae336ef5542991a06ce995d`

Question: What profession do both Phil Alden Robinson and David Nixon share?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: Two potential semantic dimensions were inventoried and tested with focused grounding — (a) role/credit dimension: director vs producer vs screenwriter/writer, and (b) entity dimension: which David Nixon. Only one dimension satisfies all explicit constraints — shared profession = film director. Producer fails as primary shared profession for Phil Alden Robinson (occupation listed as Film director, screenwriter, only Associate/Executive Producer credits). Screenwriter fails for David Nixon (director) — no evidence he is a screenwriter. Alternative David Nixon entities (magician/television presenter, choreographer, footballer) fail the "share" constraint because Phil Alden Robinson has no overlap on those professions. Variants like filmmaker/director/movie director are granularity variants of the same answer, not materially different readings.

Status: SINGLE

Canonical answer: Film director (also described as director / film and television director / filmmaker). Both Phil Alden Robinson and David Nixon (director) share the profession of film director.

Phil Alden Robinson is described as an American film director and screenwriter and his infobox lists Occupation as Film director, screenwriter. David Nixon (director) is described as an American film director and film producer.

Acceptable variants:
- director, film director, movie director, film and television director, filmmaker, film director and producer (if director is included)
Do not accept producer alone, screenwriter alone, writer alone, magician, television presenter, choreographer, footballer, or actor as the shared profession.

Necessary caveats:
- Entity disambiguation: "David Nixon" has multiple referents including magician, choreographer, footballer, and director. Only David Nixon (director) yields a shared profession with Phil Alden Robinson.
- Producer is not a primary shared profession: Phil Alden Robinson's technical roles include Associate Producer and Executive Producer credits (e.g., Freedom Song, All the Queen's Men) but his primary occupation is not listed as producer, whereas David Nixon is explicitly a film director and film producer. Grade producer alone as incorrect unless paired with director.
- Screenwriter overlap not supported: Phil Alden Robinson is a screenwriter, but David Nixon (director) is not evidenced as a screenwriter; David Nixon (magician) was described as writer and actor on different works, but that entity does not share director profession with Phil Alden Robinson. David Nixon (magician) occupations are Magician and television presenter.
- Scope: "Film director" and "director" are treated as the same profession at different granularity; do not treat as separate interpretations.

Evidence:
- Phil Alden Robinson — American film director and screenwriter whose films include Field of Dreams, Sneakers, The Sum of All Fears — supports Phil = film director.
- Phil Alden Robinson — Occupation | Film director, screenwriter — supports official occupation listing.
- David Nixon (director) — is an American film director and film producer. He is the director and producer of Letters to God — supports David Nixon (director) = film director.
- David Nixon (director) — same description duplicated in fresh search — corroborates director status.
- David Nixon may refer to: magician, choreographer, footballer, director, etc. — supports existence of multiple entities and need to select director reading.
- David Nixon (magician) — Occupations | Magician and television presenter — supports why magician reading fails shared-profession constraint.
- Phil Alden Robinson — Associate Producer technical credit — supports caveat that producer is marginal for Phil.

Confidence: HIGH — Both subjects' primary occupation descriptions converge on film director in reputable secondary sources (Wikipedia) with consistent corroboration across multiple pages; alternative readings fail the shared-profession constraint.

Grading guidance:
- Accept: "film director," "director," "movie director," "film and television director," "filmmaker," "American film director." Full phrase "film director and film producer" acceptable only if director present.
- Reject: "producer" alone, "screenwriter" alone, "writer" alone, "magician," "television presenter/host," "choreographer," "footballer," "actor." These either describe only one subject or are not evidenced as shared.
- Do not penalize for adding correct auxiliary roles (e.g., "director and screenwriter") if director is stated, but do not credit answers that omit director.
- If answer specifies "David Nixon (director)" vs generic "David Nixon," both acceptable provided profession is film director.

### Repeat 2 — clear — `5ae6168f5542996de7b71b47`

Question: Adrian the Barbarian (later Olaff the Madlander) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The wording explicitly names the entity as `Adrian the Barbarian (later Olaff the Madlander)` and describes its publication history as `in the comic The Beezer, and later the merged Beezer and Topper` as background, then asks for `first introduced in which year` unqualified. `First` selects the original absolute debut, not a debut in a later venue or under a later title. Three semantic dimensions were inventoried and tested with focused grounding: (1) year first published as Adrian in The Beezer, (2) year first appeared in the merged Beezer and Topper, (3) year first appeared under the reprint title Olaff the Madlander in The Beano. Only dimension 1 honors all explicit constraints (`Adrian` as primary name, `first` as original, `The Beezer` as origin). Dimensions 2 and 3 require reinterpreting `first introduced` to mean `first introduced in the merged title` or `first introduced as Olaff`, which violates the parenthetical disambiguation and the unqualified `first`. Both alternative dimensions have supporting evidence for different years but are ruled out by wording, so no material interpretation ambiguity remains; remaining variation (1989 vs 18 Feb 1989) is granularity.

Status: SINGLE

Canonical answer: 1989 — `Adrian the Barbarian` (later reprinted as `Olaff the Madlander`) was first introduced in The Beezer in 1989, specifically in Beezer No. 1727 dated 18 February 1989.

Acceptable variants:
- 1989
- February 1989
- 18 February 1989 / 18th Feb 1989 / Feb 18, 1989
- Beezer No. 1727 (1989)

Necessary caveats:
- A table row showing `1990` next to Adrian in List of Beezer comic strips is the End Date column, not the start date; the row appears under the heading `## 1989`. Do not treat 1990 as an conflicting start year.
- The strip continued to Beezer final issue No. 1809 (15 Sept 1990) and then ran in the combined Beezer & Topper for most of its 153 issues; the merger in September 1990 is a continuation, not the first introduction.
- The reprint title `Olaff the Madlander` was first used in July 2008 (Beano issue 3441); this is a reprint date under a new title, not the `first introduced` date requested. Evidence conflict between 1989 and 2008 is therefore a scope mismatch, not genuine ambiguity.
- Granularity: year-level and day-level answers refer to same event; treat 18 Feb 1989 as more specific form of 1989, not a separate interpretation.

Evidence:
- Dedicated article states `Olaff the Madlander (initially published as Adrian the Barbarian) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989`, repeated on same page.
- Detailed forum record gives precise debut: `Adrian the Barbarian aka Olaff the Badlander first appeared in the Beezer in No 1727 dated 18th Feb 1989 and ran on through to the comics final issue No 1809 (15/9/90) and then went on to appear in the combined Beezer & Topper for most of its 153 issues`.
- Sid Burgon began drawing Adrian for The Beezer in 1989: `began to draw for DC Thomson in 1989 drawing a revival of Biffo the Bear in The Beano and Adrian the Barbarian for The Beezer`.
- Reprint dimension tested and distinguished: `In July 2008, as of issue 3441, Adrian the Barbarian was reprinted in The Beano under the new title of Olaff the Madlander` and `Olaff the Madlander | Reprints from The Beezer and Topper | Sid Burgon | 2008 | 2008`.
- Merger dimension tested: `In September 1990, DC Thomson decided to rationalise their comics portfolio and merged the Beezer with The Topper` and `In September 1990, it was decided to merge the Topper with another ... combined as Beezer and Topper. This continued ... until 1993` showing 1990 is continuation date.
- List of Beezer and Topper strips confirms Adrian was a continuing strip from The Beezer, not newly introduced there: `Adrian the Barbarian | Sid Burgon | 1 | 153 | The Beezer | Reprinted in The Beano in 2008 as Olaff the Madlander`.

Confidence: High — multiple primary-structure sources (dedicated Wikipedia article, detailed issue-numbered forum history, artist biography) converge on 1989 with day-level precision; alternative years (1990, 2008) are well-evidenced but clearly refer to different events (merger continuation, Beano reprint) and are ruled out by wording.

Grading guidance:
- Accept any answer clearly stating 1989 as the original Beezer debut, including full date 18 February 1989 or issue No. 1727.
- Do not accept 1990 (confusion with End Date / merger year), 2008 (Olaff reprint in The Beano), 1993 (end of Beezer and Topper), or 1990-1991 (first appearance in merged title issue 1).
- Year alone is sufficient; do not require month/day but accept them as correct elaboration.
- Require exact year match; treat circa or decade approximations as incorrect.

### Repeat 3 — broad-but-single — `56e2419acd28a01900c67c2a`

Question: What structures can be viewed as both works of art and symbols of certain cultures?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question wording “structures can be viewed as both works of art and symbols of certain cultures” selects the physical-built dimension. “Works of art” requires a created artifact with aesthetic/artistic status, and “symbols of certain cultures” requires a material signifier of collective identity. Focused grounding was performed for three plausible semantic dimensions of “structures”: (A) physical/architectural structures, (B) social/organizational structures, (C) abstract/conceptual structures. Only dimension A is supported by evidence that a class of structures satisfies both predicates simultaneously. Dimensions B and C produced no evidence that such structures are described as both works of art and cultural symbols; search hits discuss culture as a symbolic system but do not describe social structures themselves as works of art. Variation between naming the general class (architecture/buildings/monuments) versus naming specific iconic examples (Eiffel Tower, Taj Mahal, etc.) is scope/granularity, not a materially different semantic interpretation, and both satisfy the same constraints. No evidence-supported alternative dimension would change the grading outcome.

Status: SINGLE
Canonical answer: Architectural works / built structures — in the material form of buildings, monuments and other physical structures — are often perceived as both cultural symbols and works of art. Historical civilizations are often identified with their surviving architectural achievements.

Definitionally, architecture is the study and practice of designing structures, especially habitable ones, and is considered a visual art. Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements. The same definition is duplicated as “Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art”.

Architecture can mean a general term to describe buildings and other physical structures, and architectural monuments are defined as buildings and structures that retain full or partial authenticity and express characteristics of a particular culture, era, style, construction technique, or represent works of renowned architects.

As international cultural statements without words, symbols of a peoples values devotion, patriotism, power symbols of a civilization's grandeur, these monuments still fascinate... Their historical message goes far beyond mere art history, for they tell us of the lives and evolution of the peoples of the past, as does the Parthenon in Athens.

A course examining monuments expressly lists as its range of famous monuments: the Eiffel Tower; the Taj Mahal; the Great Mosque of Mecca; the Statue of Liberty; the Alhambra, all belonging to the same class defined above.

Acceptable variants / Acceptance set:
Grader must accept any answer that identifies the **class** or **one or more correct instances** of that class, including paraphrases:

Class-level answers (all equivalent):
- architectural works / architecture / buildings / physical structures / monuments / monumental buildings / cultural landmarks / architectural landmarks / architectural achievements — when described as both works of art (or masterpieces, works of fine art, visual art) and cultural symbols/icons

Instance-level answers (any one is sufficient; list not exhaustive):
- Eiffel Tower (Paris/France)
- Taj Mahal (India)
- Statue of Liberty (United States)
- Great Mosque of Mecca / Mecca, Alhambra, Parthenon, Sydney Opera House, Great Wall, Pyramids of Giza, Colosseum, etc., when described as belonging to the class above

Paraphrases of the dual predicate are acceptable: “cultural symbols and works of art,” “cultural icons and works of art,” “symbols of a people’s values as monuments/monuments of genius,” “fine art monumental art,” etc. A specific monument is acceptable only if it is a physical built structure and is known to be described under both predicates via the class definition.

Necessary caveats:
- Answer must refer to physical, human-built structures. Non-physical “social structures,” “organizational structures,” “economic structures,” “sentence/language structures,” or “narrative structures” do not satisfy the question, even though culture is described as a symbolic system with patterns/structures of meaning.
- It is sufficient to name the class OR to list one or more correct examples; a comprehensive list is not required and extra incorrect examples do not invalidate a correct core unless all examples are non-physical/incorrect.
- The dual status must be present: artistic/aesthetic value AND symbolic representation of a culture/civilization/nation. An answer that mentions only one predicate is insufficient.
- Specific monument names without explanation are acceptable because they fall under the defined class “architectural works … often perceived as cultural symbols and as works of art.”
- Evidence conflicts about dating, style, or whether a particular building is the best example are granularity issues, not interpretation ambiguity, and should be noted as caveats only.

Evidence:
- Architecture is considered a visual art and is the study/practice of designing structures, especially habitable ones
- Architectural works in material form of buildings are often perceived as cultural symbols and as works of art; duplicate confirmation
- Architecture can mean buildings and other physical structures
- Architectural monuments = buildings and structures expressing characteristics of a particular culture, era, style, or representing works of renowned architects
- Monuments as international cultural statements/symbols beyond mere art history, e.g., Parthenon
- Famous monuments investigated together include Eiffel Tower, Taj Mahal, Statue of Liberty etc.
- Focused grounding for alternative dimension “social structures viewed as works of art symbols of cultures” returned no evidence that social structures themselves are described as both works of art and cultural symbols; hits discuss culture as symbolic system and art as social phenomenon, but not the dual predicate for social structures — supporting single-dimension decision.

Confidence: HIGH for the canonical class-level answer (direct, repeated definitional statements from encyclopedic sources). HIGH that architectural/monumental buildings are the intended referent of “structures” in this question. MEDIUM-HIGH that any specific iconic monument cited from the class will satisfy both predicates via the class definition; individual dual-status descriptions for specific icons are well-established but not separately cited for every possible example in this pass. LOW that an alternative social-structure reading is evidence-supported.

Grading guidance:
- ACCEPT if answer identifies the general class (architecture/architectural works/buildings/monuments/physical structures) and states they are/can be viewed as both works of art and symbols of cultures/civilizations, even if phrasing varies (“cultural icons,” “masterpieces,” “works of fine art,” “symbols of national identity”).
- ACCEPT if answer lists at least one correct built-structure example belonging to that class (Eiffel Tower, Taj Mahal, Statue of Liberty, Pyramids, Great Wall, Parthenon, Sydney Opera House, etc.).
- REJECT if answer refers solely to non-physical structures (social structures, family structures, political structures, linguistic structures) without identifying a built artifact that is both art and cultural symbol.
- REJECT if answer mentions only one side of the dual predicate (only art OR only cultural symbol).
- REJECT invented structures or generic statements about “structures” with no link to architecture/monuments/buildings and no evidence of artistic status.
- For list answers, credit requires at least one correct physical-structure answer; do not penalize inclusion of additional correct examples or omission of exhaustive list.

### Repeat 3 — broad-but-single — `56f71740711bf01900a44932`

Question: Bilateral treaties are concluded between how many states or entities?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE

**Status:** SINGLE — Unambiguous. One clear reading. The question asks for the cardinal number of contracting sides/entities in a bilateral treaty.

**Reason no branch:** Before searching, plausible semantic dimensions of "how many states or entities" were inventoried:
- Dimension 1 (D1): Number of treaty sides / parties as bilateral relationship units (the legal classification count)
- Dimension 2 (D2): Number of individual sovereign signatories counted instrument-by-instrument
- Dimension 3 (D3): Qualitative "type" of entity (state vs. international organization) — a scope detail, not a numeric dimension

The wording "between how many states or entities?" explicitly selects D1: "between" denotes the relationship sides, not a tally of signatures. Focused grounding was performed for both D1 and D2. D1 is uniformly evidenced as "two" and as "between two countries" vs. multilateral "involving more than two countries" and "strictly between two subjects" and "between two parties". D2 is evidenced only as an exceptional counting nuance: a bilateral treaty may *involve more than two parties* instrumentally but still counts as two sides/entities, e.g., seventeen parties divided into two groups. D2 therefore does not honor the question's constraint as an alternative numeric answer — it is a nested level of detail/caveat, not a materially different interpretation that would change grading. D3 does not affect the number and is scope elaboration (two nations, or two international organizations, or one nation and one international organization). No second interpretation satisfies *all* explicit constraints + has supporting evidence + would produce a different grading outcome. Per the ambiguity contract, evidence conflict / granularity is not relabeled as ambiguity.

**Canonical answer:** **Two (2).** Bilateral treaties are concluded between **two** states or entities / two parties / two sides / two subjects of public international law.

**Acceptable variants / Acceptance set:**
A grader must accept any semantic equivalent of "two" that preserves the bilateral count:
- "Two" / "2" / "2 states or entities"
- "Two states" / "Two entities" / "Two parties" / "Two sides" / "Two subjects of public international law" / "Two countries" / "Two independent governments"
- Elaborations specifying the two can be two sovereign states, two international organizations, or one state and one international organization are correct elaborations, not different answers.

Do NOT accept as correct for bilateral: "One", "Three or more", "More than two" sides/entities, "Plurilateral/multilateral (>2)" without the two-sides qualification.

**Necessary caveats:**
- **Signatory count vs. sides count:** It is possible for a bilateral treaty to have more than two signatories and still be classified as bilateral, because the signatories are divided into two groups. The canonical example is each of the bilateral treaties between Switzerland and the European Union (EU), which has seventeen parties divided into the Swiss ("on the one part") and the EU and its member states ("on the other part") — establishing rights and obligations between the two sides severally, not among members of the same side. Similarly: "It is not necessary that the treaty can only have 2 parties; there may be more than two parties, however, there should be only two states [entities] involved". This is a counting caveat, not a different answer - the correct graded number remains two sides.
- No conflicting evidence on the core definition was found; all primary sources converge on two.

**Evidence:**

- Core definition "concluded between two states or entities" — supports answer Two
- Contrastive definition bilateral = between two vs. multilateral = involving more than two — supports Two and excludes >2
- Definition as strictly between two subjects of public international law — supports strictly two
- Definition as between two parties — supports two parties
- Typology of the two entities (two nations, two international organizations, or one of each) — supports acceptable variants for entity type
- Caveat that bilateral may involve >2 signatories but only two entities/sides, with 17-party Swiss-EU example divided into two groups — supports necessary caveat and why this does not create a second interpretation and

**Confidence:** High. Convergent, reputable sources (Treaty article citing Vienna Convention literature, dedicated Bilateral treaty article, Library of Congress/PacLII definitions) agree with no authoritative contradiction.

**Grading guidance:**
- **Mark CORRECT:** Any answer containing the number two (word or numeral) to describe the number of states/entities/parties/sides/countries/subjects a bilateral treaty is between. Accept "two states or entities" and any subset phrasing ("two states", "two parties", "two sides"). Accept answers that add correct entity-type elaboration or the Swiss-EU caveat if they reaffirm the count as two sides/groups.
- **Mark CORRECT WITH CAVEAT NOTATION:** Answers stating "Two sides/groups (though the instrument may list 17 signatories, still bilateral because divided into two groups)" — correct, as the grading number remains two.
- **Mark INCORRECT:** Answers stating bilateral means one party, three or more sides/entities, or that bilateral is synonymous with multilateral/plurilateral without the two-sides qualification. Answers that only state "17" (the Swiss-EU signatory tally) without explaining "still bilateral = two sides" are incorrect as a response to "how many" in the definitional sense.
- **Scope/granularity:** Do not penalize for additional correct detail (e.g., naming state vs. organization types) or for omitting the caveat. Do not treat differing phrasing of "two" as a different interpretation.

*What each URL supports:*
- `https://en.wikipedia.org/wiki/International_treaty` (ID 2896221383307973216) lines L48-L49 support "between two states or entities"; lines L8-L9 support bilateral vs. multilateral distinction; lines L48-L52 support the 17-party/two-groups caveat.
- `http://en.wikipedia.org/wiki/Bilateral_treaty` (ID 7185218051693196380) lines L2-L4 support "strictly between two subjects"; lines L11-L13 support the two-entity typology.
- Search result ID 19447084777967599 lines L34-L36 support "between two parties" definition.
- Search result ID 8648016523870355994 lines L12-L15 support the D2 caveat that more than two signatories can still be only two entities.

### Repeat 3 — ambiguous — `57260b4389a1e219009ac188`

Question: What is an example of something that serves a function other than protection?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question explicitly predicates on "serves a function" and asks for an example where that function is other than protection. This selects the functional dimension. Alternative semantic dimensions for underspecified terms like "type/kind/form/class" (physical form, habitat, taxonomy) would not honor the explicit "serves a function" constraint. All plausible exemplars (accessory domain, integumentary domain, cellular domain) satisfy the same proposition - naming a thing and a verifiable non-protective function - and differ only in domain scope/granularity, not in reading of the request. Focused grounding checks were performed for every plausible dimension, and no evidence-supported alternative dimension produces a materially different interpretation that honors the wording and would change grading outcome.

Status: SINGLE

Canonical answer: Eyeglasses — an item that serves a function other than protection: vision correction (correcting visual acuity / refractive error), and is therefore classified as an accessory rather than clothing. Equivalent: ordinary eyeglasses / spectacles for vision correction.

Acceptable variants / Acceptance set: The question is open-ended. Any specific, verifiable non-protective function attributed to a real thing/structure is acceptable. Must name both thing and non-protective function. Flattened acceptable set includes (non-exhaustive, any one suffices):

* Eyeglasses / spectacles / ordinary eyeglasses — fully correct visual acuity or eliminate refractive error / vision correction; prescription glasses are made to correct vision, not for personal protection; regular glasses are primarily intended for vision correction
* Skin — sensation, thermoregulation, vitamin D synthesis, insulation (not just physical/chemical barrier protection): "Sensation, thermoregulation, vitamin D synthesis, and insulation are just a few of the functions of the skin"; "Sensation: skin has receptors that sense touch, pressure, pain and temperature ... Thermoregulation: various features of the skin are involved in regulating temperature"; "Protection is provided ... Sensation is provided by nerve endings ... Thermoregulation is supported ... Synthesis of vitamin D occurs"; "The skin plays important roles in protection, sensing stimuli, thermoregulation, and vitamin D synthesis"
* Plasma / cell membrane — selective transport of molecules and cell-cell recognition/signaling beyond barrier: "proteins ... carry out the specific functions of the plasma membrane, including selective transport of molecules and cell-cell recognition"; markers that allow cells to recognize one another; transport proteins, receptor proteins, cell-recognition markers
* Other verifiable non-protective functions (e.g., mucus — lubrication/hydration enabling eye closure/swallowing/olfaction; clothing — social/cultural functions, adornment) accepted if correctly attributed.
Generic "something that doesn't protect" without naming alternative function is not acceptable. Protective-function examples (safety goggles/glasses for impact protection, helmet, etc.) are not acceptable.

Necessary caveats:
* No single entity is uniquely correct; the canonical answer is the verbatim match to the source phrase "those that serve a function other than protection (eyeglasses)", but biological examples satisfy the same interpretation and must be graded as correct.
* Varying levels of detail (e.g., "eyeglasses" vs "eyeglasses correct refractive error" vs "skin regulates temperature via sweating") are scope/granularity variants, not separate interpretations.
* Evidence conflicts do not create ambiguity here; some sources also note eyeglasses can provide limited physical barrier, but primary intended function remains vision correction, distinct from protection.
* Focused grounding searches tested: (1) apparel/accessory functional classification, (2) integumentary biological functions, (3) cellular membrane functions. No plausible untested dimension remains that honors "serves a function" and yields a different answer.

Evidence:
* Exact phrase linking non-protective function to eyeglasses as accessory: "or those that serve a function other than protection (eyeglasses), are normally considered accessories rather than clothing"; repeated "or those that serve a function other than protection ([eyeglasses]), are normally considered accessories rather than clothing"; also "or those that serve a function other than protection (eyeglasses), are normally considered accessories rather than clothing"
* Vision correction definition: "the term 'ordinary eyeglasses or contact lenses' means lenses that are intended to fully correct visual acuity or eliminate refractive error"; "Prescription glasses are made to correct vision, not for personal protection"; "Regular glasses ... are primarily intended for vision correction"
* Skin non-protective functions: see citations in Acceptance set above
* Plasma membrane non-barrier functions: "including selective transport of molecules and cell-cell recognition"

Confidence: HIGH for canonical answer (eyeglasses) as direct verbatim match; HIGH that any verifiable non-protective function example in Acceptance set satisfies the single clear reading; no material ambiguity remains after tested dimensions.

Grading guidance:
* ACCEPT if answer (1) names a real object/structure and (2) states a specific verifiable function other than protection correctly attributed to it (paraphrases/synonyms accepted). At least one valid pair suffices if multiple examples given.
* REJECT if no example given, only negates protection without naming alternative function, or names a function that is protection, or misattributes function (e.g., claiming ordinary eyeglasses primarily protect against impact).
* Do not require exact phrasing "eyeglasses"; accept domain-specific correct alternatives as listed in Acceptance set.

### Repeat 3 — ambiguous — `57302972b2c2fd140056899e`

Question: What could be considered a trait of Rome's military campaigns?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE

**Status: SINGLE**

**Reason no branch:**
Before choosing SINGLE, a private inventory of plausible semantic dimensions for "trait" was made: (1) organizational/disciplinary, (2) tactical/operational, (3) engineering/logistical/infrastructure, (4) strategic-political/expansionist. Focused grounding searches were performed for each dimension:
- discipline/organization: searched and found support
- engineering/fortified camps/siege: searched and found support
- strategic innovation/flexibility: searched and found support
- expansionist/territorial: searched and found support

Each dimension is evidence-supported and yields a materially different example answer (e.g., "disciplined" vs "engineering-intensive" vs "expansionist"). However, they do **not** constitute materially different *readings* of the question. The wording "What could be considered a trait of Rome's military campaigns?" does not select a single dimension (it contains no qualifier like "tactical form" or "political objective"). All dimensions satisfy the explicit constraints (a characteristic attributable to Roman campaigns) and produce answers that are interchangeable for grading - any valid trait satisfies the question. The differences are therefore scope/granularity (multiple acceptable instances within one broad category), not interpretation ambiguity. The contract requires separate branches only when each reading honors all constraints AND would change the grading outcome by making a different set of answers correct/incorrect. Here, the grading outcome is identical: any well-supported trait is correct. No wording constraint rules out any dimension, and no dimension is untested. Branching would be listing related facts, not competing interpretations. Confidence conflicts (e.g., degree of strategic planning) are documented as caveats.

**Canonical answer:**
A trait of Rome's military campaigns is that they were highly organized, disciplined, strategically innovative and engineering-intensive, with a preponderant operational preference for direct engagement in open battle and, where necessary, overcoming fortified positions via military engineering and systematic construction of fortified marching camps.

Core phrasing supported by sources: Roman warfare was characterized by its organization, discipline, and strategic innovation. Romans also emphasized engineering, constructing roads and fortifications to support their campaigns. The preponderance of Roman campaigns exhibit a preference for direct engagement in open battle and, where necessary, the overcoming of fortified positions via military engineering. The Roman army was adept at building fortified camps for protection from enemy attack, but history shows a reluctance to sit in the camp awaiting battle and a history of seeking open battle. Roman soldiers were highly skilled in pitched battle, siegecraft, and military engineering, and Roman engineers were used extensively if the campaign ended in a siege as Rome had no equal with regards to siege warfare. The majority of Rome's campaigns were characterised by one of two types. The first is the territorial expansionist campaign, normally begun as a counter-offensive, allowing Rome to grow from a small town to 55 million.

**Acceptable variants:**
Any one historically attested characteristic that is a proper subset of the canonical description is acceptable. Grader should accept paraphrases at any level of granularity, including but not limited to:

- Discipline / *disciplina militaris* / brutal discipline / trained skill and discipline and combat training was an important element of Roman *disciplina militaris*; organization, trained skill, discipline, and technical expertise in siegewarfare admired by Josephus; strict hierarchy, legion/cohort/century structure, flexibility
- Highly trained and structured, allowing for flexibility on the battlefield; use of formations like testudo and manipular system
- Engineering emphasis: constructing roads, bridges, fortifications, ditches, palisades, siege works; camp construction overnight giving strategic advantage
- Logistics and infrastructure: well organized Roman logistics also sustained combat power, from routine resupply and storage to the construction of military roads; troops built military roads, bridges, and large permanent camps
- Systematic fortified camp (*castra*) system, standardized layout with rectangular plan and grid; superiority in siege warfare via circumvallation/contravallation
- Tactical preference for open battle and overcoming fortifications via engineering as above
- Adaptability/innovation: organization more flexible than opponents; blending tactics, technology, and logistical planning
- Expansionist/territorial: expansionist push, aggressive territorial conquest, counter-offensive campaigns bringing subjugation of large areas

Synonyms (e.g., "well-disciplined," "highly structured," "logistically capable," "road-building," "aggressive/offensive," "siege-capable") are acceptable if semantically equivalent to an item above.

**Necessary caveats:**
- **Period variation:** Traits describe mature Republic and early-to-mid Empire most strongly. Early Republic campaigns were seasonal and reactive; late Empire shifted to more defensive grand strategy. Trait need not be universal across all 753 BC - 476 AD to be considered characteristic.
- **Not universal per campaign:** "Preponderance" language indicates tendency, not absolute rule; not every campaign exhibited every trait simultaneously. Question asks "could be considered," so preponderant pattern suffices.
- **Historiographical nuance:** Modern debate continues on whether Rome had a coherent expansionist grand strategy vs reactive/defensive imperialism, but that sources characterize campaigns as expansionist is well-supported; this debate is scope detail, not interpretation ambiguity.
- **Civil war vs foreign campaign:** Sources distinguish territorial expansionist campaign vs civil war as the two main campaign types; either descriptor is acceptable within strategic dimension.

**Evidence:**

- Organization, discipline, strategic innovation as characterization: Roman warfare was characterized by its organization, discipline, and strategic innovation
- Engineering support: Romans also emphasized engineering, constructing roads and fortifications to support their campaigns
- Combined tactics/logistics: Overall, Roman warfare was a blend of tactics, technology, and logistical planning, contributing significantly to the expansion of their empire
- Operational preference: preponderance of Roman campaigns exhibit a preference for direct engagement in open battle and, where necessary, the overcoming of fortified positions via military engineering
- Fortified camps adeptness: The Roman army was adept at building fortified camps for protection from enemy attack, but history shows a reluctance to sit in the camp awaiting battle and a history of seeking open battle
- Engineering-siegecraft skill: Roman soldiers were highly skilled in pitched battle, siegecraft, and military engineering; In the course of their campaigns Roman troops built military roads, bridges, and large permanent camps
- Siege engineering superiority: Roman engineers were also used extensively if the campaign ended in a siege. Rome had no equal with regards to siege warfare, mostly due to their engineering ability
- Logistics system: Well organized Roman logistics also sustained combat power, from routine resupply and storage to the construction of military roads
- Camp standardization: rectangular plan, defensive walls, gates, and internal streets arranged in a grid pattern, reflecting Roman military discipline and engineering expertise
- Discipline ideology: The Romans represented military discipline as ensuring their conquest and subjection of other ethnicities; combat training was an important element of Roman *disciplina militaris*
- Flexibility: Roman organisation was more flexible than those of many opponents
- Expansionist characterization: The majority of Rome's campaigns were characterised by one of two types. The first is the territorial expansionist campaign, normally begun as a counter-offensive
- Each URL supports the specific factual claim indicated: nutshell100.com for organization/discipline/engineering blend; Wikipedia Strategy page for open-battle preference and fortified camps; Wikipedia Roman infantry tactics/Castra and EBSCO for engineering/logistics/camps; LibreTexts/Cambridge for disciplina; Wikipedia Campaign history for expansionist typology.

**Confidence: HIGH**
Convergence across primary institutional, encyclopedic and academic sources on the same trait cluster. No contradictory evidence that these traits are *not* characteristic; remaining uncertainty is only period-specific weighting, which does not affect validity of any single trait as an acceptable answer.

**Grading guidance:**
- **Accept** any single trait that matches Canonical answer or Acceptable variants list, including paraphrases and different granularities of the same dimension (e.g., "built fortified camps each night" and "emphasized engineering" both acceptable as engineering dimension).
- **Do not require** exhaustive enumeration; one correct trait satisfies "What could be considered a trait..."
- **Reject** answers factually incorrect for Roman campaigns (e.g., "avoided engineering," "undisciplined / reliant primarily on mercenaries as core Republic trait," "purely naval with no land component," "consistently avoided open battle").
- **Do not penalize** period qualification: early Republic seasonal levy and late Empire professional standing army are both acceptable if accurately described as a trait of campaigns in that period.
- **Do not treat** different dimensions as requiring separate interpretation branches; grading set is flat - any valid dimension counts as correct.

### Repeat 3 — ambiguous — `5ab9e3d25542994dbf019889`

Question: What type of vegetation does Lythrum and Erythrina have in common?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "MULTIPLE"}`

Interpretation decision: MULTIPLE

Status: AMBIGUOUS

Question wording `What type of vegetation does Lythrum and Erythrina have in common?` uses unqualified `type of vegetation`. Prior inventory of plausible semantic dimensions: (a) physical form / habit / life-form / duration (herb / shrub / tree, herbaceous / woody, annual / perennial, deciduous / evergreen), (b) systematic / taxonomic vegetation class (flowering plant / angiosperm, dicot / eudicot / rosid), (c) habitat / ecological vegetation type (wetland, woodland, etc.), (d) phenology (deciduous vs evergreen) - sub-dimension of (a). Wording does not select among (a) and (b). Focused grounding was performed for all dimensions. (c) fails: no common habitat evidenced. (a) and (b) both honor all explicit constraints (ask for shared plant category), are evidence-supported, and produce materially different grading answers (perennial herb vs dicot/eudicot). This is material interpretation ambiguity, not scope/granularity or evidence conflict. Ranking by fit to wording: (a) is closest to conventional `vegetation type` (herbaceous vegetation, shrub vegetation), but (b) is also described as plant type/genus type in sources and satisfies literal reading; per contract cannot collapse merely because (a) is more common.

### Interpretation 1: Growth-form / habit / life-form dimension (structural vegetation type)

**Interpretation:** `type` = Raunkiær / horticultural life-form: herb / shrub / tree, herbaceous vs woody, and duration perennial vs annual, including leaf-retention deciduous.

**Answer:** Both are perennial plants, specifically perennial herbs / herbaceous perennials (and at genus level, perennial shrubs/trees with herbaceous members). The common denominator at genus level is `perennial` habit; the most precise shared herbaceous formulation is `herbaceous perennial` / `perennial herb` (clearly exemplified by *Lythrum* throughout and by *Erythrina herbacea* and the genus description "less often herbaceous").

**Scope:** What habit / life-form category do they share as vegetation

**Supporting evidence:**
* Lythrum is a Herbaceous Perennial forming a compact clump; *L. salicaria is an erect, herbaceous, perennial herb*
* Erythrina genus: `The evergreen or deciduous, perennial shrubs or trees`; Morphology General Habit `Perennial, Not climbing, Shrub/Tree`
* *Erythrina herbacea* Morphology General Habit `Perennial, Not climbing, Herb/Shrub/Tree` - establishes herbaceous member within perennial genus
* `Herbaceous vegetation` is treated as a vegetation type; `Vegetation means collective plant life... including trees, shrubs, and grasses` confirms herb/shrub/tree are vegetation types

**Confidence:** Medium-high for Lythrum uniform; Medium for Erythrina genus-level herbaceous (genus predominantly woody perennial, herbaceous only "less often" / specific to *E. herbacea*), High for perennial commonality.

**Grading guidance for Interpretation 1:** Accept any habit/duration answer in this dimension: `perennial`, `perennials`, `perennial herb`, `perennial herbs`, `herb`, `herbs`, `herbaceous`, `herbaceous perennial`, `herbaceous perennial herb`, `perennial shrub` / `perennial herb/shrub/tree`, `deciduous perennial` (Lythrum deciduous foliage and Erythrina deciduous forms share deciduous, but deciduous alone is acceptable as phenology sub-type). Granularity differences (herb vs herbaceous perennial vs perennial) are same interpretation. Reject `annual only`, `evergreen only`, `woody only without perennial`, `monocot`.

### Interpretation 2: Systematic / taxonomic vegetation class dimension

**Interpretation:** `type` = systematic botanical class: flowering plant (Angiospermae), dicot / dicotyledon / magnoliopsid, eudicot (true dicot), rosid.

**Answer:** Both are dicotyledonous flowering plants (angiosperms), specifically eudicots (true dicots) in the rosid clade.

**Scope:** What major systematic plant group do they share

**Supporting evidence:**
* Lythrum: `Clade: Eudicots` ... `Clade: Rosids` ... `Family: Lythraceae`; also `Lythrum - loosestrife dicot genus, magnoliopsid genus - genus of flowering plants having two cotyledons`
* Erythrina: `Clade: Eudicots` ... `Clade: Rosids` ... `Family: Fabaceae`
* Both families placed in major group Angiosperms/Flowering plants is inherent to those clades (Lythrum Lythraceae angiosperm, Erythrina Fabaceae angiosperm shown by rosid eudicot placement)

**Confidence:** High - consistent across species pages for all Lythrum species and all Erythrina species.

**Grading guidance for Interpretation 2:** Accept any systematic equivalent at this dimension: `angiosperm`, `flowering plant`, `flowering plants`, `dicot`, `dicots`, `dicotyledon`, `dicotyledonous`, `magnoliopsid`, `eudicot`, `true dicot`, `rosid`, `rosid eudicot`, `dicotyledonous flowering plant`. `Angiosperm` vs `dicot` vs `eudicot` vs `rosid` are granularity within same interpretation. Reject `monocot`, `gymnosperm`, `conifer`, `fern`, `bryophyte`, `algae`.

### Acceptance set (flattened)

A response is correct if it matches ONE interpretation dimension; do not require both.

* Habit/duration/phenology set: `perennial`, `perennial plant`, `perennial herb`, `herbaceous perennial`, `herb`, `herbaceous`, `deciduous perennial`, `deciduous herbaceous perennial`, `perennial herbaceous shrub`
* Taxonomic set: `angiosperm` / `flowering plant` / `dicot` / `dicotyledon` / `dicotyledonous` / `magnoliopsid` / `eudicot` / `true dicot` / `rosid` (any combination)

Habitat answers (e.g., `wetland`, `aquatic`, `tropical forest`) not in set - no evidence of common habitat; ecological vegetation types forest/grassland/desert not supported as shared.

### Necessary caveats

* Granularity vs ambiguity: `angiosperm`/`flowering plant` vs `dicot` vs `eudicot` vs `rosid` are not separate interpretations; same for `herb` vs `herbaceous perennial` vs `perennial` vs `deciduous perennial`. Do not create new branches for those.
* Evidence conflict not ambiguity: Erythrina genus is variable `deciduous or evergreen` and predominantly woody `shrub/tree`, with herbaceous only less often / in *E. herbacea*; Lythrum is uniformly herbaceous perennial. This variation is habit granularity, not a separate taxonomic reading. Shared `perennial` is the strongest common habit; `herbaceous` is supported via genus inclusion and type species.
* Habitat dimension tested and failed: Lythrum associated with wetlands/boggy areas, Erythrina with tropical woodlands - no common ecological vegetation type evidenced; do not accept habitat answers.
* Wording fit ranking: `herbaceous perennial / perennial` is the prototypical `vegetation type` (herbaceous vegetation vs shrub vegetation); `dicot/eudicot` is a plant group type. Because phrase is underspecified, both remain plausible per contract.

### Evidence (what each URL supports)

*  supports Lythrum as Herbaceous Perennial
*  supports Lythrum salicaria as erect herbaceous perennial herb
*  supports Erythrina as perennial shrubs or trees, evergreen or deciduous
*  supports Erythrina general habit Perennial, Shrub/Tree
*  supports *E. herbacea* as Herb/Shrub/Tree perennial - establishes herbaceous perennial within genus
*  supports herbaceous vegetation as a vegetation type
*  supports vegetation includes trees/shrubs/grasses as types
*  supports Lythrum clade Eudicots/Rosids
*  supports Erythrina clade Eudicots/Rosids
*  supports Lythrum as dicot/magnoliopsid genus

### Confidence

* Overall ambiguous status: High
* Interpretation 1 (habit/perennial): Medium-high
* Interpretation 2 (dicot/eudicot angiosperm): High

### Grading guidance (global)

* Do not penalize for granularity within dimension. Mark correct if answer contains any term from Acceptance set in correct dimension, even if other dimension not mentioned.
* Do not require both dimensions in one answer.
* Mark wrong if answer asserts non-shared class: `annual only`, `monocot`, `gymnosperm`, `non-flowering`, `evergreen only`, `wetland-only shared habitat`.
* If answer mixes dimensions (e.g., `perennial dicot` / `herbaceous eudicot`), mark correct if at least one dimension term is accurate.

### Repeat 3 — clear — `5ac4e4595542996feb3fe968`

Question: Which cola drink is the leader in India, Thums Up or Ricqlès?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: Question wording "Which cola drink is the leader in India, Thums Up or Ricqlès?" imposes three conjunctive constraints: (1) entity must be a cola drink, (2) geographic scope is India, (3) requested attribute is leadership. Focused grounding was run for every plausible semantic dimension: (a) product-type dimension - is each candidate classified as a cola vs another soft-drink type; (b) market-leadership dimension - is each candidate described as largest/top/billion-dollar cola brand in India by sales/share; (c) geographic-origin dimension - French mint drink vs Indian cola. All dimensions converge on the same outcome: Thums Up satisfies cola-type + India + leader; Ricqlès fails the type constraint (mint-flavoured, not cola) and has no evidence as Indian cola leader. No second dimension yields a supported, materially different grading answer. Related facts, evidence incompleteness, or granularity differences do not constitute interpretation ambiguity.

Status: SINGLE

Canonical answer: **Thums Up** — Thums Up is the leader / largest cola drink in India; Ricqlès is not a cola drink (it is a French mint-flavoured / peppermint soft drink) and is not the cola leader in India.

Acceptable variants:
- “Thums Up” alone; “Thums Up (Thumbs Up) is the biggest/top/largest/number-one cola in India”
- “Ricqlès is a mint/peppermint soft drink from France, not a cola, therefore not leader; Thums Up is”
- Spelling variations Thums Up / Thumbs Up accepted; do not require diacritics on Ricqlès
- Explanations that Thums Up leads the cola segment specifically (even if Sprite leads overall carbonated soft drinks) remain correct

Necessary caveats:
- Thums Up is defined as Type: Cola and as “a brand of cola” introduced in 1977 after Coca-Cola withdrew from India, Origin India.
- Within the wider carbonated beverage market, Sprite (lemon-lime, not cola) has topped total carbonated share in some years according to Euromonitor cited by Mint (20.5% vs Thums Up 15.4% in year ending 2016), but Thums Up remains the largest cola brand specifically.
- Ricqlès evidence consistently classifies it as mint/peppermint, never as cola; absence of India cola-leadership evidence is not an evidence conflict but a failure to meet the type constraint.

Evidence:
- Thums Up is largest cola brand in India: “Thums Up, the carbonated beverage brand owned by Coca-Cola Co.’s Indian unit, is the largest cola brand in India”; “has dominated India’s cola market in the past four decades”; “Thums Up continues to be the top cola drink in India.”; “as the largest cola brand by far”
- Thums Up is a cola of Indian origin: Type Cola; “Thums Up is a brand of cola.”; “Thums Up is a 40-year-old home-grown cola brand”; quickly became India’s biggest cola brand
- Ricqlès is mint-flavoured soft drink, not cola, from France: “Ricqlès – mint-flavoured soft drink” listed under France; also listed as “Ricqlès – mint-flavoured soft drink” among non-cola entries; “Ricqlès is a peppermint spirit and a French soft drink.” with Alcool de Menthe created 1838 in Lyon; “France: Ricqlès … Originally a medical soda, Ricqlès is a mint-flavored soda”

Confidence: High — convergence of primary business press (Mint citing Coca-Cola and Euromonitor), encyclopedic classification, and brand history; no credible source classifies Ricqlès as cola or as Indian market leader.

Grading guidance:
- CORRECT if answer selects Thums Up as the Indian cola leader, even if it additionally explains that Ricqlès is mint/peppermint and not a cola.
- CORRECT if answer states Ricqlès is not a cola (French mint drink) and therefore Thums Up is the only valid cola leader of the two.
- INCORRECT if answer selects Ricqlès, says both are cola leaders, says neither, or claims Ricqlès is a cola leading in India.
- Do not penalize for noting broader market nuance (e.g., Sprite leading total carbonated) provided the cola-segment answer remains Thums Up.
- Minor spelling/accent variations accepted.

### Repeat 3 — clear — `5adce8c15542992c1e3a249d`

Question: Are Thyme and Cortaderia both aromatic plants?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: Before searching, two semantic dimensions for "aromatic" were inventoried: (A) botanical/horticultural aromatic plant = plant with pleasant distinctive smell / volatile essential oil / fragrance, and (B) chemical aromatic = plant containing/producing aromatic (benzene-ring) hydrocarbons. Focused grounding searches were run for both. Dimension A is lexically primary for "aromatic plant", has explicit definitions and direct evidence, and yields a different answer for the two taxa. Dimension B was tested with searches for "Cortaderia selloana aromatic compounds essential oil" and "Cortaderia aromatic" - results show only that pyrolysis of Cortaderia biomass yields fractions classified as aromatic organic compounds, not that Cortaderia is classified as an "aromatic plant" in that sense; no source describes Cortaderia as an aromatic plant via chemistry. Dimension B therefore lacks evidence that would satisfy all question constraints and produce a materially different grading answer ("both are aromatic"). The single remaining reading honors the wording and is fully supported. The Cortaderia nitida "fragrant flowers" note is a species-level scope variation, not a separate interpretation of "aromatic", and does not make the genus Cortaderia aromatic.

Status: SINGLE

Canonical answer: No. Thyme (Thymus, especially Thymus vulgaris) is an aromatic plant, Cortaderia (pampas grass, e.g., Cortaderia selloana) as a genus is not an aromatic plant. They are not both aromatic plants.

Acceptable variants:
- No / False / Not both
- Only thyme is aromatic; Cortaderia is not aromatic / is non-fragrant / is an ornamental grass not known for fragrance
- Thyme yes, Cortaderia no
- Any paraphrase that denies the conjunction while affirming thyme is aromatic and Cortaderia generally is not

Necessary caveats:
- "Aromatic plant" in standard botanical/horticultural use means a plant species having a pleasant distinctive smell with high content of aromatic volatile substances collectively known as essential oil. Thyme fits this; Cortaderia generally does not.
- Within Cortaderia, one species Cortaderia nitida is described as used "for its fragrant flowers", but the genus as a whole is described as "not known for their fragrance" and commercial/cultivar listings record "Fragrance | Non-Fragrant" for Cortaderia selloana, "Fragrance: Non-fragrant" for Cortaderia selloana 'Pumila', and "Fragrance Non-fragrant" for Cortaderia selloana 'Rosea'. This species-level exception does not make the genus aromatic and does not make the answer to "both" Yes.
- Chemical sense: pyrolysis studies classify gas/bio-oil fractions from Cortaderia selloana biomass into non-aromatic and aromatic organic compounds including "Aromatic organic compound with 1 benzene ring". This is a chemical analysis of thermal decomposition products, not a classification of Cortaderia as an "aromatic plant" and provides no support for an alternative interpretation that would change grading.

Evidence:
- Definition of aromatic plant as fragrant essential-oil plant: **Aromatic Plant** plant species having a pleasant distinctive smell. These plants have high content of aromatic volatile organic substances collectively known as essential oil - establishes Dimension A.
- Thyme is aromatic: Thyme genus comprises "fragrant aromatic undershrubs"; Thymus vulgaris description: "highly aromatic grey-green leaves"; Thymus vulgaris has "fragrant foliage" - supports thyme = aromatic.
- Cortaderia genus is not fragrant/aromatic: "Are members of the Cortaderia plant genus fragrant? *Cortaderia* species are not known for their fragrance, but their visual impact in a landscape is significant"; tabular listings: "Fragrance | Non-Fragrant" for Cortaderia selloana and "Fragrance: Non-fragrant" for Dwarf Pampas Grass 'Pumila' - supports Cortaderia = not aromatic.
- Species-level exception within genus: "*Cortaderia nitida* is used as an ornamental plant, for erosion control, and for its fragrant flowers" - evidence conflict at species vs genus scope, retained as caveat.
- Chemical aromatic dimension tested and not supporting aromatic-plant classification: "Relative composition of the gas fractions for the different Cortaderia Selloana biomass pyrolysis processes Non aromatic organic compounds Aromatic organic compounds" - shows aromatic refers to pyrolysis products, not plant type.

Confidence: High

Grading guidance:
- Grade as correct any answer that states No / False / not both, with thyme affirmed as aromatic and Cortaderia denied as aromatic (or described as generally non-fragrant, non-aromatic, not known for fragrance, ornamental grass valued for visual plumes not fragrance).
- Grade as incorrect any answer that states Yes / True / both are aromatic plants, or that claims Cortaderia as a genus is generally aromatic.
- If answer mentions Cortaderia nitida fragrant flowers but still concludes the genus is not generally considered an aromatic plant and therefore the answer to "both" is No, grade as correct; do not treat the single-species note as making the conjunction true.
- Do not accept answers that rely on chemical pyrolysis production of aromatic hydrocarbons to classify Cortaderia as an "aromatic plant" — no source supports that classification.

### Repeat 3 — broad-but-single — `5ae336ef5542991a06ce995d`

Question: What profession do both Phil Alden Robinson and David Nixon share?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: Inventoried plausible semantic dimensions for "profession" (film director/director, screenwriter/writer, film producer/filmmaker, magician/entertainer, choreographer/dancer/artistic director) and plausible referents for underspecified "David Nixon" (American film director, English magician, Canadian choreographer, footballer). Performed focused grounding searches for each. Only the film-director reading honors all explicit constraints — both persons' primary, infobox-listed occupation is film director — with strong evidence. Other dimensions either require relying on secondary IMDb writer credit for the magician (whose primary occupations are magician and television presenter) or imposing producer on Phil Alden Robinson contrary to primary occupation listing, or stretching choreographer/artistic director to film director. These are scope/granularity or evidence-conflict issues, not materially different interpretations that both satisfy the shared-primary-profession constraint, so MULTIPLE is not warranted.

Status: SINGLE

Canonical answer: Both are (film) directors — the shared profession is film director / director (filmmaker). Phil Alden Robinson and David Nixon (the American film director) share the profession of film director.

Acceptable variants:
- film director, movie director, director — all acceptable
- filmmaker / film director and producer / director-producer where director is included — acceptable as granularity variant of film director
- Not acceptable alone: screenwriter/writer alone, producer alone, magician, television presenter/personality, choreographer, dancer, artistic director (ballet/theatre), footballer/American football player — none is the shared primary profession with evidence

Necessary caveats:
- "David Nixon" is ambiguous without qualifier. Evidence identifies at least five referents: English magician and television personality, Canadian dance choreographer, Scottish footballer, American football linebacker, and American film director and film producer. Only the American film director shares a primary profession with Phil Alden Robinson. Other referents do not satisfy the shared-profession constraint under primary occupation listings.
- Phil Alden Robinson is primarily documented as Film director, screenwriter, not producer in the Wikipedia infobox, even though some secondary databases list Writer, Director, Producer. David Nixon (director) is documented as film director and film producer, not screenwriter. Therefore producer-alone or writer-alone are not supported as shared primary professions; director is the only robust overlap.
- Evidence conflict exists for David Nixon (magician): IMDb lists him as a writer and actor, while Wikipedia infobox lists Occupations as Magician and television presenter and describes him as an English magician and television personality. The writer credit is a secondary filmography credit, not a primary profession, and does not create a material interpretation ambiguity for grading as "writer."

Evidence:
- Phil Alden Robinson Occupation = Film director, screenwriter — supports canonical occupation
- Phil Alden Robinson described as "is an American film director and screenwriter whose films include Field of Dreams, Sneakers, and The Sum of All Fears" — supports film director
- Phil Alden Robinson also described as "He is a writer and director, known for Field of Dreams (1989), Sneakers (1992) and The Sum of All Fears (2002)" — supports writer+director, confirms directorial role
- David Nixon (director) described as "is an American film director and film producer. He is the director and producer of Letters to God..." — supports film director as shared profession
- Corroborating same description: "David Nixon is an American film director and film producer. He is the director and producer of Letters to God..." — supports same
- David Nixon name disambiguation list showing multiple persons must be resolved to director referent — supports need to select film director referent
- David Nixon (magician) primary occupations = Magician and television presenter — rules out director/writer as primary shared profession for that referent
- David Nixon (magician) described as English magician and television personality — confirms non-film-director
- David Nixon magician secondary writer claim on IMDb "He was a writer and actor, known for Aladdin (1951)..." — shows why writer-alone is weak/conflicted
- David Nixon choreographer described as having career as dancer, choreographer and artistic director — rules out shared film director profession for that referent

Confidence: High (0.95) — primary-occupation sources consistently align on film director as the shared profession; alternative readings rely on secondary or stretched evidence and fail the primary-occupation constraint.

Grading guidance:
- Accept any response containing director qualified for film/movies: "film director," "movie director," "director," "filmmaker" when context indicates directing. If response says "director and producer" or "film director and producer" or "filmmaker/director," accept because director is included.
- Reject responses that list only "screenwriter / writer," "producer," "author," "magician," "television presenter/personality," "choreographer," "dancer," "artistic director" (ballet), "footballer" or "actor" alone — these are not the evidenced shared primary profession.
- If response answers "writer" based on Phil Alden Robinson screenwriter + David Nixon (magician) writer credit, mark incorrect — magician's primary occupations are magician/television presenter, not writer, and that entity does not satisfy the intended shared film profession. Treat as evidence-confusion, not alternative interpretation.
- If response specifies "David Nixon (director)" + "Phil Alden Robinson" both film directors, award full credit. If it says simply "director" without "film," award full credit.

### Repeat 3 — clear — `5ae6168f5542996de7b71b47`

Question: Adrian the Barbarian (later Olaff the Madlander) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The wording contains explicit venue constraints — "in the comic The Beezer, and later the merged Beezer and Topper, first introduced" — and an explicit entity identification — "Adrian the Barbarian (later Olaff the Madlander)" as one strip. The only reading that honors all constraints is the year the strip debuted as Adrian the Barbarian in The Beezer. A reading asking for the year the Olaff the Madlander title was first used fails the venue constraint (that title was first used in The Beano as reprints in July 2008, not in The Beezer/Beezer and Topper) and a reading asking for first appearance in the merged Beezer and Topper is a nested level of detail of the same continuity, not a materially different interpretation. Focused grounding was performed for both the Adrian (1989) and Olaff (2008) dimensions; only the 1989 dimension satisfies all constraints and has venue-matched evidence.

Status: SINGLE

Canonical answer: 1989 — Adrian the Barbarian (later reprinted as Olaff the Madlander) was first introduced in The Beezer in 1989, specifically Beezer No. 1727 dated 18 February 1989, described generally as "first introduced in 1989"

Acceptable variants:
- Year alone "1989" is sufficient
- More specific entailments acceptable: "February 1989", "18 February 1989", "18th Feb 1989", "Beezer No. 1727 (18 Feb 1989)"
- Do not require issue number or exact day; year is the requested attribute

Necessary caveats:
- Do not accept 2008: In July 2008, as of issue 3441, Adrian the Barbarian was reprinted in The Beano under the new title of Olaff the Madlander. This is reprint, not first introduction in The Beezer, and venue is The Beano, violating the question's venue constraint. Reprints also noted as "Reprinted in The Beano in 2008 as Olaff the Madlander"
- Do not accept 1990: Table entries show 1990 as End Date column value but the strip is listed under the 1989 chronological heading and detailed record gives 18 Feb 1989 debut; 1990 reflects continuation/end in Beezer, not first introduction. The merged Beezer and Topper did not exist until September 1990, and Adrian continued into it, so merged-title debut is not the first introduction
- Evidence conflict is scope/granularity, not interpretation ambiguity: general year (1989) vs specific date (18 Feb 1989 No. 1727) are same answer at different granularity

Evidence:
- General year in same wording as question: "***Olaff the Madlander*** (initially published as ***Adrian the Barbarian***) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989." Also paraphrased: "**Adrian the Barbarian** (later **Olaff the Madlander**) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989."
- Precise debut issue/date: "Adrian the Barbarian aka Olaff the Badlander first appeared in the Beezer in No 1727 dated 18th Feb 1989 and ran on through to the comics final issue No 1809 (15/9/90) and then went on to appear in the combined Beezer & Topper for most of its 153 issues."
- Chronological placement supports 1989 (not 1990): List entry shows "*Adrian the Barbarian* | Sid Burgon | 1990" under 1989 heading with note "Continued in the *Beezer and Topper*, and was reprinted as Olaff the Madlander in *The Beano*" — 1990 is End Date, not first introduction
- Corroborating artist start year: "began to draw for DC Thomson in 1989 drawing a revival of Biffo the Bear in The Beano and Adrian the Barbarian for The Beezer"
- Venue distinction for 2008 alternative: "In July 2008, as of issue 3441, *Adrian the Barbarian* was reprinted in The Beano under the new title of *Olaff the Madlander*." and "Reprinted in The Beano in 2008 as Olaff the Madlander"

Confidence: High — Convergent primary description ("first introduced in 1989") across Wikipedia and mirrors, plus specific issue-level corroboration (No. 1727, 18 Feb 1989) from specialist forum, plus artist biography anchoring 1989 start. No evidence supports alternative year satisfying venue constraints.

Grading guidance:
- Grade CORRECT any answer stating 1989 or entailed specific date (February 1989, 18 February 1989, 18th Feb 1989, No. 1727)
- Grade INCORRECT any answer stating 2008, 1990, 1993, or other year; 2008 is the Beano reprint as Olaff, not Beezer debut; 1990 is end/continuation year
- Require explicit year 1989; vague "late 1980s" without year is insufficient; extra detail (issue number, day) does not invalidate if year correct

## Pilot boundary

This is a 9-question, 3-repeat pilot for prompt behavior. No full-dataset run was started.

Private cache: `artifacts/muse_ambiguity_pilot_v3_2026-08-22.json`
