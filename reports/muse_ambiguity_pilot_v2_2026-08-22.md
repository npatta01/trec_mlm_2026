# Muse ambiguity-handling pilot

- Model: `muse-spark-1.2-contributor`
- Prompt variant: `reference_generic_branches_v2`
- Endpoint: Meta native Responses API with `web_search`
- Policy tested: two calls per item; no application Search or output-length cap
- Questions: 9 (3 ambiguous, 3 broad-but-single, 3 clear)
- Repeats per question: 3
- Expected records: 27
- Completed records: 27
- Hard failures: 0
- Stage-2 finals: 27
- Cache-only rerun items skipped without API calls: 0
- Search queries recorded: 337
- URL citations recorded: 180

The signals below are mechanical indicators for review; they are not a semantic quality judgment.

## Output signals

| Repeat | ID | Category | Status signal | Branch headings | Acceptance set | Citations |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | `56e2419acd28a01900c67c2a` | broad-but-single | SINGLE | 0 | no | 14 |
| 1 | `56f71740711bf01900a44932` | broad-but-single | SINGLE | 0 | no | 8 |
| 1 | `57260b4389a1e219009ac188` | ambiguous | SINGLE | 0 | yes | 0 |
| 1 | `57302972b2c2fd140056899e` | ambiguous | SINGLE | 0 | yes | 11 |
| 1 | `5ab9e3d25542994dbf019889` | ambiguous | MULTIPLE | 2 | yes | 1 |
| 1 | `5ac4e4595542996feb3fe968` | clear | SINGLE | 0 | no | 3 |
| 1 | `5adce8c15542992c1e3a249d` | clear | SINGLE | 0 | no | 4 |
| 1 | `5ae336ef5542991a06ce995d` | broad-but-single | SINGLE | 0 | no | 11 |
| 1 | `5ae6168f5542996de7b71b47` | clear | SINGLE | 0 | no | 4 |
| 2 | `56e2419acd28a01900c67c2a` | broad-but-single | SINGLE | 0 | yes | 6 |
| 2 | `56f71740711bf01900a44932` | broad-but-single | SINGLE | 0 | yes | 5 |
| 2 | `57260b4389a1e219009ac188` | ambiguous | SINGLE | 0 | yes | 14 |
| 2 | `57302972b2c2fd140056899e` | ambiguous | SINGLE | 0 | yes | 15 |
| 2 | `5ab9e3d25542994dbf019889` | ambiguous | SINGLE | 0 | no | 6 |
| 2 | `5ac4e4595542996feb3fe968` | clear | SINGLE | 0 | no | 10 |
| 2 | `5adce8c15542992c1e3a249d` | clear | SINGLE | 0 | no | 4 |
| 2 | `5ae336ef5542991a06ce995d` | broad-but-single | SINGLE | 0 | no | 1 |
| 2 | `5ae6168f5542996de7b71b47` | clear | SINGLE | 0 | no | 5 |
| 3 | `56e2419acd28a01900c67c2a` | broad-but-single | SINGLE | 0 | yes | 5 |
| 3 | `56f71740711bf01900a44932` | broad-but-single | SINGLE | 0 | no | 6 |
| 3 | `57260b4389a1e219009ac188` | ambiguous | SINGLE | 0 | yes | 12 |
| 3 | `57302972b2c2fd140056899e` | ambiguous | SINGLE | 0 | yes | 0 |
| 3 | `5ab9e3d25542994dbf019889` | ambiguous | MULTIPLE | 2 | yes | 0 |
| 3 | `5ac4e4595542996feb3fe968` | clear | SINGLE | 0 | no | 2 |
| 3 | `5adce8c15542992c1e3a249d` | clear | SINGLE | 0 | no | 6 |
| 3 | `5ae336ef5542991a06ce995d` | broad-but-single | MULTIPLE | 0 | yes | 19 |
| 3 | `5ae6168f5542996de7b71b47` | clear | SINGLE | 0 | no | 8 |

## Answers for manual review

### Repeat 1 — broad-but-single — `56e2419acd28a01900c67c2a`

Question: What structures can be viewed as both works of art and symbols of certain cultures?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks "What structures can be viewed as both works of art and symbols of certain cultures?" The noun "structures" is formally underspecified, but the dual predicate - being viewable simultaneously as *works of art* and as *symbols of cultures* - selects a single semantic dimension: physical built structures (architecture/buildings/monuments). Abstract readings (social structures, organizational structures, grammatical structures, data structures) do not satisfy the explicit "works of art" constraint as defined in evidence, and no evidence was found that would support such a reading as a materially different, evidence-backed answer that changes grading. Alternative phrasings found (architectural works, buildings, monuments, sacred structures) are nested levels of detail / scope within the same dimension, not distinct interpretations that honor all constraints and yield different answers.

Status: SINGLE

Canonical Answer:
Architectural works / buildings and other physical structures - in particular monuments and sacred/religious architecture.

The core definition: Architecture is the study and practice of designing structures, especially habitable ones, considered a visual art, and is both the process and product of constructing buildings or other structures. Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art.

More specifically:
* **Architecture generally:** So far, we have given very little consideration to architecture, yet it is one of the most culturally significant forms of art. Often, with structures that were built for group activities, they reflect the culture, its values, style, purposes, and preferences. From pyramids to cathedrals to skyscrapers, architecture reflects societies. The practice of architecture has been used as a way of expressing culture by civilizations on all seven continents, and for this reason architecture is considered to be a form of art.
* **Monuments / landmark structures:** Since at least Göbekli Tepe 11,000 years ago, humans everywhere have built structures of size and grandeur: from the Cahokia Mounds to Chichen Itza, from Great Zimbabwe to Angkor Wat, from St. Basil's Cathedral to Petra to the Sydney Opera House. These structures and monuments stand testament to human ingenuity and culture.
* **Sacred structures:** All cultures had some kind of sacred belief and they erected different types of structures including: temples, churches, mosques, or pagodas where they observed their religion. Globally, all of the buildings were designed, constructed, decorated and enhanced with some style of art: statues, stained glass, special trim, architectural configurations, or unique materials.

Scope: The answer is a class, not a single exemplar. Any member of the class satisfies the question. Civilizations are often identified with their surviving architectural achievements.

Acceptable Variants:
Accept the class label or any correct exemplars within the class. Grader should accept:
* General terms: architecture; architectural works; buildings; structures; physical structures; monuments; landmarks; sacred architecture / religious architecture; cultural architecture
* Sub-types: temples, churches, mosques, pagodas, stupas, synagogues, cathedrals, palaces, pyramids, etc.
* Specific examples: Pyramids / Pyramids of Giza, Parthenon / Acropolis, Great Wall of China, Taj Mahal, Angkor Wat, St. Basil's Cathedral, Petra, Sydney Opera House, Cahokia Mounds, Chichen Itza, Great Zimbabwe, Machu Picchu, Eiffel Tower, Statue of Liberty, etc. Any one example is sufficient; listing more does not invalidate; alternative correct examples from the same class are acceptable.
* Equivalent phrasing: "Architectural works, such as buildings, are frequently viewed as cultural symbols and works of art".

Necessary Caveats:
* The question is broad; it does not request an exhaustive list. Evidence conflicts do not exist for the core category; variation is in granularity (general category vs. specific examples), not in interpretation.
* Do not accept non-physical "structures" (e.g., social structures, economic structures, DNA structure, data structures) as they fail the explicit "works of art" + "cultural symbols" dual requirement in retrieved definitional sources.
* Sacred vs. secular monuments are both valid; an answer focusing only on one subset is still correct if it describes built structures as art and cultural symbols.

Evidence:
* Architecture defined as designing structures/buildings and considered a visual art - supports that structures are art.
* Architectural works in material form of buildings often perceived as cultural symbols and as works of art - directly supports canonical dual property.
* Architecture as way of expressing culture on all continents, considered art - supports cultural symbol + art status.
* Architecture as culturally significant form of art reflecting cultural values - supports general answer.
* Structures/monuments from Göbekli Tepe to Sydney Opera House stand testament to culture - supports monuments/landmark structures as cultural symbols and structures.
* All cultures erected temples/churches/mosques/pagodas, all decorated with art - supports sacred structures subtype as both art and cultural symbol.
* Variant wording: Architectural works, such as buildings, are frequently viewed as cultural symbols and works of art - corroborates same definition from alternate source.
* Architecture defined as buildings and other physical structures - clarifies "structures" = buildings.

Confidence: High - Multiple independent reputable sources (Wikipedia Architecture article mirrored across domains, Humanities LibreTexts, Infoplease) converge on identical definitional phrasing "often perceived as cultural symbols and as works of art" for architectural works/buildings/structures.

Grading Guidance:
* Award full credit for any answer that identifies the class as architectural works / buildings / monuments / structures (especially monumental or sacred architecture) or that provides one or more correct exemplars (e.g., pyramids, temples, monuments, Sydney Opera House, etc.).
* Do not require exhaustive enumeration; do not penalize omission of examples or inclusion of additional valid examples.
* Reject answers that cite only abstract/non-physical structures without establishing they are viewed as works of art and cultural symbols per the definitional evidence.
* Treat general category answers and specific-example answers as equally correct; granularity differences are not interpretation errors.

### Repeat 1 — broad-but-single — `56f71740711bf01900a44932`

Question: Bilateral treaties are concluded between how many states or entities?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question explicitly asks for a count ("how many states or entities"). The term 'bilateral' selects a single semantic dimension — number of parties/sides — and all explicit constraints point to that dimension. Alternative dimensions (physical form, habitat, taxonomy, function) referenced in the ambiguity contract do not apply to a count question. The only documented edge case (Switzerland-EU treaties with 17 formal signatories still counted as bilateral because parties are divided into two groups) is a formal-party-count vs. side-count granularity caveat, not a second interpretation that honors the wording and yields a different canonical number. No evidence supports a distinct, wording-consistent answer other than two.

Status: SINGLE - One clear meaning

Canonical answer: Two. A bilateral treaty is concluded between two states or entities — i.e., strictly between two subjects/parties/sides of public international law (two states, two international organizations, or one state and one international organization). Any agreement with more than two sides is multilateral.

Acceptable variants:
- 2 / two
- two states
- two entities
- two states or entities
- two parties / two sides / two subjects of international law
- two sovereign states / two nations / two governments
- two contracting parties (one on each side)
All convey the numeric answer two for the number of sides/entities to a bilateral treaty. Phrasing variants that specify the composition of the two (state-state, organization-organization, state-organization) are acceptable if the count remains two.

Necessary caveats:
- Formal signatory count vs. side count: It is possible for a formally bilateral treaty to list more than two signatories but remain bilateral in substance because the signatories are grouped into two sides. The canonical example is each Switzerland-European Union bilateral treaty, which has seventeen formal parties divided into two groups — the Swiss ("on the one part") and the EU and its member states ("on the other part") — and does not establish rights/obligations amongst the EU and its member states. This does not change the definitional answer of two sides; it is a granularity caveat for grading, not a separate interpretation.
- No inference beyond count: The answer does not vary by treaty subject matter (trade, investment, peace, boundaries) — bilateral always denotes two sides.

Evidence:
- Core definition: "Bilateral treaties are concluded between two states or entities."
- Strict two-subject definition: "A bilateral treaty (also called a bipartite treaty) is a treaty strictly between two subjects of public international law, generally either sovereign states or international organisations established by treaty."
- Composition of the two: "These two parties can be two nations, or two international organizations, or one nation and one international organization."
- Contrast defining bilateral as two: "Treaties may be bilateral (between two countries) or multilateral (involving more than two countries)."
- Library of Congress classification: "A multilateral treaty is a treaty involving more than two parties, while a bilateral treaty involves an agreement between two parties."
- Alternative phrasing: "Treaties may be bilateral (two parties) or multilateral (between several parties)"
- Multilateral threshold: "Any agreement with more than two parties is a multilateral treaty."
- Investment-treaty confirmation: "A bilateral investment treaty ('BIT') is a reciprocal legal agreement concluded between two sovereign States"
- Edge-case caveat — still bilateral with >2 formal parties: "It is possible for a bilateral treaty to have more than two parties; for example, each of the bilateral treaties between Switzerland and the European Union (EU) has seventeen parties: The parties are divided into two groups, the Swiss ('on the one part') and the EU and its member states ('on the other part')." and "It is possible for a bilateral treaty to involve more than two parties; for example, each of the bilateral treaties between Switzerland and the European Union (EU) has seventeen parties. The parties are divided into two groups, the Swiss ('on the one part') and the EU and its member states ('on the other part')."

Confidence: High. Multiple independent primary/reputable sources (Wikipedia Treaty and Bilateral treaty articles citing international law literature, Library of Congress Research Guide, Britannica-derived content) converge unanimously on two as the definitional count. The Switzerland-EU exception is well-documented and consistently described as still bilateral with two sides, creating no conflicting evidence on the canonical number.

Grading guidance:
- Accept: Any response stating the number is two, including numeric "2" and phrasing as two states, two entities, two parties, two sides, two subjects, two sovereign states, two nations, two governments, or two contracting parties. Accept if respondent adds correct composition detail (two states, two organizations, or one state + one organization) as long as count is two.
- Accept with caveat: "Two sides (though formally may list many signatories, e.g., 17 for Switzerland-EU, but grouped into two sides)" — treat as correct because canonical count is preserved.
- Reject: "One", "three or more", "at least two", "more than two", "several", "unlimited", "many", or answers confusing bilateral with multilateral / plurilateral. Reject "17" alone without qualification that it is still two sides, because that misstates the definition.
- Do not penalize for level of detail: "two states" vs. "two states or entities" vs. "two subjects of international law" are equivalent for grading.

### Repeat 1 — ambiguous — `57260b4389a1e219009ac188`

Question: What is an example of something that serves a function other than protection?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks for an example of an entity whose function is distinct from protection (function ≠ protection). The wording does not select a specific semantic dimension such as physical form vs taxonomy vs habitat, and does not require that the entity also serves a protective function. The only plausible alternative reading — that the entity must serve protection PLUS an additional function — adds an unstated conjunctive condition not in the text. Inventory of dimensions shows all valid examples (skin, skeleton, mucus, paint, etc.) satisfy the single explicit constraint “serves a function other than protection” and differ only in domain/instantiation, which is scope/granularity, not interpretation ambiguity, and would not produce a materially different grading outcome when graded against the broad reading. No second reading honors all explicit constraints, has distinct supporting evidence, and would change pass/fail while ranking higher on fit to wording.

Status: SINGLE

Canonical answer:
An example is the human integumentary system / skin, or the human skeletal system, which serves functions other than protection. For the skin, functions distinct from protection include thermoregulation, sensation, and vitamin D synthesis, also described as protecting the body, acting as a sensory organ, modulating body temperature and synthesizing vitamin D, and as playing critical roles in protection, sensation, thermoregulation, immune defense, and vitamin D synthesis. For the skeletal system, major functions include body support, facilitation of movement, protection of internal organs, storage of minerals and fat, and blood cell formation, also summarized as support and protection, facilitation of movement, storage of minerals, production of blood cells in red bone marrow. Any other verifiable example of an entity with a documented non-protective function is also correct.

Acceptable variants / Acceptance set:
Any single correct pairing of entity + non-protective function must be accepted. Accept at least:

* Skin / integumentary system: thermoregulation (sweat glands and blood vessel dilation/constriction to cool when overheated), sensation / sensing stimuli / sensory organ, vitamin D synthesis, acting as sensory organ and modulating body temperature
* Skeletal system / bones: supports the body, facilitates movement, produces blood cells, stores and releases minerals and fat; support, facilitation of movement, storage of minerals, production of blood cells; also described as system of levers for movement, site for mineral storage and blood cell formation (haematopoiesis)
* Mucus / mucus gels: serving functions from lubrication (as seen e.g. in the eyes)
* Paint / coating: facilities are often painted for other purposes than, or in addition to, protection — these include appearance, marking, and safety

Accept also other factually accurate examples where the function is verifiably not protection, e.g., heart — pumping blood/circulation; feathers — flight; clothing — adornment; government — order/conflict resolution; ribosome — protein synthesis — provided the non-protective function is stated accurately.

Necessary caveats:
* The question is open-ended; there is no single uniquely correct example. Do not require a specific domain (biology vs manufactured object vs institution).
* One correct non-protective function suffices; answer need not list all functions and need not state a protective function. An entity that also protects (e.g., skin, skeleton, paint) and an entity that does not protect at all (e.g., heart) are both valid under the canonical reading.
* Do not penalize level of detail (e.g., “thermoregulation” vs “sweating to cool the body” vs “vasodilation”) if the function is accurately non-protective.
* Reject only if the stated function is actually protective (e.g., shielding, barrier, defense) mislabeled as non-protective, or is factually incorrect.

Evidence:
* Skin plays important roles in protection, sensing stimuli, thermoregulation, and vitamin D synthesis; sweat glands cool when overheated; thermoregulation also by vessel dilation/constriction — supports skin has non-protective functions distinct from protection
* Skin and accessory structures: protecting body, preventing dehydration, acting as sensory organ, modulating body temperature and synthesizing vitamin D — supports sensory and temperature and vitamin D as non-protective
* Skin plays critical roles in protection, sensation, thermoregulation, immune defense, and vitamin D synthesis — reinforces same
* Skeletal system performs critical functions: supports body, facilitates movement, protects internal organs, produces blood cells, stores and releases minerals and fat — supports skeleton has multiple non-protective functions
* Primary functions of skeletal system include support and protection, facilitation of movement, storage of minerals, production of blood cells in red bone marrow — supports same
* Mucus gels cover wet epithelium, serving functions from lubrication (as in eyes) to barrier formation — supports lubrication as non-protective mucus function
* Facilities are often painted for other purposes than, or in addition to, protection. These include appearance, marking, and safety — supports paint has non-protective purposes

Confidence: High — multiple primary, institutional, and textbook sources converge on skin and skeletal non-protective functions; paint and mucus examples provide independent domain corroboration.

Grading guidance:
* Accept any answer naming a specific entity/object/structure/system and accurately stating one function that is not protection, with or without mention of a protective function.
* Reject answers that only describe protection (or synonyms: shielding from injury, barrier against pathogens, defense) without a distinct non-protective function.
* Do not require citation or exhaustive list; accept any granularity that correctly identifies a non-protective function.
* If answer states a function that is factually protective but claims it is non-protective, mark incorrect; otherwise, any verifiable non-protective function in the Acceptance set or equivalent should pass.

### Repeat 1 — ambiguous — `57302972b2c2fd140056899e`

Question: What could be considered a trait of Rome's military campaigns?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question wording "What could be considered a trait of Rome's military campaigns?" uses an intentionally permissive predicate ("could be considered a trait") without selecting a narrower semantic dimension (e.g., taxonomy vs function vs physical form). All plausible fillers — campaign purpose/type, operational/tactical preference, organization/discipline, endurance — honor all explicit constraints and fall within the single dimension "characteristic property of campaigns." Although multiple distinct traits are attested with evidence, they are scope/granularity variants within that same dimension, not competing readings that redefine the question constraints or require mutually exclusive grading keys. No second dimension with different truth-conditions (e.g., trait = biological taxonomy vs trait = behavioral function) is licensed by the wording and supported by evidence such that it would produce a materially different grading outcome. Variation is therefore collapsed to one canonical answer with an Acceptance set.

Status: SINGLE

Canonical answer: Any one well-evidenced characteristic is correct. Centrally evidenced traits are:

* **Typological trait:** the majority of Rome's campaigns were characterised by one of two types — the **territorial expansionist campaign, normally begun as a counter-offensive, in which each victory brought subjugation of large areas of territory (allowing growth from a small town to ~55 million in the early empire)** and the **civil war**.

* **Operational/tactical trait:** **preference for direct engagement in open battle and, where necessary, overcoming fortified positions via military engineering; adept at building fortified camps (castra) but with a history of seeking open battle rather than sitting in camp**.

* **Organizational/disciplinary trait:** **extremely disciplined**, maintained as a standing force between conflicts, with **brutal discipline** including forms such as **decimation — every tenth man executed by his cohort for cowardice/mutiny/desertion** used by senior commanders.

* **Endurance trait:** **obstinate persistence overcoming appalling losses** — even the greatest enemies such as Pyrrhus and Hannibal would win the battle but lose the war.

Acceptable variants / Acceptance set: Grader must accept **any single** response that correctly names one trait from the following flattened set (more specific phrasings at finer granularity count as instances of the broader trait):

- territorial expansionist / expansionism / imperial expansion — normally begun as counter-offensive, subjugation of large areas
- civil war as characteristic campaign type
- preference for direct/open battle; seeking battle rather than remaining in fortified camps
- overcoming fortified positions via military engineering / siegeworks
- adept at / systematic construction of fortified camps (castra) with palisade and ditch; streets, gates, guards; 2-5 hour construction even for single-day stay
- extremely disciplined / strict discipline / disciplina militaris / well-exercised standing army
- brutal discipline — summary execution, beating to death, decimation (1 in 10 executed by comrades)
- obstinate persistence / resilience after catastrophic defeats; winning wars despite losing battles
- versatile/adaptable tactical handbook / measured, cohesive heavy-infantry tactics (quincunx, pila/scutum)
- serving triple purpose of securing borders, exploiting peripheral areas via tribute, and maintaining internal order

Necessary caveats:

- Typology caveat: "majority characterised by one of two types" is a summary typology, not exhaustive; other traits are preponderant tendencies.
- Period caveat: Republic and early Empire showed little reluctance to prosecute offensive operations and seek open battle; later Empire sources (e.g., Vegetius, 4th c.) describe a more conservative strategy of limited direct engagement — traits vary by era.
- Evidence-conflict caveat: Scholarship notes Roman warfare has been described as both restrained/disciplined and as especially bloody/brutal; decadence/decimation is attested as real but infrequent ultimate punishment — do not treat brutality and discipline as mutually exclusive, and do not require a finding that campaigns were uniquely brutal for the period.

Evidence:

- Two-type typology — majority of campaigns characterised by expansionist counter-offensive and civil war, with 55 million population claim:  — supports expansionist and civil-war answer.

- Triple purpose (securing borders, exploiting periphery via tribute, internal order) as context for campaign purpose: .

- Operational preference for open battle and engineering, adept at fortified camps, reluctance to sit in camp:  — supports tactical/engineering trait.

- Systematic camp construction detail (castra, palisade, ditch, streets, 2-5 hours, no other ancient army persisted so long):  — supports engineering/logistics trait as variant.

- Discipline — extremely disciplined, standing between conflicts, heavy infantry vs less disciplined allies:  — supports discipline trait.

- Brutal discipline via decimation — every tenth man executed by cohort members for cowardice/mutiny/desertion:  — supports brutal-discipline variant.

- Persistence — obstinate persistence overcoming appalling losses; Pyrrhus/Hannibal win battle but lose war:  — supports endurance trait.

Confidence: High that at least one trait from the Acceptance set is correct for "What could be considered a trait"; Medium-High for any single canonical phrasing, because the open-ended "could be considered" licenses multiple non-contradictory correct answers and periodization nuances apply.

Grading guidance:

- Grade CORRECT if response provides any one trait from Acceptance set with no false factual claim.

- Granularity: Accept broader categories (e.g., "expansionist") and finer instantiations (e.g., "built a fortified camp each night," "used decimation," "sought open battle with pila and scutum") as correct — these are nested levels of detail, not separate interpretations.

- Grade INCORRECT only if trait is contradicted by evidence (e.g., "primarily naval after First Punic War," "avoided open battle as a rule in Republic," "undisciplined irregular force") or is unrelated to campaigns.

- If multiple traits listed, grade CORRECT if at least one is from Acceptance set and none are disqualifying falsehoods. Do not penalize for adding historical context (dates, generals, regions) if core trait is accurate.

### Repeat 1 — ambiguous — `5ab9e3d25542994dbf019889`

Question: What type of vegetation does Lythrum and Erythrina have in common?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 2, "grading_guidance_signal": true, "status_signal": "MULTIPLE"}`

Interpretation decision: MULTIPLE

Status: AMBIGUOUS

Interpretation 1 — Vegetation **life-form / growth-form / structural type** (what kind of plant they ARE)

- Interpretation: “type of vegetation” = morphological / phenological vegetation class (herb vs shrub vs tree, woody vs herbaceous, annual vs perennial, deciduous vs evergreen, forb)
- Answer: Both are **herbaceous perennial vegetation — perennial deciduous herbs / perennial forbs** (herbaceous perennial / perennial herb is the minimal intersection; deciduous is the shared phenology)
- Scope: Genus-level life-form. Lythrum is uniformly herbaceous; Erythrina as genus is variable but *includes* that life-form (rarely perennial herbs).
- Supporting evidence:
  - Lythrum Habit: Annual, perennial herb
  - Lythrum can be annuals or herbaceous perennials
  - *Lythrum salicaria*, also known as Purple-Loosestrife, is a perennial deciduous herb
  - Lythrum salicaria Life history perennial, Woodiness herbaceous, Growth form herb, Foliage retention deciduous
  - Erythrina is a perennial deciduous herb, shrub, subshrub or tree
  - Mostly trees or shrubs, rarely perennial herbs (herbaceous resprouters), mostly deciduous
  - Trees or shrubs [or perennial herbs]
  - Erythrina Life history perennial, Woodiness herbaceous woody, Growth form herb shrub subshrub tree, Foliage retention deciduous
  - Erythrina herbacea Perennial herb or shrub or small tree
  - Erythrina herbacea Shrublike herbaceous perennial
  - Erythrina ×bidwillii classified as a woodybased perennial herb, growing ... but going deciduous
- Confidence: High
- Grading guidance for this reading: Accept any combination of {perennial} + {herb/forb/herbaceous} + optional {deciduous}. Examples: herb, forb, herbaceous plant, perennial herb, herbaceous perennial, perennial forb, perennial deciduous herb, deciduous herbaceous perennial. Accept partial “perennial” or “herbaceous” alone as less complete but not wrong. Reject “tree only”, “shrub only”, “annual only”, “evergreen only”, “woody tree” as exclusive common type.

Interpretation 2 — Ecological **vegetation community / habitat type** (what kind of vegetation they occur IN)

- Interpretation: “type of vegetation” = phytosociological / habitat / community type (wetland vs dry woodland, swamp, marsh, riparian)
- Answer: Both share **wetland vegetation — specifically freshwater swamp / marsh / riparian / wetland-margin vegetation** (low-lying freshwater swamps, riverbanks, swamps, marshes, wetlands)
- Scope: Community type at species level within each genus. Lythrum as genus is predominantly wetland; Erythrina as genus is predominantly dry sandy woodland / wooded grassland but contains well-documented wetland / swamp / riparian species (E. fusca, E. crista-galli).
- Supporting evidence:
  - Lythrum Habitat: Invades many wetland and marsh habitats including freshwater wet meadows, tidal and non-tidal marshes, river and stream banks, pond edges
  - Lythrum Habitat: Swamps, marshes, other wet places
  - E. fusca In low-lying freshwater swamps *Erythrina fusca* attains huge dimensions and sometimes develops almost pure stands
  - E. crista-galli Habitat: It is commonly found in moist, low-lying areas such as riverbanks, wetlands, and marshes
  - The natural habitat of *E. crista-galli* is forests near watercourses, swamps and wetlands
  - Contrast (shows why this reading is ambiguous): E. herbacea Habitat: Maritime forests, dry sandy woodlands, hammocks, longleaf pine sandhills and woodland and wooded grasslands — i.e. most Erythrina are not wetland.
- Confidence: Moderate — Lythrum-wetland link is high; Erythrina-wetland link is species-specific (E. fusca, E. crista-galli) not genus-wide, but still directly attested.
- Grading guidance for this reading: Accept wetland, freshwater swamp, marsh, fen, riparian wetland, swamp forest, low-lying freshwater wetland, riverbank / riparian vegetation, wetland margin. Reject dry sandy woodland, xeric woodland, savannah grassland alone as *shared* habitat, because Lythrum does not share that. If grader adopts habitat reading, do not penalize growth-form answer and vice-versa; both dimensions are valid under ambiguity.

Acceptance set (flattened — grader may accept any):
- herbaceous perennial / perennial herb / perennial forb / herbaceous deciduous perennial / deciduous herb / perennial deciduous herb / forb
- wetland vegetation / freshwater swamp vegetation / marsh / swamp / riparian wetland / riverbank wetland / low-lying freshwater swamp / wetland-margin vegetation

Necessary caveats:
- Growth-form caveat: Erythrina genus includes trees and shrubs as its predominant form; the shared form is only the herbaceous-perennial subset. An answer claiming the *only* shared form is “tree” or “shrub” fails because Lythrum is not woody tree/shrub.
- Habitat caveat: At genus level Erythrina is not universally wetland; only some species (e.g., E. fusca — swamp immortelle, E. crista-galli) are wetland/riparian, while E. herbacea and many others are dry sandy woodland / wooded grassland. Evidence conflict is not interpretation ambiguity; it limits the generality of the wetland answer to species-level overlap.
- Granularity caveat: “perennial” alone, “herb” alone, or “deciduous” alone are nested levels of detail of Interpretation 1, not separate interpretations; they should be graded as partially complete, not as wrong.
- Taxonomic fact (both are angiosperms / dicots / eudicots: Lythrum Calflora Category: angiosperm USDA PLANTS group: Dicot; Erythrina Kingdom: Plantae Clade: Angiosperms Class: Dicots) is true but is a taxonomic rank, not a “vegetation type” in the ecological/structural sense and is not required for grading under either Interpretation.

Evidence:
- Lythrum life-form evidence: see Interpretation 1 citations 3162126790970320213, 8882573586548738929, 6269371554137138560
- Erythrina life-form evidence: see Interpretation 1 citations 1658560889456624351, 4170194596918925799, 1770765675024657923, 8724850783499621741, 5565492881596150181
- Wetland habitat evidence: see Interpretation 2 citations 4664605477683143046, 3596981713570675587, 7259031380519417984, 5868229753171172061

Confidence:
- Overall decision to branch: High that genuine material ambiguity exists between structural and habitat dimensions.
- Interpretation 1 confidence: High
- Interpretation 2 confidence: Moderate

Grading guidance:
- If question is graded under SINGLE-reading assumption, prioritize Interpretation 1 (structural herbaceous perennial deciduous herb) as the best fit to “type of vegetation does [taxa] have” (what they are, not where they grow).
- Under AMBIGUOUS grading, accept any answer in the Acceptance set; do not mark a student wrong for providing the other branch’s answer.
- Do not accept related but non-shared types: annual only, woody tree only, shrub only, evergreen only, dry xeric woodland only, grassland only, xerophytic succulent vegetation.

### Repeat 1 — clear — `5ac4e4595542996feb3fe968`

Question: Which cola drink is the leader in India, Thums Up or Ricqlès?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question wording "Which cola drink is the leader in India, Thums Up or Ricqlès?" selects the semantic dimension *flavor/category = cola* and the geographic dimension *India*. Inventory of plausible dimensions (e.g., physical form, taxonomy, market segment) shows only this dimension satisfies all explicit constraints: "cola drink" requires a cola-type beverage. Evidence supports only Thums Up as a cola; Ricqlès is evidenced as a mint-flavoured soft drink / peppermint spirit from France, not a cola. No alternative dimension that honors all constraints yields a materially different answer that would change grading. Differences between "cola leader" vs "overall carbonated beverage leader" (Sprite) are scope/granularity, not interpretation ambiguity, because the question explicitly restricts to cola.

Status: SINGLE

Canonical answer: Thums Up

Acceptable variants:
- Accept "Thums Up" as canonical; accept case-insensitive and common historical spelling variants "Thumps Up" / "Thumbs Up" when clearly referring to the Indian cola brand
- Accept phrasing "Thums Up is the largest / No.1 / leading cola (brand) in India"
- Do not accept Ricqlès, "Ricqlès is the leader," "both," "neither," or an answer that treats Ricqlès as a cola leader in India

Necessary caveats:
- The premise that both options are cola drinks is inaccurate. Only Thums Up satisfies the "cola drink" constraint. Ricqlès is evidenced as a mint-flavoured soft drink, not a cola.
- Within the broader Indian carbonated soft-drink market, Thums Up consistently rules the *cola* scene but is described as being topped overall by Sprite in some reports; this does not affect the answer to the cola-specific question.

Evidence:
- Thums Up is a brand of cola: Infobox Type: Cola; Text: **Thums Up** is a brand of cola
- Thums Up is the largest cola brand in India: "Thums Up, the carbonated beverage brand owned by Coca-Cola Co.'s Indian unit, is the largest cola brand in India"; same statement in second result
- Thums Up rules the cola scene in India: "today Thums Up rules the cola scene in India, only topped in the carbonated beverage arena by Sprite"
- Thums Up has been India's No. 1 cola brand by volume: "Thums Up has been India's **No. 1 cola brand by volume** for over a decade"
- Ricqlès is not a cola – classified as mint-flavoured soft drink: "Ricqlès – mint-flavoured soft drink"; same list entry; further description: "**Ricqlès** is a peppermint spirit and a French soft drink"
- Additional mint description: "France: Ricqlès Originally a medical soda, Ricqlès is a mint-flavored soda"

Confidence: High

Grading guidance:
- CORRECT: Any answer that selects Thums Up as the Indian cola leader (including variants above) and does not select Ricqlès as leader.
- INCORRECT: Answer selects Ricqlès, selects both, selects neither, claims Ricqlès is a cola leader in India, or claims Thums Up is not a cola.
- Do not penalize an answer that adds correct context that Sprite is the overall carbonated beverage leader while Thums Up remains the cola leader, as this is a consistent scope clarification.
- Minor spelling variations of Thums Up (Thumbs Up / Thumps Up) should not be penalized if intent is unambiguous.

### Repeat 1 — clear — `5adce8c15542992c1e3a249d`

Question: Are Thyme and Cortaderia both aromatic plants?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE

**Status:** SINGLE - One clear reading; no material interpretation ambiguity

**Reason no branch:** Before searching, inventoried plausible semantic dimensions for "aromatic plants": (a) horticultural/botanical sense = fragrant plants producing essential oils/volatile compounds with pleasant scent; (b) chemistry sense = plants containing aromatic (benzenoid) hydrocarbons; (c) functional sense = culinary/spice use. Dimension (a) is the established definition for the collocation "aromatic plants" - definitions state aromatic = having noticeable and pleasant smell fragrant and aromatic plants are those that produce essential oils or compounds that emit a strong fragrance. Dimension (b) has no supporting definition that classifies plants as "aromatic plants" by chemistry; sources describing Cortaderia pyrolysis bio-oils containing polycyclic aromatic hydrocarbons describe industrial products, not a plant class, and no source defines "aromatic plant" as "plant containing aromatic rings." Dimension (c) converges with (a) in evidence and grading outcome. Because only dimension (a) honors all question constraints, has definitional support, and yields a different answer for the pair, no second branch satisfies the contract's requirement that each branch honors wording, has supporting evidence, and would change grading. The Cortaderia nitida "fragrant flowers" claim is an isolated species-level conflict, not a different semantic dimension.

**Canonical answer:** No. Thyme (Thymus) is an aromatic plant/herb; Cortaderia (pampas grass) is not. They are not both aromatic plants.

Thyme is explicitly classified as aromatic: any species of the genus Thymus, aromatic herbs; Thyme is a genus of about 350 species of aromatic perennial herbs; Thymus pulegioides is a small spreading subshrub with strongly aromatic leaves; Thymus vulgaris forms a woody subshrub with small and highly aromatic grey-green leaves; and Thyme is a low-growing, aromatic member of the mint family.

Cortaderia is explicitly classified as not aromatic: Are members of the Cortaderia plant genus fragrant? *Cortaderia* species are not known for their fragrance; product specifications list Botanical Name Cortaderia selloana | Fragrance Non-Fragrant; Dwarf Pampas Grass Fragrance: Non-fragrant; Pampas Grass Fragrant No; and Fragrance Odorless.

**Acceptable variants:**
- No / False
- Only Thyme is aromatic; Cortaderia is not
- Thyme yes, Cortaderia no / non-fragrant / odorless / not known for fragrance
- Thyme is aromatic herb, Cortaderia is ornamental grass with no significant scent

**Necessary caveats:**
- Scope is genus-level as asked. Thyme evidence applies to genus Thymus and best-known species T. vulgaris/pulegioides; Cortaderia evidence applies to genus and to dominant horticultural species C. selloana. The answer holds at both species and genus levels.
- Conflicting evidence: One secondary commercial entry lists Cortaderia selloana Rosea variants as "wonderfully fragrant" or "many fragrant flowers," conflicting with majority sources (RHS, Missouri Botanical Garden, multiple nursery specs, and Gardening HQ) stating non-fragrant/odorless. This is an evidence conflict, not an interpretation ambiguity, and does not support classifying the genus as aromatic plants.
- Chemical interpretation does not create alternative answer: no source defines "aromatic plant" by content of aromatic hydrocarbons; all definitional sources use the fragrance/essential-oil sense.
- Granularity caveat: describing Thyme as "highly/strongly aromatic" vs "aromatic" is same grading outcome.

**Evidence:**
- Definition of aromatic in plant context: aromatic = having a noticeable and pleasant smell : fragrant ... aromatic herbs; aromatic plants are those that produce essential oils or compounds that emit a strong fragrance
- Thyme classification: any species of the genus Thymus, aromatic herbs or shrubby plants of the family Labiatae; Thyme is a genus of about 350 species of aromatic perennial herbs; Thyme is a low-growing, aromatic member of the mint family; with strongly aromatic leaves; with highly aromatic grey-green leaves
- Cortaderia classification: In the genus *Cortaderia* are a couple of dozen species of grasses; Cortaderia species are not known for their fragrance; Fragrance | Non-Fragrant for Cortaderia selloana; Non-fragrant for 'Pumila'; Fragrant | No; Odorless

**Confidence:** High. Multiple independent primary horticultural, encyclopedic, and commercial specification sources converge. Definitional sources for "aromatic plant" align unambiguously with fragrance sense; no credible source supports a chemistry-based plant classification that would make both genera aromatic.

**Grading guidance:**
- **Accept as correct:** "No" or any paraphrase stating only Thyme is aromatic and Cortaderia is not / non-fragrant / odorless / not known for fragrance. Accept elaborations adding family (Lamiaceae vs Poaceae), essential oil content (thymol), or ornamental grass description if core distinction is correct.
- **Reject as incorrect:** "Yes, both are aromatic (plants/herbs)" or claiming Cortaderia as genus is aromatic.
- **Partial/outlier handling:** If answer notes isolated "fragrant" claim for a single cultivar/species (e.g., C. selloana Rosea) but still correctly concludes genus is not considered aromatic and answer is No, accept as correct. If answer generalizes from that outlier to claim genus is aromatic, reject.
- **Do not penalize** for synonym variation in fragrance description for Thyme (aromatic, highly aromatic, strongly aromatic, fragrant herb) or for omitting citations.

### Repeat 1 — broad-but-single — `5ae336ef5542991a06ce995d`

Question: What profession do both Phil Alden Robinson and David Nixon share?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The requested attribute "profession" has one clear semantic dimension (occupational role). Alternative phrasings "director" / "film director" / "filmmaker" are nested granularity variants within the same dimension, not distinct dimensions. Entity-level ambiguity for "David Nixon" was inventoried: disambiguation lists David Nixon (magician), David Nixon (choreographer), David Nixon (footballer), David Nixon (director), David Nixon (American football). Only David Nixon (director) shares a primary, infobox-level occupation with Phil Alden Robinson that fully honors the question wording. The alternative writer overlap via David Nixon (magician) relies on a secondary IMDb credit as writer and actor while his primary occupations are listed as Magician and television presenter, not a primary profession match; it is an evidence-conflict/secondary-credit issue, not a distinct supported interpretation that ranks equally on fit to wording.

Status: SINGLE

Canonical answer: Film director — both are film directors (also acceptable as director / movie director / filmmaker)

Supporting evidence:
- Phil Alden Robinson occupation is Film director, screenwriter
- Phil Alden Robinson is described as an American film director and screenwriter
- Phil Alden Robinson is also described as a writer and director
- David Nixon (director) is an American film director and film producer, specifically director and producer of *Letters to God* and co-producer and assistant director of *Fireproof* and *Facing the Giants*
- David Nixon name is ambiguous and includes other primary professions (e.g., Magician and television presenter for David Nixon (magician)), which do not share the film-director profession.

Acceptable variants:
- director
- film director
- movie director
- filmmaker / film-maker
- film director and producer / director-producer (if includes director as shared element)
Do not accept as sole answer: magician, television presenter, choreographer, dancer, football player / linebacker, screenwriter alone, writer alone, or producer alone without director. Granularity variants within directing (e.g., "Hollywood director") are acceptable; narrowing to "television director" alone is insufficient without film-director coverage.

Necessary caveats:
- The question requires disambiguation to David Nixon (director), the American film director and film producer. Other David Nixons (magician, choreographer, footballer, American football linebacker) do not share the film-director profession as primary occupation. The David Nixon (magician) secondary credit as writer and actor is not his primary occupation and does not constitute the primary shared profession with Phil Alden Robinson.
- There is no material semantic-dimension ambiguity between e.g., "director" vs "screenwriter" vs "producer" — those are values within the same occupational-role dimension, with only director supported as primary shared occupation for both subjects in reputable sources. Conflict between secondary credits vs infobox primary occupations is an evidence-weighting caveat, not an interpretation branch.

Evidence:
- Phil Alden Robinson — Occupation: Film director, screenwriter
- Phil Alden Robinson — American film director and screenwriter
- Phil Alden Robinson — writer and director (IMDb)
- David Nixon (director) — American film director and film producer
- David Nixon disambiguation — shows multiple persons with that name including David Nixon (director), American film director and producer
- David Nixon (magician) — Occupations: Magician and television presenter
- David Nixon (magician) — was a writer and actor (secondary credit)

Confidence: High — primary occupation for both as film director is consistently supported by Wikipedia infobox/lead and corroborated by filmography/biographical sources; alternative entity reading based on secondary writing credit is weak and not primary.

Grading guidance:
- Accept any response that identifies the shared profession as directing film (director, film director, movie director, filmmaker equivalent).
- Accept responses that phrase as "both are directors/film directors/filmmakers" even if additional true details added (e.g., also screenwriters/producers).
- Reject responses that name only a non-shared profession (e.g., magician, choreographer, football player, television presenter) or claim no shared profession.
- Reject responses that answer only "writer" or "screenwriter" or "producer" without director, because primary shared profession evidenced is director; writer overlap relies on secondary non-primary credit for David Nixon (magician) and is not the intended primary-occupation match.
- Do not penalize for not specifying "David Nixon (director)" disambiguator, but if disambiguator is given it must be the director/film producer.

### Repeat 1 — clear — `5ae6168f5542996de7b71b47`

Question: Adrian the Barbarian (later Olaff the Madlander) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks for the single temporal attribute 'first introduced in which year' for the strip identified as 'Adrian the Barbarian (later Olaff the Madlander)' in The Beezer (later Beezer and Topper). The wording explicitly selects the original debut year, not a type/kind/form dimension or the later reprint title. No alternative semantic dimension satisfies all explicit constraints and yields a materially different supportable year. Potential distractors (1990 as end-date/cover-star year, 2008 as Olaff reprint in The Beano, 1990 as merge year) are evidence-conflict / granularity differences, not distinct interpretations that honor the 'first introduced as Adrian in The Beezer' constraint.

Status: SINGLE

Canonical answer: 1989

Acceptable variants: 1989; February 1989; 18 February 1989; Beezer No. 1727 (18 Feb 1989) — all specify same calendar year 1989. Do not accept 1990, 2008, or ranges centered on other years.

Necessary caveats:
- Precise debut is documented as Beezer No. 1727 dated 18 Feb 1989, running through final Beezer issue No. 1809 (15 Sept 1990) and then into Beezer and Topper for most of its 153 issues. This explains why tables list 1990 as an end-date, not a start-date.
- The 2008 date refers to reprints as 'Olaff the Madlander' in The Beano (issue 3441, July 2008), not the first introduction as 'Adrian the Barbarian'.
- List tables that show '1990' in the adjacent column are labeling 'End Date' under a section heading '## 1989' that denotes release year, not start year 1990.

Evidence:
- Definition and year: "***Olaff the Madlander*** (initially published as ***Adrian the Barbarian***) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989."
- Chronological organization of Beezer strips confirms heading is release year: "The following is a **list of *Beezer* comic strips**, organized by *year of release date* and in *chronological order*:"
- Placement under 1989 heading: "## 1989 | *Adrian the Barbarian* | Sid Burgon | 1990 |  | Was the comic's cover star. Continued in the *Beezer and Topper*, and was reprinted as Olaff the Madlander" — showing 1990 is End Date, not start.
- Specific issue date: "Adrian the Barbarian aka Olaff the Badlander first appeared in the Beezer in No 1727 dated 18th Feb 1989 and ran on through to the comics final issue No 1809 (15/9/90) and then went on to appear in the combined Beezer & Topper"
- Corroborating career chronology: "began to draw for DC Thomson in 1989 drawing a revival of Biffo the Bear in The Beano and Adrian the Barbarian for The Beezer (which was recently reprinted in The Beano as Olaff the Madlander)."

Confidence: High — multiple independent sources (Wikipedia article text, chronological Beezer list, dated issue forum research, artist biography) converge on 1989 with exact issue/date; conflicting 1990 occurrences are explainable as End Date column.

Grading guidance:
- Grade CORRECT if answer states 1989 alone or with more specific date within 1989 (e.g., 18 Feb 1989, Feb 1989, Beezer #1727).
- Grade INCORRECT if answer states 1990 (misreading End Date), 2008 (Olaff reprint), 1988, or any year other than 1989, even if citing a table row that shows 1990 without the heading context.
- Do not penalize spelling variants of the strip name if year is correct.
- Ranges (e.g., 1989-1990) are INCORRECT unless 1989 is explicitly identified as first-introduction year.

### Repeat 2 — broad-but-single — `56e2419acd28a01900c67c2a`

Question: What structures can be viewed as both works of art and symbols of certain cultures?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: Before searching, plausible semantic dimensions of "structures" were inventoried: (a) physical built structures (buildings/monuments/architecture), (b) abstract/social structures (social hierarchies, organizational structures), (c) formal/symbolic structures (data structures, linguistic structures, geometric patterns). Wording requires both "works of art" AND "symbols of certain cultures." Only dimension (a) satisfies both constraints with supporting evidence. Dimensions (b) and (c) lack evidence that they are collectively described as both works of art and cultural symbols, would require stretching "works of art" beyond the question's built-environment framing, and would produce no materially different grading-relevant answer grounded in sources. Scope variation within (a) — category name vs. specific exemplars — is granularity, not interpretation ambiguity per contract.

Status: SINGLE

Canonical answer: **Architectural works / (built) structures — in the material form of buildings and monuments / architectural heritage** - which are often perceived as both works of art and cultural (and political) symbols. Historical civilizations are often identified with / known by their surviving architectural achievements.

This category includes public monuments and landmark buildings. Well-documented exemplars of the same category include the Egyptian pyramids, Roman Colosseum, Greek Parthenon/Acropolis, Great Wall, Taj Mahal, Moai, Statue of Liberty, Eiffel Tower, and equivalent historic/iconic architectural monuments. Any correct identification of the category or one or more valid exemplars satisfies the question.

Acceptable variants / Acceptance set:
A grader should accept any of the following as correct (granularity variants of the single interpretation):
- Generic category: "architecture," "architectural works," "buildings," "built structures," "physical structures / nonbuilding structures designed/constructed," "monuments," "public monuments," "historic monuments," "cultural monuments," "architectural monuments," "landmarks / iconic buildings / heritage structures"
- Specific correct exemplars (illustrative, not exhaustive): Egyptian pyramids, Parthenon / Acropolis of Athens, Roman Colosseum, Great Wall of China, Taj Mahal, Moai of Easter Island, Statue of Liberty, Eiffel Tower, Angkor Wat, Borobudur, Hagia Sophia, Sydney Opera House, etc.
- Combined answers: e.g., "Architectural works — e.g., Eiffel Tower for France, Taj Mahal for India/Mughal culture"
Reject answers that refer only to non-physical structures (social structures, organizational structures, data structures, sentence structures, chemical structures) without linking to built architecture/monuments, or to non-structural art forms alone (paintings, music) that do not answer "structures."

Necessary caveats:
- The claim is generic ("often perceived as"), not universal: not every individual building is both an artwork and a cultural symbol, but the class of architectural works/monuments is characterized this way.
- Heritage/legal definitions (e.g., UNESCO) restrict "monuments" to those of outstanding universal value from the point of view of history, art or science — a narrower subset of the broader cultural-symbol usage, not a conflicting interpretation.
- Exemplar lists are open-ended; absence of one exemplar does not invalidate an answer that provides another valid exemplar or the correct category.

Evidence:
- Architecture defined as designing structures and considered a visual art: "Architecture is the study and practice of designing structures, especially habitable ones. It utilizes civil engineering techniques, but is considered a visual art. It is both the process and the product of sketching, conceiving, planning, designing, and constructing buildings or other structures."
- Architecture as expression of culture and form of art: "The practice of architecture itself, which began in the prehistoric era, has been used as a way of expressing culture by civilizations on all seven continents. For this reason, architecture is considered to be a form of art."
- Central dual-perception definition: "Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements."
- Duplicate attestation of same definition for architectural works
- Architecture as general term for buildings and other physical structures
- Architectural works perceived as cultural and political symbols and works of art; cities/regions/cultures known by their architectural monuments
- Such buildings as the pyramids of Egypt and the Roman Colosseum are cultural symbols
- Public monuments defined as structures that are works of art and symbols: "Public monuments are structures created to commemorate significant events, individuals, or cultural values within society. These works of art serve not only as symbols of remembrance but also as expressions of civic pride and identity"
- General definition monuments are structures built to commemorate: "Monuments are structures or statues that are built to commemorate a person, event, or significant historical moment."
- Cultural symbols arise from attaching meaning to structures: "Cultural symbols exist around the world because people attach meaning to structures, making them icons to represent a place, a region, or a historical period."

Confidence: High — the core statement that architectural works in material form are often perceived as both cultural symbols and works of art is directly and repeatedly attested in encyclopedic and academic sources, with convergent definitions for monuments/public monuments and multiple independent exemplar confirmations.

Grading guidance:
- ACCEPT any answer that identifies the category as built/architectural structures/monuments/buildings/heritage landmarks OR that lists one or more valid exemplars known to be described as cultural symbols and artworks. Do not require both category and exemplar.
- ACCEPT singular or plural phrasing, and qualifiers such as historic, iconic, monumental, heritage, cultural.
- ACCEPT granularity variants: category-only, exemplar-only, or combined are all correct under the single interpretation.
- REJECT answers that exclusively refer to abstract, social, economic, or computational structures with no built-physical referent, as they fail the "works of art + cultural symbols" dual constraint and lack evidence.
- Do not penalize additional correct detail (e.g., specifying culture: "Eiffel Tower — French culture") or partial lists; do not require exhaustive enumeration.

### Repeat 2 — broad-but-single — `56f71740711bf01900a44932`

Question: Bilateral treaties are concluded between how many states or entities?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks a single numeric attribute — "how many states or entities" are party to a bilateral treaty. The term "bilateral" selects one semantic dimension (number of contracting sides). No other plausible dimension (e.g., physical form, habitat, taxonomy, function) is invoked by the wording, and no alternative reading honors the explicit "how many" constraint while yielding a materially different correct number. The documented edge case of >2 signatories grouped into two sides does not produce a different canonical answer; it is a counting-method caveat, not a competing interpretation that would change grading.

Status: SINGLE - Unambiguous
Canonical answer: Two — a bilateral treaty is concluded between two states or entities / two parties / two sides.

Acceptable variants / Acceptance set:
- "2", "two", "two parties", "two states", "two entities", "two subjects of public international law", "two sides" or "two groups" are all acceptable.
- Formulations such as "between two sovereign states", "between two states or international organizations", "between two state parties / two state entities", or "bipartite treaty — strictly between two" are acceptable.
- "Two countries" acceptable for generic answer.

Necessary caveats:
- Classification is by sides, not by total count of signature instruments. It is possible for a bilateral treaty to have more than two signatories but still be classified as bilateral because the parties are divided into two groups, e.g., each of the bilateral treaties between Switzerland and the European Union has seventeen parties: the Swiss ("on the one part") and the EU and its member states ("on the other part") and establishes rights and obligations between the two sides severally, not among EU members inter se. This does not change the canonical count of two sides/parties; treaties with more than two independent sides are multilateral.

Evidence:
- Core definition — concluded between two states or entities: "Bilateral treaties are concluded between two states or entities."; same wording: "Bilateral treaties are concluded between two states or entities." and "Bilateral treaties are concluded between two states or entities."
- Formal definition — strictly between two subjects/parties: "A bilateral treaty (also called a bipartite treaty) is a treaty strictly between two subjects of public international law, generally either sovereign states or international organisations established by treaty. It is an agreement made by negotiations between two parties"; variant "A bilateral treaty (also called a bipartite treaty) is a treaty strictly between two state entities. It is an agreement made by negotiations between two parties"
- Enumeration of permissible party types: "A bilateral treaty (also called a bipartite treaty) is a treaty strictly between two state parties. These two parties can be two states, or two international organizations, or one state and one international organization." and summary "These two parties can be two states, or two international organizations, or one state and one international organization."
- US Government definition: "Bilateral treaty: An international agreement concluded between two states."
- Binary typology with multilateral: "Treaties may be bilateral (between two countries) or multilateral (involving more than two countries)."
- Caveat on Swiss-EU example still counted as bilateral (two sides): see  above.

Confidence: High — definition is consistent across encyclopedic (Wikipedia Treaty), legal (Practical Law, Oxford Public International Law paraphrase), and official U.S. Government (Senate Committee Print via govinfo.gov) sources with no conflicting definition for the generic numeric count.

Grading guidance:
- Accept any answer stating the number is two / 2, or that a bilateral treaty is between two parties/sides/states/entities/subjects.
- Accept qualified answers that add the Swiss-EU 17-signatory nuance provided they affirm the treaty is still bilateral because it has two sides/groups.
- Do not accept answers stating one, three, "more than two," "three or more," "plurilateral," or "multilateral" as the definition of bilateral.
- Numerical answer "2" alone is fully correct; phrasing "two states or entities" exactly matching source is fully correct.

### Repeat 2 — ambiguous — `57260b4389a1e219009ac188`

Question: What is an example of something that serves a function other than protection?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks for an existential example of an entity ("something") whose function ≠ "protection". The underspecified domain of "something" (biological structure, system, artifact, tissue, etc.) and the open set of qualifying functions creates broad *scope/granularity* - many different correct examples satisfy the same reading - not a semantic ambiguity in the requested attribute. The term "function" selects the functional dimension, and no alternative dimension (e.g., physical form vs habitat vs taxonomy) is forced by the wording that both honors all explicit constraints ("serves a function other than protection") and yields a materially different answer requiring a separate branch. Answering with a different example or naming the function itself (e.g., "support") is a level-of-detail variant within the same interpretation, not a competing interpretation.

Status: SINGLE

Canonical answer: A structure/system that also has a protective role but whose documented functions include one or more non-protective functions qualifies. Examples:

* **Skeletal system / bone** - besides protection, it supports the body, facilitates movement, stores and releases minerals and fat, and produces blood cells (hematopoiesis); summarized as body support, facilitation of movement, protection of internal organs, storage of minerals and fat, and hematopoiesis.
* **Skin / integumentary system** - with three main functions: protection, regulation, and sensation, its other functions are insulation, temperature regulation, sensation, and the production of vitamin D.
* **Painting / coating of facilities** - facilities are often painted for other purposes than, or in addition to, protection. These include appearance, marking, and safety.

Any one verifiable example of a thing + a specific non-protective function satisfies the question.

Acceptable variants / Acceptance set: The question is open-ended; a grader should accept any specific entity paired with a specific documented function distinct from protection, including but not limited to:
- Skeletal system/bones: support, facilitate movement (bones act as levers for muscle attachment), store minerals (calcium, phosphorus, potassium) and fat/triglycerides, produce blood cells - hematopoiesis - in red bone marrow
- Skin/integument: insulation, temperature regulation / modulating body temperature, sensation / sensory organ (touch, heat), synthesis/production of vitamin D, preventing dehydration, electrolyte balance, storing fats/cushion
- Painting/coatings: appearance / decoration, marking (airfield runways, streets, parking lots), safety / safety colors and designs, sanitation, preservation, identification
- Other valid non-protective functions if sourced: cell membrane - selective transport, receptor signaling; skeletal muscle - producing movement, posture, heat; roots - starch storage; feathers/wings - flight, thermoregulation, display

An answer may be phrased as the entity ("the skeleton"), the function ("support and movement"), or the pair ("the skeleton supports the body and enables movement"). All are acceptable if the function is explicitly distinct from protection.

Necessary caveats:
- "Protection" includes defending/covering internal organs (e.g., ribs protect lungs/heart, vertebral column protects spinal cord, cranium protects brain; skin protects against pathogens and excessive water loss). An answer listing only such protective functions does not satisfy the constraint.
- Vague answers without a concrete entity or concrete function (e.g., "something useful") should not be accepted.
- No exhaustive list required; one correct example is sufficient.

Evidence:
- Skeletal system is body system of bones and cartilage performing critical functions: supports the body, facilitates movement, protects internal organs, produces blood cells, stores and releases minerals and fat.
- On metabolic level bone matrix acts as reservoir for minerals especially calcium and potassium released to bloodstream; calcium essential for muscle contractions and nerve impulses.
- Bone marrow types: yellow marrow contains adipose tissue/triglycerides as energy source; red marrow is where hematopoiesis - production of blood cells - takes place, producing red blood cells, white blood cells, platelets.
- Major functions summarized as body support, facilitation of movement, protection of internal organs, storage of minerals and fat, and hematopoiesis.
- Skin is layer with three main functions: protection, regulation, and sensation.
- Skin's other functions are insulation, temperature regulation, sensation, and production of vitamin D folates.
- Painting for purposes other than protection: Facilities are often painted for other purposes than, or in addition to, protection. These include appearance, marking, and safety.

Confidence: High. Multiple primary, reputable sources (OpenStax/Medicine LibreTexts Anatomy & Physiology; Wikipedia Skin with cited references; Unified Facilities Guide Specifications - Painting for Purposes Other Than Protection) converge on the same non-protective functions for structures that also have protective roles; no conflicting evidence on these functional inventories.

Grading guidance:
- ACCEPT: Any answer naming a specific thing and stating a specific sourced non-protective function (or clearly implying the pair), e.g., "bones support the body / enable movement / store calcium / make blood cells", "skin regulates temperature / senses touch / makes vitamin D", "paint is used for appearance/marking/safety", "heart pumps blood", "lungs exchange gases", "roots store starch" - if supported by evidence that the function ≠ protection.
- REJECT: Answers that only describe protection (e.g., "skull protects brain", "skin protects from bacteria"); answers that give no concrete example or function; answers that mislabel a protective function as non-protective without source support.
- For borderline phrasing (e.g., "prevents water loss"), treat as protection unless source explicitly distinguishes it from protection; skin's "preventing dehydration" is listed alongside protection as a separate essential function and may be accepted if distinguished.

### Repeat 2 — ambiguous — `57302972b2c2fd140056899e`

Question: What could be considered a trait of Rome's military campaigns?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks for any characteristic considered a trait of Rome's campaigns. Phrased as "What could be considered a trait" it explicitly permits any valid characteristic exemplar and does not select a particular semantic dimension. Candidate traits such as organizational discipline, tactical preference for open battle/engineering, and strategic purpose of territorial expansion are not competing readings that honor the wording in mutually exclusive ways; they are parallel exemplars within the single broad reading "characteristic property of how campaigns were conducted/purposed." They differ in granularity/scope and topical focus, not in word-sense, and are interchangeable as correct answers under one interpretation with an Acceptance set. No wording constraint forces a choice between e.g. 'how organized' vs 'what purpose' – both satisfy "trait of campaigns." Therefore SINGLE with a flattened Acceptance set correctly covers grading without needing separate interpretation branches.

Status: SINGLE

Canonical answer: A trait is that Rome's military campaigns/military were **highly disciplined and organized** — characterized by superior organization, discipline and training, with a characteristic operational preference for direct engagement in open battle and overcoming fortified positions via military engineering/systematic fortified camp construction, and characterized historically as predominantly territorial-expansionist (exploiting/securing borders and peripheries), with civil war as the other major type. Providing any one of these factually supported characteristics satisfies the question.

Acceptable variants / Acceptance set:
Any one or combination of the following supported characteristics should be accepted (phrasing variants allowed):
- highly disciplined and organized; superior organization, discipline and training and lines L206-L208
- brutal/strict discipline,Summary execution/decimation, highest standards of unit cohesion, constant combat drills
- preference for direct engagement in open battle
- overcoming fortified positions via military engineering; adept at building fortified camps with palisade/ditch/castra ; engineering was integral part of any Roman campaign
- systematic camp construction at end of day's march (castra) – no other ancient army persisted so long in systematic camp construction
- territorial expansionist campaign (majority of campaigns characterised by one of two types, the first territorial expansionist, normally begun as counter-offensive, allowing growth from small town to 55 million) ; second type is civil war
- securing borders, exploiting peripheral areas via tribute, maintaining internal order
- adaptability, flexibility, logistics/infrastructure (roads, fortifications), obstinate persistence overcoming losses

Necessary caveats:
- "Could be considered" is open-ended; a single valid trait is sufficient, no exhaustive list required and no need to enumerate all campaign periods.
- Traits evolved over ~13 centuries (Kingdom to Western Empire fall 5th c. AD). Generalizations about discipline/expansionism apply most clearly to Republic and early Empire; late Empire shifted toward perimeter defense, heavy cavalry and greater barbarian recruitment, but this does not invalidate the general traits as correct answers.
- A specific weapon name (pilum/gladius), formation name alone, or commander/battle name without framing as a characteristic trait of campaigns is insufficient unless linked to the disciplined/organized/engineering/expansionist character.
- Conflicting emphasis across sources (some emphasize discipline, others expansionism/engineering) is evidence breadth, not interpretation ambiguity; all are valid under same reading.

Evidence:
- Military and social cultures of Rome were closely related and highly disciplined and organized — supports canonical answer "highly disciplined and organized."
- Undergirding tactics and weapons was the superior organization, discipline, and training that each legionnaire went through — supports discipline/training as defining trait.
- The preponderance of Roman campaigns exhibit a preference for direct engagement in open battle and, where necessary, the overcoming of fortified positions via military engineering. The Roman army was adept at building fortified camps — supports tactical/engineering trait.
- No other ancient army persisted over such a long period in systematic camp construction like the Romans, even if the army rested for only a single day and Romans would typically establish a strong field camp called a castra, complete with palisade and deep ditch — supports systematic engineering trait.
- From the outset, Rome's military typified this pattern, and the majority of Rome's campaigns were characterised by one of two types. The first is the territorial expansionist campaign, normally begun as a counter-offensive, in which each victory brought subjugation of large areas of territory and allowed Rome to grow from a small town to a population of 55 million — supports expansionist trait; second is civil war.
- Rome's military served the triple purpose of securing its borders, exploiting peripheral areas through measures such as imposing tribute on conquered peoples, and maintaining internal order — supports border-security/exploitation trait.

Confidence: High — convergent reputable sources (EBSCO Research Starters academic overview, Wikipedia Strategy and Campaign history syntheses citing scholarly literature) agree on discipline/organization, engineering/open-battle preference, and expansionist characterization as canonical traits.

Grading guidance:
- Accept any response that contains one or more items from the Acceptance set or semantically equivalent paraphrase (e.g., "well-organized," "well-disciplined," "rigorous training," "engineering-focused," "built fortified camps nightly," "sought open battle," "expansionist/imperial conquest," "aggressive territorial expansion").
- Do not penalize for granularity differences ("disciplined" vs "brutally disciplined with decimation" vs "highly organized legion/cohort/hierarchy") – all are same interpretation.
- Reject responses that assert the opposite trait (e.g., "unorganized," "avoided engineering," "purely defensive with no expansion") or that provide only an isolated fact without trait framing.
- If a response lists a correct trait plus an additional incorrect claim, fail only if the incorrect claim is presented as the central trait and contradicts evidence; minor embellishment alongside a correct core trait should still pass.

### Repeat 2 — ambiguous — `5ab9e3d25542994dbf019889`

Question: What type of vegetation does Lythrum and Erythrina have in common?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The wording "type of vegetation" selects the growth-form/habit dimension (herb/shrub/tree, herbaceous/woody, annual/perennial, deciduous/evergreen). A taxonomic reading such as "dicot/angiosperm/eudicot" is a systematic rank, not a vegetation type as used in horticulture, flora checklists and vegetation science, and therefore does not honor the explicit requested attribute as well as the habit reading. Apparent alternatives such as "herb" vs "perennial herb" vs "deciduous perennial herb" are nested levels of detail within that single habit dimension, not materially different semantic dimensions that would change grading.

Status: SINGLE

Canonical answer:
Lythrum and Erythrina share the herbaceous perennial (perennial herb / perennial forb) habit. At genus level the overlap is described as **Herbs / herbaceous perennials** - both genera contain perennial herbs, and in the concrete overlap both are deciduous perennials. The most concise common type is **herbaceous perennial herb / forb**, exemplified by *Lythrum* spp. and *Erythrina herbacea*.

Acceptable variants:
Any equivalent habit term within the same dimension should be accepted: herb, forb, herbaceous perennial, perennial herb, perennial forb, herbaceous perennial forb, perennial herbaceous plant, deciduous perennial herb/herbaceous perennial, shrublike herbaceous perennial. Specific phrasings such as "herbaceous perennial forming a clump" or "perennial forb" are same answer at different granularity. Do not require mention of deciduous, but deciduous perennial herbaceous is acceptable as more specific.

Necessary caveats:
* Neither genus is uniform: *Lythrum* "can be annuals or herbaceous perennials" and includes strictly perennial species such as *Lythrum virgatum* described as "**Herbs perennial,** slender". *Erythrina* is described broadly as "trees, shrubs, and perennial herbs" and as "Erythrina can be deciduous or evergreen subshrubs, shrubs or trees" with species-level variation: e.g., *Erythrina edulis* Tree, *Erythrina humeana* Shrub, *Erythrina herbacea* Perennial, and overall "Perennial, Not climbing, Herb/Shrub/Tree". The common intersection is the perennial-herb component.
* Both shared taxa are also deciduous: *Lythrum* "Foliage, Deciduous" and *Erythrina* "is a deciduous Shrub growing to 2 m", so "deciduous herbaceous perennial" is a more precise but not required variant.
* Evidence conflicts on woody vs herbaceous wording for *Erythrina* reflect species differences, not a separate interpretation; the genus-level inclusion of perennial herbs resolves the conflict.

Evidence:
* Lythrum is an herbaceous perennial: "Herbaceous Perennial *Lythrum* *salicaria* 'Feuerkerze' purple loosestrife 'Feuerkerze' Herbaceous perennial forming a compact clump"; also "Lythrum can be annuals or herbaceous perennials, with simple leaves in opposite pairs"; *Lythrum virgatum* "**Herbs perennial,** slender"; *Lythrum lanceolatum* listed as "Lythrum lanceolatum | loosestrife | Herbs".
* Erythrina contains perennial herbs and overlap as herb: "A genus of about 100 species, trees, shrubs, and perennial herbs"; *Erythrina herbacea* table entry "Erythrina herbacea | Cardinal Spear, Redcardinal | Perennial"; general habit "Perennial, Not climbing, Herb/Shrub/Tree"; genus description "Erythrina can be deciduous or evergreen subshrubs, shrubs or trees".
* Deciduous overlap supporting perennial-herb reading: Lythrum "Foliage, Deciduous"; Erythrina "is a deciduous Shrub".

Confidence: High — habit classification as herbaceous perennial / perennial herb/forb for both genera is explicit, repeated across RHS, Flora of North America, PFAF and regional floras, and the deciduous perennial detail is also explicitly attested.

Grading guidance:
* Accept: herb / forb / herbaceous plant / herbaceous perennial / perennial herb / perennial forb / perennial herbaceous forb / deciduous herbaceous perennial / shrublike herbaceous perennial. Any of these satisfies the canonical answer.
* Accept granularity variants: "perennial herb" without "deciduous" or "herbaceous" without "perennial" if clearly referring to herb habit; best to require perennial when possible but do not penalize "herb/herbaceous" alone as it is the core type.
* Reject: tree only, shrub only, woody tree, annual only, monocot, gymnosperm/conifer, fern, bryophyte, aquatic-only vegetation type, or habitat-only answers like "wetland" without habit linkage.
* Do not treat "dicot / angiosperm / eudicot / flowering plant" alone as correct under this wording; if given together with herbaceous perennial, grade as correct but taxonomy alone does not satisfy "type of vegetation" as asked.

### Repeat 2 — clear — `5ac4e4595542996feb3fe968`

Question: Which cola drink is the leader in India, Thums Up or Ricqlès?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question has one clear reading that satisfies all explicit constraints: "Which [of the two named drinks] is the [market-]leader in India in the category cola drink?" Thums Up satisfies the type constraint (cola), origin/market constraint (India), and leadership attribute with strong evidence. Ricqlès fails the type constraint - evidence consistently classifies it as peppermint / mint-flavoured spirit/soft drink, French, not a cola - and has no evidence of cola leadership in India. No alternative semantic dimension (physical form, habitat, taxonomy, function) honoring all constraints yields a different supported answer that would change grading. The previous draft's distinction is evidence about cola vs. mint type, not a competing interpretation; it confirms the same answer. This is intensity/scope granularity within the cola segment, not material interpretation ambiguity.

Status: SINGLE - Unambiguous

Canonical answer:
**Thums Up** is the leader - the largest / No.1 / dominant cola brand in India. **Ricqlès is not a cola drink** (it is a French peppermint / mint-flavoured spirit and soft drink) and is not the cola leader in India.

Acceptable variants:
- "Thums Up" (including spelling variant "Thumbs Up" without the 'b') alone is sufficient; acceptable paraphrases include: largest cola brand in India, India's biggest cola brand, India's No.1 cola, dominates/rules the cola scene in India, billion-dollar cola brand in India.
- Answer may optionally note Ricqlès is a French peppermint/mint-flavoured soft drink / peppermint spirit / Alcool de Menthe / mint-flavoured soda, not a cola, and therefore the distractor.
- Not acceptable: Ricqlès, both equally, neither, or any answer claiming Ricqlès is a cola leader in India.

Necessary caveats:
- Leadership refers to the **cola segment** of the Indian carbonated soft-drink market. Within the overall carbonated beverage market, Sprite (also Coca-Cola) has at times topped overall share (e.g., Euromonitor 2016: Sprite 20.5% vs Thums Up 15.4%), but within cola specifically, Thums Up remains the leader. Some sources phrase as "largest cola brand", others as "largest cola brand by far" or "rules the cola scene in India" - intensity differences do not change the binary answer.
- Figures are time-bound (e.g., Rs5,000 crore annual sales in 2017, billion-dollar brand in 2021, 85% share at 1993 sale) but directionally consistent across decades that Thums Up has dominated India's cola market in the past four decades.
- Ricqlès classification as mint/peppermint does not preclude it being listed among soft drinks, but it is consistently listed as mint-flavoured, not cola.

Evidence:
- Thums Up is a brand of cola; Type | Cola; Thums Up is a cola brand; Thums Up is a brand of cola introduced in 1977 to offset withdrawal of Coca-Cola from India
- Thums Up is the largest cola brand in India with Rs5,000 crore in annual sales; same statement repeated; Thums Up has dominated India's cola market in the past four decades; Thums Up remains India's largest cola brand by far; Thums Up quickly became India's biggest cola brand; today Thums Up rules the cola scene in India, only topped in the carbonated beverage arena by Sprite
- Ricqlès is a French manufacturing products based on peppermint; Ricqlès is a peppermint spirit and a French soft drink; The liquor Alcool de Menthe de Ricqlès was created in 1838; Ricqlès – mint-flavoured soft drink

Confidence: HIGH
Multiple independent, reputable sources (Wikipedia infobox and article text, Livemint citing Coca-Cola India VP Marketing, FoodNavigator-Asia, Atlas Obscura) unanimously describe Thums Up as India's largest/dominant cola over decades. Ricqlès unanimously described as French peppermint/mint product across Wikipedia and soft-drink taxonomy lists; no source describes it as a cola or as Indian market leader. No genuine conflicting evidence on the binary question.

Grading guidance:
- Full credit: Answer identifies Thums Up (or Thumbs Up) as the cola leader in India. Mention of Ricqlès as French mint/peppermint and not a cola is correct supplementary information but not required.
- Partial credit: Correct identification of Thums Up despite minor ancillary error (e.g., wrong year, wrong parent company) if core leadership claim is intact.
- No credit: Answer identifies Ricqlès as the cola leader, claims both are equal leaders, claims neither, or denies Thums Up is a cola. Do not penalize for "Thumbs Up" spelling, for adding ownership/history (Coca-Cola Company, 1977 introduction, 1993 acquisition from Parle), or for paraphrasing "largest/best-selling/No.1/dominant" - all satisfy.

### Repeat 2 — clear — `5adce8c15542992c1e3a249d`

Question: Are Thyme and Cortaderia both aromatic plants?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question uses the established collocation “aromatic plants” (plural noun phrase categorizing plants). In horticultural, botanical, and essential-oil literature this uniformly selects the organoleptic/functional dimension: plants possessing odorous volatile substances/essential oils and valued for fragrance/flavour. A chemical-aromaticity dimension (plants containing or yielding aromatic-hydrocarbon compounds such as benzene-ring structures upon analysis/pyrolysis) does not honour the explicit wording “aromatic plants” – sources phrase the latter as “plants containing aromatic compounds” or “bio-oils containing aromatic hydrocarbons,” not as classification as an aromatic plant. No reputable source classifies Cortaderia as an “aromatic plant” under the correct collocation, and all sources classify Cortaderia as a non-fragrant ornamental grass. The alternative dimension therefore fails the wording-fit test and would not produce a supportable materially different answer under the ambiguity contract. No other dimension (taxonomy, physical form, habitat, function) honours all constraints and yields a different grading outcome.

Status: SINGLE - Unambiguous question with one canonical answer

Canonical answer: No. Thyme and Cortaderia are not both aromatic plants. Thyme is an aromatic plant; Cortaderia (pampas grass) is not.

Decomposition:
- Thyme (genus Thymus, family Lamiaceae) = Yes, aromatic
- Cortaderia (genus Cortaderia, e.g., Cortaderia selloana, pampas grass, family Poaceae) = No, not aromatic, explicitly non-fragrant ornamental grass
Therefore the conjunction “both are aromatic plants” is false.

Acceptable variants:
- “No – only Thyme is aromatic; Cortaderia is not”
- “Thyme yes, Cortaderia no”
- “Thyme is a highly aromatic/fragrant/essential-oil herb; Cortaderia is a non-fragrant ornamental grass (pampas grass)”
Any wording that denies the conjunction and correctly assigns Thyme to aromatic and Cortaderia to non-aromatic is acceptable.

Not acceptable:
- “Yes, both are aromatic plants”
- “Cortaderia is aromatic / fragrant / an aromatic herb / an essential-oil plant”
- “Neither is aromatic” or “Thyme is not aromatic”

Necessary caveats:
- “Aromatic plants” is defined functionally as plants possessing odorous volatile substances occurring as essential oil, gum exudate, balsam and oleoresin, and whose essential oils are concomitant to fragrance/perfumes, and more generally as species that produce volatile oils (essential oils) responsible for their fragrance; the term “aromatic” refers to the characteristic fragrance emitted which arises from the presence of essential oils.
- Some Poaceae are aromatic (e.g., Cymbopogon citratus lemongrass, Vetiveria), but Cortaderia is not among them. Absence from aromatic-plant lists and explicit non-fragrant classification applies specifically to Cortaderia.
- Pyrolysis studies noting that Cortaderia selloana biomass bio-oils contain “nonaromatic and monoaromatic hydrocarbons” or “aromatic organic compounds with 1 benzene ring” describe thermal decomposition chemistry, not horticultural classification as an aromatic plant, and do not make Cortaderia an aromatic plant. This is an evidence conflict in terminology only, not a material interpretation branch.

Evidence:
- Thyme is a genus of about 350 species of aromatic perennial herbs in the family Lamiaceae
- The genus Thymus contains about 350 species of aromatic perennial herbaceous plants and subshrubs in the family Lamiaceae
- Thymus vulgaris forms a woody subshrub with small and highly aromatic grey-green leaves
- Thyme is highly aromatic with a hint of clove and mint fragrance
- Thyme is an aromatic perennial herb in the mint family
- Thyme (Thymus vulgaris) known for the aroma and flavour of its dried leaves and flowering tops
- Thymus are aromatic herbs or shrubby plants of the mint family
- Definition of aromatic plants: possess odorous volatile substances which occur as essential oil, gum exudate, balsam and oleoresin; aromatic plants are species that produce volatile oils commonly known as essential oils responsible for fragrance; aromatic refers to characteristic fragrance from essential oils
- Cortaderia selloana product attributes: Fragrance | Non-Fragrant; Pampas grass is a very popular ornamental grass
- In the genus Cortaderia are a couple of dozen species of grasses
- Cortaderia species are not known for their fragrance, but their visual impact is significant
- Cortaderia described as striking tall plumes and lush green foliage, ideal for adding height and texture, not fragrance

Confidence: High (0.96). Convergent high-quality sources (university extension, Britannica, North Carolina Extension Toolbox, Missouri Botanical Garden, RHS, commercial horticultural databases with explicit “Non-Fragrant” attribute, and definitional literature) consistently classify Thyme as archetypal aromatic and Cortaderia as non-fragrant ornamental grass. No reputable source classifies Cortaderia as an aromatic plant under the correct sense.

Grading guidance:
- Correct: Any answer that states No/False to “both aromatic” and identifies Thyme as aromatic (or highly aromatic, fragrant, essential-oil bearing) and Cortaderia/pampas grass as not aromatic / non-fragrant / ornamental grass / not an aromatic plant. Synonyms for aromatic (fragrant, scented, pungent herb known for aroma, produces thymol-rich essential oil) are acceptable for Thyme.
- Incorrect: Any answer asserting Yes/True that both are aromatic, asserting Cortaderia is aromatic/fragrant/an aromatic herb, or asserting Thyme is not aromatic.
- Do not require exact species names; genus-level reference (Thyme = Thymus, Cortaderia = pampas grass) is sufficient.
- Additional true adjacent information (family Lamiaceae vs Poaceae, culinary/medicinal uses of thyme, invasive status of Cortaderia, pyrolysis yields of aromatic hydrocarbons) does not affect correctness if core classification is preserved.
- Ambiguous or non-committal answers that do not explicitly negate “both” (e.g., “Thyme is aromatic, Cortaderia unclear”) do not satisfy the question and should be graded incorrect.

### Repeat 2 — broad-but-single — `5ae336ef5542991a06ce995d`

Question: What profession do both Phil Alden Robinson and David Nixon share?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: Entity "David Nixon" is ambiguous (5+ persons on disambiguation page), but only David Nixon (director) honors the explicit constraint that the profession is *shared* with Phil Alden Robinson's primary occupations. Alternative David Nixons have primary occupations magician/television presenter, choreographer/artistic director, footballer, linebacker - none match Phil's film director/screenwriter at the same level of description. Fresh search confirms Phil Alden Robinson is Film director, screenwriter and American film director and screenwriter, while David Nixon (director) is American film director and film producer. The magician's primary occupations are magician and television presenter (not writer/director), so no alternative yields a different supported shared profession. "Writer" or "producer" would be granularity variants within the same film-profession dimension, not a different semantic dimension.

Status: SINGLE

Canonical answer: Film director (director / filmmaker) — both are film directors. Phil Alden Robinson is an American film director and screenwriter and David Nixon (director) is an American film director and film producer; the shared profession is director (film director / filmmaker).

Acceptable variants:
- director
- film director / movie director / film and television director
- filmmaker / film director and producer

Not acceptable as canonical (though partially related): magician, television presenter, choreographer, dancer, artistic director, footballer, linebacker, actor. "Screenwriter/writer" alone is incomplete as David Nixon (director) is not described as screenwriter; "producer" alone is secondary (Phil's primary label is director/screenwriter, not producer).

Necessary caveats:
- Name ambiguity: "David Nixon" may refer to multiple people including English magician and television personality, Canadian choreographer/artistic director, Scottish footballer, American football linebacker, and American film director and producer. Only the film director shares a profession with Phil Alden Robinson.
- Both are American film professionals; Phil Alden Robinson also described as writer and director, David Nixon (director) also as film producer and co-producer/assistant director, but director is the most specific shared primary occupation.
- Granularity: "film director," "director," and "filmmaker" are the same answer at different specificity levels, not separate interpretations.

Evidence:
- Phil Alden Robinson — Occupation Film director, screenwriter
- Phil Alden Robinson is an American film director and screenwriter
- Phil Alden Robinson is a writer and director known for Field of Dreams etc.
- David Nixon is an American film director and film producer
- David Nixon may refer to: magician, choreographer, footballer, director (American film director and producer), American football linebacker
- David Nixon (magician) — Occupations Magician and television presenter — does not share with Phil
- David Nixon choreographer — long career as a dancer, choreographer and artistic director — Artistic Director of Northern Ballet — primary label is choreographer, not film director

Confidence: High (0.95) — Primary infobox/lede occupations for both subjects converge unambiguously on film director; disambiguation page and occupation details exclude other David Nixons from satisfying the shared-profession constraint; no conflicting primary evidence.

Grading guidance:
- Accept: Any response stating shared profession is (film/movie) director, filmmaker, or film director/producer (e.g., "film director," "director," "movie director," "filmmaker," "American film director").
- Accept with lower weight if needed: "producer/film producer" — both have producer credits, but director is more precise to primary occupations.
- Reject: Answers naming a profession held by only one subject (magician, television presenter/personality, choreographer, dancer, artistic director, footballer/football player, linebacker, actor alone) or answers that identify the wrong David Nixon without a shared film-director role, or "writer/screenwriter" alone (David Nixon director not described as screenwriter).
- Normalize case and modifiers: "director" = "film director" = "filmmaker" for grading.

### Repeat 2 — clear — `5ae6168f5542996de7b71b47`

Question: Adrian the Barbarian (later Olaff the Madlander) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks for the year a single identified strip — "Adrian the Barbarian (later Olaff the Madlander) ... in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?" — was first introduced. The requested attribute is unambiguously temporal (year of first appearance in The Beezer). No semantic dimension ambiguity exists between physical form / taxonomy / function etc. The only plausible alternative dimension — the year the strip was first published under the renamed title Olaff the Madlander in The Beano (2008) — fails to satisfy explicit constraints: the question specifies location as "The Beezer, and later the merged Beezer and Topper," while sources explicitly describe the Beano Olaff version as a reprint, not a first introduction. Table values showing "1990" are End Date, not first appearance, so conflict is source presentation, not wording ambiguity. No second reading honors all wording, has supporting evidence as a first introduction, and would change grading.

Status: SINGLE

Canonical answer: 1989 — Adrian the Barbarian (initially published as Adrian the Barbarian, later reprinted as Olaff the Madlander) was first introduced in The Beezer in 1989, specifically in No 1727 dated 18 February 1989.

Acceptable variants:
- 1989
- 18 February 1989 / 18 Feb 1989 / February 1989
- No 1727, 18 Feb 1989

Necessary caveats:
- Do not accept 1990 as first introduction year. In List of Beezer comic strips, Adrian the Barbarian is grouped under the header "## 1989"with the row listing "1990"; per the table column definition that column is End Date, so 1990 is the end/cover-star period, not the debut.
- Do not accept 2008. In July 2008, as of issue 3441, Adrian the Barbarian was reprinted in The Beano under the new title Olaff the Madlander, and that version has only ever appeared in The Beano as reprints, never as new strips. The question explicitly limits scope to The Beezer / merged Beezer and Topper.
- The UK Comics Wiki entry "- "Adrian the Barbarian" (1989-83)"contains a typographical end date but corroborates 1989 as start.
- Year alone is sufficient; exact day/month not required.

Evidence:
- Identity and first-introduction year/construction: "Olaff the Madlander (initially published as Adrian the Barbarian) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989." — supports strip identity, publication venue (Beezer + merged Beezer and Topper), and canonical year 1989.
- Corroboration of same sentence from search index: "Olaff the Madlander (initially published as Adrian the Barbarian) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989." — supports year.
- Alternative phrasing same fact: "**Adrian the Barbarian** (later **Olaff the Madlander**) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989." — supports same.
- Precise debut issue/date: "Adrian the Barbarian aka Olaff the Badlander first appeared in the Beezer in No 1727 dated 18th Feb 1989 and ran on through to the comics final issue No 1809 (15/9/90)" — supports exact day/month/year and issue number, and duration.
- Artist chronology supporting 1989: "Burgon began to draw for DC Thomson in 1989 drawing a revival of Biffo the Bear in The Beano and Adrian the Barbarian for The Beezer" — supports that creation occurred in 1989, not later.
- Reprint vs first introduction distinction: Reprint statementand reprint-only statement— supports why 2008 is not first introduction.

Confidence: High — Multiple primary-style descriptions converge on 1989, reinforced by an exact issue number and date (No 1727, 18 Feb 1989) and by creator chronology; apparent 1990 conflict is resolved as End Date.

Grading guidance:
- Accept any answer stating 1989 as first introduction year (including full date 18 February 1989).
- Reject answers stating 1990 or 2008 or any year other than 1989 as the first introduction in The Beezer.
- Accepting range: only if 1989 is clearly given as start year (e.g., "1989-1990", "1989 to 1990") — accept; if range excludes 1989, reject.
- Do not penalize formatting variations (e.g., '89, 1989 AD) if year is unambiguously 1989.

### Repeat 3 — broad-but-single — `56e2419acd28a01900c67c2a`

Question: What structures can be viewed as both works of art and symbols of certain cultures?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question predicates "structures" with two simultaneous attributes: "works of art" AND "symbols of certain cultures." Inventory of semantic dimensions for "structures" includes (a) physical/built form (buildings, monuments), (b) social/organizational structures, (c) chemical/molecular structures, (d) narrative/linguistic structures. Only dimension (a) satisfies BOTH constraints together with supporting evidence. Dimensions (b)-(d) honor at most one constraint and lack evidence that they are viewed as works of art *and* cultural symbols in this sense. Varying between a generic class answer ("architecture") and specific exemplars ("Eiffel Tower") is scope/granularity, not a materially different interpretation that changes grading outcome. No second reading honors all explicit constraints with distinct supporting evidence.

Status: SINGLE

Canonical answer:
Architectural works — in the material form of buildings and other physical structures / architecture / monuments / landmarks — which are often perceived as both cultural symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements.

Correctly instantiated by iconic examples including Eiffel Tower (France/Paris), Taj Mahal (India), Great Wall of China, Egyptian Pyramids / Great Pyramid of Giza, Greek Parthenon, Sydney Opera House (Australia), Statue of Liberty (USA), and comparable monumental landmarks such as the Colosseum, Big Ben, Moai, etc., that have become symbols of their civilizations.

Acceptable variants / Acceptance set:
Grader should accept ANY of the following as correct:

* Generic class: architecture / architectural works / buildings / physical structures / monuments / landmarks / iconic structures / cultural architecture described as both works of art and cultural symbols
* Any one or more valid exemplar where sources describe it as both artistic and culturally symbolic, including but not limited to: Eiffel Tower, Taj Mahal, Great Wall of China, Egyptian Pyramids/Giza Pyramids, Parthenon/Acropolis, Sydney Opera House, Statue of Liberty, Colosseum, Big Ben, St. Basil's Cathedral, Torii Gate, Moai Statue, Machu Picchu, Brandenburg Gate, Christ the Redeemer, Forbidden City, Petra, Angkor Wat
* Paraphrases such as "iconic buildings," "famous monuments," "world landmarks," "architectural icons" when used to convey the same class

Listing a single valid exemplar is sufficient; listing multiple is also correct. No exhaustive list required.

Necessary caveats:
* The source phrasing is generic and perception-based: "often perceived as cultural symbols and as works of art" — not every building qualifies, but architecturally significant / monumental / surviving works do.
* Civilization identification is linked to surviving architecture: "Historical civilizations are often identified with their surviving architectural achievements".
* Variant phrasing "cultural and political symbols and as works of art" is attested as equivalent.
* Evidence conflicts or incompleteness (e.g., which landmarks are most iconic) is scope difference, not interpretation ambiguity; do not treat alternative valid exemplars as conflicting answers.

Evidence:
* Definition — Architecture is considered a visual art of designing structures: "Architecture is the study and practice of designing structures, especially habitable ones. It utilizes civil engineering techniques, but is considered a visual art".
* Core dual predicate — "Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art"; identically in Outline: "Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art".
* Cultural expression — "The practice of architecture itself, which began in the prehistoric era, has been used as a way of expressing culture by civilizations on all seven continents. For this reason, architecture is considered to be a form of art".
* Exemplar set as civilization symbols — "the Egyptian Pyramids, the Greek Parthenon, the Great Wall of China, Islamic Indian Taj Mahal or the Moai of Easter Island have become symbols of their civilizations".
* Eiffel Tower as French cultural symbol — "For 130 years, the Eiffel Tower has been a powerful and distinctive symbol of the city of Paris, and by extension, of France" and "Like the soaring vaults of Gothic cathedrals, the Eiffel Tower is a uniquely French symbol".
* Taj Mahal as Indian cultural symbol and art masterpiece — "The very mention of the Taj Mahal, today perhaps the most famous symbol of India" and "The Taj Mahal was designated as a UNESCO World Heritage Site in 1983 for being 'the jewel of Muslim art in India" and "It is regarded by many as the best example of Mughal architecture and a symbol of India's rich history".
* Sydney Opera House as national symbol and masterpiece — "The Sydney Opera House is the symbol of modern Australia, a World Heritage-listed masterpiece".
* Statue of Liberty as American symbol and masterpiece — "Possibly the most iconic symbol of the United States in modern times, The Statue of Liberty" and "The Statue of Liberty is a masterpiece of colossal statuary".
* Parthenon as cultural symbol and artistic pinnacle — "The Parthenon stands as an enduring symbol of Ancient Greece, Western Civilization, and classical architecture" and decorative sculptures "considered to be the pinnacle of ancient Greek art" with Parthenon as "symbol of ... Athenian democracy, and is regarded as one of the world's greatest cultural monuments" (lines 73-76 of that source).
* Landmark icon set corroboration — editorial set contains icons: "Eiffel Tower, Giza Pyramids, Statue of Liberty, St. Basil's Cathedral, Big Ben, Taj Mahal, Great Wall Of China, Torii Gate, Moai Statue, Acropolis, Maya pyramids".
* Alternative phrasing confirmation — "Architectural works are often perceived as cultural and political symbols and as works of art".

Confidence: High for the canonical class answer and for the core exemplar set (Eiffel Tower, Taj Mahal, Great Wall, Pyramids, Parthenon, Sydney Opera House, Statue of Liberty). High that the question has one clear reading (architectural structures). Medium that any additional landmark beyond those explicitly cited as "symbol" will be judged correct — graders should verify dual artistic + symbolic description, but the class answer covers all.

Grading guidance:
* Accept generic answer alone (architecture/buildings/monuments/landmarks perceived as cultural symbols and works of art) as fully correct, even without examples.
* Accept any single valid exemplar plus indication that it is both artistic and culturally symbolic as fully correct; do not require exhaustive list.
* Accept synonyms: iconic structures, famous monuments, world heritage landmarks, architectural icons when tied to dual predicate.
* Reject answers that interpret "structures" as non-physical entities (social structures, chemical structures, story structures, organizational structures) without built-environment referent, or that name structures not attested as both art and cultural symbol.
* Do not penalize for granularity differences (e.g., "buildings" vs. "Eiffel Tower") — both are same interpretation at different levels of detail.

### Repeat 3 — broad-but-single — `56f71740711bf01900a44932`

Question: Bilateral treaties are concluded between how many states or entities?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks for a cardinal count ("how many states or entities") for the class "bilateral treaties." The term "bilateral" lexically and legally selects the numeric dimension "two-sided." Inventory of plausible semantic dimensions was:
1) count dimension (how many sides/parties/groups),
2) type-of-entity dimension (state vs international organization vs mixed entity).

Both dimensions converge on the same numeric answer in the evidence: the definition is consistently two sides/entities, where the two sides may be instantiated as two states, two organizations, or one state and one organization. The Switzerland-EU 17-party example does not create a competing interpretation that honors the wording "bilateral" as "more than two" — sources explicitly classify it as still bilateral because it has two groups/sides, preserving the count "two." No second dimension yields a materially different supported count that would change grading outcome. Evidence conflict about formal signatory count is granularity/caveat, not interpretation ambiguity.

Status: SINGLE - clear question with one canonical meaning

Canonical answer: Two — a bilateral treaty is concluded between two states or entities (two parties/sides/groups).

Acceptable variants:
- “2” / “two” / “a pair”
- “two states” / “two entities” / “two states or entities”
- “two parties” / “two sides” / “two groups (on the one part / on the other part)”
- “between two sovereign states/entities” / “between two nations/countries/governments” specifying the bilateral relationship
All express exactly two contracting sides. Numeric form (digit vs word) is equivalent.

Necessary caveats:
- Entity type flexibility: the two sides may be two states, two international organizations, or one state and one international organization — the number remains two.
- Formal signatory vs bilateral classification: Sources note it is possible for a treaty classified as bilateral to have more than two formal parties if parties are divided into two groups, e.g., each Switzerland-EU bilateral treaty has seventeen parties divided into two groups. This does not change the definitional answer “two sides/groups” and is a scope/granularity note, not a different answer. The treaty “establishes rights and obligations between the Swiss and the EU and the member states severally — it does not establish any rights and obligations amongst the EU and its member states.”
- Contrast with multilateral: multilateral is defined as involving more than two countries / more than two parties; any agreement with more than two parties (in the multilateral sense) is not bilateral.

Evidence:
- Core definition — bilateral between two states or entities: "Bilateral treaties are concluded between two states or entities."
- Bilateral vs multilateral count distinction: "Treaties may be bilateral (between two countries) or multilateral (involving more than two countries)."
- Official U.S. Senate glossary definition: "Bilateral treaty: An international agreement concluded between two states."
- Strictly between two entities/parties: "A bilateral treaty (also called a bipartite treaty) is a treaty strictly between two state entities. It is an agreement made by negotiations between two parties" and "Any agreement with more than two parties is a multilateral treaty."
- Entity-type instantiation still two: "These two parties can be two nations, or two international organizations, or one nation and one international organization, or two persons."
- Supportive restatement: "Treaties made between two independent governments are called bilateral treaties."
- Caveat illustration of retained bilateral classification despite many signatories: "It is possible for a bilateral treaty to have more than two parties; for example, each of the bilateral treaties between Switzerland and the European Union (EU) has seventeen parties: The parties are divided into two groups, the Swiss ("on the one part") and the EU and its member states ("on the other part")."

Confidence: High — convergent primary/official and reputable secondary sources give verbatim identical count with no material definitional conflict.

Grading guidance:
- ACCEPT any response stating exactly “two” with any of the accepted type labels above, or numeric “2.”
- ACCEPT “two sides/groups” explanations even if mentioning the 17-party Switzerland-EU example, provided the respondent clarifies it is still bilateral due to two groups.
- REJECT: “one,” “three,” “several,” “many,” “more than two,” “multilateral / at least three,” or answers defining only multilateral treaties. Answers stating “more than two parties” without the two-groups qualification misclassify bilateral.
- No alternative interpretation branch to accept; do not accept “17” or “more than two” as standalone correct.

### Repeat 3 — ambiguous — `57260b4389a1e219009ac188`

Question: What is an example of something that serves a function other than protection?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks for any example of an entity that has at least one function that is not protection. The wording selects the semantic dimension "function" explicitly ("serves a function other than protection") and does not ask for a kind/type/form along another dimension such as physical form, habitat or taxonomy. Variation in domain (integumentary, skeletal, artifact, molecule) is scope/granularity within that single functional dimension, not a distinct interpretation. The only plausible alternative reading — that the entity must serve exclusively non-protective functions and not also protect — adds an exclusivity constraint ("only") not present in the wording and is therefore ruled out; the question is satisfied by existence of a non-protective function even if the entity also protects. No second reading both honors all explicit constraints and yields a materially different supportable answer.

Status: SINGLE

Canonical answer: An example is human skin (integumentary system) — while it does provide protection, it also serves functions other than protection, including sensory function, thermoregulation, and vitamin D synthesis. Equivalent examples in other domains are equally correct, e.g., the skeletal system (support, movement, hematopoiesis, mineral/fat storage) or painting/coating (appearance, marking, safety).

* The skin and accessory structures perform a variety of essential functions, such as protecting the body from invasion by microorganisms, chemicals, and other environmental factors; preventing dehydration; acting as a sensory organ; modulating body temperature and electrolyte balance; and synthesizing vitamin D.
* Sensory detail: The skin acts as a sense organ because the epidermis, dermis, and the hypodermis contain specialized sensory nerve structures that detect touch, surface temperature, and pain.
* Thermoregulation detail: The integumentary system helps regulate body temperature through its tight association with the sympathetic nervous system... sweat glands, accessory structures to the skin, secrete water, salt, and other substances to cool the body when it becomes warm.
* Vitamin D detail: The epidermal layer of human skin synthesizes vitamin D when exposed to UV radiation.

Acceptable variants / Acceptance set:
Any specific entity cited with a verifiable non-protective function is acceptable. Non-exhaustive acceptance set includes:

- Skin/integumentary: sensation / sensory organ, thermoregulation / temperature regulation / sweating / vasodilation, vitamin D synthesis, excretion/electrolyte balance, insulation, fat storage
- Skeleton/bone: support/structure, movement/facilitation of movement via muscle attachment/levers, production of blood cells / hematopoiesis in red marrow, mineral storage/release (calcium, phosphorus, potassium), fat/triglyceride storage in yellow marrow
- Artifact/structure: painting/coating for appearance / decoration / cosmetic appearance, marking (airfield runways, streets, parking lots), safety colors/designs, reflective finishes, sanitation/preservation, identification/colour-coding
- Biological ornament/molecule: peacock feathers for mating display/sexual ornament, chlorophyll for photosynthesis/light absorption and energy transfer, Bacillus spore coat for other functions beyond genome protection, immune system for development/repair beyond protection — provided the non-protective function is correctly stated.

Necessary caveats:
- The prompt is open-ended; no single required example exists. Grader must accept any well-evidenced non-protective function.
- The cited function must be distinct from protection. Synonyms of protection (shielding, barrier, defense, armor) alone do not satisfy without an additional non-protective function. An answer that lists protection plus a non-protective function is correct if the non-protective function is accurate.
- Granularity does not affect correctness: "sensation" vs "detecting touch via Meissner corpuscle" are the same interpretation at different specificity; do not branch.
- Evidence conflicts are not interpretation ambiguities; treat disagreements about mechanism or relative importance as caveats, not separate branches.

Evidence:
- Skin multi-function statement (protection vs sensory, thermoregulation, vitamin D) —  supports that skin serves functions other than protection and lists the specific alternatives.
- Skin sensory function definition —  supports sensation as a non-protective sensory function.
- Skin thermoregulation mechanism —  supports thermoregulation via sweat glands and nervous regulation.
- Skin vitamin D synthesis —  supports synthesis of vitamin D in epidermis upon UV exposure.
- Skeletal system critical functions list — skeletal system performs: supports the body, facilitates movement, protects internal organs, produces blood cells, stores and releases minerals and fat supports skeletal non-protective functions (support, movement, hematopoiesis, storage).
- Skeletal mineral storage and hematopoiesis detail — bone matrix acts as a reservoir for minerals... especially calcium and Red marrow is where hematopoiesis—the production of blood cells—takes place supports storage and blood-cell production as functions other than protection.
- Painting purposes other than protection — Facilities are often painted for other purposes than, or in addition to, protection. These include appearance, marking, and safety supports non-biological example where painting serves appearance/marking/safety rather than protection.

Confidence: High — Multiple primary/institutional sources (OpenStax via LibreTexts, Medicine LibreTexts, US military handbook) converge on existence of well-documented non-protective functions for skin, skeleton, and coatings. No evidence conflicts on the existence of these functions.

Grading guidance:
- Accept any answer naming a concrete entity and accurately describing at least one function that is not protection, with support from evidence or established knowledge.
- Reject answers that cite only protection / barrier / defense, that rephrase protection as the answer, or that attribute a factually incorrect non-protective function.
- Do not penalize domain choice (biological vs artifact) or level of detail; "skin — thermoregulation" and "femur marrow — makes red blood cells" and "paint — marking runways for safety" are equally correct.
- If answer includes both protective and non-protective functions, grade as correct provided the non-protective claim is accurate and evidenced.

### Repeat 3 — ambiguous — `57302972b2c2fd140056899e`

Question: What could be considered a trait of Rome's military campaigns?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks for "a trait" (characteristic feature) of Rome's military campaigns with no restrictive modifier selecting a semantic dimension. Inventoried plausible dimensions — strategic purpose, tactical method, organizational/logistical structure, behavioral/cultural resilience — all fall under the single ordinary reading "characteristic feature of the campaigns." They are not mutually exclusive readings that each honor distinct constraints and yield contradictory true answers; they are scope/granularity variants within one reading. The wording "could be considered" explicitly licenses any well-supported trait, so variation is handled by an Acceptance set, not by branching interpretations. No wording forces a choice between e.g. physical form vs function vs taxonomy, and no second reading honors all constraints while producing a materially different grading outcome.

Status: SINGLE

Canonical answer: A trait of Rome's military campaigns is that they were **characterized as territorial expansionist campaigns** (normally begun as a counter-offensive in which each victory brought subjugation of large areas of territory and allowed growth from a small town to ~55 million in the early empire) and more generally as **highly disciplined, organized, flexible and persistent**, with a **preference for direct engagement in open battle combined with systematic military engineering and siege capability and systematic construction of fortified camps (castra)**,,,,

Acceptable variants / Acceptance set: Any one correct trait description counts. Grader must accept any of the following (or synonymous phrasing) as correct:

* Territorial expansionist typology; majority of campaigns characterised by one of two types — territorial expansionist (normally begun as counter-offensive) vs civil war
* Served triple purpose of securing borders, exploiting peripheral areas through measures such as imposing tribute on conquered peoples, and maintaining internal order
* Preference for direct engagement in open battle and, where necessary, overcoming fortified positions via military engineering
* Systematic building of strong fortified camp (castra) with palisade and deep ditch at end of each day's march; no other ancient army persisted so long in systematic camp construction even when resting only a single day
* Disciplined, organized and systematized force sustaining combat effectiveness; more flexible organization than opponents; effectively handled challenges ranging from cavalry to guerrillas to siege warfare
* Strict, uniform discipline (including harsh punishments) making commanding and replacement consistent
* Engineering mastery second to none; mastery of offensive and defensive siege warfare, construction and investment of fortifications
* Obstinate persistence overcoming appalling losses; history of campaigning is if nothing else a history of persistence despite catastrophic defeats; more persistent and willing to absorb and replace losses than opponents,

Necessary caveats:
* "Could be considered" is open-ended: one accurately described trait is sufficient; exhaustive list not required.
* Sources note Roman armies were not invincible and produced incompetents who led to catastrophic defeats; persistence is the trait, not invincibility.
* Different period emphases (Kingdom/Republic/Empire) reflect granularity, not conflicting interpretations — e.g., annual levies in early period vs standing professional army later, and Vegetius's later preference for stratagem vs earlier eagerness for open battle do not contradict the preponderant trait.
* Do not require exact term "expansionist" — accept conqueror, imperialist, territory-acquiring, etc., if meaning matches subjugation of large territory after victories.

Evidence:
* Triple purpose and two-type characterization, with expansionist definition and 55 million figure
* Obstinate persistence overcoming appalling losses
* Romans produced share of incompetents leading to catastrophic defeats alongside victories — persistence context
* Preponderance exhibits preference for direct open battle and overcoming fortified positions via military engineering
* Typical establishment of castra with palisade and deep ditch; systematic camp construction unique in duration
* Roman organisation more flexible; handled cavalry, guerrillas, siege warfare; discipline, organization and systematization sustained combat effectiveness
* More persistent and willing to absorb and replace losses than opponents, illustrated by wars with Carthage, Parthians and Pyrrhus
* Roman engineering skills second to none; mastery of offensive and defensive siege warfare and fortification

Confidence: High — convergent primary institutional and reputable secondary sources (Campaign history of the Roman military, Strategy of the Roman military, Roman legion articles, citing Luttwak, Trigger, Vegetius, Polybius) consistently identify expansionist typology, open-battle preference, engineering/camp systematization, discipline/organization/flexibility, and persistence.

Grading guidance:
* Accept any answer naming or accurately paraphrasing at least one trait in Acceptance set, even if phrased at different granularity (e.g., "disciplined legions," "built camps every night," "sought pitched battles," "expert siege engineers," "expansionist/conquest-driven," "never gave up after losses").
* Accept additional correct traits beyond set if at least one set trait is present and no contradiction is made.
* Reject answers claiming opposite traits (e.g., avoided open battle, lacked engineering/siege ability, undisciplined/disorganized, unable to sustain losses, purely naval or purely defensive with no expansion) or describing traits with no evidence.
* Vague answers without specific characteristic ("they were good at fighting") are insufficient — require a concrete trait name or description.

### Repeat 3 — ambiguous — `5ab9e3d25542994dbf019889`

Question: What type of vegetation does Lythrum and Erythrina have in common?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 2, "grading_guidance_signal": true, "status_signal": "MULTIPLE"}`

Interpretation decision: MULTIPLE

Status: AMBIGUOUS - `What type of vegetation` does not select a single semantic dimension for `type`. `Type/kind/form` can plausibly refer to (a) morphological life-form / life-history (growth form, duration, woodiness, Raunkiær type, taxonomic group) or (b) ecological formation / habitat affinity (hydrology, physiognomy of vegetation formation). Both dimensions satisfy all explicit question constraints (`Lythrum` + `Erythrina` + `have in common` + `vegetation type`), each is supported by primary evidence, and each yields a materially different grading outcome (herbaceous perennial forb vs. wetland emergent hydrophyte). This is material interpretation ambiguity, not scope/granularity or evidence conflict.

Interpretation 1 — Growth-form / life-history dimension (morphology + phenology + major clade)
Scope: Botanical habit / vegetation type as structural growth form. Asks what structural life-form both genera exhibit.
Answer: Both include **herbaceous perennial flowering plants (perennial forbs/herbs)** - perennial, non-woody (herbaceous) forbs, more broadly **perennial dicotyledonous angiosperms / eudicots** (flowering plants).
Evidence:
- *Lythrum* genus: "Lythrum can be annuals or herbaceous perennials, with simple leaves in opposite pairs"
- *Lythrum* specifics: "Herbs perennial, slender, 5–12 dm"; "Herbaceous, noncreeping, perennial forb"; "L. salicaria is a perennial, herbaceous plant"
- *Erythrina* general genus description: "Mostly trees or shrubs, rarely perennial herbs (herbaceous resprouters), mostly deciduous"; "Erythrina can be deciduous or evergreen subshrubs, shrubs or trees, often spiny"
- *Erythrina* herbaceous-perennial representatives: "Intro: Shrublike herbaceous perennial found in maritime forests, dry sandy woodlands" for *Erythrina herbacea*; "Unlike most *Erythrina* species, which usually are woody trees or shrubs, *Erythrina leptorhiza* is a perennial which is herbaceous"
- Both as flowering plants / eudicots: "Lythraceae is a family of flowering plants" that includes *Lythrum*; *Erythrina* "Family: Fabaceae"; "Clade: Eudicots ... Order: Myrtales ... Family: Lythraceae" and "Clade: Eudicots ... Order: Fabales ... Fabaceae" corroborating both are eudicot (dicot) angiosperms.
Confidence: High that both genera contain herbaceous perennial representatives sharing the dicot angiosperm ground plan; Medium that this is the examiner's intended whole-genus generalization because most *Erythrina* species are woody trees/shrubs, not herbs.
Grading guidance (for this branch): Accept any granularity within this dimension: `herbaceous perennial`, `perennial forb`, `perennial herb`, `herb / herbaceous`, `perennial` (if not limited to woody), `forb`, `clump-forming herbaceous perennial`, `flowering plant / angiosperm`, `dicot / dicotyledonous / eudicot / rosid eudicot`. Reject `annual only`, `woody tree/shrub only for both`, `monocot`, `gymnosperm`, `non-flowering`.

Interpretation 2 — Habitat / vegetation-formation dimension (ecological hydrology)
Scope: Vegetation type as ecological formation / site affinity (hydrology, wetland formation, hydrophytic vegetation).
Answer: Both include **hydrophytic / wetland vegetation — emergent / riparian swamp-margin wetland plants** (obligate-wetland, freshwater swamp / marsh / riparian emergent vegetation).
Evidence:
- *Lythrum* as wetland hydrophyte/emergent: "*Lythrum salicaria* is a wetland hydrophyte"; "Purple loosestrife is an emergent aquatic plant of Eurasian origin"; "We studied the invasion of exotic emergent plant purple loosestrife (*Lythrum salicaria*) in a freshwater wetland"; "Hab: Swamps, marshes, other wet places"; "Wetland Indicator Status ... OBL" for *Lythrum salicaria*; "*Lythrum lineare* Wetland status Occurs only in wetlands. (Wetland indicator code: OBL)"; "serious invader of many types of wetlands, including wet meadows, prairie potholes, river and stream banks"
- *Erythrina* wetland representatives: "Habitat: ... Margins of swamps. Up to 1560 meters" for *Erythrina fusca*; "In low-lying freshwater swamps *Erythrina fusca* attains huge dimensions and sometimes develops almost pure stands"; "E. fusca is a pioneer species, most commonly found on the outer margins of swamps on sites that are not permanently" flooded; "Riparian, Wetland" habitat list for *Erythrina crista-galli*
Confidence: High that both genera contain wetland / swamp-margin hydrophytic species; Medium that this is the intended `vegetation type` in an ecological sense because *Erythrina* as a whole is not uniformly wetland (e.g., *E. herbacea* in dry sandy woodlands) — wetland affinity is species-specific, not genus-uniform.
Grading guidance (for this branch): Accept `wetland`, `swamp`, `swamp-margin`, `freshwater swamp`, `marsh`, `freshwater wetland`, `riparian wetland`, `emergent (aquatic) hydrophyte`, `obligate-wetland (OBL) hydrophyte`, `hydric / hydrophytic vegetation`, `moist / hydric swamp forest margin`. Reject `strictly xerophytic / desert upland only`, `truly submerged aquatic only`, `upland dry forest only` as sole common type.

Acceptance set (flattened - any satisfies grader):
- herbaceous perennial / perennial forb / perennial herb / perennial herbaceous forb / herbaceous / herb
- perennial flowering plant / perennial angiosperm / dicot / eudicot / dicotyledonous flowering plant
- wetland / swamp / marsh / riparian wetland / freshwater swamp / swamp-margin / emergent aquatic / emergent hydrophyte / hydrophytic / obligate-wetland (OBL) / hydric wetland vegetation

Necessary caveats:
- Genus-level heterogeneity is real and is not grounds for collapsing ambiguity. *Lythrum* is relatively uniform (herbaceous perennial wetland forbs), while *Erythrina* is predominantly woody ("Mostly trees or shrubs, rarely perennial herbs"; "Unlike most *Erythrina* species, which usually are woody trees or shrubs") and ecologically broad (dry woodlands vs. swamp margins). Answers that claim `all *Erythrina* are herbaceous wetland herbs` overgeneralize; acceptable answers acknowledge `both genera contain / include species that are ...` or name exemplar species (*Lythrum salicaria / lineare / alatum* and *Erythrina fusca / herbacea / leptorhiza / crista-galli*).
- Nested granularity within a dimension (e.g., `flowering plant` → `eudicot` → `herbaceous perennial eudicot forb`) is not a separate interpretation; all levels within Interpretation 1 are acceptable variants of one dimension.
- Similarly `wetland` → `swamp` → `emergent hydrophyte` → `OBL` are granularities within Interpretation 2.

Evidence (what each URL supports):
- Growth-form and family evidence as cited above for Interpretation 1; habitat/hydrology and wetland-indicator evidence as cited above for Interpretation 2. Primary sources used: RHS generic habit, Flora of North America, Pesticide Safety Education / Resource Database descriptions for *Lythrum*; WorldFloraDB / Wikipedia / PFAF / FSUS descriptions for *Erythrina* habit; Lythraceae/Fabaceae family pages for flowering-plant/eudicot placement; USGS, Invasive Plant Atlas, FSUS wetland indicator lists for OBL status; World Agroforestry / ResearchGate / ThisIsModest for *E. fusca* swamp-margin ecology.

Confidence (overall): Medium-High that both interpretations are legitimate readings of ambiguous wording; overall grading should accept either dimension.

Grading guidance (overall):
- Do not penalize for choosing either ecological or morphological dimension. If answer matches any term in Acceptance set, mark correct.
- Do not treat evidence conflicts (e.g., most *Erythrina* are woody vs. some are herbaceous; most *Erythrina* not wetland vs. *E. fusca* is swamp) as interpretation ambiguity—treat as caveat qualified at species level.
- Penalize only answers that contradict evidence or introduce unrelated dimensions (e.g., `monocot`, `annual only for both`, `gymnosperm`, `cactus succulent desert vegetation as sole shared type`).

### Repeat 3 — clear — `5ac4e4595542996feb3fe968`

Question: Which cola drink is the leader in India, Thums Up or Ricqlès?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question explicitly limits candidates to a binary choice and explicitly scopes the requested attribute to "cola drink" and geography to "in India". Thums Up satisfies both constraints as a cola-type drink and the documented cola leader in India; Ricqlès fails the explicit "cola drink" constraint (evidence consistently classifies it as mint-flavoured/peppermint, French-origin, not cola) and has no evidence as a cola leader in India. No other semantic dimension invited by the wording (e.g., general soft-drink leadership vs cola-segment leadership) honors all constraints and yields a materially different supportable answer - both readings resolve to Thums Up for the cola segment. Therefore no genuine interpretation ambiguity remains.

Status: SINGLE

Canonical answer: Thums Up is the cola drink leader in India; Ricqlès is not (Ricqlès is not a cola drink and is not the leader).

Acceptable variants:
- "Thums Up" (accepted spelling variants: Thums Up, Thumps Up if clearly referring to the Indian Coca-Cola-owned brand) is correct.
- Any phrasing that identifies Thums Up as the answer and explicitly or implicitly rejects Ricqlès.
- Not acceptable: Ricqlès, "both", "neither", or any answer stating Ricqlès is a cola leader in India.

Necessary caveats:
- Scope caveat: Thums Up is the largest *cola* brand in India; within the broader *carbonated beverage* market, Sprite (also Coca-Cola) is sometimes cited as topping Thums Up overall, but this does not change the cola-segment leadership.
- Category caveat: Ricqlès is categorically not a cola - listed as mint-flavoured soft drink, described as a peppermint spirit and French soft drink and as French manufacturing products based on peppermint - so it cannot satisfy the "cola drink" constraint.
- No evidence conflict creates ambiguity; the cola-leader evidence for Thums Up is consistent across sources.

Evidence:
- Thums Up is a brand of cola; Type = Cola
- Thums Up is the largest cola brand in India with Rs5,000 crore in annual sales, which has dominated India's cola market
- Thums Up quickly became India's biggest cola brand, trumping competitors such as Campa Cola and Double Seven
- Thums Up is the largest beverage brand in Coca-Cola's India portfolio as well as the largest soft-drink brand in the country
- Thums Up rules the cola scene in India, only topped in the carbonated beverage arena by Sprite - supports cola-segment vs overall market distinction
- Ricqlès – mint-flavoured soft drink
- Ricqlès is a peppermint spirit and a French soft drink
- Ricqlès is a French manufacturing products based on peppermint

Confidence: HIGH

Grading guidance:
- Award credit if response selects Thums Up as the leader and does not select Ricqlès as cola leader. Full credit requires Thums Up as the sole correct choice for the India cola leadership.
- Reject responses that answer Ricqlès, claim Ricqlès is a cola drink leader in India, claim both are co-leaders, or claim neither.
- Accept minor misspellings (Thumbs Up / Thumps Up) if unambiguous reference to the Indian cola brand is clear.
- Do not penalize additional correct context (e.g., owned by Coca-Cola since 1993, introduced 1977, billion-dollar brand in 2021) if Thums Up remains the identified leader.

### Repeat 3 — clear — `5adce8c15542992c1e3a249d`

Question: Are Thyme and Cortaderia both aromatic plants?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The question asks "Are Thyme and Cortaderia both aromatic plants?" - requiring a single binary attribute (aromatic) applied to two genera. Inventory of plausible dimensions of "aromatic": (1) sensory/horticultural = fragrant, producing essential oils/volatile aroma compounds, pleasant smell (the established meaning of "aromatic plant/herb"); (2) organic-chemistry = containing or yielding aromatic compounds with benzene rings (e.g., via pyrolysis). Dimension 2 is ruled out by wording and evidence: no source classifies a living plant genus as an "aromatic plant" based on pyrolysis bio-oil chemistry, sources describing Cortaderia pyrolysis describe thermal decomposition products, not a plant type, and all botanical/horticultural sources use dimension 1 to define Thyme as aromatic and to evaluate Cortaderia for fragrance. Evidence conflicts at species-level granularity (Cortaderia nitida noted for fragrant flowers) do not create a second semantic dimension that honors all constraints and changes the genus-level answer. No other dimension (taxonomy, habitat, form) satisfies the wording "aromatic." Therefore only one materially different reading remains.

Status: SINGLE

Canonical answer: No. Thyme (genus *Thymus*, e.g., *Thymus vulgaris*) is an aromatic plant/herb, Cortaderia (pampas grass, e.g., *Cortaderia selloana*) as a genus is not classified as an aromatic/fragrant plant and is described as non-fragrant / not known for fragrance. They are not both aromatic plants.

Evidence support:
- Thyme is defined as aromatic herbs or shrubby plants of the mint family (any species of genus *Thymus*)
- *Thymus vulgaris* forms a subshrub with small and highly aromatic grey-green leaves and more generally the genus *Thymus* comprises fragrant aromatic undershrubs
- Thyme is a genus of about 350 species of aromatic perennial herbs in the family Lamiaceae, and *Thymus pulegioides* has strongly aromatic leaves
- By horticultural definition, aromatic = having a noticeable and pleasant smell: fragrant … aromatic herbs
- Cortaderia species are not known for their fragrance, but their visual impact in a landscape is significant
- Commercial/horticultural data for the most grown species *Cortaderia selloana* list Fragrance | Non-Fragrantand again Fragrance | Non-Fragrant; other cultivars likewise listed as Fragrance: Non-fragrant
- Chemistry sources describing "aromatic organic compounds" from *Cortaderia selloana* refer to pyrolysis gas fractions (non-aromatic vs aromatic organic compounds with 1 benzene ring etc.), not to classification of the living plant as an aromatic plant.

Acceptable variants:
- No / False — only Thyme is aromatic, Cortaderia is not.
- Thyme yes, Cortaderia no / Thyme highly aromatic, Cortaderia non-fragrant / not known for fragrance / not classified as aromatic plant.
- Answer stating genus-level distinction (Thymus aromatic, Cortaderia pampas grass not aromatic) is acceptable; genus vs species naming variations (*Thymus vulgaris*, *Cortaderia selloana*) acceptable if genus-level conclusion preserved.
- Simple "No" without elaboration is acceptable.

Necessary caveats:
- Scope is genus level. Thyme = genus *Thymus* (Lamiaceae). Cortaderia = genus *Cortaderia* (Poaceae; ~25 species, most known *C. selloana* pampas grass).
- "Aromatic plant" here is the sensory/horticultural sense (fragrant, essential-oil bearing), not the organic-chemistry sense of benzenoid aromaticity. Plants yielding aromatic hydrocarbons on pyrolysis are not thereby "aromatic plants."
- Species-level granularity caveat: While the genus as a whole is described as not known for fragrance and listed as Non-Fragrant, one source notes *Cortaderia nitida* is used for its fragrant flowers. This species-specific exception does not make the genus Cortaderia as a whole classified as aromatic plants in standard usage and does not change the answer to the "both" question.
- Do not treat evidence incompleteness or a single outlier species description as creating a second interpretation branch.

Evidence:
- Thyme aromatic herb definition:
- Thyme highly aromatic leaves:
- Thyme fragrant aromatic undershrubs:
- Thyme genus aromatic perennial herbs:
- Aromatic = fragrant:
- Cortaderia not known for fragrance:
- Cortaderia selloana Non-Fragrant (product spec 1):
- Cortaderia selloana Non-Fragrant (product spec 2):
- Cortaderia pyrolysis aromatic compounds = chemistry, not plant type:
- Cortaderia nitida fragrant outlier (caveat):

Confidence: High. Consistent, convergent evidence from encyclopedic, extension, and horticultural primary sources defines Thymus as aromatic and describes Cortaderia genus as not known for fragrance / listed as Non-Fragrant.

Grading guidance:
- Accept: Any answer concluding "No" and correctly distinguishing Thyme as aromatic/fragrant and Cortaderia as not aromatic / non-fragrant / not known for fragrance. Accept genus-level answers citing *Thymus vulgaris* or *Cortaderia selloana* as exemplars.
- Reject: "Yes, both are aromatic plants," "Both are fragrant/aromatic herbs," "Cortaderia is an aromatic plant" as a genus-level claim.
- Do not give credit for answers that switch to chemical aromaticity (e.g., claiming Cortaderia is aromatic because its pyrolysis yields benzene-ring compounds) — that does not satisfy "aromatic plant" as asked.
- Mention of *Cortaderia nitida* fragrant flowers may be noted as a nuance but should not be accepted as making the statement "both are aromatic plants" true; genus-level answer remains No.
- Partial: If answer affirms Thyme aromatic but is uncertain/silent on Cortaderia, treat as incomplete/incorrect unless it explicitly denies Cortaderia aromatic status.

### Repeat 3 — broad-but-single — `5ae336ef5542991a06ce995d`

Question: What profession do both Phil Alden Robinson and David Nixon share?

Signals: `{"acceptance_set_signal": true, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "MULTIPLE"}`

Interpretation decision: MULTIPLE
Interpretation 1
Answer: Both are film directors (director / filmmaker) — Phil Alden Robinson and David Nixon (American film director)
Scope: Refers to David Nixon (director), American film director and film producer. Shared primary occupation is directing films.
Evidence:
- Phil Alden Robinson Occupation | Film director, screenwriter; Phil Alden Robinson is an American film director and screenwriter; also described as writer and director
- David Nixon is an American film director and film producer; same description repeated: David Nixon is an American film director and film producer
- David Nixon may refer to: David Nixon (magician)... David Nixon (director), American film director and producer — establishes the director referent exists distinct from magician
Confidence: High — primary occupation for both is directly listed as film director in infobox/opening sentence with consistent corroboration across Wikipedia, TMDB and IMDb mirrors.
Grading guidance: Accept director, film director, movie director, filmmaker, film/television director. Accept director plus producer/writer qualifiers if director is included (e.g., director-producer). Do not accept producer alone, screenwriter alone, or writer alone for this referent. This is the most direct fit to the wording because it matches the primary profession for both individuals.

Interpretation 2
Answer: Both are writers / screenwriters (television/film writers) — Phil Alden Robinson and David Nixon (English magician, David Porter Nixon 1919-1978)
Scope: Refers to David Nixon (magician), English magician and television personality. Shared secondary but documented occupation is writer.
Evidence:
- Phil Alden Robinson Occupation | Film director, screenwriter; Phil Alden Robinson is an American film director and screenwriter
- David Nixon (magician) Occupations | Magician and television presenter; was an English magician and television personality
- Same David Porter Nixon described as writer and actor on IMDb: David Nixon was born on 29 December 1919... He was a writer and actor, known for Aladdin (1951); writer credits include The David Nixon Show - script and David Nixon's Magic Box - writer
- Disambiguation confirms both referents share the name: David Nixon may refer to: - David Nixon (magician) (1919–1978), English magician ... - David Nixon (director), American film director and producer
Confidence: Medium — writer is Phil Alden Robinson's primary occupation, but for the magician referent writer is secondary to magician/television presenter in Wikipedia infobox (not listed as primary occupation) and relies on IMDb biography/writer filmography rather than Wikipedia occupation field. Still supported as a documented profession.
Grading guidance: Accept writer, screenwriter, television writer, scriptwriter if answer is clearly tied to writing profession. Do not accept magician, entertainer, television presenter, actor alone, or film director for this referent. Ranked second because it requires using a secondary credit for the magician and does not match his primary public identity, whereas Interpretation 1 matches primary professions for both.

Acceptance set
A grader may accept any one of the following as correct depending on referent disambiguation, provided the answer includes the relevant profession term:
- film director / director / movie director / filmmaker (Interpretation 1)
- writer / screenwriter / television writer (Interpretation 2)
Do not accept magician, television presenter/personality, actor, choreographer, football player/linebacker, or producer/screenwriter without director/writer context that fails to include the shared term above.

Necessary caveats
- The name "David Nixon" is ambiguous and has multiple notable bearers: English magician and television personality, American film director and producer, Canadian choreographer, American football linebacker, Scottish footballer. The question presupposes a shared profession, which selects only referents where overlap exists.
- Interpretation 1 (film director) is the only overlap of primary occupations; Interpretation 2 (writer) is an overlap of a primary occupation for Robinson and a secondary documented occupation for the magician. Evidence conflicts in source hierarchy: Wikipedia infobox for the magician lists magician and television presenter, while IMDb lists writer and actor. Both are retained as evidence rather than resolved.
- Scope/granularity variants such as "director and producer," "filmmaker," "American film director" are the same profession at different specificity and do not constitute separate interpretations.

Evidence
- Phil Alden Robinson occupation film director, screenwriter: table lines cited above
- Phil Alden Robinson definition as American film director and screenwriter
- Phil Alden Robinson writer and director descriptor
- David Nixon (director) American film director and film producer
- David Nixon may refer to disambiguation list
- David Nixon (magician) occupations magician and television presenter and description as English magician and television personality
- David Nixon (magician) IMDb writer and actor and writer credits

Confidence
- Overall (that at least one interpretation is correct): High
- Interpretation 1 (film director): High
- Interpretation 2 (writer): Medium (supported but secondary)

Grading guidance
- For Interpretation 1: Award credit for any answer containing director/film director/filmmaker. Do not penalize for adding producer. Do not award for magician/television presenter/actor alone.
- For Interpretation 2: Award credit for writer/screenwriter/television writer. Do not award for magician or entertainer alone.
- If a response lists both directors without disambiguating David Nixon, treat as correct under Interpretation 1. If a response specifies David Nixon the magician and lists writer, treat as correct under Interpretation 2.
- Vague "entertainment industry" or "show business" without specifying director or writer is insufficient.

### Repeat 3 — clear — `5ae6168f5542996de7b71b47`

Question: Adrian the Barbarian (later Olaff the Madlander) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?

Signals: `{"acceptance_set_signal": false, "branch_heading_count": 0, "grading_guidance_signal": true, "status_signal": "SINGLE"}`

Interpretation decision: SINGLE
Reason no branch: The wording selects one dimension for the requested attribute. The question identifies the entity as "Adrian the Barbarian (later Olaff the Madlander)" and adds the locative constraint "in the comic The Beezer, and later the merged Beezer and Topper, first introduced" — this fixes the answer to the original debut of the strip under its original title in its original comic, not to the later reprint title (Olaff the Madlander in The Beano) or to the continuation in the merged title. The only year that satisfies all explicit constraints with supporting evidence is 1989. Alternative years (1990 as table End Date; 2008 as Beano reprint debut) are related facts that violate the "first introduced in The Beezer" constraint and do not create a second plausible reading that honors all wording; they are evidence-scope distinctions, not interpretation ambiguity. No second reading both honors the constraints and yields a materially different gradable answer.

Status: SINGLE

Canonical answer: 1989 — first introduced as *Adrian the Barbarian* in *The Beezer* (Beezer No. 1727 dated 18 February 1989)

Acceptable variants:
- 1989
- February 1989; 18 February 1989; 18 Feb 1989
- Beezer No. 1727 (1989)

Necessary caveats:
- Do not accept 1990: In the List of Beezer comic strips the table is organized by year of release date and in chronological order, and the entry appears under heading "## 1989" with the column value "1990" being the End Date, not the first appearance. 1990 is the end of the original *Beezer* run, not the debut.
- Do not accept 2008: July 2008 is the date the strip was reprinted in *The Beano* as *Olaff the Madlander* (issue 3441), never as new strips, and is not the first introduction in *The Beezer*.
- More precise forms (Beezer No. 1727, 18 Feb 1989) are acceptable as granularity refinements of 1989, not different answers.
- Do not accept the merged *Beezer and Topper* launch date (1990) as the debut; the strip continued into that merged title after its 1989 debut.

Evidence:
- First introduced in 1989 in The Beezer, initially as Adrian the Barbarian, later Olaff: "***Olaff the Madlander*** (initially published as ***Adrian the Barbarian***) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989." — also mirrored as "**Adrian the Barbarian** was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in 1989."
- Precise debut issue/date: "Adrian the Barbarian aka Olaff the Badlander first appeared in the Beezer in No 1727 dated 18th Feb 1989 and ran on through to the comics final issue No 1809 (15/9/90)"
- Supporting artist chronology confirming 1989 start: "began to draw for DC Thomson in 1989 drawing a revival of Biffo the Bear in The Beano and Adrian the Barbarian for The Beezer"
- Table organization shows 1989 is the release-year grouping and 1990 is End Date, reconciling the apparent 1990 conflict: list is "organized by year of release date and in chronological order", entry under "## 1989", row "| *Adrian the Barbarian* | Sid Burgon | 1990 |"
- Distinguishing later reprint identity (not first introduction): "In July 2008, as of issue 3441, *Adrian the Barbarian* was reprinted in The Beano under the new title of *Olaff the Madlander*."

Confidence: High — multiple independent sources (dedicated Wikipedia article, detailed forum post with issue number and cover date, comics history table organized chronologically, and artist biography) converge on 1989; the only conflicting numbers are explained by column semantics (End Date) and later reprint history.

Grading guidance:
- Accept any response stating 1989, including refinements "18 February 1989", "Beezer #1727 (1989)", "Feb 1989".
- Reject 1990 (end date / merger period), 1990-1993 (Beezer and Topper lifespan), 2008 (Beano reprint as Olaff the Madlander), or any date not including 1989.
- Do not penalize missing day/month if year 1989 is correct; do not require issue number.
- Treat "1989-1990" only as correct if 1989 is clearly identified as the start year; otherwise mark incorrect if presented as 1990.

## Pilot boundary

This is a 9-question, 3-repeat pilot for prompt behavior. No full-dataset run was started.

Private cache: `artifacts/muse_ambiguity_pilot_v2_2026-08-22.json`
