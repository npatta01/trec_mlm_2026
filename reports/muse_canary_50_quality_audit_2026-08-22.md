# Muse 50-question canary: quality audit

Date: 2026-08-22
Model: `muse-spark-1.2-contributor` via Meta Model API
Dataset: 50-question deterministic sample from the 321-question development set

## Executive result

- 50/50 questions completed.
- 2 model rounds per question; 100 model API calls total.
- Search was not limited to two searches: 411 search actions, 310 page opens, and 8 in-page finds, or 729 web-search actions total.
- 50/50 final answers included URL citations.
- No clear answer-level factual failure was found in this manual audit.
- Wikipedia and secondary sources are acceptable for this reference use when the claims are supported and the sources are not fabricated. The main remaining issue is making necessary qualifications explicit without treating every optional detail as part of the canonical answer.

“Pass with caveat” means the answer is materially correct but should preserve a qualification, identify an ambiguity, or make the canonical answer easier to extract. It does not mean the answer is wrong, and long presentation is acceptable.

## Item-by-item verdict

| # | Short answer | Verdict | Audit note |
|---:|---|---|---|
| 1 | *Senior*: 8 Sep Japan; 13 Sep UK/Europe | Pass with caveat | Release dates vary by territory; the final answer handled the conflict correctly. |
| 2 | 1989 | Pass | Specific issue chronology supports the year. |
| 3 | Fiorello La Guardia | Pass | Correct; fusion-voting history is a useful caveat. |
| 4 | 1873 | Pass | Correctly identifies Enrico Caruso’s birth year. |
| 5 | Amazon | Pass with caveat | Twitch was acquired by Amazon after the first tour; the answer explains the timing. |
| 6 | Architecture / monumental architecture | Pass | Correct and appropriately illustrated. |
| 7 | Jack Rose | Pass | Correct; the supporting source mix is weaker than ideal. |
| 8 | American / United States | Pass | Correct, with a harmless honorary-citizenship nuance. |
| 9 | Classic Case | Pass | 2002 precedes Panic! at the Disco’s 2004 formation. |
| 10 | Wetland / riparian vegetation | Pass with caveat | The overlap is valid for relevant species, not every *Erythrina* species; the answer says this. |
| 11 | Thylakoid membrane/network | Pass | Supported by primary plant-biology literature. |
| 12 | Department of Health, Education, and Welfare | Pass | Supported by Social Security Administration material. |
| 13 | 1929 | Pass | Correct establishment year for the Serie B reference. |
| 14 | American | Pass with caveat | Aronofsky’s later Polish citizenship is time-sensitive but correctly noted. |
| 15 | Israel | Pass with caveat | “Most” depends on the historical period and wording; the answer distinguishes the source’s intended Israel/aliyah answer from North-American migration. |
| 16 | *The Moon-Spinners* | Pass | 1964 precedes *The Fighting Prince of Donegal* (1966). |
| 17 | China | Pass | Both Jiexiu and Luohe are in China. |
| 18 | Teenage depression and suicide | Pass | Correctly matches the song’s subject. |
| 19 | China | Pass | Both locations are in China, in different provinces. |
| 20 | *The Closer* and *Hairspray* | Pass | David Gabriel is from *The Closer*; Seaweed is from *Hairspray*. |
| 21 | Skin / skeletal system / feathers, etc. | Pass with caveat | The question is open-ended; the examples are valid. For evaluation, the canonical answer should identify one accepted example clearly. |
| 22 | Nerf Herder | Pass | 1994 precedes The Academy Is... (2003). |
| 23 | United States | Pass | Both bands originated in the U.S. |
| 24 | Luke Greenfield | Pass | Correct director of the 2004 film. |
| 25 | *Matilda* | Pass | Crunchem Hall is the school in the 1996 film. |
| 26 | Christer Fuglesang | Pass | ESA material confirms the astronaut identity. |
| 27 | Pro Kabaddi League | Pass | Correct league for UP Yoddha(s). |
| 28 | Vera Karalli; ballet dancer | Pass | Correctly disambiguates the 1921 film from later films with the same English title. |
| 29 | Korea and Vietnam | Pass | Correct pair for the standard Cold War question. |
| 30 | Jeff Flake | Pass | Correct Arizona junior senator alongside John McCain in 2013. |
| 31 | No; only thyme is aromatic | Pass with caveat | Correct in ordinary horticultural usage; “aromatic” should be defined as fragrant/essential-oil producing. |
| 32 | No | Pass with caveat | Correct at the genus-level core range, though *Evolvulus* has pantropical introductions and disputed native-range treatment for some species. |
| 33 | False | Pass | *Otello* is an opera; *Lucrezia Borgia* is also a play title and an opera title. |
| 34 | Holiday Bowl; since 1978 | Pass with caveat | Correct, with the answer noting COVID-era cancellations despite the “annually” wording. |
| 35 | Vandenberg Air Force Base, California | Pass | NASA TIMED records confirm the launch site and date. |
| 36 | Chicken-fried steak / country-fried steak | Pass with caveat | Naming varies by whether it is deep-fried or pan-fried; the answer explains this. |
| 37 | Miguel Ali / Grizz Ali | Pass | Writer-director identity and alternate credit are correctly connected. |
| 38 | Two | Pass with caveat | Bilateral normally means two parties; treaty structures can contain multiple entities grouped into two sides. |
| 39 | Sheila Jackson Lee | Pass with caveat | The answer is consistent with the source trail, but this item should receive a primary hearing-transcript citation before production use. |
| 40 | MTV | Pass with caveat | Correct for the historical seasons; the franchise’s later platform history is separate. |
| 41 | Boston, Massachusetts | Pass | Dorchester is the Boston neighborhood commonly given as the group’s origin. |
| 42 | John von Neumann | Pass | Correctly identifies the Murray–von Neumann affiliated-operator reference. |
| 43 | Film directors | Pass with caveat | “David Nixon” is ambiguous; the answer correctly specifies the director, not the magician. |
| 44 | New York | Pass | De La Soul formed in Amityville, New York. |
| 45 | Discipline, engineering, logistics, expansion, etc. | Pass with caveat | The long answer is acceptable; for evaluation, the canonical answer should identify the supported trait(s) rather than require matching the whole list. |
| 46 | House Rule XIX, clause 2; historical Rule XIII protection | Pass with caveat | The answer correctly explains that the “with instructions” form was removed at the start of the 117th Congress. |
| 47 | No; Rooney American, Brocka Filipino | Pass | Correct. |
| 48 | *Cissus* | Pass | It has substantially more known species than *Digitalis*. |
| 49 | *Battlefield Earth* | Pass | Correct film and source novel relationship. |
| 50 | Thums Up | Pass with caveat | Correct within the cola category; market-leadership claims are time-sensitive and should be dated. |

## Recommendation

Proceed to a larger run if we retain the two-call workflow and use a reference-answer format that separates the canonical answer, acceptable variants, necessary qualifications, and evidence. Prefer primary/official, academic, government, or reputable news sources when available, but do not reject Wikipedia merely because it is Wikipedia. This does not require an arbitrary answer-length cap.

## Independent spot-checks

- NASA’s TIMED fact sheet confirms the 7 December 2001 date and Vandenberg launch site: <https://spdf.gsfc.nasa.gov/pub/data/timed/documentation/mdc/timed_factsheet.pdf>
- The plastoglobule claim is supported by the primary research article: <https://pmc.ncbi.nlm.nih.gov/articles/PMC1488921/>
- The UN treaty handbook defines a bilateral treaty as an agreement between two parties: <https://treaties.un.org/doc/source/publications/treatyhandbook/TreatyHandbookEng.pdf>
- Congress’s current CRS summary explains the historical motion-to-recommit rule and the 117th-Congress change: <https://www.congress.gov/crs-product/R44330>
- Kew’s plant record supports the *Adenium* range used in the audit: <https://powo.science.kew.org/taxon/urn:lsid:ipni.org:names:76369-1>
- The official Holiday Bowl site dates the bowl to 1978: <https://holidaybowl.com/about-us/>
