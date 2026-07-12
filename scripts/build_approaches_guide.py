"""Build the source-backed, mobile-first TREC 2026 approaches guide artifact.

The portable HTML is generated from the canonical Data Analytics report payload.
This script keeps local-data facts executable and the literature synthesis reviewable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from statistics import mean, median
from textwrap import dedent
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "data/dev/llm_dev_data.tsv"
QRELS = ROOT / "data/dev/llm_dev_qrels.txt"
DENSE_RESULTS = ROOT / "reports/trec_2026_dense_query_knn_results.json"
DISCOVERY_AUDIT = ROOT / "reports/trec_2026_discovery_profile_audit.json"
LLM_PROFILES = ROOT / "reports/trec_2026_llm_profiles.json"
ARTIFACT = ROOT / "reports/trec_2026_approaches_guide.artifact.json"
SOURCE_NOTES = ROOT / "reports/trec_2026_approaches_guide_source_notes.md"
GENERATED_AT = "2026-07-11T00:00:00Z"


RUNS_2025: list[dict[str, Any]] = [
    {"rank": 1, "run": "submission_4", "team": "SRCB", "ndcg10": 0.1745, "mrr": 0.3126, "family": "Weak labels + neural scorer", "idea": "Run page names Qwen3-Embedding-8B; the paper does not fully map runs to configurations"},
    {"rank": 2, "run": "infolab_UD_run2", "team": "UDInfo", "ndcg10": 0.1691, "mrr": 0.2725, "family": "Weak-label query kNN", "idea": "Judged neighbors, global prior, and query-domain clusters"},
    {"rank": 3, "run": "r-ef-rm3-1000-ss", "team": "GenAIus", "ndcg10": 0.1680, "mrr": 0.1980, "family": "Response retrieval", "idea": "Extra-filtered response BM25/RM3 with score aggregation"},
    {"rank": 4, "run": "r-f-rm3-1000-rr-ss", "team": "GenAIus", "ndcg10": 0.1608, "mrr": 0.2066, "family": "Response retrieval", "idea": "Response BM25/RM3, SPLADE reranking, score aggregation"},
    {"rank": 5, "run": "r-f-rm3-1000-ss", "team": "GenAIus", "ndcg10": 0.1545, "mrr": 0.1843, "family": "Response retrieval", "idea": "Filtered response BM25/RM3 with score aggregation"},
    {"rank": 6, "run": "infolab_UD_run4", "team": "UDInfo", "ndcg10": 0.1534, "mrr": 0.2579, "family": "Neural router", "idea": "Weak-label MLP classifier"},
    {"rank": 7, "run": "ensemble_v1", "team": "SRCB", "ndcg10": 0.1522, "mrr": 0.2605, "family": "Ensemble", "idea": "SRCB ensemble; exact component mapping is not documented in the paper"},
    {"rank": 8, "run": "infolab_UD_run1", "team": "UDInfo", "ndcg10": 0.1517, "mrr": 0.2760, "family": "Answer profiles", "idea": "Per-model answer centroids plus nearest-answer refinement"},
    {"rank": 9, "run": "ensemble_v3", "team": "SRCB", "ndcg10": 0.1485, "mrr": 0.2510, "family": "Ensemble", "idea": "SRCB ensemble; exact component mapping is not documented in the paper"},
    {"rank": 10, "run": "ensemble_v2", "team": "SRCB", "ndcg10": 0.1477, "mrr": 0.2541, "family": "Ensemble", "idea": "SRCB ensemble; exact component mapping is not documented in the paper"},
    {"rank": 11, "run": "submission_5", "team": "SRCB", "ndcg10": 0.1339, "mrr": 0.1971, "family": "Weak labels + neural scorer", "idea": "Qwen3-14B annotations, DeBERTa-v3, fully connected decoder, BCE"},
    {"rank": 12, "run": "non-neural", "team": "indelab", "ndcg10": 0.1309, "mrr": 0.2130, "family": "Engineered ensemble", "idea": "Bradley-Terry, random forest, and LightGBM meta-ranker"},
    {"rank": 13, "run": "lightgbm_job266431", "team": "uvairlab", "ndcg10": 0.1233, "mrr": 0.1433, "family": "Confidence LTR", "idea": "LambdaRank over query embeddings, model ID, and logprob summaries"},
    {"rank": 14, "run": "q-f-rm3-100-ss", "team": "GenAIus", "ndcg10": 0.1203, "mrr": 0.1966, "family": "Query retrieval", "idea": "Question BM25/RM3 with score aggregation"},
    {"rank": 15, "run": "llm-f-100", "team": "GenAIus", "ndcg10": 0.1060, "mrr": 0.1890, "family": "Whole-model profiles", "idea": "One concatenated BM25 profile per model"},
    {"rank": 16, "run": "infolab_UD_run5", "team": "UDInfo", "ndcg10": 0.1021, "mrr": 0.2114, "family": "Neural router", "idea": "Deeper weak-label MLP router"},
    {"rank": 17, "run": "infolab_UD_run3", "team": "UDInfo", "ndcg10": 0.1018, "mrr": 0.1832, "family": "Answer profiles", "idea": "Clustered per-model answer profiles"},
    {"rank": 18, "run": "mllm-indelab-09-17", "team": "indelab", "ndcg10": 0.0814, "mrr": 0.1479, "family": "Dual encoder", "idea": "Query/model towers with metadata-derived weak labels"},
    {"rank": 19, "run": "lambdamart_profiles", "team": "paulphys", "ndcg10": 0.0549, "mrr": 0.0704, "family": "Behavior profiles", "idea": "Hand-engineered behavioral profiles with LambdaMART"},
]


APPROACH_LADDER = [
    {"stage": 0, "approach": "Global candidate prior", "needs": "Dev qrels", "run_type": "Internal", "effort": "Hours", "role": "Sanity floor", "priority": "Keep"},
    {"stage": 1, "approach": "Sparse query-neighbor transfer", "needs": "Dev queries + qrels", "run_type": "Internal", "effort": "Hours", "role": "Lexical control", "priority": "Keep"},
    {"stage": 2, "approach": "Dense query-neighbor transfer", "needs": "Dev queries + qrels + frozen encoder", "run_type": "Likely external", "effort": "Hours", "role": "Current strongest baseline", "priority": "Keep"},
    {"stage": 3, "approach": "Query-to-candidate-name match", "needs": "Readable candidate IDs", "run_type": "Internal or external", "effort": "Hours", "role": "Weak feature only", "priority": "Test"},
    {"stage": 4, "approach": "Response-level BM25/RM3", "needs": "Discovery responses", "run_type": "Internal", "effort": "Days", "role": "Strong 2025-style baseline", "priority": "Build next"},
    {"stage": 5, "approach": "Dense response retrieval + profiles", "needs": "Discovery responses + encoder", "run_type": "Likely external", "effort": "Days", "role": "Semantic behavioral evidence", "priority": "Build next"},
    {"stage": 6, "approach": "Logprob/confidence LTR", "needs": "Discovery metadata", "run_type": "Internal", "effort": "Days", "role": "Cheap complementary features", "priority": "Add"},
    {"stage": 7, "approach": "Weak-label hybrid router", "needs": "Discovery responses + judge/reranker", "run_type": "Likely external", "effort": "1–2 weeks", "role": "High-upside supervision", "priority": "Add carefully"},
    {"stage": 8, "approach": "Nested fusion / ensemble", "needs": "Out-of-fold scores from diverse systems", "run_type": "Depends", "effort": "Days", "role": "Final submission portfolio", "priority": "Finish"},
]


SOURCES: list[dict[str, Any]] = [
    {"id": "track_2026", "label": "Official TREC 2026 Million LLMs track page", "href": "https://trec-mllm.github.io/"},
    {"id": "track_2025", "label": "Archived TREC 2025 Million LLMs track page", "href": "https://trec-mllm.github.io/2025.html"},
    {"id": "overview_2025", "label": "Official TREC 2025 Million LLMs overview", "href": "https://trec.nist.gov/pubs/trec34/papers/Overview_mllm.pdf"},
    {"id": "runs_2025", "label": "Official TREC 2025 run appendix and trec_eval files", "href": "https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main.html"},
    {"id": "srcb_2025", "label": "SRCB at TREC 2025", "href": "https://trec.nist.gov/pubs/trec34/papers/SRCB.mllm.tot.pdf"},
    {"id": "udinfo_2025", "label": "UDInfo at TREC 2025", "href": "https://trec.nist.gov/pubs/trec34/papers/UDInfo.mllm.pdf"},
    {"id": "genaius_2025", "label": "GenAIus at TREC 2025", "href": "https://trec.nist.gov/pubs/trec34/papers/GenAIus.mllm.pdf"},
    {"id": "uvairlab_2025", "label": "UvA IRLab at TREC 2025", "href": "https://trec.nist.gov/pubs/trec34/papers/uvairlab.mllm.pdf"},
    {"id": "srcb_submission5", "label": "Official submission_5 run description", "href": "https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main/submission_5.html"},
    {"id": "indelab_neural", "label": "Official indelab neural run description", "href": "https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main/mllm-indelab-09-17.html"},
    {"id": "indelab_non_neural", "label": "Official indelab non-neural run description", "href": "https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main/non-neural.html"},
    {"id": "paulphys_run", "label": "Official paulphys run description", "href": "https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main/lambdamart_profiles.html"},
    {"id": "paulphys_repo", "label": "Author repository and behavioral-profiles report", "href": "https://github.com/eigenpaul/trec2025-mllm"},
    {"id": "deep_dive_2025", "label": "Existing interactive TREC 2025 methods deep dive", "path": "reports/trec_2025_mllm_interactive_report.html"},
    {"id": "local_data", "label": "Official 2026 development queries and qrels", "path": "data/dev/llm_dev_qrels.txt"},
    {"id": "local_readme", "label": "Captured 2026 data release notes and manifest", "path": "data/dev/README.md"},
    {"id": "dense_results", "label": "Reviewed 2026 nested dense-query baseline", "path": "reports/trec_2026_dense_query_knn_results.json"},
    {"id": "dense_code", "label": "Exact nested leave-one-query-out implementation", "path": "trec_mlm/dense_experiment.py"},
    {"id": "expert_finding", "label": "Balog et al. (2006), Formal Models for Expert Finding", "href": "https://doi.org/10.1145/1148170.1148181"},
    {"id": "expert_voting", "label": "Macdonald and Ounis (2006), Voting for Candidates", "href": "https://doi.org/10.1145/1183614.1183671"},
    {"id": "redde", "label": "Si and Callan (2003), ReDDE distributed resource selection", "href": "https://doi.org/10.1145/860435.860490"},
    {"id": "clarity", "label": "Cronen-Townsend, Zhou, and Croft (2002), Predicting query performance", "href": "https://doi.org/10.1145/564376.564429"},
    {"id": "bm25", "label": "Robertson and Zaragoza (2009), The Probabilistic Relevance Framework: BM25 and Beyond", "href": "https://doi.org/10.1561/1500000019"},
    {"id": "rm3", "label": "Abdul-Jaleel et al. (2004), UMass at TREC 2004: Novelty and HARD", "href": "https://trec.nist.gov/pubs/trec13/papers/umass.novelty.hard.pdf"},
    {"id": "splade", "label": "Formal et al. (2021), SPLADE v2", "href": "https://arxiv.org/abs/2107.05720"},
    {"id": "sbert", "label": "Reimers and Gurevych (2019), Sentence-BERT", "href": "https://aclanthology.org/D19-1410/"},
    {"id": "dpr", "label": "Karpukhin et al. (2020), Dense Passage Retrieval", "href": "https://aclanthology.org/2020.emnlp-main.550/"},
    {"id": "mpnet", "label": "Song et al. (2020), MPNet", "href": "https://proceedings.neurips.cc/paper/2020/hash/c3a690be93aa602ee2dc0ccab5b7b67e-Abstract.html"},
    {"id": "e5", "label": "Wang et al. (2022), Text Embeddings by Weakly-Supervised Contrastive Pre-training", "href": "https://arxiv.org/abs/2212.03533"},
    {"id": "bge", "label": "Xiao et al. (2023), C-Pack / BGE embeddings", "href": "https://arxiv.org/abs/2309.07597"},
    {"id": "bge_m3", "label": "Chen et al. (2024), BGE-M3", "href": "https://arxiv.org/abs/2402.03216"},
    {"id": "colbert", "label": "Khattab and Zaharia (2020), ColBERT", "href": "https://arxiv.org/abs/2004.12832"},
    {"id": "rrf", "label": "Cormack, Clarke, and Büttcher (2009), Reciprocal Rank Fusion", "href": "https://doi.org/10.1145/1571941.1572114"},
    {"id": "mteb", "label": "Muennighoff et al. (2023), MTEB", "href": "https://aclanthology.org/2023.eacl-main.148/"},
    {"id": "beir", "label": "Thakur et al. (2021), BEIR", "href": "https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html"},
    {"id": "lambdamart", "label": "Burges (2010), From RankNet to LambdaRank to LambdaMART", "href": "https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/"},
    {"id": "lightgbm", "label": "Ke et al. (2017), LightGBM", "href": "https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html"},
    {"id": "listnet", "label": "Cao et al. (2007), ListNet", "href": "https://doi.org/10.1145/1273496.1273513"},
    {"id": "xgboost", "label": "Chen and Guestrin (2016), XGBoost", "href": "https://doi.org/10.1145/2939672.2939785"},
    {"id": "catboost", "label": "Prokhorenkova et al. (2018), CatBoost", "href": "https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html"},
    {"id": "frugalgpt", "label": "Chen, Zaharia, and Zou (2023), FrugalGPT", "href": "https://arxiv.org/abs/2305.05176"},
    {"id": "llm_blender", "label": "Jiang, Ren, and Lin (2023), LLM-Blender", "href": "https://arxiv.org/abs/2306.02561"},
    {"id": "routerbench", "label": "Hu et al. (2024), RouterBench", "href": "https://arxiv.org/abs/2403.12031"},
    {"id": "routellm", "label": "Ong et al. (2024), RouteLLM", "href": "https://arxiv.org/abs/2406.18665"},
    {"id": "graphrouter", "label": "Feng, Shen, and You (2025), GraphRouter", "href": "https://openreview.net/forum?id=eU39PDsZtT"},
    {"id": "contextualrouter", "label": "Varangot-Reille, Bouvard, and Gourru (2026), ContextualRouter", "href": "https://aclanthology.org/2026.eacl-srw.22/"},
    {"id": "llmrouterbench", "label": "Li et al. (2026), LLMRouterBench", "href": "https://aclanthology.org/2026.findings-acl.1881/"},
    {"id": "bradley_terry", "label": "Bradley and Terry (1952), Rank analysis of incomplete block designs", "href": "https://doi.org/10.1093/biomet/39.3-4.324"},
    {"id": "instructgpt", "label": "Ouyang et al. (2022), InstructGPT", "href": "https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html"},
    {"id": "snorkel", "label": "Ratner et al. (2017), Snorkel", "href": "https://doi.org/10.14778/3157794.3157797"},
    {"id": "mtbench", "label": "Zheng et al. (2023), MT-Bench and Chatbot Arena", "href": "https://arxiv.org/abs/2306.05685"},
    {"id": "judge_bias", "label": "Wang et al. (2023), Large Language Models are not Fair Evaluators", "href": "https://arxiv.org/abs/2305.17926"},
    {"id": "self_preference", "label": "Panickssery, Bowman, and Feng (2024), LLM evaluators favor their own generations", "href": "https://papers.nips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html"},
    {"id": "judge_jury", "label": "Verga et al. (2024), Replacing judges with juries", "href": "https://arxiv.org/abs/2404.18796"},
    {"id": "geval", "label": "Liu et al. (2023), G-Eval", "href": "https://arxiv.org/abs/2303.16634"},
    {"id": "llm_rubric", "label": "Hashemi et al. (2024), LLM-Rubric", "href": "https://aclanthology.org/2024.acl-long.745/"},
    {"id": "maple_judges", "label": "Jinji et al. (2026), MAPLE multi-aspect evaluator panels", "href": "https://aclanthology.org/2026.findings-acl.1351/"},
    {"id": "saja", "label": "Kola et al. (2026), SAJA judge alignment", "href": "https://aclanthology.org/2026.acl-industry.45/"},
    {"id": "rubric_position_bias", "label": "Xu et al. (2026), Position bias in rubric-based LLM judging", "href": "https://arxiv.org/abs/2602.02219"},
    {"id": "calibration", "label": "Guo et al. (2017), On Calibration of Modern Neural Networks", "href": "https://proceedings.mlr.press/v70/guo17a.html"},
    {"id": "self_knowledge", "label": "Kadavath et al. (2022), Language Models (Mostly) Know What They Know", "href": "https://arxiv.org/abs/2207.05221"},
    {"id": "semantic_entropy", "label": "Farquhar et al. (2024), Detecting hallucinations using semantic entropy", "href": "https://www.nature.com/articles/s41586-024-07421-0"},
    {"id": "nested_cv", "label": "Cawley and Talbot (2010), selection bias and nested evaluation", "href": "https://www.jmlr.org/papers/v11/cawley10a.html"},
    {"id": "leakage", "label": "Kaufman et al. (2012), Leakage in data mining", "href": "https://doi.org/10.1145/2382577.2382579"},
    {"id": "covariate_shift", "label": "Sugiyama et al. (2007), importance-weighted cross-validation", "href": "https://www.jmlr.org/papers/v8/sugiyama07a.html"},
    {"id": "shift_detection", "label": "Rabanser, Günnemann, and Lipton (2019), Failing Loudly", "href": "https://papers.nips.cc/paper_files/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html"},
    {"id": "wilds", "label": "Koh et al. (2021), WILDS", "href": "https://proceedings.mlr.press/v139/koh21a.html"},
    {"id": "ndcg", "label": "Järvelin and Kekäläinen (2002), Cumulated gain-based evaluation", "href": "https://doi.org/10.1145/582415.582418"},
]


def clean(text: str) -> str:
    return dedent(text).strip()


def md(block_id: str, body: str, source_id: str | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {"id": block_id, "type": "markdown", "body": clean(body)}
    if source_id:
        block["sourceId"] = source_id
    return block


def teaching_visual(
    block_id: str,
    title: str,
    intro: str,
    body: str,
    audio_summary: str,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Create a script-free teaching figure for the sandboxed report reader."""

    visual_css = """
    <style>
      :root {
        color-scheme: light dark;
        --bg: #ffffff;
        --surface: #f8fafc;
        --surface-blue: #eaf2ff;
        --surface-gold: #fff7df;
        --surface-green: #eaf8ef;
        --surface-red: #fff0f0;
        --ink: #172033;
        --muted: #475569;
        --line: #cbd5e1;
        --blue: #174ea6;
        --gold: #7a4d00;
        --green: #17633a;
        --red: #9b2525;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: transparent;
        color: var(--ink);
        font: 16px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .lesson {
        margin: 0;
        padding: clamp(16px, 4vw, 24px);
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--bg);
      }
      .eyebrow {
        margin: 0 0 5px;
        color: var(--blue);
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      h2 { margin: 0; font-size: clamp(1.2rem, 4vw, 1.5rem); line-height: 1.25; }
      .lede { max-width: 72ch; margin: 8px 0 16px; color: var(--muted); }
      .card {
        min-width: 0;
        padding: 13px;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: var(--surface);
        overflow-wrap: anywhere;
      }
      .card h3, .card strong { color: var(--ink); }
      .card h3 { margin: 0 0 6px; font-size: 1rem; line-height: 1.3; }
      .card p { margin: 5px 0 0; }
      .card ul, .card ol { margin: 7px 0 0; padding-left: 1.25rem; }
      .card li + li { margin-top: 4px; }
      .blue { border-color: var(--blue); background: var(--surface-blue); }
      .gold { border-color: var(--gold); background: var(--surface-gold); }
      .green { border-color: var(--green); background: var(--surface-green); }
      .red { border-color: var(--red); background: var(--surface-red); }
      .badge {
        display: inline-block;
        margin-top: 8px;
        padding: 3px 8px;
        border: 1px solid currentColor;
        border-radius: 999px;
        color: var(--muted);
        font-size: 0.95rem;
        font-weight: 750;
      }
      .source-grid, .branch-grid, .compare-grid, .phase-grid, .metric-grid, .cache-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }
      .phase-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .flow-row, .triple {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 32px minmax(0, 1fr) 32px minmax(0, 1fr);
        align-items: stretch;
        gap: 6px;
      }
      .arrow { align-self: center; color: var(--muted); font-size: 1.35rem; font-weight: 800; text-align: center; }
      .arrow .narrow { display: none; }
      .down { margin: 8px 0; color: var(--muted); font-size: 1.35rem; font-weight: 800; text-align: center; }
      .divider {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 15px 0;
        color: var(--muted);
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .divider::before, .divider::after { content: ""; flex: 1; border-top: 1px solid var(--line); }
      .callout {
        margin: 13px 0 0;
        padding: 11px 13px;
        border-left: 4px solid var(--blue);
        border-radius: 0 9px 9px 0;
        background: var(--surface-blue);
        font-weight: 700;
      }
      .grade-list, .ranking, .signal-list { margin: 9px 0 0; padding: 0; list-style: none; }
      .grade-list li, .ranking li, .signal-list li {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        padding: 8px 0;
        border-bottom: 1px solid var(--line);
      }
      .grade-list li:last-child, .ranking li:last-child, .signal-list li:last-child { border-bottom: 0; }
      .grade {
        flex: 0 0 auto;
        min-width: 2rem;
        padding: 2px 7px;
        border: 1px solid currentColor;
        border-radius: 999px;
        font-weight: 800;
        text-align: center;
      }
      .grade-2 { color: var(--green); }
      .grade-1 { color: var(--blue); }
      .grade-0 { color: var(--muted); }
      .stage-list { margin: 8px 0 0; padding-left: 1.45rem; }
      .stage-list li + li { margin-top: 9px; }
      .stage-list span { display: block; color: var(--muted); font-size: 1rem; }
      .gate { margin: 12px 0; padding: 10px 12px; border: 1px dashed var(--gold); border-radius: 10px; background: var(--surface-gold); }
      .caption { margin: 15px 0 0; padding-top: 12px; border-top: 1px solid var(--line); color: var(--muted); }
      code { font-size: 1em; overflow-wrap: anywhere; }
      @media (max-width: 600px) {
        .source-grid, .branch-grid, .compare-grid, .phase-grid, .metric-grid, .cache-grid { grid-template-columns: 1fr; }
        .flow-row, .triple { grid-template-columns: 1fr; }
        .arrow .wide { display: none; }
        .arrow .narrow { display: inline; }
        .card { padding: 12px; }
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --bg: #111827;
          --surface: #1f2937;
          --surface-blue: #172c4f;
          --surface-gold: #422f0c;
          --surface-green: #143b2a;
          --surface-red: #482020;
          --ink: #f8fafc;
          --muted: #d1d5db;
          --line: #64748b;
          --blue: #b8d7ff;
          --gold: #ffe09a;
          --green: #a7e8c2;
          --red: #ffb9b9;
        }
      }
      @media (forced-colors: active) {
        .card, .badge, .grade, .gate { border: 2px solid CanvasText; }
        .callout { border-left-color: CanvasText; }
      }
    </style>
    """
    block: dict[str, Any] = {
        "id": block_id,
        "type": "html",
        "layout": "full",
        "body": clean(
            f"""
            {visual_css}
            <figure class="lesson" lang="en" role="group" aria-labelledby="{block_id}-title" aria-describedby="{block_id}-caption">
              <p class="eyebrow">Beginner visual</p>
              <h2 id="{block_id}-title">{title}</h2>
              <p class="lede">{intro}</p>
              {body}
              <figcaption class="caption" id="{block_id}-caption"><strong>Audio summary:</strong> {audio_summary}</figcaption>
            </figure>
            """
        ),
    }
    if source_id:
        block["sourceId"] = source_id
    return block


def _load_local_data() -> dict[str, Any]:
    queries: dict[str, str] = {}
    for line_number, raw in enumerate(QUERIES.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split("\t", 1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"invalid query row {line_number}")
        query_id, query = parts
        if query_id in queries:
            raise ValueError(f"duplicate query id {query_id}")
        queries[query_id] = query

    grades: Counter[int] = Counter()
    candidates: set[str] = set()
    per_query_positive: defaultdict[str, int] = defaultdict(int)
    pairs: set[tuple[str, str]] = set()
    for line_number, raw in enumerate(QRELS.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if len(parts) != 4:
            raise ValueError(f"invalid qrel row {line_number}")
        query_id, iteration, candidate, grade_text = parts
        if iteration != "0" or query_id not in queries:
            raise ValueError(f"invalid qrel identity at row {line_number}")
        pair = (query_id, candidate)
        if pair in pairs:
            raise ValueError(f"duplicate qrel pair {pair}")
        pairs.add(pair)
        grade = int(grade_text)
        if grade not in {0, 1, 2}:
            raise ValueError(f"invalid grade {grade}")
        grades[grade] += 1
        candidates.add(candidate)
        if grade > 0:
            per_query_positive[query_id] += 1

    if len(queries) != 322 or len(candidates) != 150 or len(pairs) != 48_300:
        raise ValueError("unexpected 2026 development matrix shape")
    if set(per_query_positive) != set(queries):
        raise ValueError("every query should have at least one positive candidate")

    positive_counts = sorted(per_query_positive.values())
    word_counts = sorted(len(text.split()) for text in queries.values())
    return {
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "qrel_count": len(pairs),
        "grade_counts": dict(sorted(grades.items())),
        "positive_count": grades[1] + grades[2],
        "positive_share": (grades[1] + grades[2]) / len(pairs),
        "grade2_share_of_positive": grades[2] / (grades[1] + grades[2]),
        "exactly_one_positive": sum(value == 1 for value in positive_counts),
        "at_most_five_positive": sum(value <= 5 for value in positive_counts),
        "at_least_fifty_positive": sum(value >= 50 for value in positive_counts),
        "positive_min": min(positive_counts),
        "positive_median": median(positive_counts),
        "positive_mean": mean(positive_counts),
        "positive_max": max(positive_counts),
        "query_words_min": min(word_counts),
        "query_words_median": median(word_counts),
        "query_words_mean": mean(word_counts),
        "query_words_max": max(word_counts),
        "candidate_names_with_underscore": sum("_" in value for value in candidates),
    }


def _dense_metrics() -> dict[str, dict[str, float]]:
    data = json.loads(DENSE_RESULTS.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    expected = {"dense_nested", "tfidf_fixed_k10", "global_prior"}
    if set(metrics) != expected:
        raise ValueError("unexpected dense-result systems")
    return metrics


def _discovery_profile_audit() -> dict[str, Any]:
    data = json.loads(DISCOVERY_AUDIT.read_text(encoding="utf-8"))
    plain = data["plain"]
    metadata = data["metadata"]
    expected = {
        "query_count": 7_759,
        "candidate_count": 150,
        "response_count": 1_163_850,
    }
    for field, value in expected.items():
        if plain[field] != value:
            raise ValueError(f"unexpected discovery audit {field}: {plain[field]}")
    if metadata["response_count"] != plain["response_count"]:
        raise ValueError("discovery audit response counts disagree")
    if data["sources"]["ordered_response_tuple_sha256"] != plain["tuple_sha256"]:
        raise ValueError("discovery audit tuple digest is inconsistent")
    return data


def _llm_profiles() -> dict[str, Any]:
    data = json.loads(LLM_PROFILES.read_text(encoding="utf-8"))
    if data["profile_version"] != "discovery-behavior-topic-v1":
        raise ValueError("unexpected candidate-profile version")
    scope = data["scope"]
    expected = {
        "query_count": 7_759,
        "candidate_count": 150,
        "response_count": 1_163_850,
    }
    for field, value in expected.items():
        if scope[field] != value:
            raise ValueError(f"unexpected candidate-profile {field}: {scope[field]}")
    profiles = data["profiles"]
    candidate_ids = [row["candidate_id"] for row in profiles]
    if len(profiles) != 150 or candidate_ids != sorted(set(candidate_ids)):
        raise ValueError("candidate profiles are missing, duplicated, or unsorted")
    for row in profiles:
        if row["history"]["record_count"] != scope["query_count"]:
            raise ValueError(f"incomplete profile for {row['candidate_id']}")
        coverage = row["coverage"]
        if coverage["attempt_count"] + coverage["exact_refusal_count"] != scope["query_count"]:
            raise ValueError(f"coverage counts do not reconcile for {row['candidate_id']}")
    return data


def _execute_sqlite_queries(
    tables: dict[str, list[dict[str, Any]]], queries: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    """Materialize reviewed rows and execute every SQL statement named in provenance."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        for table_name, rows in tables.items():
            if not rows:
                raise ValueError(f"cannot materialize empty table {table_name}")
            columns = list(rows[0])
            if any(list(row) != columns for row in rows):
                raise ValueError(f"inconsistent columns in {table_name}")

            def sqlite_type(column: str) -> str:
                values = [row[column] for row in rows if row[column] is not None]
                if values and all(isinstance(value, (bool, int)) for value in values):
                    return "INTEGER"
                if values and all(isinstance(value, (bool, int, float)) for value in values):
                    return "REAL"
                return "TEXT"

            quoted_table = '"' + table_name.replace('"', '""') + '"'
            definitions = ", ".join(
                f'"{column.replace(chr(34), chr(34) * 2)}" {sqlite_type(column)}'
                for column in columns
            )
            connection.execute(f"CREATE TABLE {quoted_table} ({definitions})")
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {quoted_table} VALUES ({placeholders})",
                [[row[column] for column in columns] for row in rows],
            )

        output: dict[str, list[dict[str, Any]]] = {}
        for name, sql in queries.items():
            cursor = connection.execute(sql)
            output[name] = [dict(row) for row in cursor.fetchall()]
        return output
    finally:
        connection.close()


def _profile_table_rows(profile_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for profile in profile_data["profiles"]:
        coverage = profile["coverage"]
        style = profile["attempt_style"]
        metadata = profile["metadata_quality"]
        terms = profile["attempted_query_terms"][:3]
        term_summary = " · ".join(
            f"{term['term']} ({term['support_queries']:,} queries; {term['lift']:.1f}×)"
            for term in terms
        )
        rows.append(
            {
                "candidate_id": profile["candidate_id"],
                "attempt_count": coverage["attempt_count"],
                "attempt_share": coverage["attempt_share"],
                "exact_refusal_share": coverage["exact_refusal_share"],
                "attempt_word_median": style["word_count_median"],
                "attempt_context_template_share": style["context_template_share"],
                "attempt_insufficiency_share": style["insufficiency_phrase_share"],
                "attempt_token_logprob_usable_share": metadata[
                    "attempt_token_logprob_usable_share"
                ],
                "top_term_summary": term_summary or "Insufficient support",
            }
        )
    return rows


def build_artifact(
    local: dict[str, Any],
    dense: dict[str, dict[str, float]],
    discovery_audit: dict[str, Any],
    llm_profiles: dict[str, Any],
) -> dict[str, Any]:
    sql_queries = {
        "data_headline": "SELECT query_count, median_query_words, candidate_count, qrel_count, positive_share, exactly_one FROM data_headline",
        "dense_headline": "SELECT dense_ndcg, dense_delta FROM dense_headline",
        "baseline_metrics": "SELECT system_order, metric_order, system, metric, value, query_count, protocol FROM baseline_metrics ORDER BY system_order, metric_order",
        "baseline_display": "SELECT system, metric, printf('%.4f / 1', value) AS value_display, protocol FROM baseline_metrics ORDER BY system_order, metric_order",
        "runs_display": "SELECT rank, run, team, printf('%.4f / 1', ndcg10) AS ndcg10_display, printf('%.4f / 1', mrr) AS mrr_display, family, idea FROM runs_2025 ORDER BY rank",
        "data_files": "SELECT file, status, size, contains, use FROM data_files ORDER BY \"order\"",
        "literature_map": "SELECT family, plain, evidence, fit, risk FROM literature_map ORDER BY \"order\"",
        "approach_ladder": "SELECT CAST(stage AS TEXT) AS stage_display, approach, needs, run_type, effort, role, priority FROM approach_ladder ORDER BY stage",
        "portfolio": "SELECT CAST(slot AS TEXT) AS slot_display, run, type, why FROM portfolio ORDER BY slot",
        "candidate_profiles": "SELECT candidate_id, attempt_count, attempt_share, exact_refusal_share, attempt_word_median, attempt_context_template_share, attempt_insufficiency_share, attempt_token_logprob_usable_share, top_term_summary FROM candidate_profiles_raw ORDER BY candidate_id",
    }
    dense_primary = dense["dense_nested"]
    tfidf = dense["tfidf_fixed_k10"]
    prior = dense["global_prior"]
    guide_source = {
        "id": "guide_notes",
        "label": "Evidence inventory, surface map, and literature-review notes",
        "path": "reports/trec_2026_approaches_guide_source_notes.md",
        "query": {
            "language": "text",
            "description": (
                "Reviewed Python/SQLite synthesis joining the official 2025 overview, participant papers, "
                "run descriptions and trec_eval files with the official 2026 release notes, "
                "local development matrix, reproducible baselines, and cited primary literature."
            ),
            "executed_at": GENERATED_AT,
            "tables_used": [
                "data/dev/llm_dev_data.tsv",
                "data/dev/llm_dev_qrels.txt",
                "data/dev/llm_discovery_data.json",
                "data/dev/llm_discovery_metadata.json",
                "reports/trec_2026_discovery_profile_audit.json",
                "reports/trec_2026_llm_profiles.json",
                "scripts/build_llm_profiles.py",
                "reports/trec_2026_dense_query_knn_results.json",
                "TREC 2025 official overview and per-run trec_eval files",
            ],
            "filters": [
                "2025 official Million LLM runs only for the prior-year leaderboard",
                "2026 local checkout state as reviewed on 2026-07-11",
                "primary research papers and official proceedings prioritized",
            ],
            "metric_definitions": [
                "nDCG@10: macro query mean using the repository's trec_eval-style linear 0/1/2 gain through rank 10.",
                "MRR: macro query mean reciprocal rank of the first candidate with grade greater than zero.",
                "Positive pair: a query-candidate qrel with grade 1 or grade 2.",
            ],
        },
    }
    discovery_audit_source = {
        "id": "discovery_profile_audit",
        "label": "2026 discovery response and metadata profile audit",
        "path": "reports/trec_2026_discovery_profile_audit.json",
        "query": {
            "language": "python",
            "description": (
                "Streams both official discovery JSON files to audit response coverage, "
                "two worked candidate profiles, and token/logprob array alignment."
            ),
            "executed_at": GENERATED_AT,
            "tables_used": [
                "data/dev/llm_discovery_data.json",
                "data/dev/llm_discovery_metadata.json",
                "scripts/audit_discovery_profiles.py",
            ],
            "filters": [
                "all 7,759 discovery questions",
                "all 150 candidate responses per question",
                "worked candidates: Anthropology and Applied_mathematics",
            ],
            "metric_definitions": [
                discovery_audit["definitions"]["exact_refusal"],
                discovery_audit["definitions"]["attempt"],
                discovery_audit["definitions"]["logprob_usable"],
            ],
        },
    }
    derived_sources = [
        {
            "id": "local_headline_sql",
            "label": "2026 development-matrix headline extraction",
            "path": "data/dev/llm_dev_qrels.txt",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql_queries["data_headline"],
                "description": "Selects audited headline fields computed from the development TSV and complete qrel matrix.",
                "executed_at": GENERATED_AT,
                "tables_used": ["data_headline", "data/dev/llm_dev_data.tsv", "data/dev/llm_dev_qrels.txt"],
                "filters": ["all 322 development queries", "all 150 candidates", "positive means grade > 0"],
                "metric_definitions": ["Positive qrel share = count(grade > 0) / count(all query-candidate qrels)."],
            },
        },
        {
            "id": "dense_headline_sql",
            "label": "Dense-baseline headline extraction",
            "path": "reports/trec_2026_dense_query_knn_results.json",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql_queries["dense_headline"],
                "description": "Selects the reviewed nested dense nDCG@10 and its paired mean difference from fixed TF-IDF.",
                "executed_at": GENERATED_AT,
                "tables_used": ["dense_headline", "reports/trec_2026_dense_query_knn_results.json"],
                "filters": ["322 strict outer leave-one-query-out predictions"],
                "metric_definitions": ["dense_delta = dense nested nDCG@10 - fixed TF-IDF k=10 nDCG@10."],
            },
        },
        {
            "id": "baseline_metrics_sql",
            "label": "Three-system 2026 baseline comparison",
            "path": "reports/trec_2026_dense_query_knn_results.json",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql_queries["baseline_metrics"],
                "description": "Selects long-form nDCG@10 and MRR rows for the dense, TF-IDF, and prior systems.",
                "executed_at": GENERATED_AT,
                "tables_used": ["baseline_metrics", "reports/trec_2026_dense_query_knn_results.json"],
                "filters": ["322 strict outer leave-one-query-out predictions"],
                "metric_definitions": ["nDCG@10 uses linear 0/1/2 gains; MRR treats any grade > 0 as relevant."],
            },
        },
        {
            "id": "baseline_display_sql",
            "label": "Exact 2026 baseline score table",
            "path": "reports/trec_2026_dense_query_knn_results.json",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql_queries["baseline_display"],
                "description": "Formats the reviewed nDCG@10 and MRR values to four decimals for an accessible exact-score comparison table.",
                "executed_at": GENERATED_AT,
                "tables_used": ["baseline_metrics", "reports/trec_2026_dense_query_knn_results.json"],
                "filters": ["322 strict outer leave-one-query-out predictions"],
                "metric_definitions": ["nDCG@10 uses linear 0/1/2 gains; MRR treats any grade > 0 as relevant."],
            },
        },
        {
            "id": "runs_2025_sql",
            "label": "Official 2025 run-score extraction",
            "href": "https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main.html",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql_queries["runs_display"],
                "description": "Formats reviewed official scores from the 19 per-run NIST trec_eval files to four decimals and joins conservative method annotations from official papers/run pages.",
                "executed_at": GENERATED_AT,
                "tables_used": ["runs_2025", "TREC 2025 per-run .trec_eval files"],
                "filters": ["Million LLM main task", "all 19 automatic official runs", "official aggregate values"],
                "metric_definitions": ["rank is descending official nDCG@10 order; MRR is the official recip_rank aggregate; display fields preserve four decimals."],
            },
        },
        {
            "id": "data_files_sql",
            "label": "2026 file-status inventory",
            "path": "data/dev/file_manifest.tsv",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql_queries["data_files"],
                "description": "Selects reviewed local status, expected size, content, and intended use for the four official development/discovery files.",
                "executed_at": GENERATED_AT,
                "tables_used": ["data_files", "data/dev/file_manifest.tsv", "data/dev/README.md"],
                "filters": ["official 2026 files listed on the captured track page"],
            },
        },
        {
            "id": "literature_map_sql",
            "label": "Reviewed literature-family map",
            "path": "reports/trec_2026_approaches_guide_source_notes.md",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql_queries["literature_map"],
                "description": "Selects the reviewed method-family synthesis whose primary papers are linked in the report and source inventory.",
                "executed_at": GENERATED_AT,
                "tables_used": ["literature_map", "primary literature inventory in artifact sources"],
                "filters": ["methods relevant to query-to-candidate expertise ranking"],
            },
        },
        {
            "id": "approach_ladder_sql",
            "label": "Competition-specific experiment ladder",
            "path": "reports/trec_2026_approaches_guide_source_notes.md",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql_queries["approach_ladder"],
                "description": "Selects the evidence-derived build order, data dependency, likely run class, effort band, and experimental role.",
                "executed_at": GENERATED_AT,
                "tables_used": ["approach_ladder", "2025 official methods", "2026 local data audit"],
                "filters": ["methods feasible for the released 2026 data"],
            },
        },
        {
            "id": "portfolio_sql",
            "label": "Five-run portfolio synthesis",
            "path": "reports/trec_2026_approaches_guide_source_notes.md",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql_queries["portfolio"],
                "description": "Selects five methodologically distinct run concepts under the official five-run allowance.",
                "executed_at": GENERATED_AT,
                "tables_used": ["portfolio", "official 2026 run allowance", "2025 method evidence"],
                "filters": ["maximum five official runs"],
            },
        },
        {
            "id": "candidate_profiles_sql",
            "label": "First-pass 2026 candidate behavior/topic profiles",
            "path": "reports/trec_2026_llm_profiles.json",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql_queries["candidate_profiles"],
                "description": (
                    "Projects the 150 reproducible profile rows built by streaming every "
                    "official discovery response and metadata record."
                ),
                "executed_at": GENERATED_AT,
                "tables_used": [
                    "candidate_profiles_raw",
                    "reports/trec_2026_llm_profiles.json",
                    "scripts/build_llm_profiles.py",
                    "data/dev/llm_discovery_data.json",
                    "data/dev/llm_discovery_metadata.json",
                    "data/dev/llm_dev_qrels.txt",
                ],
                "filters": [
                    "all 7,759 discovery questions",
                    "all 150 candidates",
                    "discovery candidate IDs exactly match development-qrel candidate IDs",
                    "query terms require at least five attempted-query occurrences, ten corpus occurrences, and 1.25x smoothed lift",
                ],
                "metric_definitions": [
                    llm_profiles["definitions"]["exact_refusal"],
                    llm_profiles["definitions"]["attempt"],
                    llm_profiles["definitions"]["enriched_attempted_query_term"],
                    llm_profiles["definitions"]["attempt_token_logprob_usable"],
                ],
            },
        },
    ]
    source_list = [*SOURCES, guide_source, discovery_audit_source, *derived_sources]

    data_files = [
        {
            "order": 1,
            "file": "llm_dev_data.tsv",
            "status": "Present locally",
            "size": "40.6 kB",
            "contains": "322 query IDs and query text",
            "use": "Inputs and query features",
        },
        {
            "order": 2,
            "file": "llm_dev_qrels.txt",
            "status": "Present locally",
            "size": "2.18 MB",
            "contains": "48,300 complete 0/1/2 grades",
            "use": "Validation, training, metric checks",
        },
        {
            "order": 3,
            "file": "llm_discovery_data.json",
            "status": "Present locally (shared)",
            "size": "137.3 MB",
            "contains": "Discovery queries and stored responses",
            "use": "Response retrieval and model profiles",
        },
        {
            "order": 4,
            "file": "llm_discovery_metadata.json",
            "status": "Present locally (shared)",
            "size": "1.46 GB",
            "contains": "Responses plus logprobs/metadata",
            "use": "Confidence and response-quality features",
        },
    ]
    baseline_rows = []
    for order, (key, label) in enumerate(
        [
            ("dense_nested", "Frozen dense query-kNN"),
            ("tfidf_fixed_k10", "TF-IDF query-kNN"),
            ("global_prior", "Fold-local global prior"),
        ],
        1,
    ):
        for metric_order, (field, metric_label) in enumerate(
            [("ndcg_at_10", "nDCG@10"), ("mrr", "MRR")], 1
        ):
            baseline_rows.append(
                {
                    "system_order": order,
                    "metric_order": metric_order,
                    "system": label,
                    "metric": metric_label,
                    "value": dense[key][field],
                    "query_count": local["query_count"],
                    "protocol": "strict query-level leave-one-out",
                }
            )

    portfolio_rows = [
        {"slot": 1, "run": "Response BM25/RM3", "type": "Internal", "why": "Strong, explainable evidence-first control"},
        {"slot": 2, "run": "Frozen dense retrieval", "type": "Likely external", "why": "Preserve the strongest current semantic baseline"},
        {"slot": 3, "run": "Metadata + sparse LambdaMART", "type": "Internal", "why": "Tests whether logprobs add complementary signal"},
        {"slot": 4, "run": "Calibrated weak-label router", "type": "Likely external", "why": "High-upside pseudo-supervision with an audit trail"},
        {"slot": 5, "run": "Cross-fitted hybrid / RRF", "type": "Depends", "why": "Combines diverse errors without reusing 2025 priors"},
    ]
    literature_rows = [
        {"order": 1, "family": "Expert finding", "plain": "Retrieve evidence, then let it vote for its expert", "evidence": "Balog 2006; Macdonald & Ounis 2006", "fit": "Very high", "risk": "Evidence-count bias"},
        {"order": 2, "family": "Sparse retrieval", "plain": "Match distinctive words, entities, and phrases", "evidence": "BM25; RM3; SPLADE", "fit": "High", "risk": "Vocabulary mismatch / drift"},
        {"order": 3, "family": "Dense retrieval", "plain": "Match meaning using vectors", "evidence": "SBERT; DPR; E5; BGE; ColBERT", "fit": "High", "risk": "External model and domain shift"},
        {"order": 4, "family": "Rank fusion", "plain": "Combine systems that make different mistakes", "evidence": "RRF; learned fusion", "fit": "High", "risk": "Weight overfitting"},
        {"order": 5, "family": "Learning to rank", "plain": "Learn which clues should move candidates upward", "evidence": "ListNet; LambdaMART; LightGBM", "fit": "High if cross-fitted", "risk": "Only 322 query groups"},
        {"order": 6, "family": "Weak supervision", "plain": "Use cheap, imperfect teachers to label responses", "evidence": "Bradley-Terry; Snorkel; LLM judges", "fit": "Potentially high", "risk": "Judge and style bias"},
        {"order": 7, "family": "Confidence", "plain": "Use logprobs as a clue, not as truth", "evidence": "Calibration; self-knowledge; semantic entropy", "fit": "Medium as support", "risk": "Confident hallucination"},
        {"order": 8, "family": "Candidate prototypes", "plain": "Build a résumé or several centroids per expert", "evidence": "Expert profiles; prototypes", "fit": "Medium", "risk": "One profile blurs specialties"},
        {"order": 9, "family": "Modern LLM routing", "plain": "Predict per-query model utility", "evidence": "RouterBench; RouteLLM; GraphRouter", "fit": "Framing + features", "risk": "Usually far fewer models"},
        {"order": 10, "family": "Shift and validation", "plain": "Test whether the exam differs from practice", "evidence": "Nested CV; shift detection; WILDS", "fit": "Essential", "risk": "Optimistic random folds"},
    ]
    raw_data_headline = [
        {
            "query_count": local["query_count"],
            "median_query_words": local["query_words_median"],
            "candidate_count": local["candidate_count"],
            "qrel_count": local["qrel_count"],
            "positive_share": local["positive_share"],
            "exactly_one": local["exactly_one_positive"],
        }
    ]
    raw_dense_headline = [
        {
            "dense_ndcg": dense_primary["ndcg_at_10"],
            "dense_delta": dense_primary["ndcg_at_10"] - tfidf["ndcg_at_10"],
        }
    ]
    query_results = _execute_sqlite_queries(
        {
            "data_headline": raw_data_headline,
            "dense_headline": raw_dense_headline,
            "baseline_metrics": baseline_rows,
            "runs_2025": RUNS_2025,
            "data_files": data_files,
            "literature_map": literature_rows,
            "approach_ladder": APPROACH_LADDER,
            "portfolio": portfolio_rows,
            "candidate_profiles_raw": _profile_table_rows(llm_profiles),
        },
        sql_queries,
    )
    runs_display = query_results["runs_display"]
    runs_top10 = runs_display[:10]
    runs_bottom9 = runs_display[10:]
    if len(runs_top10) != 10 or len(runs_bottom9) != 9:
        raise ValueError("unexpected 2025 run-table split")

    cards = [
        {
            "id": "query_count",
            "description": "Independent development groups; all split and bootstrap operations must stay at query level.",
            "dataset": "data_headline",
            "sourceId": "local_headline_sql",
            "metrics": [
                {"label": "Development queries", "field": "query_count", "format": "number"},
                {"label": "Median words", "field": "median_query_words", "format": "number"},
            ],
        },
        {
            "id": "candidate_count",
            "description": "Every query has a complete grade for the same candidate set.",
            "dataset": "data_headline",
            "sourceId": "local_headline_sql",
            "metrics": [
                {"label": "Candidates per query", "field": "candidate_count", "format": "number"},
                {"label": "Total qrels", "field": "qrel_count", "format": "compact"},
            ],
        },
        {
            "id": "positive_share",
            "description": "Share of query-candidate pairs graded 1 or 2; accuracy would mostly reward predicting zero.",
            "dataset": "data_headline",
            "sourceId": "local_headline_sql",
            "metrics": [
                {"label": "Positive qrel share", "field": "positive_share", "format": "percent"},
                {"label": "Queries with one positive", "field": "exactly_one", "format": "number"},
            ],
        },
        {
            "id": "dense_ndcg",
            "description": "Strict nested leave-one-query-out estimate; the comparison chip is versus fixed TF-IDF k=10.",
            "dataset": "dense_headline",
            "sourceId": "dense_headline_sql",
            "metrics": [
                {"label": "Dense nDCG@10", "field": "dense_ndcg", "format": "number"},
                {"label": "vs TF-IDF", "field": "dense_delta", "format": "number", "signed": True},
            ],
        },
    ]

    charts = [
        {
            "id": "baseline_comparison",
            "title": "2026 development metrics for current baselines",
            "subtitle": "322 strict query-level outer predictions; the exact four-decimal values appear in the immediately following table.",
            "showDescription": True,
            "type": "bar",
            "dataset": "baseline_metrics",
            "sourceId": "baseline_metrics_sql",
            "intent": "comparison",
            "question": "How much does frozen dense query retrieval improve on sparse query retrieval and a global prior?",
            "rationale": "A compact grouped bar reveals the same-direction gain across both same-scale metrics; the adjacent table is the accessible exact-value path.",
            "comparisonContext": {"grain": "one macro score per system and metric", "unit": "0–1 score", "baseline": "TF-IDF and global prior"},
            "encodings": {
                "x": {"field": "system", "type": "nominal", "label": "System"},
                "y": {"field": "value", "type": "quantitative", "label": "Score", "format": "number"},
                "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                "tooltip": [
                    {"field": "query_count", "type": "quantitative", "label": "Queries"},
                    {"field": "protocol", "type": "text", "label": "Protocol"},
                ],
            },
            "yAxisTitle": "Score",
            "valueFormat": "number",
            "palette": {"kind": "categorical"},
            "settings": {"groupMode": "grouped", "orientation": "vertical", "showValues": True, "sort": "custom"},
            "labels": {"values": "all"},
            "surface": {"viewMode": "both", "showControls": True},
            "layout": "full",
            "maxRows": 6,
        }
    ]

    tables = [
        {
            "id": "baseline_display_table",
            "title": "Exact 2026 development scores for current baselines",
            "subtitle": "All values are strict query-level outer predictions over the same 322 development queries.",
            "dataset": "baseline_display",
            "sourceId": "baseline_display_sql",
            "defaultSort": {"field": "system", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "system", "label": "System", "type": "text"},
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "value_display", "label": "Exact score (0–1)", "type": "text"},
                {"field": "protocol", "label": "Protocol", "type": "text"},
            ],
        },
        {
            "id": "data_files",
            "title": "2026 files and their local availability",
            "subtitle": "Both discovery JSON files now resolve from Git-shared local storage and pass a complete streaming JSON parse.",
            "dataset": "data_files",
            "sourceId": "data_files_sql",
            "defaultSort": {"field": "file", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "file", "label": "File", "type": "text"},
                {"field": "status", "label": "Local status", "type": "text"},
                {"field": "size", "label": "Expected size", "type": "text"},
                {"field": "contains", "label": "Contains", "type": "text"},
                {"field": "use", "label": "Best use", "type": "text"},
            ],
        },
        {
            "id": "runs_top10_table",
            "title": "Official TREC 2025 runs ranked 1–10",
            "subtitle": "Exact four-decimal official scores; method descriptions conservatively summarize papers and run pages.",
            "dataset": "runs_top10",
            "sourceId": "runs_2025_sql",
            "defaultSort": {"field": "rank", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "rank", "label": "Rank", "format": "number"},
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "team", "label": "Team", "type": "text"},
                {"field": "ndcg10_display", "label": "nDCG@10 (0–1)", "type": "text"},
                {"field": "mrr_display", "label": "MRR (0–1)", "type": "text"},
                {"field": "family", "label": "Family", "type": "text"},
                {"field": "idea", "label": "Evidence-backed summary", "type": "text"},
            ],
        },
        {
            "id": "runs_bottom9_table",
            "title": "Official TREC 2025 runs ranked 11–19",
            "subtitle": "The second half completes the official set without truncating the no-script or assistive reading path.",
            "dataset": "runs_bottom9",
            "sourceId": "runs_2025_sql",
            "defaultSort": {"field": "rank", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "rank", "label": "Rank", "format": "number"},
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "team", "label": "Team", "type": "text"},
                {"field": "ndcg10_display", "label": "nDCG@10 (0–1)", "type": "text"},
                {"field": "mrr_display", "label": "MRR (0–1)", "type": "text"},
                {"field": "family", "label": "Family", "type": "text"},
                {"field": "idea", "label": "Evidence-backed summary", "type": "text"},
            ],
        },
        {
            "id": "literature_map",
            "title": "Literature map: what each family contributes",
            "subtitle": "A decision table for this dataset; citations and caveats are expanded in the narrative above it.",
            "dataset": "literature_map",
            "sourceId": "literature_map_sql",
            "defaultSort": {"field": "family", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "family", "label": "Family", "type": "text"},
                {"field": "plain", "label": "Plain-English idea", "type": "text"},
                {"field": "evidence", "label": "Anchor literature", "type": "text"},
                {"field": "fit", "label": "2026 fit", "type": "text"},
                {"field": "risk", "label": "Main risk", "type": "text"},
            ],
        },
        {
            "id": "approach_ladder",
            "title": "Recommended experiment ladder",
            "subtitle": "Build in this order so each expensive method must beat a simpler, auditable control.",
            "dataset": "approach_ladder",
            "sourceId": "approach_ladder_sql",
            "defaultSort": {"field": "stage_display", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "stage_display", "label": "Stage", "type": "text"},
                {"field": "approach", "label": "Approach", "type": "text"},
                {"field": "needs", "label": "Needs", "type": "text"},
                {"field": "run_type", "label": "Likely type", "type": "text"},
                {"field": "effort", "label": "Rough effort", "type": "text"},
                {"field": "role", "label": "Role", "type": "text"},
                {"field": "priority", "label": "Action", "type": "text"},
            ],
        },
        {
            "id": "portfolio",
            "title": "A diverse five-run submission portfolio",
            "subtitle": "The track allows up to five runs; preserve methodological diversity instead of five tiny variants.",
            "dataset": "portfolio",
            "sourceId": "portfolio_sql",
            "defaultSort": {"field": "slot_display", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "slot_display", "label": "Slot", "type": "text"},
                {"field": "run", "label": "Run concept", "type": "text"},
                {"field": "type", "label": "Type", "type": "text"},
                {"field": "why", "label": "Why keep it", "type": "text"},
            ],
        },
        {
            "id": "candidate_profiles_table",
            "title": "First-pass behavior profiles for all 150 candidates",
            "subtitle": (
                "One saved row per candidate across all 7,759 discovery queries. "
                "These are behavior and coverage clues, not correctness grades."
            ),
            "dataset": "candidate_profiles",
            "sourceId": "candidate_profiles_sql",
            "defaultSort": {"field": "candidate_id", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "candidate_id", "label": "Candidate ID", "type": "text"},
                {
                    "field": "top_term_summary",
                    "label": "Unusually common words in queries attempted",
                    "type": "text",
                },
                {"field": "attempt_count", "label": "Attempts", "format": "number"},
                {"field": "attempt_share", "label": "Attempt share", "format": "percent"},
                {
                    "field": "exact_refusal_share",
                    "label": "Exact-refusal share",
                    "format": "percent",
                },
                {
                    "field": "attempt_word_median",
                    "label": "Median attempted words",
                    "format": "number",
                },
                {
                    "field": "attempt_context_template_share",
                    "label": "Context-template share",
                    "format": "percent",
                },
                {
                    "field": "attempt_token_logprob_usable_share",
                    "label": "Attempts with usable metadata",
                    "format": "percent",
                },
            ],
        },
    ]

    blocks = build_blocks(
        local, dense_primary, tfidf, prior, discovery_audit, llm_profiles
    )
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Finding the Right Expert LLM: A 2026 Beginner’s Field Guide",
            "description": "A phone-friendly visual learning path into TREC 2025 approaches, a reproducible 150-candidate behavior/topic profile, ranking features, the broader routing literature, and a leakage-safe 2026 experiment plan.",
            "generatedAt": GENERATED_AT,
            "filters": [
                {
                    "id": "team_filter",
                    "label": "Explore 2025 runs by team",
                    "dataset": "runs_2025",
                    "field": "team",
                    "includeAll": True,
                    "targets": [
                        {"dataset": "runs_top10", "field": "team"},
                        {"dataset": "runs_bottom9", "field": "team"},
                    ],
                }
            ],
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [
                {key: value for key, value in source.items() if key != "query"}
                for source in source_list
            ],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "data_headline": query_results["data_headline"],
                "dense_headline": query_results["dense_headline"],
                "data_files": query_results["data_files"],
                "baseline_metrics": query_results["baseline_metrics"],
                "baseline_display": query_results["baseline_display"],
                "runs_2025": RUNS_2025,
                "runs_top10": runs_top10,
                "runs_bottom9": runs_bottom9,
                "literature_map": query_results["literature_map"],
                "approach_ladder": query_results["approach_ladder"],
                "portfolio": query_results["portfolio"],
                "candidate_profiles": query_results["candidate_profiles"],
            },
        },
        "sources": source_list,
    }


def build_blocks(
    local: dict[str, Any],
    dense: dict[str, float],
    tfidf: dict[str, float],
    prior: dict[str, float],
    discovery_audit: dict[str, Any],
    llm_profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    discovery_plain = discovery_audit["plain"]
    discovery_metadata = discovery_audit["metadata"]
    candidate_audit_profiles = {
        row["candidate"]: row for row in discovery_plain["candidate_examples"]
    }
    candidate_metadata = {
        row["candidate"]: row for row in discovery_metadata["candidate_examples"]
    }
    anthropology = candidate_audit_profiles["Anthropology"]
    applied_math = candidate_audit_profiles["Applied_mathematics"]
    anthropology_metadata = candidate_metadata["Anthropology"]
    applied_math_metadata = candidate_metadata["Applied_mathematics"]
    built_profiles = {
        row["candidate_id"]: row for row in llm_profiles["profiles"]
    }
    anthropology_profile = built_profiles["Anthropology"]
    politics_location_profile = built_profiles["Politics_by_location"]
    anthropology_term_items = "".join(
        (
            f"<li><span><code>{term['term']}</code></span>"
            f"<strong>{term['support_queries']:,} queries · {term['lift']:.1f}×</strong></li>"
        )
        for term in anthropology_profile["attempted_query_terms"][:3]
    )
    anthropology_term_words = ", ".join(
        term["term"] for term in anthropology_profile["attempted_query_terms"][:3]
    )
    pair_decisions_per_query = (
        discovery_plain["candidate_count"] * (discovery_plain["candidate_count"] - 1) // 2
    )
    all_pair_decisions = pair_decisions_per_query * discovery_plain["query_count"]
    blocks = [
        md("title", "# Finding the Right Expert LLM: A 2026 Beginner’s Field Guide"),
        md(
            "technical_summary",
            f"""
            ## The short answer: retrieve evidence first, then learn how to combine it

            **Plain-language map:** every promising system does two jobs. First, it finds useful clues in past questions, past responses, names, or metadata. Second, it turns those clues into a ranked list of candidate models. The bullets below give the exact research answer; every technical term is unpacked in the next few sections.

            - **Keep the current dense query-neighbor baseline—meaning-based search over earlier questions.** Its strict nested estimate, with settings selected only inside the training data, is **{dense['ndcg_at_10']:.4f} nDCG@10 / {dense['mrr']:.4f} MRR**, clearly ahead of the TF-IDF keyword baseline and the default prior recomputed without each held-out query. nDCG@10 evaluates the ordered top ten; MRR focuses on the first useful candidate.
            - **Build response-level retrieval next—search past answers one by one, then credit their authors.** In 2025, retrieving individual historical responses and aggregating them to their producing model nearly matched the best weak-label neural systems while remaining explainable.
            - **A real first-pass profile now exists for every 2026 candidate.** The builder turns all discovery records into 150 behavior/topic rows after auditing **{discovery_plain['exact_refusal_share']:.1%}** exact “I don't know” or “I do not know” defaults and **{discovery_metadata['token_logprob_mismatch_share']:.2%}** token/logprob length mismatches. It measures coverage and attempted-query associations—not correctness—and keeps exact defaults out of future semantic centroids.
            - **Use readable candidate names only as a weak feature.** Their wording looks helpful, but actual qrels contain surprising matches; observed behavior is more trustworthy than the label on the door.
            - **Treat token-confidence values, called logprobs, and LLM judges as noisy teachers.** They can add signal, but confidence is not correctness and judge bias can be learned by the router.
            - **Split, tune, and bootstrap by whole query.** Bootstrapping means resampling whole queries to measure score uncertainty. The independent sample size is 322 queries—not 48,300 query-candidate rows—and every feature that touches qrels, the relevance answer key, must be recomputed using only that fold’s training queries.
            """,
            "guide_notes",
        ),
        md(
            "reader_map",
            """
            ## Contents — choose the shortest route you need

            **Profiles start immediately below this box.** Read the definition, follow one visual from 1,163,850 response records to 150 saved rows, then inspect the real candidate table. If that answers your question, stop there.

            - **Profiles:** *An expert profile is stored evidence about one candidate*.
            - **Two-minute task introduction:** *The story in two minutes*.
            - **Data and measured baselines:** *The current data favors retrieval and regularization over a giant classifier*.
            - **What teams tried in 2025:** *What 2025 can teach—and what it cannot transfer*.
            - **Literature and LLM judges:** *What “literature foundation” means in this guide* and *An LLM judge grades one stored response*.
            - **Safe experiment plan:** *Safe validation is a rehearsal* and *Build cheap evidence before expensive supervision*.
            - **Quick reference:** *Pocket glossary*, followed by limitations and open questions.

            These are exact visible heading names. Use your screen reader’s heading list, or the page’s Find command, to locate one. Tables are sortable and paginated; open a table full-screen and use browser Find for a candidate. On a narrow screen, swipe sideways instead of shrinking the text.
            """,
        ),
        md(
            "walking_summary",
            """
            ## The story in two minutes

            This is one uninterrupted, audio-friendly block. A **candidate model** is one of the 150 IDs we must rank. A **query** is the new question. A **router** is the system that scores and ranks the candidate models. Start your phone’s built-in **Read Aloud**, **Select to Speak**, or screen reader here; the page itself does not synthesize or download audio.

            Imagine a help desk with 150 specialist desks. A new question arrives. You may not ask every desk for a fresh answer, so you study how the desks behaved on earlier questions and rank which ones are most likely to help now.

            Last year, the strongest idea was not “trust the expert’s name.” It was “find relevant past behavior.” Some teams retrieved old answers. Others found similar old questions and transferred quality labels. Others built résumés for each expert, used token confidence, or asked stronger models to create imperfect training labels.

            For our 2026 data, start simple. Keep a default ranking based on who usually did well. Find similar labeled development questions. Match the query to candidate names as a small extra clue. The discovery files are now local, so retrieve individual past responses with both keyword and meaning-based search, then add their evidence to the candidate that produced them. Finally, let a small model learn how much each clue should count, using predictions made for questions that component did not train on.

            The safety rule is simple: hide an entire query and all 150 of its labels whenever you validate. If any part of that answer key influences its features, tuning, tie-breaking, or confidence adjustment, the score is optimistic.
            """,
        ),
        teaching_visual(
            "beginner-visual-competition-flow",
            "The whole competition in one picture",
            "First learn from past behavior and labels. At test time, use only the new question to rank the fixed candidates.",
            """
            <div class="source-grid">
              <section class="card gold">
                <h3>1. Discovery history</h3>
                <ul>
                  <li>Past question</li>
                  <li>Stored candidate response</li>
                  <li>Candidate ID</li>
                  <li>Log probabilities and metadata</li>
                </ul>
                <span class="badge">Present locally in shared storage</span>
              </section>
              <section class="card blue">
                <h3>2. Development labels</h3>
                <ul>
                  <li>322 questions</li>
                  <li>150 candidates per question</li>
                  <li>Grades 0, 1, or 2</li>
                </ul>
                <span class="badge">Present locally</span>
              </section>
            </div>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card">
              <h3>3. Build the router</h3>
              <ol>
                <li>Retrieve useful past evidence.</li>
                <li>Turn evidence into candidate scores.</li>
                <li>Learn or choose how to combine the clues.</li>
              </ol>
            </section>
            <div class="divider">At test time</div>
            <div class="flow-row">
              <section class="card"><h3>4. New question only</h3><p>No target answer and no test labels.</p></section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card"><h3>5. Score all 150 IDs</h3><p>Estimate which candidate is most likely to help.</p></section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card green"><h3>6. Return ranks 1–150</h3><p>The most promising candidate goes first.</p></section>
            </div>
            <p class="callout">The system is choosing <em>who should answer</em>. It is not writing the answer.</p>
            """,
            "Before the test, discovery history and the complete 322-by-150 development labels provide evidence for building a router. At test time, only a new question enters. The router scores and ranks all 150 candidate IDs. The task chooses a likely expert; it does not generate the answer.",
            "guide_notes",
        ),
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["query_count", "candidate_count", "positive_share", "dense_ndcg"]},
        md(
            "task_explained",
            """
            ## The task is expert search, not ordinary answer generation

            A normal search engine ranks documents. This track ranks **candidate model IDs** by how likely each is to answer a query well—*before* seeing a target answer. The discovery set supplies earlier queries, stored responses, and metadata. The development set supplies query-level expertise labels. The test set supplies only new queries.

            The cleanest research analogy is [expert finding](https://doi.org/10.1145/1148170.1148181): retrieve evidence connected to a person—or here, a candidate model—then combine that evidence into an expert score. Modern LLM-routing work asks a related question, but often chooses between only two or a few real models and optimizes cost as well as quality. TREC 2026 instead requires a complete ranking of 150 fixed candidates.

            **Submission:** one six-column TREC run line per query-candidate rank. **Primary metrics:** nDCG@10 rewards highly graded candidates near the top ten; MRR rewards finding the first grade-1-or-2 candidate as early as possible. Up to five runs are allowed.
            """,
            "track_2026",
        ),
        md(
            "one_training_example",
            """
            ## One example: what a relevance grade belongs to

            In the **development set**—the practice questions with an answer key—a grade belongs to a *question–candidate pair*. Grade 0 means no relevance credit, grade 1 means relevant, and grade 2 means highly relevant. The complete answer-key table is called **qrels**, short for *query relevance judgments*.

            For the real question “What does CFL stand for?”, `Energy` has grade 2, `The_arts` has grade 1, and `Mathematics` has grade 0. These are expertise grades for candidate models, not proposed answers to the question.

            One query therefore creates 150 rows that share the same context. Treat those 150 rows as one group when training, splitting, and scoring.
            """,
            "local_data",
        ),
        teaching_visual(
            "beginner-visual-query-candidate-pair",
            "A grade belongs to a question–candidate pair",
            "This exact local example shows why one question becomes a 150-row training group.",
            """
            <div class="triple">
              <section class="card blue">
                <h3>Query</h3>
                <p><code>56dfb498231d4119001abc8d</code></p>
                <p>“What does CFL stand for?”</p>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">+</span><span class="narrow">↓</span></div>
              <section class="card gold"><h3>Candidate</h3><p><code>Energy</code></p></section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card green"><h3>Grade</h3><p><span class="grade grade-2">2</span> highly relevant</p></section>
            </div>
            <section class="card" style="margin-top:12px">
              <h3>Same question, different candidates</h3>
              <ul class="grade-list">
                <li><code>The_arts</code><span><span class="grade grade-1">1</span> relevant</span></li>
                <li><code>Mathematics</code><span><span class="grade grade-0">0</span> not relevant</span></li>
                <li><strong>147 other candidate rows</strong><span>complete the group</span></li>
              </ul>
            </section>
            <p class="callout">One query becomes 150 pair rows. Keep the whole group together.</p>
            """,
            "In the development data, a grade does not belong to a question alone or a candidate alone. It belongs to their pair. For the question ‘What does CFL stand for?’, Energy has grade 2, The arts has grade 1, and Mathematics has grade 0. The same question has a row for every one of the 150 candidates.",
            "local_data",
        ),
        md(
            "three_evidence_sources",
            """
            ## Almost every approach uses three kinds of evidence

            **What the question resembles.** Find earlier questions with similar words or meaning, then inspect which candidates did well. This is *query-neighbor transfer*.

            **What candidates did before.** Search individual stored responses, or summarize a candidate’s response history into a profile. Matching responses can vote for the candidate that produced them.

            **How trustworthy that behavior looks.** Use confidence, refusals, length, repetition, judge scores, or other metadata as supporting clues—not as proof of correctness.

            A final ranking system combines these clues. The named approaches mainly differ in what they retrieve, how evidence becomes a candidate score, and how safely they learn the combination.
            """,
        ),
        md(
            "approaches_in_one_breath",
            """
            ## The approaches in one breath

            Last year’s systems mostly mixed six moves: **use a default ranking**, **copy evidence from similar old questions**, **search individual past answers**, **compress a candidate’s history into a résumé**, **create imperfect practice grades with judges or proxies**, and **learn or vote on how much each clue counts**.

            Keyword systems match distinctive words. Dense systems match meaning. Confidence and behavior metadata can change how much to trust either one. An ensemble combines systems that make different mistakes.

            That is the field’s basic shape. The later nine-part primer separates these moves carefully; each team section then shows a particular combination.
            """,
        ),
        md(
            "profiles_features_orientation",
            """
            ## The missing mental model: profile → feature row → score

            **An expert profile is a stored résumé for one candidate.** It is built before the new query arrives from that candidate’s historical responses, embeddings, confidence, refusals, topics, or other behavior.

            **A ranking feature is one clue in one query–candidate row.** When a new query arrives, the system creates 150 rows—one per candidate. Query features repeat across the rows, candidate profiles change by row, and interaction features measure how well this query matches this candidate’s history.

            **The ranker turns each row into one score.** Sort the 150 scores and you have the run ranking. The qrel grade is the training answer key; it is not an input feature and stays hidden at prediction time.

            The profile quickstart near the top already defines and builds one real saved row. The deeper continuation just before the team methods compares 2025 profile shapes, shows how a query turns profile memory into features, and gives the grouped-table shape a ranker would receive.
            """,
            "guide_notes",
        ),
        md(
            "metrics_gentle_intro",
            """
            ## The two metrics ask different questions about the top of the list

            **MRR**, pronounced “em-arr-arr,” means *mean reciprocal rank*. For each query it asks: how far must we travel before finding the first candidate with grade 1 or 2? First place earns 1; fifth place earns one-fifth. The final MRR averages that value across queries.

            **nDCG@10** means *normalized discounted cumulative gain at ten*, pronounced “en-dee-see-gee at ten.” The name is less important than its plain meaning: quality of the ordered top ten. It gives more credit for placing grade-2 candidates above grade-1 candidates and for putting both near the top, then normalizes against the best possible ordering for that query.

            A system can find one useful candidate immediately yet organize the rest of the top ten badly. That is why the guide reports both metrics.
            """,
            "guide_notes",
        ),
        teaching_visual(
            "beginner-visual-metrics",
            "MRR notices the first useful hit; nDCG@10 reads the whole top ten",
            "The toy lists below are explanations, not measured competition runs.",
            """
            <div class="metric-grid">
              <section class="card blue">
                <h3>Ranking A: first hit is early, but stronger items are buried</h3>
                <ol class="ranking">
                  <li><strong>Rank 1</strong><span><span class="grade grade-1">1</span> relevant</span></li>
                  <li><strong>Ranks 2–3</strong><span>0, 0</span></li>
                  <li><strong>Rank 4</strong><span><span class="grade grade-2">2</span> highly relevant</span></li>
                  <li><strong>Rank 5</strong><span><span class="grade grade-1">1</span> relevant</span></li>
                  <li><strong>Ranks 6–10</strong><span>0, 0, 0, 0, 0</span></li>
                </ol>
                <p><strong>MRR:</strong> excellent, because the first useful candidate is first.</p>
                <p><strong>nDCG@10:</strong> notices that the grade-2 candidate and another useful candidate are buried.</p>
              </section>
              <section class="card gold">
                <h3>Ranking B: the same useful set is ordered well</h3>
                <ol class="ranking">
                  <li><strong>Rank 1</strong><span><span class="grade grade-2">2</span> highly relevant</span></li>
                  <li><strong>Rank 2</strong><span><span class="grade grade-1">1</span> relevant</span></li>
                  <li><strong>Rank 3</strong><span><span class="grade grade-1">1</span> relevant</span></li>
                  <li><strong>Ranks 4–5</strong><span>0, 0</span></li>
                  <li><strong>Ranks 6–10</strong><span>0, 0, 0, 0, 0</span></li>
                </ol>
                <p><strong>MRR:</strong> also sees a useful candidate at rank 1.</p>
                <p><strong>nDCG@10:</strong> is higher because the same useful candidates are concentrated and correctly ordered near the top.</p>
              </section>
            </div>
            <p class="callout">Both lists contain the same useful grades: 2, 1, and 1. Both have query-level MRR = 1 because a useful candidate is first. nDCG@10 prefers Ranking B’s stronger ordering.</p>
            """,
            "Both conceptual rankings contain the same relevant grades: one grade 2 and two grade 1s; all other top-ten positions are grade 0. Both place a useful candidate first, so their query-level MRR is 1. Ranking B puts the grade-2 candidate first and keeps all useful candidates in the top three, so nDCG is higher than for Ranking A.",
        ),
        md(
            "data_reality",
            f"""
            ## The current data favors retrieval and regularization over a giant classifier

            The local matrix is complete: **{local['query_count']} queries × {local['candidate_count']} candidates = {local['qrel_count']:,} grades**. Yet **{local['grade_counts'][0] / local['qrel_count']:.1%}** are grade 0. The median query has **{local['positive_median']:.0f}** positive candidate, while the mean is **{local['positive_mean']:.1f}** because a small tail has many positives. Exactly **{local['exactly_one_positive']} queries ({local['exactly_one_positive'] / local['query_count']:.1%})** have one positive; **{local['at_most_five_positive']} ({local['at_most_five_positive'] / local['query_count']:.1%})** have five or fewer.

            A system that predicts grade 0 for every pair would look about 90.9% accurate, yet it would produce no useful ranking. This is why plain accuracy is almost meaningless here.

            The imbalance also makes a 150-class neural router easy to **overfit**, meaning it memorizes practice patterns that do not survive new questions. Use query-grouped ranking objectives, shallow **regularized** models whose complexity is deliberately limited, and **hard negatives**—plausible but wrong candidates that teach fine distinctions. Give each query equal weight. Always report both ranking metrics: MRR asks “how fast did we find one usable expert?” while nDCG@10 asks “how good is the whole top ten?”

            Query text ranges from **{local['query_words_min']} to {local['query_words_max']} words** (median {local['query_words_median']:.0f}). That range supports both exact lexical matching and semantic encoders; neither should be assumed to dominate every query type.
            """,
            "local_data",
        ),
        {"id": "data_files_block", "type": "table", "tableId": "data_files", "layout": "full"},
        md(
            "missing_discovery",
            """
            ## The discovery-data blocker has been removed

            Both official discovery files are now available at their expected `data/dev/` paths in every local worktree. One shared, Git-ignored copy supplies each checkout: the 137 MB response file supports response indexing and behavioral profiles, and the 1.46 GB metadata file supports token-logprob and confidence features. Both files match the official byte sizes, pass a complete streaming JSON parse, and contain 7,759 top-level discovery queries. The first 150-row behavior/topic profile and metadata-alignment audit now exist. Factual correctness, broader refusal families, near-duplicates, language, and template families remain to audit. The test queries are not present; the official schedule lists their release for **September 5, 2026**.

            The behavior/topic profile fields have now been materialized, but no result should be inferred before they are evaluated inside the router. Response retrieval, semantic centroids, full confidence summaries, and judge-label experiments remain to build.
            """,
            "guide_notes",
        ),
        md(
            "candidate_names",
            """
            ## Candidate names are useful clues—and unreliable ground truth

            The 2026 IDs are readable labels such as `Anthropology` and `Applied_mathematics`; 99 of 150 contain underscores. That makes query-to-name keyword matching—such as BM25—or embedding similarity an unusually cheap feature.

            But the behavior matrix warns against trusting the name alone. The BeiDou satellite-launch query’s sole grade-2 candidate is `Mathematics`. The Chopin music-festival query’s sole grade-2 candidate is `Politics_by_location`. These examples are not errors we can correct by hand; they are evidence that the model’s observed expertise may differ from the human-readable label.

            **Recommendation:** implement candidate-name similarity as a diagnostic prior, then ask whether it adds out-of-fold value beside dense query neighbors and response evidence. Never make it the main router.
            """,
            "local_data",
        ),
        md(
            "sparse_dense_intro",
            """
            ## Word matching and meaning matching are complementary search habits

            **Sparse retrieval** keeps a large vocabulary in which most word weights are zero. In practice, systems such as BM25 reward shared distinctive words and phrases. **Dense retrieval** compresses a whole text into an **embedding**—a vector of numbers that acts like a location on a meaning map—then finds nearby vectors.

            Sparse search is excellent for exact names and rare technical language. Dense search can connect paraphrases that share few words. Because their raw scores use different scales, keep the rankings separate and combine rank positions, for example with Reciprocal Rank Fusion, before trying learned weights.
            """,
        ),
        teaching_visual(
            "beginner-visual-sparse-dense",
            "Two search habits catch different clues",
            "The exact query below comes from the local development data. The paraphrase is only an illustration.",
            """
            <section class="card">
              <h3>New query</h3>
              <p>“Who did Chopin attend the Lower Rhenish Music Festival with?”</p>
            </section>
            <div class="down" aria-hidden="true">↓</div>
            <div class="branch-grid">
              <section class="card blue">
                <h3>Sparse retrieval: the word detective</h3>
                <p>Looks for shared rare words and phrases such as <strong>Chopin</strong>, <strong>Rhenish</strong>, and <strong>Music Festival</strong>.</p>
                <p><strong>Good at:</strong> names, dates, entities, exact technical language.</p>
                <p><strong>Can miss:</strong> a useful paraphrase with few shared words.</p>
              </section>
              <section class="card gold">
                <h3>Dense retrieval: the meaning map</h3>
                <p>Connects the query to meanings like “Which companion went with Frédéric Chopin to a festival on the Rhine?”</p>
                <p><span class="badge">Illustrative paraphrase—not a stored record</span></p>
                <p><strong>Good at:</strong> the same idea expressed with different words.</p>
                <p><strong>Can miss:</strong> niche meanings represented poorly by the frozen encoder.</p>
              </section>
            </div>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card green">
              <h3>Use both</h3>
              <p>Keep their lists separate, then combine rank positions—for example with <strong>Reciprocal Rank Fusion</strong>. Do not directly average BM25 and cosine scores as though the numbers had the same units.</p>
            </section>
            """,
            "Sparse retrieval matches important words. Dense retrieval matches overall meaning. For the Chopin question, sparse retrieval notices words such as Chopin and Rhenish, while dense retrieval can connect the question to a differently worded paraphrase. The two systems make different errors, so their ranks can be fused.",
            "local_data",
        ),
        md(
            "baseline_interpretation",
            f"""
            ## Frozen dense query retrieval is the baseline to beat

            **Nested leave-one-query-out** means hiding one complete query for evaluation while choosing the model settings only inside the remaining queries. Under that strict protocol, dense query-neighbor transfer reaches **{dense['ndcg_at_10']:.4f} nDCG@10** and **{dense['mrr']:.4f} MRR**. Fixed TF-IDF query neighbors reach {tfidf['ndcg_at_10']:.4f} / {tfidf['mrr']:.4f}; the fold-local global prior reaches {prior['ndcg_at_10']:.4f} / {prior['mrr']:.4f}.

            Dense semantics improve both objectives, but this is development cross-validation, not hidden-test performance. The pinned MPNet encoder was selected within an exact nested loop; changing the encoder list after reading these results creates a new experiment. An **automatic run** is produced without manual intervention on individual test queries. **Internal** runs use only permitted track resources; **external** runs add outside models or data. This repository conservatively treats a pretrained outside encoder as a **likely external automatic** run; the 2025 metadata shows classification inconsistencies, so confirm borderline cases with the organizers.
            """,
            "dense_results",
        ),
        {"id": "baseline_chart_block", "type": "chart", "chartId": "baseline_comparison", "layout": "full"},
        {"id": "baseline_table_block", "type": "table", "tableId": "baseline_display_table", "layout": "full"},
        md(
            "prior_year_context",
            """
            ## What 2025 can teach—and what it cannot transfer

            The first Million LLMs edition used **1,131 simulated experts**: one shared generation setup backed by different document clusters, not 1,131 independent foundation models. Six teams submitted **19 automatic runs**. The 2026 candidate set is different, so a candidate’s 2025 prior, profile, or ID embedding is unusable this year. Only algorithms and feature ideas transfer.

            The final 2025 evaluation was also fragile. NIST authored 50 difficult questions, found ground-truth answers for 37, and only **16** had at least one relevant candidate response. Planned pooled human judging was not completed; GPT‑5 supplied the final relevance judgments. Treat the leaderboard as method evidence, not a precise ordering that will reproduce in 2026.
            """,
            "overview_2025",
        ),
        md(
            "evidence_warnings",
            """
            ## Evidence warning: the 2025 sources do not agree on every detail

            - The official overview totals 14,950 discovery queries; several participant papers say 14,940.
            - The overview describes 50 intended test topics, while scoring ultimately used the 16 topics with any positive answer.
            - 1,131 candidate clusters is the consistent construction count; one overview sentence says 1,113.
            - “Internal/external” flags conflict between an overview table, individual run forms, and methods that use pretrained encoders or LLM judges.
            - The overview calls leading systems largely unsupervised even though SRCB and UDInfo Run 2 use extensive pseudo- or weak supervision.

            These discrepancies do not erase the method lessons. They do mean the guide keeps exact run claims narrow, prefers per-run NIST pages over summary flags, and refuses to silently reconcile incompatible numbers.
            """,
            "guide_notes",
        ),
        md(
            "leaderboard_interpretation",
            """
            ## Three very different ideas occupied the 2025 top three

            **In normal words:** first place trained a neural scorer from imperfect practice grades. Second place asked what worked on similar earlier questions and balanced that local advice with stable averages. Third place searched individual old answers and let each match vote for its candidate.

            SRCB’s weak-label neural family led official nDCG@10. UDInfo’s best hybrid transferred judged expertise from similar historical queries and mixed local, cluster, and global evidence. GenAIus came within 0.0065 nDCG using classical response-level BM25/RM3 and score aggregation.

            That diversity is the real result. Expensive neural routing was not the only route to the top, and a transparent evidence-retrieval system was competitive. The two searchable, sortable tables preserve all 19 official runs with four-decimal scores and support the team filter at the top of the report.
            """,
            "runs_2025",
        ),
        md(
            "method_family_primer",
            """
            ## Nine reusable building blocks make the 2025 systems easier to understand

            The team papers use different names, but most systems combine the same small set of ideas. Read these as ingredients, not nine mutually exclusive recipes.

            ### 1. Global prior — “Who is usually good?”

            Average each candidate’s past grades and use that as the default ranking. A **prior** is simply a belief used before seeing local evidence for the new query. It is stable and cheap, but it does not react to the question. During validation, recompute it without the held-out query.

            ### 2. Candidate-name matching — “Does the label on the door fit?”

            Compare the query with labels such as `Applied_mathematics`, using distinctive words or embeddings. This is nearly free. The local qrels show why it is weak evidence: readable names can disagree with observed expertise.

            ### 3. Similar-question transfer — “What worked on a question like this?”

            Find the nearest earlier queries. Their complete 150-candidate grade vectors vote on the new ranking. **k-nearest neighbors**, shortened to **kNN**, means consulting the *k* closest examples. Sparse kNN matches words; dense kNN matches meaning. Poor neighbors and answer-key leakage are the main risks.

            ### 4. Response-level retrieval — “Search the work samples”

            An **index** is a structure that makes a collection searchable. Index each historical response separately, retrieve responses related to the new query, then credit the candidate that produced each response. **SumScore** adds retrieval scores. **Rank voting** gives more credit to higher retrieved positions. A per-candidate **cap** prevents candidates with more surviving records from winning on evidence volume alone.

            ### 5. Profiles and centroids — “Build a résumé”

            Combine a candidate’s history into one profile, or into several representative centers called **centroids**. This is compact and fast. One broad résumé can blur a candidate that is excellent in several unrelated specialties; several centroids preserve more of that shape.

            ### 6. Confidence and behavior features — “How reliable does the work look?”

            A token **log probability**, often shortened to **logprob**, records how probable the model considered a generated token. Summaries of logprobs, refusals, response length, repetition, and structure can adjust trust. They cannot prove correctness: a model can be confidently wrong.

            ### 7. Weak labels — “Ask imperfect teachers for practice grades”

            A **weak label**, or pseudo-label, is a cheap and imperfect substitute for a gold judgment. An LLM judge, a **reranker** that rescores retrieved items, reference similarity, or a pairwise “which answer is better?” comparison can create these labels. Several independent teachers are safer than one, because a router can otherwise learn a judge’s style preference instead of factual expertise.

            ### 8. Learning to rank — “Learn how much each clue should count”

            A learning-to-rank model combines retrieval scores, priors, confidence, and other features specifically to improve an ordered list. Each training group must contain all 150 candidates for one query. Despite 48,300 rows, this dataset contains only 322 independent query groups, so the model should stay shallow and regularized.

            ### 9. Fusion or ensemble — “Let different judges vote”

            An **ensemble** combines systems that make different mistakes. Reciprocal Rank Fusion combines rank positions without pretending different score scales match. Learned fusion can be stronger, but its weights must be trained on **out-of-fold predictions**: predictions made for examples that the component did not train on.

            **Team map:** SRCB combined weak labels, neural scoring, and ensembles. UDInfo combined similar queries, answer profiles, and weak labels. GenAIus centered on response retrieval and aggregation. UvA combined confidence features with learning to rank. indelab combined weak supervision and an engineered ensemble. paulphys combined behavior profiles with learning to rank.
            """,
            "guide_notes",
        ),
        md(
            "expert_profiles_deep_dive",
            """
            ## An expert profile is stored evidence about one candidate

            Think of a profile as a résumé kept in a filing cabinet. It is built from one candidate’s historical discovery behavior **before** a new query arrives. The profile itself usually stays fixed across test queries; the query-dependent clues appear when the router compares the new query with that stored profile.

            The word **profile** covered several different objects in 2025:

            1. **Text profile:** one giant searchable document containing a candidate’s old queries and responses.
            2. **Semantic profile:** several answer-embedding centroids, sometimes backed by an index of every answer vector.
            3. **Behavioral profile:** fixed numbers for response length, confidence, refusals, repetition, structure, domains, tasks, and semantic spread.
            4. **Topic-local profile:** separate quality or behavior summaries for each query cluster.

            These are not interchangeable. A text profile is searched with words. A centroid profile is compared with vector similarity. A behavioral vector is joined to a query–candidate row and given to a ranker.
            """,
            "deep_dive_2025",
        ),
        teaching_visual(
            "beginner-visual-profile-anatomy",
            "One candidate can have several kinds of résumé",
            "All four profiles are built before the new test query. They summarize different aspects of the same candidate’s history.",
            """
            <section class="card gold">
              <h3>Start with one candidate’s discovery history</h3>
              <p>Historical questions, stored responses, candidate ID, log probabilities, and cleaning flags.</p>
            </section>
            <div class="down" aria-hidden="true">↓</div>
            <div class="source-grid">
              <section class="card">
                <h3>Clean and measure</h3>
                <p>Flag failures, refusals, missing values, length, repetition, structure, and confidence. Keep counts so filtering remains auditable.</p>
              </section>
              <section class="card">
                <h3>Embed and organize</h3>
                <p>Turn responses into vectors, cluster related answers, store centroids, dispersion, support counts, and optionally every answer vector.</p>
              </section>
            </div>
            <div class="down" aria-hidden="true">↓</div>
            <div class="source-grid">
              <section class="card blue">
                <h3>Text profile</h3>
                <p>One concatenated document. Search it with BM25 or another text retriever.</p>
              </section>
              <section class="card blue">
                <h3>Semantic profile</h3>
                <p>Several centroids or prototypes. Compare the new query with each specialty center.</p>
              </section>
              <section class="card green">
                <h3>Behavioral profile</h3>
                <p>A numeric résumé of confidence, refusals, length, structure, overlap, and semantic statistics.</p>
              </section>
              <section class="card green">
                <h3>Topic-local profile</h3>
                <p>Separate statistics for music, science, law, or learned query clusters, each with its own evidence count.</p>
              </section>
            </div>
            <p class="callout">A profile is precomputed memory indexed by candidate—and, for a local profile, by topic. Query-to-profile comparisons turn that memory into query–candidate ranking features.</p>
            """,
            "An expert profile is a precomputed summary of one candidate’s past behavior. It may be a searchable text document, several semantic centroids, a numeric behavioral résumé, or separate statistics by query topic. The profile is stored before the new query; comparing the query with the profile creates ranking features.",
            "deep_dive_2025",
        ),
        teaching_visual(
            "beginner-visual-real-discovery-profile",
            "The first real profile signal is coverage—not expertise",
            "The downloaded 2026 data shows why cleaning, refusal handling, and support counts must come before centroids, judges, or a learned ranker.",
            f"""
            <div class="metric-grid">
              <section class="card red"><h3>{discovery_plain['exact_refusal_share']:.1%} exact refusals</h3><p>{discovery_plain['exact_refusals']:,} of {discovery_plain['response_count']:,} responses normalize to only “I don’t know” or “I do not know.”</p></section>
              <section class="card green"><h3>{discovery_plain['non_refusal_attempt_share']:.1%} non-exact attempts</h3><p>{discovery_plain['non_refusal_attempts']:,} responses contain something else. “Attempt” does not mean relevant or correct.</p></section>
              <section class="card blue"><h3>Median {discovery_plain['median_non_refusal_attempts_per_query']:.0f} attempts</h3><p>For a typical discovery question, only about ten of the 150 candidates produce a non-exact-refusal response.</p></section>
              <section class="card gold"><h3>{discovery_plain['queries_with_zero_non_refusal_attempts_share']:.1%} have no attempt</h3><p>{discovery_plain['queries_with_zero_non_refusal_attempts']:,} discovery questions have zero non-exact attempts from all 150 candidates.</p></section>
            </div>
            <div class="divider">Two observed candidate behavior cards</div>
            <div class="compare-grid">
              <section class="card blue">
                <h3><code>Anthropology</code></h3>
                <ul class="signal-list">
                  <li><span>Historical records</span><strong>{anthropology['records']:,}</strong></li>
                  <li><span>Exact-refusal share</span><strong>{anthropology['exact_refusal_share']:.1%}</strong></li>
                  <li><span>Non-exact attempts</span><strong>{anthropology['attempts']:,}</strong></li>
                  <li><span>Median attempted words</span><strong>{anthropology['attempt_word_median']:.0f}</strong></li>
                  <li><span>Attempt uses context template</span><strong>{anthropology['attempt_context_template_share']:.1%}</strong></li>
                  <li><span>Token/logprob mismatches</span><strong>{anthropology_metadata['mismatched']:,}</strong></li>
                </ul>
              </section>
              <section class="card gold">
                <h3><code>Applied_mathematics</code></h3>
                <ul class="signal-list">
                  <li><span>Historical records</span><strong>{applied_math['records']:,}</strong></li>
                  <li><span>Exact-refusal share</span><strong>{applied_math['exact_refusal_share']:.1%}</strong></li>
                  <li><span>Non-exact attempts</span><strong>{applied_math['attempts']:,}</strong></li>
                  <li><span>Median attempted words</span><strong>{applied_math['attempt_word_median']:.0f}</strong></li>
                  <li><span>Attempt uses context template</span><strong>{applied_math['attempt_context_template_share']:.1%}</strong></li>
                  <li><span>Token/logprob mismatches</span><strong>{applied_math_metadata['mismatched']:,}</strong></li>
                </ul>
              </section>
            </div>
            <div class="gate"><strong>Do not read these as expertise scores.</strong> They are coverage and behavior summaries conditional on the discovery-query mix. A non-exact attempt may still be irrelevant, incomplete, or wrong; a readable candidate name remains only a prior.</div>
            <div class="source-grid">
              <section class="card green"><h3>Profile implication</h3><p>Remove exact default answers from semantic centroids and response retrieval, but preserve refusal/default rate as a separate candidate and topic feature.</p></section>
              <section class="card green"><h3>Judge implication</h3><p>Deterministic screening reduces the response set that needs expensive quality judging from {discovery_plain['response_count']:,} to at most {discovery_plain['non_refusal_attempts']:,} non-exact attempts before further sampling.</p></section>
              <section class="card red"><h3>Metadata implication</h3><p>{discovery_metadata['token_logprob_mismatched']:,} records ({discovery_metadata['token_logprob_mismatch_share']:.2%}) have different token and logprob lengths. Mask them before token-level summaries.</p></section>
              <section class="card blue"><h3>Evidence implication</h3><p>Store support beside every rate, centroid, and topic statistic. The same average based on two responses is not as reliable as one based on two hundred.</p></section>
            </div>
            """,
            f"Across {discovery_plain['response_count']:,} official discovery responses, {discovery_plain['exact_refusal_share']:.1%} are exact refusals and {discovery_plain['non_refusal_attempts']:,} are non-exact attempts. The typical query has ten attempts, while {discovery_plain['queries_with_zero_non_refusal_attempts']:,} queries have none. Anthropology attempts more often than Applied mathematics in this discovery mix, but neither coverage nor a readable name proves correctness. Filter exact defaults out of response retrieval and centroids, keep refusal rate and support as profile fields, and mask the {discovery_metadata['token_logprob_mismatched']:,} token/logprob length mismatches before confidence summaries.",
            "discovery_profile_audit",
        ),
        md(
            "profile_build_explained",
            f"""
            ## Profiles — the actual build: one saved row per candidate

            **A profile is not another LLM, a prompt, or a mysterious object.** This first version is an ordinary saved JSON record. The builder reads the discovery history, groups records by candidate ID, calculates summaries, and writes exactly **{llm_profiles['scope']['candidate_count']} profile rows**.

            1. **Read:** stream all **{llm_profiles['scope']['response_count']:,}** stored response records without loading the 1.46 GB metadata file into memory.
            2. **Flag:** mark exact defaults, response length, context-template language, insufficiency language, and whether token/logprob arrays line up.
            3. **Group:** collect those measurements by candidate ID and count words that occur unusually often in the queries that candidate attempted.
            4. **Save:** write `reports/trec_2026_llm_profiles.json`, with the recipe in `scripts/build_llm_profiles.py` so the rows can be rebuilt.

            **What version 1 knows:** observed coverage, response style, metadata usability, and lexical associations with attempted queries. **What it does not know:** whether an answer was correct, whether the candidate is truly an expert, or where its semantic answer centroids should be. Those require a response encoder or calibrated judge labels and should be added as separately evaluated profile versions.

            **Short-path stop:** after the next visual and candidate table, you will understand what was built. The later profile chapter adds 2025 variants, semantic centroids, and ranking-feature details only if you want them.
            """,
            "candidate_profiles_sql",
        ),
        teaching_visual(
            "beginner-visual-profile-build-pipeline",
            "How 1,163,850 records become 150 actual profiles",
            "The profile is the saved output of a group-and-summarize job. The worked row below is measured from the downloaded 2026 discovery files.",
            f"""
            <div class="flow-row">
              <section class="card blue">
                <h3>1. Raw history</h3>
                <p>{llm_profiles['scope']['query_count']:,} questions × {llm_profiles['scope']['candidate_count']} candidates = {llm_profiles['scope']['response_count']:,} response records.</p>
                <span class="badge">one row per question–candidate pair</span>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card gold">
                <h3>2. Measure each record</h3>
                <p>Exact default? Word count? Template or insufficiency phrase? Usable token/logprob arrays? Which cleaned words occur in the query?</p>
                <span class="badge">still no correctness label</span>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card green">
                <h3>3. Group by candidate and save</h3>
                <p>Aggregate counts, rates, quartiles, metadata quality, and supported query-word associations.</p>
                <span class="badge">exactly 150 JSON profile rows</span>
              </section>
            </div>

            <div class="divider">One real saved row: <code>Anthropology</code></div>
            <div class="source-grid">
              <section class="card blue">
                <h3>Coverage and style</h3>
                <ul class="signal-list">
                  <li><span>Discovery records</span><strong>{anthropology_profile['history']['record_count']:,}</strong></li>
                  <li><span>Non-exact attempts</span><strong>{anthropology_profile['coverage']['attempt_count']:,} · {anthropology_profile['coverage']['attempt_share']:.1%}</strong></li>
                  <li><span>Exact-refusal share</span><strong>{anthropology_profile['coverage']['exact_refusal_share']:.1%}</strong></li>
                  <li><span>Median attempted length</span><strong>{anthropology_profile['attempt_style']['word_count_median']:.0f} words</strong></li>
                  <li><span>Attempts using context template</span><strong>{anthropology_profile['attempt_style']['context_template_share']:.1%}</strong></li>
                </ul>
              </section>
              <section class="card green">
                <h3>Data quality and lexical clues</h3>
                <ul class="signal-list">
                  <li><span>Attempts with usable token metadata</span><strong>{anthropology_profile['metadata_quality']['attempt_token_logprob_usable_count']:,} · {anthropology_profile['metadata_quality']['attempt_token_logprob_usable_share']:.1%}</strong></li>
                  {anthropology_term_items}
                </ul>
                <p><strong>How to read a term:</strong> “{anthropology_profile['attempted_query_terms'][0]['term']}” appeared in {anthropology_profile['attempted_query_terms'][0]['support_queries']:,} attempted queries and was {anthropology_profile['attempted_query_terms'][0]['lift']:.1f}× as common there as in the full discovery set.</p>
              </section>
            </div>
            <div class="compare-grid" style="margin-top:12px">
              <section class="card green">
                <h3>Safe statement</h3>
                <p>This candidate attempted discovery queries containing {anthropology_term_words} unusually often, and its observed attempt rate was {anthropology_profile['coverage']['attempt_share']:.1%}.</p>
              </section>
              <section class="card red">
                <h3>Unsafe statement</h3>
                <p>“This proves the candidate is an expert in those topics.” It does not: attempts may be irrelevant or wrong, and the discovery mix is only one sample.</p>
              </section>
            </div>

            <div class="divider">Later, when a new query arrives</div>
            <div class="flow-row">
              <section class="card"><h3>New query</h3><p>For example, a question containing a supported profile term.</p></section>
              <div class="arrow" aria-hidden="true"><span class="wide">+</span><span class="narrow">+</span></div>
              <section class="card"><h3>Saved profile row</h3><p>Carry coverage/style fields and compare query words with the stored lexical associations.</p></section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card green"><h3>Ranking features</h3><p>Repeat for all 150 candidates. A ranker can use these clues beside stronger response-retrieval and semantic evidence.</p></section>
            </div>
            """,
            (
                "The builder streams one million one hundred sixty-three thousand eight hundred fifty response records, measures simple behavior and metadata flags, groups by candidate, and saves one hundred fifty profile rows. The real Anthropology row records its coverage, style, metadata usability, and supported words from queries it attempted. Those words are associations, not verified expertise. At query time, the saved row supplies candidate clues and query-to-profile comparisons; stronger response retrieval and semantic evidence should remain separate inputs."
            ),
            "candidate_profiles_sql",
        ),
        {
            "id": "candidate_profiles_table_block",
            "type": "table",
            "tableId": "candidate_profiles_table",
            "layout": "full",
        },
        md(
            "profile_shapes_2025",
            """
            ## What the reported 2025 profiles actually contained

            **GenAIus — one giant text document.** It concatenated each candidate’s historical query–response pairs into one profile document and searched profiles with BM25/RM3. This was simple but weaker than retrieving individual responses, because one document blurred unrelated specialties.

            **UDInfo — ten semantic centers plus answer-level refinement.** It embedded and normalized each candidate’s answers with BGE, ran k-means with ten centroids per candidate, and used maximum query-to-centroid similarity as a broad score. A per-candidate FAISS index then supplied the ten nearest individual answers for a refined similarity score. The two scores were mixed with a reported parameter of 0.2.

            **paulphys — a 64-dimensional behavioral résumé.** Reported feature groups covered handcrafted domains and task types; token, character, logprob, punctuation, repetition, sentence, list, code, formatting, uppercase, and digit behavior; query–response overlap; and semantic centroid, dispersion, and similarity summaries. The public material does **not** provide a trustworthy column-by-column dictionary for all 64 dimensions, so this guide does not invent one.

            **UvA — a compact confidence résumé.** Its official feature row joined a 384-dimensional query embedding with a scalar candidate ID and two candidate-level statistics: mean and standard deviation over the candidate’s top 1% most confident responses. A post-submission extension described candidate-by-topic confidence, refusal, length, and support statistics, plus query-to-cluster distances to the three nearest centroids. Its stated total dimensionality conflicts with the listed components.

            **indelab — learned identity and engineered summaries.** Its neural system learned a candidate embedding; its stronger non-neural system used Bradley–Terry skill, random-forest query/model/profile features, and roughly 17 meta-ranking features.
            """,
            "deep_dive_2025",
        ),
        teaching_visual(
            "beginner-visual-single-vs-multi-profile",
            "One broad résumé can hide a real specialty",
            "Profiles compress history. More compression is cheaper, but it can erase the small pocket of evidence that matters for one new question.",
            """
            <div class="gate"><strong>Conceptual teaching example—not a measured 2026 result:</strong> imagine one candidate has stored answers in three unrelated areas. The question is how much of that shape survives when we summarize the history.</div>
            <div class="phase-grid">
              <section class="card blue"><h3>History pile A</h3><p>Astronomy responses</p><span class="badge">one specialty</span></section>
              <section class="card gold"><h3>History pile B</h3><p>Music responses</p><span class="badge">another specialty</span></section>
              <section class="card green"><h3>History pile C</h3><p>Legal responses</p><span class="badge">another specialty</span></section>
            </div>
            <div class="down" aria-hidden="true">↓</div>
            <div class="compare-grid">
              <section class="card red">
                <h3>Compress everything into one profile</h3>
                <ul>
                  <li>One long searchable document or one average vector.</li>
                  <li><strong>Good:</strong> compact, stable, and fast.</li>
                  <li><strong>Risk:</strong> distinctive words are diluted and one average can sit between specialties.</li>
                  <li><strong>Result:</strong> a narrow but real specialty can disappear.</li>
                </ul>
              </section>
              <section class="card green">
                <h3>Keep several supported drawers</h3>
                <ul>
                  <li>Several centroids, topic-local rows, and links back to individual responses.</li>
                  <li><strong>Good:</strong> the new query can open its nearest specialty drawer.</li>
                  <li><strong>Risk:</strong> tiny clusters are noisy and the number of drawers must be tuned.</li>
                  <li><strong>Result:</strong> detail survives, but each drawer needs an evidence count.</li>
                </ul>
              </section>
            </div>
            <div class="callout"><strong>Recommended hierarchy:</strong> keep the response bank, derive several specialty summaries, and keep a global fallback. Store support count, spread, missingness, and profile version beside every summary. Blend a low-support local drawer back toward the safer global profile.</div>
            """,
            "A profile compresses a candidate's response history. One document or one average vector is compact, but it can blur unrelated specialties. Several centroids or topic drawers preserve more detail, while individual responses preserve the most. Keep raw responses as the source of truth, store evidence counts and spread with every summary, and fall back toward a global profile when a local drawer has little support.",
            "guide_notes",
        ),
        md(
            "profile_recommendation_2026",
            """
            ## For 2026, keep profiles modular and keep response evidence primary

            **The key idea is that a profile is a container, not one magic vector.** One candidate can simultaneously have a searchable text document, several centroids, numeric behavior statistics, topic-local rows, and a pointer to every raw response. A practical data design uses separate tables: a response bank with one historical response per row; a global profile with one row per candidate; a centroid table with one row per candidate–centroid pair; a topic-local table with one row per candidate–query-cluster pair; and a ranking table with one row per new query–candidate pair.

            **My recommendation is to treat the response bank as the source of truth and profiles as cached views.** The raw work sample keeps narrow evidence that compression can erase. Global summaries add stability. Several centroids and topic-local rows add speed and a useful middle level. Every derived field should carry its support count, spread or cluster radius, missingness flag, cleaning version, encoder version, and—if judges were used—judge/rubric version.

            Store **support counts** and **dispersion** beside every local statistic. A confidence mean of 0.9 based on two examples should not be trusted like the same mean based on 200 examples. Blend an uncertain local estimate toward the global value:

            `shrunk local = (local count × local value + trust constant × global value) ÷ (local count + trust constant)`

            This shrinkage equation is a recommended 2026 pattern, not a claim about one exact 2025 implementation.

            The 2025 evidence argues against making one giant profile the only system. Individual response retrieval was stronger for GenAIus, so use profiles for stability, complementary features, and diagnosis. Run three ablations—response retrieval only, profile only, and their combination—so the profile must prove that it adds value. Candidate IDs, learned candidate-ID embeddings, and candidate profile values from 2025 do not transfer to the new 150-candidate set.

            The current boundary still matters: global coverage, refusal, response-style, metadata-usability, and attempted-query word associations are now materialized for all 150 candidates. Response-retrieval scores, token-confidence summaries, semantic answer centroids, topic-local quality rows, and judge labels remain designs to build next—not measurements already present in this repository.
            """,
            "guide_notes",
        ),
        teaching_visual(
            "beginner-visual-profile-to-score",
            "A profile is stored memory; the query turns it into a score",
            "Candidate-only memory gives a default ordering. Query-to-profile comparisons are what make the ranking change for each new question.",
            """
            <section class="card gold">
              <h3>1. Stored once for Candidate M</h3>
              <ul>
                <li>Global behavior: refusal, length, confidence, and missingness summaries</li>
                <li>Several semantic centroids with support count and spread</li>
                <li>Topic-local summaries and a pointer to the raw response bank</li>
              </ul>
              <span class="badge">candidate memory—not a final score</span>
            </section>
            <div class="down" aria-hidden="true">+</div>
            <section class="card blue">
              <h3>2. A new query arrives</h3>
              <p>The query supplies its words, embedding, apparent type, and nearest topic or discovery neighbors.</p>
              <span class="badge">changes for every question</span>
            </section>
            <div class="down" aria-hidden="true">↓</div>
            <div class="source-grid">
              <section class="card">
                <h3>3A. Semantic interaction</h3>
                <p>Compare the query with every centroid. Keep the best and top-k similarities, the winning drawer, its support, and its spread.</p>
              </section>
              <section class="card">
                <h3>3B. Topic interaction</h3>
                <p>Open the nearest topic row. Carry its judged quality or behavior summaries, support count, and distance from the query.</p>
              </section>
              <section class="card">
                <h3>3C. Raw evidence interaction</h3>
                <p>Retrieve matching responses. Aggregate BM25 or dense scores, hit count, rank votes, and evidence diversity back to Candidate M.</p>
              </section>
              <section class="card">
                <h3>3D. Global fallback</h3>
                <p>Carry stable candidate-only values. Missing is not zero, and a local estimate with little support should move toward this fallback.</p>
              </section>
            </div>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card green">
              <h3>4. One query–candidate feature row becomes one relative score</h3>
              <p>A shallow ranker or transparent fusion combines these clues. Repeat for all 150 candidates, then sort the scores.</p>
            </section>
            <div class="compare-grid" style="margin-top:12px">
              <section class="card red"><h3>Do not confuse the score with truth</h3><p>It need not be a calibrated probability. Raw score scales from different systems are not automatically comparable.</p></section>
              <section class="card red"><h3>Keep the answer key separate</h3><p>The qrel grade teaches the model during training; it is never copied into the input row. Candidate-ID order is not evidence either.</p></section>
            </div>
            """,
            "Build one stored memory package for each candidate. When a new query arrives, compare it with the candidate's centroids, nearest topic, and raw response bank, while carrying global fallback statistics and evidence counts. Those comparisons become one query-candidate feature row and one relative score. Repeat for all one hundred fifty candidates and sort. The score is not automatically a probability, and the hidden qrel grade is never an input feature.",
            "guide_notes",
        ),
        md(
            "ranking_features_deep_dive",
            """
            ## A ranking feature is one clue for one query–candidate pair

            A feature is simply an input available before the correct grade is revealed. For one new query, the ranking table contains 150 rows. Each row joins four kinds of clues:

            - **Query-only:** length, question type, embedding dimensions, or distances to query clusters. These repeat across all 150 candidate rows and cannot distinguish candidates by themselves.
            - **Candidate-only:** fold-local prior, candidate name, global confidence, refusal rate, response statistics, or a behavioral profile. These distinguish candidates but tend to produce the same default ordering for every query.
            - **Query × candidate interactions:** name similarity, query-to-centroid cosine similarity, domain/task match, keyword overlap, or learned query/candidate interaction. These make the ranking query-specific.
            - **Retrieved evidence and support:** response BM25 SumScore, hit count, nearest-answer similarity, neighbor-label average, margins, support counts, and component ranks. These show what historical evidence backs the score.

            The **target** is separate. A development qrel grade teaches the ranker what should move upward. It must never be copied into its own feature row. A judge score is usually a weak target too; it becomes a feature only when a deliberately designed meta-ranker consumes it as one noisy signal.
            """,
            "guide_notes",
        ),
        teaching_visual(
            "beginner-visual-ranking-feature-row",
            "A worked query–candidate row, with the answer key kept separate",
            "The local query/qrel values and saved behavior-profile values are exact. Response retrieval and semantic interactions are still unmeasured.",
            f"""
            <div class="triple">
              <section class="card blue">
                <h3>Query</h3>
                <p>“Who did Chopin attend the Lower Rhenish Music Festival with?”</p>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">+</span><span class="narrow">↓</span></div>
              <section class="card gold">
                <h3>Candidate</h3>
                <p><code>Politics_by_location</code></p>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card">
                <h3>One feature row</h3>
                <p>The ranker sees clues about this pair—not the grade.</p>
              </section>
            </div>
            <div class="source-grid" style="margin-top:12px">
              <section class="card">
                <h3>Query-only fields</h3>
                <ul class="signal-list">
                  <li><span>Word count</span><strong>10</strong></li>
                  <li><span>“Who” question flag</span><strong>1</strong></li>
                </ul>
              </section>
              <section class="card">
                <h3>Candidate-only fields</h3>
                <ul class="signal-list">
                  <li><span>Fold-local mean grade</span><strong>0.1215</strong></li>
                  <li><span>Fold-local positive rate</span><strong>0.0654</strong></li>
                </ul>
                <p>Both exclude this query: 39 grade points and 21 positives over the other 321 queries.</p>
              </section>
              <section class="card blue">
                <h3>Pair interaction</h3>
                <ul class="signal-list">
                  <li><span>Exact query/name token overlap</span><strong>0</strong></li>
                  <li><span>Query/profile top-term overlap</span><strong>0</strong></li>
                  <li><span>Query-to-centroid similarity</span><strong>not measured yet</strong></li>
                </ul>
              </section>
              <section class="card blue">
                <h3>Discovery evidence</h3>
                <ul class="signal-list">
                  <li><span>Response BM25 SumScore</span><strong>not measured yet</strong></li>
                  <li><span>Saved exact-refusal share</span><strong>{politics_location_profile['coverage']['exact_refusal_share']:.1%}</strong></li>
                  <li><span>Saved non-exact attempts</span><strong>{politics_location_profile['coverage']['attempt_count']:,}</strong></li>
                  <li><span>Attempts with usable metadata</span><strong>{politics_location_profile['metadata_quality']['attempt_token_logprob_usable_share']:.1%}</strong></li>
                </ul>
              </section>
            </div>
            <section class="card red" style="margin-top:12px">
              <h3>Training target—locked away while features are built</h3>
              <p>Actual qrel: <span class="grade grade-2">2</span> highly relevant. The surprising name mismatch is exactly why behavioral evidence can rescue a candidate that name matching would miss.</p>
            </section>
            <div class="divider">Repeat for all candidates</div>
            <div class="flow-row">
              <section class="card"><h3>150 feature rows</h3><p>One complete query group.</p></section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card"><h3>Ranker</h3><p>One predicted score per row.</p></section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card green"><h3>Sorted ranking</h3><p>Candidate positions 1 through 150.</p></section>
            </div>
            """,
            f"For the real Chopin query and candidate Politics by location, the query has ten words, the name overlap is zero, and the candidate’s fold-local mean grade is 0.1215 with a 0.0654 positive rate. The saved discovery profile adds an {politics_location_profile['coverage']['exact_refusal_share']:.1%} exact-refusal share, {politics_location_profile['coverage']['attempt_count']:,} non-exact attempts, and {politics_location_profile['metadata_quality']['attempt_token_logprob_usable_share']:.1%} usable attempt metadata. Response retrieval and semantic-centroid comparisons remain unmeasured. The actual grade 2 is the hidden training target, never an input feature. Repeat the row for all 150 candidates, score each row, and sort.",
            "guide_notes",
        ),
        md(
            "ranking_features_2025",
            """
            ## Three reported feature rows make the idea concrete

            **UvA’s compact row had 387 fields:** 384 query-embedding dimensions, one scalar candidate ID, the candidate’s top-1%-confidence mean, and its confidence standard deviation. The query vector repeated across candidates; the ID and confidence fields changed. A tree could learn candidate-specific interactions, but the numeric order of IDs was arbitrary and should not be treated as meaningful in a new system.

            **paulphys joined a 64-dimensional candidate profile with query descriptors.** It added domain/task match indicators and elementwise interaction products. The submitter described a LambdaMART-style LightGBM ranker and specified `rank_xendcg`, LightGBM’s XE_NDCG_MART objective. The candidate profile supplied stable memory; the match and interaction fields made it query-specific.

            **indelab’s meta-ranker combined about 17 higher-level features.** Reported ingredients included Bradley–Terry skill, query characteristics, judge-like or component scores, random-forest outputs, discovery-profile features, and interactions.

            These examples explain why “number of rows” and “number of features” are different. In the 2025 track, one query generated one row per 1,131 candidate experts, while the number of fields in each row varied by method.
            """,
            "deep_dive_2025",
        ),
        md(
            "ranker_grouping_explained",
            """
            ## LambdaMART compares candidates inside one query group

            For the local development matrix, a grouped ranking table has this shape:

            ```text
            X: 48,300 rows × F features
            y: 48,300 grades
            group: [150, 150, ..., 150]  # one entry for each of 322 queries
            ```

            The group array tells LightGBM: “compare these 150 candidates with one another.” Its entries are sequential row counts, so all 150 rows for one query must be contiguous; never shuffle individual rows across query boundaries. LambdaMART then builds many shallow decision trees. Its training signal gives larger weight to candidate swaps that would improve nDCG more—for example, moving a candidate with relevance grade 2 from rank 30 to rank 2 matters much more than swapping two candidates with grade 0 near the bottom.

            The model does **not** need one human-written weighted-sum formula. Trees can learn interactions such as: high centroid similarity is useful when it has enough historical support and the refusal rate is low. Query-only fields cannot separate candidates alone; candidate-only fields cannot adapt much to the query; pair and evidence features connect the two.

            During exact outer leave-one-query-out evaluation, the training side has 321 × 150 = 48,150 rows and the held-out side has 150 rows. Hyperparameter tuning, qrel-derived priors, feature selection, calibration, supervised clusters, and fusion weights stay inside the 321-query training side. Despite 48,300 rows, there are only 322 independent query groups, so a compact, shallow ranker is safer than a huge feature space.
            """,
            "guide_notes",
        ),
        md(
            "srcb",
            """
            ## SRCB: manufacture labels, then train an efficient scorer

            **In one sentence:** SRCB created a practice answer key with several imperfect teachers, then trained a smaller scorer to imitate useful parts of that signal.

            **Everyday picture:** several tutors grade the same practice answers for different reasons. The student model sees their grades, while the audit checks whether the tutors reward polish instead of truth.

            **Plain idea:** the discovery responses had no gold training labels, so SRCB created several noisy versions of an answer-quality ranking. One used LLM pairwise comparisons in a tournament-style, Quicksort-like procedure; one used Qwen reranker scores; others used embedding similarity to a generated reference or to the query; a combination fused label sources.

            The team froze a pretrained **encoder**—a text-to-vector model whose weights were not updated—and trained lightweight ranking, classification, or regression **heads** that emit one score per candidate. Classification chooses a label category; regression predicts a numeric score. Their paper’s development study found the strongest variants around Qwen3-Embedding-8B with Quicksort or combined labels. The official `submission_4` run page names Qwen3-Embedding-8B but does not give a full configuration. [`submission_5`](https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main/submission_5.html) explicitly used Qwen3-14B annotations, DeBERTa-v3, a fully connected decoder, and binary cross-entropy (BCE) loss. The three ensemble memberships remain unpublished.

            **2026 use:** weak-label only a **high-recall** response subset designed to retain nearly every plausible answer, keep judge/reranker/reference signals separate, and **calibrate** each by checking how its confidence matches held-out qrels before fusion. The danger is learning a judge’s preference for polished style instead of factual expertise.
            """,
            "srcb_2025",
        ),
        teaching_visual(
            "beginner-visual-srcb-pipeline",
            "SRCB: build a practice answer key, then train the scorer",
            "No gold labels existed for the discovery answers, so several imperfect teachers created alternative practice grades.",
            """
            <div class="flow-row">
              <section class="card blue">
                <h3>1. Make one practice example</h3>
                <p>Join a stored question with one candidate response.</p>
                <span class="badge">Input: query + response</span>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card gold">
                <h3>2. Ask four imperfect teachers</h3>
                <ul>
                  <li>Pairwise LLM judge</li>
                  <li>Qwen reranker</li>
                  <li>Similarity to a generated reference</li>
                  <li>Similarity to the query</li>
                </ul>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card green">
                <h3>3. Train a small scoring head</h3>
                <p>Keep the text encoder frozen. Learn a ranking, classification, or regression head from each label version.</p>
              </section>
            </div>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card green">
              <h3>4. Score every candidate and sort</h3>
              <p>Each trained variant emits one candidate score. Compare the variants separately; combine only versions that survive held-out validation.</p>
            </section>
            <div class="callout"><strong>What can fool it:</strong> the student can copy a teacher's preference for polished writing instead of learning factual expertise. Audit every teacher against held-out qrels.</div>
            <div class="gate"><strong>2026 data check:</strong> this needs discovery responses, which are listed in the release but absent from this checkout. This is the documented SRCB method-family flow; the paper does not fully map the winning run or three ensembles to exact configurations.</div>
            """,
            "SRCB turns each stored question and response into a practice example. Four imperfect teachers supply alternative labels: a pairwise LLM judge, a reranker, similarity to a generated reference, and similarity to the query. A frozen text encoder plus a small learned head then scores every candidate. The main risk is copying a teacher's style bias rather than learning expertise, and the exact winning-run configuration was not fully published.",
            "srcb_2025",
        ),
        md(
            "udinfo",
            """
            ## UDInfo: combine similar-query evidence with stable global context

            **In one sentence:** UDInfo balanced local advice—what worked on questions like this—with a stable global estimate of what each candidate usually knows.

            **Everyday picture:** ask nearby neighbors first, but fall back toward the town-wide reputation when those neighbors disagree or look unfamiliar.

            **Method 1** embedded each candidate’s historical answers, compressed them into ten centroids for broad expertise, and refined the score with nearest answers from a per-candidate FAISS vector-search index. **Method 2**, corresponding strongly but not explicitly to their best run, used an OLMo judge to assign 0/1/2 weak labels, retrieved 100 similar historical queries, and mixed their labels with a global prior and a 40-**cluster** domain estimate, where each cluster is a group of similar questions.

            Three additional methods tested centroid-only profiles and shallow/deeper multilayer-perceptron (MLP) routers. The hybrid local/global design was stronger and easier to inspect than the pure classifier family.

            **2026 use:** this is the closest blueprint to the repository’s current dense query-kNN. Replace raw qrel transfer with response-derived weak labels when discovery data is available, then learn how strongly to **shrink** uncertain local evidence by blending it toward a safer fold-local prior.
            """,
            "udinfo_2025",
        ),
        teaching_visual(
            "beginner-visual-udinfo-pipeline",
            "UDInfo: ask three advisers, then steady the noisy answer",
            "For every candidate, the hybrid method combines nearby-question evidence with broader estimates that change more slowly.",
            """
            <section class="card blue">
              <h3>1. Embed the new query</h3>
              <p>Turn the question into a meaning vector so similar historical questions and a broad domain can be found.</p>
            </section>
            <div class="down" aria-hidden="true">↓</div>
            <div class="phase-grid">
              <section class="card blue">
                <h3>2A. Local adviser</h3>
                <p>Retrieve 100 similar historical queries and average this candidate's similarity-weighted weak grades.</p>
                <span class="badge">Specific, but noisy</span>
              </section>
              <section class="card gold">
                <h3>2B. Global adviser</h3>
                <p>Use this candidate's average weak grade across the discovery data.</p>
                <span class="badge">Stable fallback</span>
              </section>
              <section class="card green">
                <h3>2C. Domain adviser</h3>
                <p>Find the closest of 40 query clusters and use this candidate's average grade inside that cluster.</p>
                <span class="badge">Topic-local context</span>
              </section>
            </div>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card green">
              <h3>3. Blend, score, and sort</h3>
              <p>First steady the local estimate with the global prior. Then combine that result with the domain estimate and rank all candidates.</p>
            </section>
            <div class="divider">A second UDInfo route</div>
            <section class="card">
              <h3>Answer-profile alternative</h3>
              <p>For each candidate, ten answer centroids provide a broad expertise match; ten nearest individual answers refine it. This is a different method, not another step in the three-adviser hybrid.</p>
            </section>
            <div class="callout"><strong>What can fool it:</strong> judge errors enter both the neighbor and cluster estimates. When debugging, show the retrieved questions, their weights, and the assigned cluster.</div>
            <div class="gate"><strong>2026 data check:</strong> dense neighbor transfer already works with the local development qrels, and the discovery responses are now local. Replacing gold-grade transfer with response-derived weak labels still requires a judge pipeline and end-to-end validation. The run-to-method mapping is strongly implied, not explicitly tabulated.</div>
            """,
            "UDInfo embeds a new query and consults three advisers for every candidate. One uses one hundred nearby historical questions, one uses the candidate's global average weak grade, and one uses the candidate's average grade in the closest of forty query clusters. The method steadies noisy local evidence with the broader estimates and then ranks candidates. A separate profile method used ten answer centroids and ten nearest answers.",
            "udinfo_2025",
        ),
        md(
            "genaius",
            """
            ## GenAIus: retrieve individual responses, then aggregate to experts

            **In one sentence:** GenAIus searched individual work samples first, then let every strong match vote for the candidate that produced it.

            **Everyday picture:** do not hire from one broad résumé. Search a filing cabinet of past work, pull the most relevant pages, and add the evidence to their authors.

            GenAIus started from about **16.9M** query-response records. A basic failure filter retained 2.83M; a stricter refusal/low-information filter retained 1.24M. They built three indexes: one giant document per candidate, one document per historical question, and one document per historical response.

            Response-level evidence was strongest. For a new query, BM25 plus RM3 relevance-model expansion—which adds likely related terms from the initial matches—retrieved 1,000 response records. Each record’s retrieval score was added to the candidate that produced it. One run inserted SPLADE learned sparse-expansion reranking before aggregation. Whole-candidate profiles were much weaker, consistent with classic expert finding: retrieve evidence first; aggregate second.

            **2026 use:** this should be the first discovery-data system because it is internal, inexpensive, debuggable, and complementary to dense query-kNN. Compare raw SumScore, rank-weighted voting, and per-candidate caps so prolific or generic responders do not win merely by leaving more indexed evidence.
            """,
            "genaius_2025",
        ),
        teaching_visual(
            "beginner-visual-genaius-pipeline",
            "GenAIus: search past answers first, then credit their authors",
            "The central move is to retrieve individual work samples instead of searching one giant résumé per candidate.",
            """
            <div class="divider">Build a cleaner response library</div>
            <div class="flow-row">
              <section class="card blue">
                <h3>1. Start broad</h3>
                <p>About 16.9 million stored query-response records.</p>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card gold">
                <h3>2. Remove obvious failures</h3>
                <p>About 2.83 million responses remain after the basic failure filter.</p>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card green">
                <h3>3. Remove refusals</h3>
                <p>About 1.24 million remain after the stricter low-information filter.</p>
              </section>
            </div>
            <div class="divider">Rank a new query</div>
            <div class="flow-row">
              <section class="card blue">
                <h3>4. Index each response</h3>
                <p>One historical response becomes one searchable document, still linked to its producing candidate.</p>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card gold">
                <h3>5. Retrieve up to 1,000</h3>
                <p>BM25 plus RM3 finds lexical evidence. One run optionally reranked the matches with SPLADE.</p>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card green">
                <h3>6. SumScore by candidate</h3>
                <p>Add every retrieved response score to its author, then sort the candidate totals.</p>
              </section>
            </div>
            <div class="callout"><strong>Why this mattered:</strong> response-level evidence was stronger than one giant candidate-profile document. The narrow work sample preserved a specialty that a broad résumé could blur.</div>
            <div class="gate"><strong>2026 data check:</strong> this is the strongest first system to build once the discovery response file is local. Test score caps or rank-weighted votes because prolific or generic responders can accumulate too much evidence, and lexical retrieval can miss paraphrases.</div>
            """,
            "GenAIus filtered about sixteen point nine million stored responses to smaller usable corpora, indexed each retained response separately, and retrieved up to one thousand matches with BM25 and RM3. Every retrieved response contributed its score to the candidate that produced it. Individual-response retrieval was stronger than treating each candidate's entire history as one giant document, but prolific candidates can collect too much evidence.",
            "genaius_2025",
        ),
        md(
            "uvairlab",
            """
            ## UvA IRLab: confidence features can help, but the sources disagree

            **In one sentence:** UvA treated confidence as a résumé feature that might adjust trust, not as proof that an answer was correct.

            **Everyday picture:** a speaker’s certainty changes how cautiously you listen, but it does not turn a confident claim into a fact.

            UvA averaged token log probabilities for every stored response, converted those values into pseudo-grades, and trained **LightGBM**, a boosted-decision-tree system, with **LambdaRank**, a ranking-focused training objective. Their base vector combined a 384-dimensional MiniLM query embedding, a numeric candidate ID, and the candidate’s mean and standard deviation over its top 1% most confident responses. A later topic-aware extension added cluster-local statistics.

            The official overview reports `lightgbm_job266431` at **0.1233 nDCG@10**. The participant paper reports roughly **0.36–0.37** for nominally submitted variants. It calls a post-submission topic vector 408-dimensional, although the listed components sum to 395, and it describes one response-based baseline that appears unavailable under the official test-time inputs. No source establishes a clean metric/query-set reconciliation, so this guide uses the official number and treats the participant values as non-comparable.

            **2026 use:** preserve mean, dispersion, lower-tail confidence, length, refusal, and local-topic versions as features. Never use logprob as the target truth; confident hallucinations exist.
            """,
            "guide_notes",
        ),
        teaching_visual(
            "beginner-visual-uvairlab-pipeline",
            "UvA: turn confidence into résumé clues, not truth",
            "Two evidence streams create one query-candidate row; a tree ranker learns how to use them.",
            """
            <div class="source-grid">
              <section class="card blue">
                <h3>1A. Describe the incoming query</h3>
                <p>MiniLM turns the query into 384 meaning-based numbers.</p>
                <span class="badge">Query evidence</span>
              </section>
              <section class="card gold">
                <h3>1B. Summarize one candidate</h3>
                <ol class="stage-list">
                  <li>Average token log probabilities within each stored response.</li>
                  <li>Keep the candidate's top 1% most confident responses.</li>
                  <li>Compute their mean and standard deviation.</li>
                </ol>
                <span class="badge">Candidate evidence</span>
              </section>
            </div>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card green">
              <h3>2. Make one 387-field row</h3>
              <p><strong>384</strong> query fields + <strong>1</strong> numeric candidate ID + <strong>1</strong> confidence mean + <strong>1</strong> confidence standard deviation.</p>
              <p>The candidate ID is an arbitrary code, not a meaningful measurement.</p>
            </section>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card green">
              <h3>3. Learn to rank and sort</h3>
              <p>LightGBM with LambdaRank scores one row for every candidate. Sorting those scores produces the final ranking.</p>
            </section>
            <div class="gate"><strong>Training-only proxy:</strong> UvA also turned average response log likelihood into practice grades: below −2 became 0, −2 to −1 became 1, and above −1 became 2. These thresholds create supervision; they do not establish correctness.</div>
            <div class="callout"><strong>What can fool it:</strong> confident hallucinations exist, and a numeric candidate-ID split can memorize last year's arbitrary identities. Rebuild candidate features for the new 150-ID set and treat confidence only as supporting evidence.</div>
            <div class="gate"><strong>Evidence warning:</strong> this diagram shows the submitted base-vector family. The official score and participant-side scores do not reconcile, and the later topic-aware extension has a dimensionality inconsistency, so neither is silently mixed into this flow.</div>
            """,
            "UvA embeds the new query into three hundred eighty-four numbers. For every candidate it adds an arbitrary numeric ID and two stored confidence summaries: the mean and standard deviation of that candidate's top one percent most confident responses. These three hundred eighty-seven fields enter a LightGBM LambdaRank model. Confidence is only supporting evidence because an answer can be confidently wrong, and last year's candidate ID should not transfer to the new set.",
            "uvairlab_2025",
        ),
        md(
            "indelab_paulphys",
            """
            ## indelab and paulphys: useful feature ideas, thinner reproducibility

            **In one sentence:** these teams tested whether many engineered signals or a learned summary could represent expertise; their results show that more features or examples do not repair unreliable supervision.

            **Everyday picture:** a detailed résumé can still describe the wrong abilities, and millions of practice cards can still teach the wrong rule if their labels are poor.

            **indelab in plain language:** it tried both a large two-tower representation and a stronger engineered ensemble that combined pairwise skill, random-forest, and ranking features.

            **paulphys in plain language:** it built a compact behavioral résumé for each candidate and trained a tree ranker over those résumés.

            The official [`non-neural`](https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main/non-neural.html) description reports that indelab’s stronger submission fused a pairwise Bradley–Terry skill estimate, a 150-tree random forest, and a 17-feature LightGBM meta-ranker. Its [`mllm-indelab-09-17`](https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main/mllm-indelab-09-17.html) **dual encoder**—one vector tower for queries and one for candidates—used about 4.45M examples, including crude weak labels that treated responses longer than 50 characters as relevant. The ensemble beat the neural run officially, a warning that scale does not repair weak supervision.

            The [`lambdamart_profiles`](https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main/lambdamart_profiles.html) run description and [author report](https://github.com/eigenpaul/trec2025-mllm) describe a 64-dimensional behavioral profile using Qwen3 embeddings, response statistics, refusal/structure cues, and hand-built domain/task matches. It sampled half the discovery responses and trained what the submitter described as a LambdaMART-style LightGBM ranker, specifying the `rank_xendcg` (XE_NDCG_MART) objective, on the top 200 development candidates. It finished last officially. The transferable lesson is not “profiles never work”; it is that one coarse résumé can blur several unrelated specialties and hand-built categories can shift.

            These details are submitter-reported in official run pages, but neither indelab system has a proceedings paper or public reproducibility package in the reviewed sources. Treat architecture-level detail with less confidence than GenAIus’s fully mapped runs.
            """,
            "runs_2025",
        ),
        teaching_visual(
            "beginner-visual-indelab-paulphys-pipelines",
            "indelab and paulphys: two ways to summarize an expert",
            "One team compared learned and engineered routes; the other built a behavioral résumé. Both expose the same bottleneck: the summary is only as useful as its labels and categories.",
            """
            <div class="compare-grid">
              <section class="card blue">
                <h3>indelab: learn a match or combine experts</h3>
                <ol class="stage-list">
                  <li><strong>Neural path</strong><span>Use about 4.45 million examples, including a weak rule that called responses longer than 50 characters relevant. A query tower and learned candidate-ID tower meet through similarity.</span></li>
                  <li><strong>Engineered path</strong><span>Combine a Bradley–Terry skill estimate, a 150-tree random forest, and an approximately 17-feature LightGBM meta-ranker.</span></li>
                  <li><strong>Official pattern</strong><span>The engineered ensemble ranked above the neural run.</span></li>
                </ol>
                <div class="gate"><strong>Risk:</strong> the length rule can reward verbosity, and the neural run's exact training loss was not published.</div>
              </section>
              <section class="card green">
                <h3>paulphys: build one behavioral résumé</h3>
                <ol class="stage-list">
                  <li><strong>Sample and embed</strong><span>Use half of the discovery responses with Qwen3 embeddings.</span></li>
                  <li><strong>Build 64 profile fields</strong><span>Summarize domains and tasks, length and logprob behavior, lexical and structural cues, overlap, and semantic behavior.</span></li>
                  <li><strong>Join query and candidate</strong><span>Add query descriptors, domain/task matches, and interaction features.</span></li>
                  <li><strong>Rank</strong><span>Train LightGBM with the <code>rank_xendcg</code> objective on the top 200 development candidates.</span></li>
                </ol>
                <div class="gate"><strong>Risk:</strong> one coarse profile can blur unrelated specialties, and hand-built categories can shift.</div>
              </section>
            </div>
            <div class="callout"><strong>Shared lesson:</strong> millions of examples and dozens of fields cannot repair an unreliable target. Use these profiles as complementary evidence and diagnostics, not as automatic truth.</div>
            <div class="gate"><strong>2026 data check:</strong> do not carry over last year's candidate IDs or résumés. A simple behavior/topic profile has now been rebuilt for all 150 new candidates; semantic answer profiles and judged-quality fields remain to build and validate. indelab lacks a reviewed proceedings paper or public reproduction package in the reviewed sources; paulphys details are submitter-reported.</div>
            """,
            "indelab compared a neural dual encoder with an engineered ensemble. Its weak neural examples treated responses longer than fifty characters as relevant, while its stronger official system combined Bradley–Terry skill, a random forest, and a LightGBM meta-ranker. paulphys sampled half the responses, summarized each candidate in a sixty-four-field behavioral profile, joined that résumé with query-match features, and trained a tree ranker. Both warn that complex models cannot rescue weak labels or overly coarse profiles.",
            "guide_notes",
        ),
        {"id": "runs_top10_block", "type": "table", "tableId": "runs_top10_table", "layout": "full"},
        {"id": "runs_bottom9_block", "type": "table", "tableId": "runs_bottom9_table", "layout": "full"},
        md(
            "literature_onramp",
            """
            ## What “literature foundation” means in this guide

            A foundation paper is not necessarily a paper about this exact competition. It is earlier research that solved a similar-shaped problem and gives us a reusable idea, vocabulary, or warning.

            Each section below now follows the same listening path: **one-sentence idea → analogy → simple algorithm → why it fits this data → research lineage and failure modes**. The technical paper names remain because they tell you where the idea came from and what to read when you want the full derivation.

            None of the cited scores from other benchmarks should be compared directly with TREC nDCG@10 or MRR. The literature supports design choices; our query-grouped validation must decide whether those choices work here.
            """,
        ),
        md(
            "expert_lit",
            """
            ## Literature foundation 1: expert finding says “evidence first”

            **In one sentence:** search the candidate’s work samples before judging the candidate as a whole.

            **Analogy:** one résumé can hide the exact experience you need. Searching individual work samples lets the most relevant examples vote for their author.

            **Simple algorithm:** retrieve matching responses; connect each response to the candidate that produced it; aggregate those response scores into a candidate ranking. The central risk is evidence-count bias: a prolific candidate can collect more votes merely by having more indexed records.

            **Where the idea comes from:** [Balog, Azzopardi, and de Rijke (2006)](https://doi.org/10.1145/1148170.1148181) distinguish a **candidate model**—build one profile per expert—from a **document model**—retrieve relevant evidence, then associate it with experts. The evidence-first formulation maps almost perfectly onto response retrieval and helps explain why GenAIus’s response index beat its giant profile documents.

            [Macdonald and Ounis (2006)](https://doi.org/10.1145/1183614.1183671) compare voting and score-fusion rules for candidates. [ReDDE](https://doi.org/10.1145/860435.860490) treats each collection as a resource and estimates its relevant-document mass. These lines suggest three 2026 ablations: SumScore, rank-discounted voting, and a normalized/capped score that corrects for different amounts of surviving evidence.

            [Query clarity](https://doi.org/10.1145/564376.564429) adds a gating idea: when retrieved neighbors are diffuse or the top score margin is small, trust local evidence less and shrink more strongly toward a global prior.
            """,
        ),
        md(
            "retrieval_lit",
            """
            ## Literature foundation 2: sparse and dense retrieval should be partners

            **In one sentence:** use both an exact-word librarian and a meaning-based librarian.

            **Analogy:** the exact-word librarian is excellent when you know a name or technical phrase. The meaning-based librarian can recognize a paraphrase. Neither librarian is best for every question.

            **Simple algorithm:** build separate sparse and dense indexes; retrieve from each; keep the score scales separate; fuse the rank positions. The main risks are **query drift**, where expansion changes the intended topic, and **domain shift**, where an encoder’s learned meaning map does not fit these questions.

            **Where the idea comes from:** [BM25](https://doi.org/10.1561/1500000019) is strong for names, dates, entities, and rare technical phrases. [RM3-style relevance feedback](https://trec.nist.gov/pubs/trec13/papers/umass.novelty.hard.pdf) can bridge vocabulary gaps but risks query drift when the first retrieval is wrong. [SPLADE](https://arxiv.org/abs/2107.05720) is a learned sparse-expansion method that retains inverted-index behavior.

            [Sentence-BERT](https://aclanthology.org/D19-1410/), [DPR](https://aclanthology.org/2020.emnlp-main.550/), [E5](https://arxiv.org/abs/2212.03533), [BGE-M3](https://arxiv.org/abs/2402.03216), and [ColBERT](https://arxiv.org/abs/2004.12832) offer increasingly rich semantic matching. The local MPNet result proves that one-vector semantics already help, but [BEIR](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html) and [MTEB](https://aclanthology.org/2023.eacl-main.148/) warn that no encoder dominates every domain.

            Build separate historical-query, response, and `query + response` indexes. Fuse their ranks with [Reciprocal Rank Fusion](https://doi.org/10.1145/1571941.1572114) before learning weights; it avoids pretending BM25 and cosine scores share a scale.
            """,
        ),
        md(
            "ltr_lit",
            """
            ## Literature foundation 3: learning to rank fits the output—but data is small

            **In one sentence:** learning to rank is a trainable scorecard for deciding how much every clue should move a candidate up or down.

            **Analogy:** a hiring panel already has résumé match, work samples, references, and confidence notes. A ranking model learns how much weight each clue deserves for the ordered shortlist.

            **Simple algorithm:** for one query, place all 150 candidate rows in one training group; compute only leakage-safe features; learn which swaps improve the ranked list. The catch is sample size: 48,300 rows come from only 322 independent questions.

            **Where the idea comes from:** [ListNet](https://doi.org/10.1145/1273496.1273513) optimizes a list rather than isolated labels. [LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) is a boosted-tree ranker that emphasizes swaps that matter to nDCG. [LightGBM](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [XGBoost](https://doi.org/10.1145/2939672.2939785), and [CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) are practical tree-learning systems for combining retrieval scores, priors, evidence counts, confidence summaries, query features, and interactions.

            One training group should contain all 150 candidates for one query. Give query groups equal weight, use shallow trees and minimum leaf sizes, and train only on out-of-fold component scores. A 48,300-row table is still only 322 independent query groups.

            Useful row features include sparse/dense response aggregation; query-neighbor grades; candidate-name similarity; fold-local global and cluster priors; max/mean/dispersion/count of matching evidence; logprob quantiles; refusal and repetition rates; query length/type; neighbor margin; and system disagreement.
            """,
        ),
        md(
            "routing_lit",
            """
            ## Literature foundation 4: a router is a dispatcher, but the job rules matter

            ### Start here: what is a router?

            A **router** looks at a new question and predicts which candidate model is most suitable. Think of a dispatcher matching a job with the worker most likely to handle that kind of job well.

            - A **candidate pool** is the fixed menu of models or IDs being considered.
            - **Query difficulty** describes clues that a question may be easy, specialized, ambiguous, or demanding.
            - A **pairwise preference** says candidate A should rank above candidate B for one example.
            - A **query–candidate interaction** is a clue measuring how this question matches this candidate.
            - A **test-time call** means sending the new question to a candidate and seeing its answer before deciding where to place it.

            **The central idea transfers:** different candidates can excel on different questions.

            **The task shape does not transfer automatically:** many routing studies choose from a smaller pool, can inspect live candidate outputs, or balance answer quality against cost. This competition requires a complete ordering of 150 fixed candidate IDs without calling them on the test question.
            """,
        ),
        teaching_visual(
            "beginner-visual-router-task-shapes",
            "Same routing idea, two different decision problems",
            "Both systems try to match a question with capable models. What they may observe, what they optimize, and what they must output are different.",
            """
            <div class="callout"><strong>Shared premise:</strong> no candidate is best for every question, so the decision should depend on the query.</div>

            <div class="compare-grid" style="margin-top:12px">
              <section class="card gold">
                <h3>A common setup in routing research</h3>
                <ol class="stage-list">
                  <li><strong>Candidate pool</strong><span>Often a smaller menu of real models.</span></li>
                  <li><strong>Available clues</strong><span>The query, historical quality, price or latency, and in some studies the candidates' current outputs.</span></li>
                  <li><strong>Decision</strong><span>Select one model, use a cheap-to-expensive cascade, or combine several outputs.</span></li>
                  <li><strong>Goal</strong><span>Often balance answer quality with money or response time.</span></li>
                </ol>
              </section>

              <section class="card green">
                <h3>This TREC 2026 task</h3>
                <ol class="stage-list">
                  <li><strong>Candidate pool</strong><span>A fixed list of 150 candidate IDs.</span></li>
                  <li><strong>Available clues</strong><span>The new query plus stored questions, responses, metadata, profiles, and development grades. No live candidate calls for the test query.</span></li>
                  <li><strong>Decision</strong><span>Score every candidate and submit the complete ordered list.</span></li>
                  <li><strong>Goal</strong><span>Place relevant candidates near the top according to hidden relevance grades.</span></li>
                </ol>
              </section>
            </div>

            <div class="divider">What transfers, and what must be rebuilt?</div>
            <div class="compare-grid">
              <section class="card blue">
                <h3>Ideas worth transferring</h3>
                <ul>
                  <li>Estimate the query's topic and difficulty.</li>
                  <li>Retrieve similar historical questions.</li>
                  <li>Learn auditable pairwise preferences.</li>
                  <li>Build query–candidate match features.</li>
                  <li>Compare every complex method with simple baselines.</li>
                </ul>
              </section>

              <section class="card red">
                <h3>Do not copy unchanged</h3>
                <ul>
                  <li>Headline gains from differently shaped benchmarks</li>
                  <li>Methods requiring candidate answers at test time</li>
                  <li>Cost thresholds when cost is not the official objective</li>
                  <li>Architectures tuned for only a few candidates</li>
                  <li>Candidate representations learned for last year's pool</li>
                </ul>
              </section>
            </div>

            <div class="gate"><strong>Practical reading rule:</strong> borrow a paper's evidence signal, then test it against query retrieval and stable-prior baselines under this competition's exact 150-candidate, no-test-time-call protocol. Do not import the paper's reported gain as an expected gain here.</div>
            <div class="gate"><strong>Data currently local:</strong> 322 development queries and complete grades for all 150 candidates. The discovery responses and metadata needed for history-based routing are listed in the release but absent from this checkout.</div>
            """,
            "Both ordinary routing research and TREC ask which candidate best fits a question. A common routing study may choose from a smaller model menu and may use known quality, cost, latency, or even generated outputs. It often selects one model, a cascade, or a final mixture. In this TREC task, the system cannot call candidate models for the test query and must order all 150 fixed candidate IDs using stored evidence and development grades. Query-difficulty clues, similar-query retrieval, pairwise preferences, and query-candidate match features can transfer. Published gains, cost rules, live-output methods, and last year's candidate representations cannot be assumed to transfer.",
            "guide_notes",
        ),
        md(
            "routing_lit_research",
            """
            ### The router literature, grouped by the job each paper studies

            **Cost, cascades, and output comparison.** [FrugalGPT](https://arxiv.org/abs/2305.05176) learns cascades that trade cost against quality. [LLM-Blender](https://arxiv.org/abs/2306.02561) pairwise-ranks candidate outputs before fusing them. Both support query-dependent model choice, but their available actions differ from ranking fixed IDs without observing current outputs.

            **Learning relationships between queries and models.** [RouterBench](https://arxiv.org/abs/2403.12031) studies a large query-by-model outcome matrix—a historical table of questions, models, and observed quality. [RouteLLM](https://arxiv.org/abs/2406.18665) learns routing from preference data. [GraphRouter](https://openreview.net/forum?id=eU39PDsZtT) represents relationships among queries, tasks, and models as a graph.

            Together, these works support model complementarity, query-difficulty features, pairwise preferences, and query/model interactions learned from data. But most published systems route among a smaller pool, can observe real model outputs, or optimize cost. They do not directly solve a fixed 150-candidate ranking with no test-time calls.

            **Why simple retrieval still matters.** [ContextualRouter (2026)](https://aclanthology.org/2026.eacl-srw.22/) predicts model performance by retrieving similar past queries. [LLMRouterBench (2026)](https://aclanthology.org/2026.findings-acl.1881/) finds many sophisticated routers difficult to distinguish from simple baselines under unified evaluation. These results strengthen the case for a disciplined baseline ladder.
            """,
        ),
        md(
            "routing_lit_transfer",
            """
            ### What to transfer into the 2026 experiment plan

            **Borrow the evidence ideas:**

            1. Describe the new query's topic and apparent difficulty.
            2. Retrieve similar historical queries and inspect which candidates performed well.
            3. Represent query–candidate compatibility explicitly.
            4. Use pairwise preferences only when they come from auditable historical evidence.
            5. Test whether graph or learned interaction features add information beyond simpler retrieval.

            **Do not copy a paper's result as a forecast:** candidate count, available evidence, test-time rules, and the optimization objective differ. A gain reported for live output selection, a cost-aware cascade, or a smaller model pool is not an expected gain for this track.

            **Decision rule:** require every sophisticated router to beat simple query retrieval and stable-prior baselines under the same leakage-safe evaluation. Complexity earns its place only when it adds repeatable evidence under this task's exact rules.
            """,
        ),
        md(
            "weak_label_lit",
            """
            ## Literature foundation 5: weak labels are practice grades, not truth

            ### Start here: what do the words mean?

            - A **qrel** is a trusted answer-key grade supplied in the labeled competition data, such as 0, 1, or 2.
            - A **weak label** is an estimated practice grade created where a qrel is unavailable. It may come from an AI judge, reranker, similarity rule, or another imperfect shortcut.
            - A **jury** means using several judge setups instead of declaring one teacher correct.
            - An **audit trail** is the saved record of the inputs, prompt, settings, raw votes, and disagreements that produced a weak label.
            - The **router** is the final model trained to rank the 150 candidate IDs for a new question.

            **Tutor analogy:** official qrels are the answer key. Weak labels are practice grades supplied by several tutors. One tutor may reward a longer answer, the first answer shown, or writing that resembles its own style.

            **The safety rule:** agreement is useful evidence, but it is not automatic truth. Keep disagreement visible. A small human- or reference-checked discovery subset can audit response labels directly; held-out development qrels test the resulting router end to end. The qrels do not directly reveal whether one historical discovery response is factually correct.
            """,
        ),
        md(
            "judge_labels_bridge",
            """
            ## An LLM judge grades one stored response; a profile remembers the pattern

            The 2026 discovery set contains a question and 150 stored responses for that question, but it contains no trusted response-quality grade. A judge can therefore create a **weak response label**: an imperfect opinion about one response on one historical question. It is not yet a permanent grade for the candidate and it is not a development qrel.

            There are three useful judgment shapes:

            1. **Pointwise:** inspect one response and assign separate rubric labels such as relevance, correctness, coverage, refusal, and “cannot verify.” This is the cheapest way to get several aspects in one structured evaluation.
            2. **Pairwise:** compare response A with response B on one criterion and return A, B, tie, or cannot decide. The decision is often easier, but comparisons multiply quickly and can depend on presentation order.
            3. **Derived:** combine many raw judgments into a candidate’s topic-local mean, pairwise win rate, refusal rate, disagreement rate, and support count. These derived values become profile fields or router inputs.

            **Important distinction:** response length, formatting, and logprob summaries are observations or features. They should not silently become “good answer” labels. A long answer can be wrong, and a confident answer can hallucinate.
            """,
            "guide_notes",
        ),
        teaching_visual(
            "beginner-visual-judge-label-menu",
            "One judgment can keep several labels instead of one vague score",
            "Separate what the response did, how good it appears, and how uncertain the judge was. Combine later, after validation.",
            """
            <section class="card blue">
              <h3>Input: one historical question + one stored response</h3>
              <p>Hide the candidate ID. Ask for a fixed structured record, not a free-form overall impression.</p>
            </section>
            <div class="down" aria-hidden="true">↓</div>
            <div class="source-grid">
              <section class="card">
                <h3>A. Mechanical flags</h3>
                <ul class="signal-list">
                  <li><span>Empty or failed</span><strong>yes / no</strong></li>
                  <li><span>Refusal</span><strong>yes / no</strong></li>
                  <li><span>Duplicate or template</span><strong>yes / no</strong></li>
                  <li><span>Language / truncation</span><strong>category</strong></li>
                </ul>
                <p>Prefer deterministic code where possible; no LLM is needed to count length.</p>
              </section>
              <section class="card green">
                <h3>B. Pointwise quality aspects</h3>
                <ul class="signal-list">
                  <li><span>Answers the question</span><strong>0 / 1 / 2</strong></li>
                  <li><span>Factual correctness</span><strong>0 / 1 / 2 / unsure</strong></li>
                  <li><span>Coverage</span><strong>0 / 1 / 2</strong></li>
                  <li><span>Unsupported-claim risk</span><strong>low / mid / high</strong></li>
                </ul>
                <p>Without a trusted reference, correctness remains a weak opinion based on the judge's knowledge.</p>
              </section>
              <section class="card gold">
                <h3>C. Pairwise outcome for hard cases</h3>
                <ul class="signal-list">
                  <li><span>Correctness winner</span><strong>A / B / tie</strong></li>
                  <li><span>Coverage winner</span><strong>A / B / tie</strong></li>
                  <li><span>Cannot decide</span><strong>allowed</strong></li>
                  <li><span>Reversed-order result</span><strong>saved separately</strong></li>
                </ul>
                <p>Do not force one winner when the evidence is unclear.</p>
              </section>
              <section class="card red">
                <h3>D. Reliability and audit labels</h3>
                <ul class="signal-list">
                  <li><span>Judge confidence</span><strong>low / mid / high</strong></li>
                  <li><span>Jury disagreement</span><strong>rate or margin</strong></li>
                  <li><span>Rubric / prompt / model</span><strong>version IDs</strong></li>
                  <li><span>Rationale or evidence span</span><strong>audit only</strong></li>
                </ul>
                <p>A fluent rationale is not proof; preserve it for review rather than treating it as truth.</p>
              </section>
            </div>
            <div class="divider">Aggregate only after keeping the aspects</div>
            <div class="flow-row">
              <section class="card"><h3>Many response records</h3><p>One candidate across many discovery questions.</p></section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card"><h3>Candidate × topic profile</h3><p>Quality means, win rate, refusal rate, disagreement, and support count.</p></section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card green"><h3>Router features</h3><p>The new query opens its nearest topic drawer and uses the values as noisy evidence.</p></section>
            </div>
            <div class="gate"><strong>Why separate labels?</strong> A fictional long answer could have high coverage but low correctness. One overall score may hide that conflict and teach the router to reward verbosity.</div>
            """,
            "A structured judge record can keep mechanical flags, several pointwise quality aspects, pairwise outcomes for hard cases, and reliability metadata. These labels describe one response on one historical question. Aggregate many records by candidate and topic to build profile fields such as judged quality, win rate, refusal rate, disagreement, and support. Keep the aspects separate until validation so a long but incorrect response cannot hide behind one overall score.",
            "guide_notes",
        ),
        md(
            "judge_label_scale",
            f"""
            ## More labels should mean more information—not every possible judgment

            The downloaded data contains **{discovery_plain['query_count']:,} discovery questions × {discovery_plain['candidate_count']} candidates = {discovery_plain['response_count']:,} stored responses**. Deterministic screening finds {discovery_plain['exact_refusals']:,} exact “I don't know” or “I do not know” defaults, leaving at most **{discovery_plain['non_refusal_attempts']:,} non-exact attempts** for quality judging before further sampling. Three judge configurations over every stored response would create {discovery_plain['response_count'] * 3:,} evaluation cases. Comparing every unordered response pair would create **{pair_decisions_per_query:,} pairs per query and {all_pair_decisions:,} pair decisions for one judge** before order reversal or a jury.

            The efficient design is a funnel. Run deterministic quality screens across everything. Preserve default rate as a profile feature, but keep exact defaults out of semantic centroids and expensive factuality judging. Use a structured pointwise rubric on a stratified pilot of the remaining responses that covers topics, candidates, lengths, languages, context templates, and random controls. Send only ambiguous or high-value cases to pairwise comparison, with reversed order. Expand the judged set only when held-out router performance improves and slice audits remain stable.

            The totals above are exact dataset arithmetic; the funnel is a recommended experiment design, not an official requirement. Batch size and API calls may differ from the number of judgment cases.
            """,
            "guide_notes",
        ),
        teaching_visual(
            "beginner-visual-judge-label-funnel",
            "Do not begin by judging every pair",
            "The data is large enough that selective, auditable labeling is a modeling decision—not just a cost optimization.",
            f"""
            <div class="phase-grid">
              <section class="card blue"><h3>{discovery_plain['query_count']:,} questions</h3><p>Discovery questions with no trusted response grade.</p></section>
              <section class="card blue"><h3>{discovery_plain['candidate_count']} responses each</h3><p>The same candidate set answers every discovery question.</p></section>
              <section class="card gold"><h3>{discovery_plain['response_count']:,} responses</h3><p>One possible pointwise judgment unit per stored response.</p></section>
            </div>
            <div class="gate"><strong>Pairwise explosion:</strong> every pair means {pair_decisions_per_query:,} comparisons for one question and {all_pair_decisions:,} across the discovery set—for one judge and one order.</div>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card green">
              <h3>Stage 0 — screen every response cheaply</h3>
              <p>Detect empty output, exact refusal, duplicates, truncation, language, and length. The audit removes {discovery_plain['exact_refusals']:,} exact defaults from expensive quality judging and leaves at most {discovery_plain['non_refusal_attempts']:,} non-exact attempts. Keep refusal rate as profile evidence; do not call “long” relevant.</p>
            </section>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card blue">
              <h3>Stage 1 — pointwise multi-aspect pilot</h3>
              <p>Sample across candidate, topic, response length, refusal state, language, and random controls. One structured case returns separate relevance, correctness, coverage, and uncertainty labels.</p>
            </section>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card gold">
              <h3>Stage 2 — pairwise hard cases</h3>
              <p>Compare close, contradictory, or decision-important responses. Reverse A/B order and allow tie or cannot decide. Keep a random-pair control so the shortlist does not become a blind spot.</p>
            </section>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card red">
              <h3>Stage 3 — audit the judges</h3>
              <p>Use a small human- or verified-reference subset for direct response-label checks. Measure order, verbosity, language, and self-preference bias. Retain disagreement.</p>
            </section>
            <div class="down" aria-hidden="true">↓</div>
            <section class="card green">
              <h3>Stage 4 — validate the whole router</h3>
              <p>Aggregate surviving labels into profiles or response scorers. Judge success by leakage-safe development nDCG@10 and MRR, not by the judge's confidence in itself.</p>
            </section>
            <div class="callout"><strong>Expansion rule:</strong> add more judged cases where the model is uncertain, systems disagree, evidence is sparse, or a held-out slice is weak. Stop when more labels add cost without repeatable ranking gain.</div>
            """,
            f"Start with {discovery_plain['response_count']:,} stored responses, not {all_pair_decisions:,} all-pairs comparisons. Deterministic screening identifies {discovery_plain['exact_refusals']:,} exact defaults and leaves at most {discovery_plain['non_refusal_attempts']:,} non-exact attempts. Collect multi-aspect pointwise labels on a stratified pilot, send ambiguous cases to order-balanced pairwise judging, directly audit a small reference or human subset, and validate the resulting router on held-out development qrels. Expand labels where uncertainty, disagreement, sparse evidence, or weak slices show that more information is needed.",
            "guide_notes",
        ),
        teaching_visual(
            "beginner-visual-weak-label-jury",
            "Weak labels: turn imperfect opinions into an auditable training hint",
            "A weak label is a temporary practice grade, not ground truth. Trust comes from checking bias, saving every vote, and testing the signal against separate official grades.",
            """
            <div class="callout"><strong>Definition:</strong> a qrel is a trusted answer-key grade in labeled data. A weak label is a cheaper estimate used where that trusted grade is missing. It may help train the router; it never becomes factual truth.</div>
            <div class="gate"><strong>Fictional teaching example—not competition evidence:</strong> imagine a short correct answer A and a longer, polished answer B that contains one factual error.</div>

            <div class="flow-row">
              <section class="card blue">
                <h3>1. Prepare a blind case</h3>
                <ul>
                  <li>Show the question and one response, or a pair of responses.</li>
                  <li>Replace candidate names with neutral labels A and B.</li>
                  <li>Give a rubric for factual correctness and coverage.</li>
                </ul>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card gold">
                <h3>2. Ask a small jury</h3>
                <ul>
                  <li>Judge A/B, then show the same pair as B/A.</li>
                  <li>Use several prompts or genuinely different judge models.</li>
                  <li>Allow a tie or “cannot decide.”</li>
                </ul>
              </section>
              <div class="arrow" aria-hidden="true"><span class="wide">→</span><span class="narrow">↓</span></div>
              <section class="card green">
                <h3>3. Make a provisional grade</h3>
                <ul>
                  <li>Combine votes only after checking judge reliability.</li>
                  <li>Keep the vote count, margin, and disagreement.</li>
                  <li>Mark the result as a weak label, not an official qrel.</li>
                </ul>
              </section>
            </div>

            <div class="divider">Why the audit matters</div>
            <div class="compare-grid">
              <section class="card red">
                <h3>Common judge biases</h3>
                <ul>
                  <li><strong>Order:</strong> prefers the answer shown first.</li>
                  <li><strong>Verbosity:</strong> rewards length despite an error.</li>
                  <li><strong>Style or self-preference:</strong> favors familiar wording.</li>
                  <li><strong>False diversity:</strong> several prompts repeat one model's bias.</li>
                </ul>
              </section>
              <section class="card green">
                <h3>Calibration check</h3>
                <ul>
                  <li>Directly audit a small discovery subset with human or verified-reference labels where possible.</li>
                  <li>Then test the complete router on held-out development qrels.</li>
                  <li>Check failures by query type, language, length, and candidate—not only overall agreement.</li>
                  <li>Reduce or reject a source that repeatedly fails.</li>
                </ul>
              </section>
            </div>

            <div class="divider">Keep two linked records</div>
            <div class="source-grid">
              <section class="card blue">
                <h3>Immutable vote log</h3>
                <ul>
                  <li>Query and response IDs or hashes</li>
                  <li>Judge model, prompt, and rubric version</li>
                  <li>A/B presentation order and temperature</li>
                  <li>Raw output, parsed vote, ties, errors, and reference used</li>
                </ul>
              </section>
              <section class="card gold">
                <h3>Derived weak-label record</h3>
                <ul>
                  <li>Practice grade or score</li>
                  <li>Number of contributing votes</li>
                  <li>Disagreement or vote margin</li>
                  <li>Combination rule, fold, and calibration version</li>
                </ul>
              </section>
            </div>

            <div class="down" aria-hidden="true">↓</div>
            <section class="card green">
              <h3>4. Train only after the audit</h3>
              <p>Freeze the prompts, thresholds, and vote-combination rule. Then use the surviving weak labels as training hints for the router that ranks all 150 candidates.</p>
            </section>
            <div class="callout"><strong>Never overwrite the raw votes.</strong> The derived label should be reproducible from the saved evidence, and disagreement should remain available to the model or reviewer.</div>
            <div class="gate"><strong>2026 competition fit:</strong> create weak labels from historical discovery responses. A response-label audit and end-to-end qrel validation answer different questions; do both inside the training side of each evaluation fold. Outside judges, rerankers, or search are likely external; confirm the final declaration with the organizers.</div>
            """,
            "A weak label is an imperfect practice grade created where a trusted response grade is unavailable. Hide candidate identity, use a factual-correctness and coverage rubric, judge each response pair in both orders, and collect votes from more than one teacher. Preserve every raw vote because judges may prefer the first, longer, or more familiar-looking answer. Directly audit a small discovery subset with human or verified-reference labels when possible, then test the complete router on held-out development qrels. Store the raw inputs, judge setup, prompt, order, temperature, output, and errors separately from the combined practice grade, support count, disagreement, and calibration version. Weak labels can train the router, but they are never ground truth.",
            "guide_notes",
        ),
        md(
            "weak_label_lit_research",
            """
            ### Research behind the picture

            [Bradley–Terry](https://doi.org/10.1093/biomet/39.3-4.324) turns pairwise wins into a hidden, or **latent**, skill score. [InstructGPT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) is a canonical example of learning a reward model from preference judgments. [Snorkel](https://doi.org/10.14778/3157794.3157797) motivates combining several noisy labeling functions instead of declaring one imperfect proxy correct.

            LLM judges can be useful: [MT-Bench](https://arxiv.org/abs/2306.05685) and [G-Eval](https://arxiv.org/abs/2303.16634) are prominent examples. But research documents **order bias**, **verbosity bias**, and **self-preference bias**. See [Wang et al. (2023)](https://arxiv.org/abs/2305.17926), [Panickssery et al. (2024)](https://papers.nips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html), and [Replacing Judges with Juries](https://arxiv.org/abs/2404.18796).

            Recent work strengthens the case for keeping aspects separate and learning how much to trust them. [LLM-Rubric](https://aclanthology.org/2024.acl-long.745/) asks multiple rubric questions and calibrates their outputs. [MAPLE](https://aclanthology.org/2026.findings-acl.1351/) combines criterion-wise pairwise evaluations from several models while estimating criterion importance and evaluator reliability. [SAJA](https://aclanthology.org/2026.acl-industry.45/) extracts a multidimensional rubric vector in one call and maps it through a lightweight calibration head. A 2026 [rubric-position study](https://arxiv.org/abs/2602.02219) warns that even the order of score options can bias a pointwise rubric, so balance or permute rubric choices during an audit rather than assuming only A/B order matters.
            """,
        ),
        md(
            "weak_label_lit_audit",
            """
            ### Complete 2026 audit checklist

            1. Blind candidate identity by replacing candidate IDs with neutral labels.
            2. Keep relevance, factual correctness, coverage, refusal, unsupported-claim risk, and judge uncertainty as separate fields.
            3. Include `cannot verify`, `tie`, and `cannot decide`; do not force confidence.
            4. Compare hard pairs in both A/B and B/A orders—the same two answers with their positions reversed.
            5. Balance or permute the order of rubric score options during a smaller audit.
            6. Use several genuinely different model families where possible; several prompts to one model are not full jury diversity.
            7. Preserve every raw judgment, including aspect scores, disagreement, ties, errors, rationale, and evidence spans.
            8. Directly audit a small discovery subset with human or verified-reference labels when possible.
            9. Validate the downstream profile or router against held-out development qrels inside the training side of the evaluation split; qrels are an end-to-end check, not direct response labels.
            10. Record the model, prompt, rubric, **temperature**—the sampling-randomness setting—candidate order, score-option order, and any reference answer.
            11. Treat outside judges, rerankers, generated references, or search as likely external, and confirm the final declaration with the organizers.

            **Main failure mode:** without these checks, the router may learn judge bias more efficiently than it learns factual expertise.
            """,
        ),
        md(
            "uncertainty_lit",
            """
            ## Literature foundation 6: confidence should control trust, not replace truth

            **In one sentence:** confidence is a dimmer switch for trust, not a truth detector.

            **Analogy:** a weather forecast is calibrated when events called 70-percent likely occur about 70 percent of the time. A single confident forecast can still be wrong.

            **Simple algorithm:** summarize confidence and behavior; test whether confidence matches held-out correctness; trust strong local evidence more when it is stable; otherwise shrink the score toward a safer prior. The required output remains a full ranking, so uncertainty changes weighting rather than allowing abstention.

            **Where the idea comes from:** [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html) separate confidence from **calibration**, the agreement between predicted certainty and observed correctness. [Kadavath et al. (2022)](https://arxiv.org/abs/2207.05221) show that language-model self-evaluation can predict answerability, but imperfectly. [Semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) groups generations by meaning before measuring uncertainty.

            The released metadata can support mean, median, minimum, standard deviation, quantiles, length-normalized likelihood, low-confidence-token share, response length, refusal/default flags, and cluster-local versions. Use these to shrink uncertain local scores toward a prior, weight ensemble components, or flag unstable queries. The official task still requires a full ranking; abstention is diagnostic, not an output.
            """,
        ),
        md(
            "shift_lit",
            """
            ## Literature foundation 7: the hidden test may not resemble development

            **In one sentence:** a good practice score may fail when the exam asks different kinds of questions.

            **Analogy:** randomly mixing similar practice cards across study and rehearsal can feel reassuring. Holding out an entire topic is a harder rehearsal for an exam that changes topic mix.

            **Simple algorithm:** leave semantic query clusters out; separate terse facts from long research questions; measure strong and weak slices; report the worst meaningful cluster beside the overall mean. The caution is not to tune repeatedly on the same stress slices and then reuse them as supposedly fresh evidence.

            **Where the idea comes from:** the 2025 development questions and NIST-authored test questions differed sharply. [Importance-weighted cross-validation](https://www.jmlr.org/papers/v8/sugiyama07a.html), [shift detection](https://papers.nips.cc/paper_files/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html), and [WILDS](https://proceedings.mlr.press/v139/koh21a.html) all emphasize that in-domain validation can be optimistic under natural distribution shifts.

            Add stress tests that leave one semantic query cluster out; separate terse factoids from long research questions; slice by positive-candidate count (1, 2–5, 6+); report the worst cluster as well as the **macro mean**, an equal-weight average across queries; and measure distance to discovery/development neighbors. Do not tune on these slices repeatedly and then call the same overall score unbiased.
            """,
        ),
        md(
            "leakage_preview",
            """
            ## Safe validation is a rehearsal with one complete answer key locked away

            A random split of the 48,300 query–candidate rows is unsafe. It can place some candidates for one query in training and other candidates for that same query in validation, exposing pieces of one 150-grade answer key on both sides.

            Instead, hold out a whole query and every one of its 150 grades. Choose **hyperparameters**—settings selected during tuning, such as the encoder or neighbor count—only inside the remaining training queries, predict the held-out ranking once, then reveal its grades. Repeat across queries or grouped outer folds. The current dense baseline uses exact leave-one-query-out outer prediction.
            """,
            "dense_code",
        ),
        teaching_visual(
            "beginner-visual-leakage-safe-validation",
            "The rule that protects your validation score",
            "Keep every 150-row query group together, and put all tuning inside the outer holdout.",
            """
            <div class="compare-grid">
              <section class="card red">
                <h3>Wrong: random pair-row split</h3>
                <ol>
                  <li>Mix the 48,300 question–candidate rows.</li>
                  <li>Put some candidates for Query Q in training.</li>
                  <li>Put other candidates for Query Q in validation.</li>
                  <li>Expose pieces of the same 150-grade answer key on both sides.</li>
                </ol>
                <p><strong>Outcome:</strong> the score can look better than real test performance.</p>
              </section>
              <section class="card green">
                <h3>Right: hold out the whole query</h3>
                <ol>
                  <li>Choose Query Q.</li>
                  <li>Lock away all 150 of its grades.</li>
                  <li>Tune only with the other training queries.</li>
                  <li>Refit without Query Q.</li>
                  <li>Rank Query Q’s 150 candidates once.</li>
                  <li>Reveal its grades and compute the score.</li>
                </ol>
              </section>
            </div>
            <div class="cache-grid" style="margin-top:12px">
              <section class="card blue">
                <h3>May be cached globally</h3>
                <p>Frozen embeddings that never use qrels, plus profiles built only from discovery data.</p>
              </section>
              <section class="card gold">
                <h3>Must be recomputed inside the fold</h3>
                <p>Priors, supervised clusters, calibration, feature selection, tie-breaks, and selected encoder, neighbor count, or fusion weights.</p>
              </section>
            </div>
            """,
            "A random row split is unsafe because candidates from one query can appear on both sides, exposing pieces of the same answer key. Safe validation removes the whole query and all 150 grades. Hyperparameters are chosen only inside the remaining data, then the held-out ranking is predicted once before its labels are revealed.",
            "dense_code",
        ),
        md(
            "leakage_protocol",
            """
            ## The non-negotiable evaluation protocol

            1. Split by query, never by query-candidate row.
            2. In each outer fold, remove the target query’s entire 150-grade vector.
            3. Recompute priors, topic means, tie-break values, calibration, feature selection, and any supervised clusters on training queries only.
            4. Fit TF-IDF vocabulary/IDF and learned scalers inside the fold. Qrel-free frozen embeddings and discovery-only profiles may be cached globally.
            5. Select encoder, neighbor count, centroid count, response-filter threshold, shrinkage constant, feature set, and fusion weight in an inner loop.
            6. Train a final fusion model only from out-of-fold component predictions.
            7. Deduplicate or group near-duplicate discovery/development queries.
            8. Predeclare the primary configuration; label later searches exploratory.
            9. Bootstrap paired metric differences by query and show intervals, not only point estimates.
            10. Generate the final test run without manual changes after seeing test queries.

            [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) explain why model-selection overfitting can be as serious as fitting overfitting; [Kaufman et al.](https://doi.org/10.1145/2382577.2382579) give the broader leakage framing. The current dense implementation includes an exact nested loop and a mutation test showing that an outer qrel cannot affect that fold’s selection or prediction.
            """,
            "dense_code",
        ),
        {"id": "literature_map_block", "type": "table", "tableId": "literature_map", "layout": "full"},
        md(
            "recommended_sequence",
            """
            ## Build cheap evidence before expensive supervision

            The ladder below is deliberately cumulative. Every stage keeps the previous controls, creates query-level out-of-fold predictions, and earns its place through paired metric gains and failure-slice evidence. Response retrieval comes before a judge because it is cheaper, internal, and easier to debug. The judge comes before a large neural router because label quality is the bottleneck.
            """,
            "guide_notes",
        ),
        teaching_visual(
            "beginner-visual-approach-ladder",
            "A build order that teaches you something at every step",
            "This is an experiment sequence, not a claim that every later method will score higher.",
            """
            <div class="phase-grid">
              <section class="card blue">
                <h3>Phase A — questions and labels already present</h3>
                <ol class="stage-list" start="0">
                  <li><strong>Global candidate prior</strong><span>Sanity floor</span></li>
                  <li><strong>Sparse query-neighbor transfer</strong><span>Lexical control</span></li>
                  <li><strong>Dense query-neighbor transfer</strong><span>Current strongest baseline</span></li>
                  <li><strong>Query-to-candidate-name match</strong><span>Weak feature only</span></li>
                </ol>
              </section>
              <section class="card gold">
                <h3>Phase B — past responses and metadata</h3>
                <p class="gate"><strong>Discovery files are present and behavior/topic v1 profiles are built.</strong> Use those rows as an auditable control; response retrieval, semantic profiles, and confidence summaries are next.</p>
                <ol class="stage-list" start="4">
                  <li><strong>Response-level BM25/RM3</strong><span>Strong 2025-style baseline; build next</span></li>
                  <li><strong>Dense response retrieval plus profiles</strong><span>Semantic behavioral evidence; build next</span></li>
                  <li><strong>Logprob/confidence learning to rank</strong><span>Complementary features, not truth</span></li>
                </ol>
              </section>
              <section class="card green">
                <h3>Phase C — careful supervision and finalization</h3>
                <ol class="stage-list" start="7">
                  <li><strong>Weak-label hybrid router</strong><span>High upside; audit judge bias</span></li>
                  <li><strong>Nested fusion or ensemble</strong><span>Combine only out-of-fold component scores</span></li>
                </ol>
              </section>
            </div>
            """,
            "Start with a global prior, then sparse and dense similar-question retrieval, then candidate-name similarity. A first behavior and topic profile now exists for every candidate. Next add sparse response retrieval, semantic response profiles, and confidence summaries, and test whether the saved profile adds anything beyond the raw evidence. Add weak labels only after the evidence systems work. Finish by combining out-of-fold component predictions.",
            "guide_notes",
        ),
        {"id": "approach_ladder_block", "type": "table", "tableId": "approach_ladder", "layout": "full"},
        md(
            "five_runs_intro",
            """
            ## Use the five run slots to hedge model risk

            Five tiny hyperparameter variants teach less than five meaningfully different systems. Preserve at least one internal sparse run, one frozen semantic run, one metadata/LTR run, one carefully documented weak-label run, and one fusion. If a component fails late, the remaining runs still answer distinct scientific questions.
            """,
            "track_2026",
        ),
        {"id": "portfolio_block", "type": "table", "tableId": "portfolio", "layout": "full"},
        md(
            "glossary",
            """
            ## Pocket glossary

            - **Candidate model:** one fixed ID that the system must place in the ranking.
            - **Query:** the question for which candidates are ranked.
            - **Discovery set:** past questions, stored candidate responses, and metadata; it supplies behavior evidence but not official gold grades.
            - **Development set:** practice queries with qrels, used for training and leakage-safe validation.
            - **Test set:** unseen queries used for the official ranking; their labels are hidden.
            - **Qrels:** query relevance judgments—the answer-key grades for every query–candidate pair.
            - **Prior:** a default belief before local evidence, such as a candidate’s fold-local average grade.
            - **Index:** a data structure that makes stored questions or responses searchable.
            - **Sparse retrieval:** match distinctive words in a very large mostly-zero vocabulary.
            - **Dense retrieval:** match vectors that compress the overall meaning of text.
            - **BM25:** find old answers that share important words with the new question.
            - **Embedding:** place text on a meaning map; nearby points discuss similar things.
            - **kNN:** ask what worked on the most similar earlier questions.
            - **Candidate profile:** saved data made by grouping one candidate’s past response records and summarizing them; it is not automatically an expertise grade.
            - **Centroid:** a representative center of several answer embeddings; multiple centroids act like several résumé specialties.
            - **Ranking feature:** one input clue stored in a query–candidate row before the correct grade is revealed.
            - **Query-only feature:** a clue such as query length that repeats across every candidate row for one query.
            - **Candidate-only feature:** a stored profile or prior that changes by candidate but not by the new query.
            - **Interaction feature:** a query-dependent comparison such as query-to-centroid similarity or query/name overlap.
            - **Support count:** how many historical examples back a statistic; larger support usually makes a local estimate more trustworthy.
            - **Dispersion:** how spread out a candidate’s response embeddings or behavior values are.
            - **Training target:** the grade a model learns to predict or rank; it is not an input feature.
            - **Query group:** all 150 candidate rows that belong to one query and must be ranked against one another.
            - **Evidence aggregation:** matching past answers vote for the candidate that produced them.
            - **Reranker:** a second system that rescores an initial retrieved list.
            - **RRF:** Reciprocal Rank Fusion; combine list positions without assuming raw score scales match.
            - **Learning to rank:** learn how much each clue should move a candidate up or down.
            - **Weak label:** a cheap, imperfect teacher-generated score.
            - **Pointwise judgment:** grade one response against a rubric without comparing it with another response.
            - **Pairwise comparison:** ask which of two responses is better rather than assigning an absolute grade.
            - **Multi-aspect labels:** keep relevance, correctness, coverage, refusal, and uncertainty as separate fields instead of immediately collapsing them into one score.
            - **Abstention:** allow a judge to say “cannot verify” or “cannot decide” rather than forcing a label.
            - **Logprob:** a token log probability; a model-confidence clue, not correctness.
            - **Calibration:** among predictions called 70% confident, about 70% should be right.
            - **Out-of-fold prediction:** score an example using a component that did not train on that example.
            - **Nested validation:** tune settings inside an inner training loop, then evaluate once in a separate outer holdout.
            - **Ablation:** add, remove, or replace one component to learn whether it actually helps.
            - **Bootstrap interval:** repeatedly resample whole queries to estimate the uncertainty of a metric or score difference.
            - **Domain shift:** the exam questions differ from the practice questions.
            - **Leakage:** the answer key accidentally influenced training, selection, or validation.
            - **nDCG@10:** reward the right experts near the top, especially grade-2 experts.
            - **MRR:** reward how quickly the first usable expert appears.
            """,
        ),
        md(
            "limitations",
            """
            ## Limits that should change how you interpret this guide

            - The 2025 leaderboard rests on only 16 test questions with at least one relevant answer and GPT‑5 final judging.
            - Several 2025 run-to-configuration mappings are incomplete; exact claims are limited to participant papers and public run descriptions.
            - UvA participant-reported values conflict with the official evaluation and remain unreconciled.
            - Official internal/external flags conflict across summary tables, run forms, and described use of pretrained encoders or judges.
            - The two 2026 discovery JSON files are present locally and structurally valid. A reproducible builder now creates all 150 behavior/topic profiles and audits exact defaults plus token/logprob alignment. These rows still do not measure factual correctness; broader refusal language, duplicates, template families, language, and topic quality require deeper audits.
            - The 2026 dense result is cross-validated development evidence, not hidden-test performance.
            - Modern routing papers usually study fewer real models, cost-aware decisions, or test-time outputs; their gains are not directly comparable with TREC scores.
            """,
            "guide_notes",
        ),
        md(
            "next_steps",
            f"""
            ## Recommended next steps

            1. Materialize a clean response table: separate the {discovery_plain['exact_refusals']:,} exact defaults, audit broader refusal/templates/duplicates/languages, and mask the {discovery_metadata['token_logprob_mismatched']:,} token/logprob length mismatches.
            2. Add candidate-name lexical and dense similarity as a small, leakage-safe ablation.
            3. Build a filtered response-level BM25 baseline over the remaining work samples with SumScore, rank voting, and per-candidate caps.
            4. Add dense response retrieval and sparse–dense RRF while keeping the current nested dense query-kNN untouched as a control.
            5. Extract logprob/refusal/length features, then train a shallow query-grouped LambdaMART model on out-of-fold component scores.
            6. Only then pilot one structured, multi-aspect pointwise judge on a stratified subset; use order-balanced pairwise juries for ambiguous cases and validate the complete router on held-out qrels.
            7. Lock a five-run portfolio before test release and retain exact run validators and metric regression tests.
            """,
            "guide_notes",
        ),
        md(
            "further_questions",
            f"""
            ## Questions worth answering with the discovery data

            - Among the {discovery_plain['non_refusal_attempts']:,} non-exact attempts, how many are broader refusals, near-duplicates, context templates, truncated, or off-topic?
            - Does response-level BM25 still lead profile retrieval with only 150 candidates?
            - Which queries are won uniquely by sparse, dense, candidate-name, and global-prior signals?
            - Does capping evidence per candidate reduce generic-answer count bias?
            - Are local logprob profiles better than global top-percentile confidence?
            - How much performance survives leave-one-query-cluster-out evaluation?
            - Can a transparent RRF ensemble match a learned LambdaMART fusion on hidden-style stress tests?
            """,
        ),
    ]
    profile_quickstart_ids = [
        "expert_profiles_deep_dive",
        "beginner-visual-profile-anatomy",
        "beginner-visual-real-discovery-profile",
        "profile_build_explained",
        "beginner-visual-profile-build-pipeline",
        "candidate_profiles_table_block",
    ]
    profile_quickstart = [
        block for block in blocks if block["id"] in profile_quickstart_ids
    ]
    if [block["id"] for block in profile_quickstart] != profile_quickstart_ids:
        raise ValueError("profile quickstart blocks are missing or out of order")
    remaining = [
        block for block in blocks if block["id"] not in set(profile_quickstart_ids)
    ]
    reader_map_index = next(
        index for index, block in enumerate(remaining) if block["id"] == "reader_map"
    )
    return (
        remaining[: reader_map_index + 1]
        + profile_quickstart
        + remaining[reader_map_index + 1 :]
    )


def build_source_notes(
    local: dict[str, Any],
    discovery_audit: dict[str, Any],
    llm_profiles: dict[str, Any],
) -> str:
    return clean(
        f"""
        # TREC 2026 Approaches Guide — Source and QA Notes

        Generated: 2026-07-11

        ## Reporting job

        - Audience: technical beginner.
        - Primary decision: which 2025 methods and wider literature should guide a 2026 experiment sequence.
        - Delivery: one complete canonical report artifact rendered through the Data Analytics MCP app.
        - Transform: Python audits local files, streams both large discovery JSON files, materializes reviewed rows in in-memory SQLite, and executes the exact source queries stored in the artifact.
        - Comparison baseline: fold-local global prior, fixed TF-IDF query-kNN, and exact nested dense query-kNN.
        - Current evidence boundary: local development TSV/qrels, reviewed baseline outputs, and both official discovery JSON files are present. Exact defaults, token/logprob alignment, and all {llm_profiles['scope']['candidate_count']} behavior/topic profile rows are materialized; correctness, semantic centroids, response retrieval, and judge labels are not yet measured.

        ## Required-structure mapping

        - Title: visible artifact title and first heading.
        - Technical summary: “The short answer”, followed by a compact contents block, an immediate six-block profile quickstart, and then the two-minute task story.
        - Key findings with accessible evidence: twenty-one code-native teaching figures, headline metric cards, the exact 2026 baseline table, a complete 150-row profile table, and two complete 2025 run tables.
        - Scope/data/metric definitions: competition-flow visual, exact query–candidate example, worked metric explanation, task, data reality, file inventory, and expanded pocket glossary.
        - Methodology/model specification: nine-family beginner primer, a reproducible 150-row profile build plus worked row and exact lookup table, a ranking-feature module, five narrative-and-visual playbooks covering all six 2025 teams, and seven literature foundations with plain-language leads; the profile and weak-label foundations are layered into vocabulary, visual workflows, research, and practical checklists.
        - Limitations/uncertainty/robustness: prior-year evaluation caveat, leakage protocol, shift section, and limitations.
        - Recommended next steps: experiment ladder, five-run portfolio, and numbered next steps.
        - Further questions: final open-question section.

        ## Local-data audit

        - Queries: {local['query_count']}.
        - Candidates: {local['candidate_count']}.
        - Qrels: {local['qrel_count']} complete pairs.
        - Grade counts: {local['grade_counts']}.
        - Positive share: {local['positive_share']:.8f}.
        - Exactly one positive: {local['exactly_one_positive']} queries.
        - At most five positives: {local['at_most_five_positive']} queries.
        - Discovery responses: {discovery_audit['plain']['response_count']:,} across {discovery_audit['plain']['query_count']:,} queries and {discovery_audit['plain']['candidate_count']} candidates.
        - Exact normalized defaults: {discovery_audit['plain']['exact_refusals']:,} ({discovery_audit['plain']['exact_refusal_share']:.2%}).
        - Non-exact attempts: {discovery_audit['plain']['non_refusal_attempts']:,}; this is a coverage proxy, not correctness.
        - Queries with zero non-exact attempts: {discovery_audit['plain']['queries_with_zero_non_refusal_attempts']:,} ({discovery_audit['plain']['queries_with_zero_non_refusal_attempts_share']:.2%}).
        - Token/logprob length mismatches: {discovery_audit['metadata']['token_logprob_mismatched']:,} ({discovery_audit['metadata']['token_logprob_mismatch_share']:.4%}).
        - Saved candidate profiles: {llm_profiles['scope']['candidate_count']} rows, one per candidate, built from {llm_profiles['scope']['response_count']:,} response records using profile version `{llm_profiles['profile_version']}`.
        - Discovery files are present through Git-ignored shared symlinks; byte sizes and SHA-256 values match the saved audit, and the plain/metadata ordered response-tuple digests agree.

        ## Evidence-surface map

        1. `baseline_comparison` plus `baseline_display_table`
           - Question: how does nested dense query retrieval compare with TF-IDF and the prior?
           - Form: compact grouped bar followed immediately by a six-row exact lookup table, split by system and metric.
           - Takeaway: dense improves both nDCG@10 and MRR.
           - Source: `reports/trec_2026_dense_query_knn_results.json`.

        2. `runs_top10_table` and `runs_bottom9_table`
           - Question: which prior-year runs led official nDCG@10, and what did every team submit?
           - Form: two complete sortable/filterable tables with exact four-decimal nDCG@10 and MRR.
           - Takeaway: three different method families occupied the top three.
           - Source: official NIST run appendix and individual `trec_eval` files.

        3. `candidate_profiles_table`
           - Question: what was actually saved for each of the 150 current candidates?
           - Form: one sortable, paginated row per candidate with coverage, style, metadata usability, and supported attempted-query word associations.
           - Takeaway: the profile is real stored data, but version 1 describes behavior rather than correctness.
           - Source: `reports/trec_2026_llm_profiles.json`, rebuilt by `scripts/build_llm_profiles.py`.

        ## Beginner teaching-figure map

        These figures explain relationships that are harder to learn from dense prose. Nineteen are conceptual teaching diagrams; `beginner-visual-real-discovery-profile` and `beginner-visual-profile-build-pipeline` display audited quantitative values. Each is deterministic HTML/CSS with semantic text, a complete audio summary, a 16-pixel base font, dark-mode tokens, forced-color borders, and a single-column layout below 600 pixels. No raster image, remote asset, script, or second chart runtime is used.

        1. `beginner-visual-competition-flow`
           - Question: what enters the system during learning and test time?
           - Takeaway: the task ranks candidate IDs; it does not generate the answer.
           - Adjacent prose: `walking_summary`.

        2. `beginner-visual-query-candidate-pair`
           - Question: what does one qrel grade belong to?
           - Takeaway: one query is a 150-row group, and the grade belongs to each question–candidate pair.
           - Adjacent prose: `one_training_example`.

        3. `beginner-visual-metrics`
           - Question: how do MRR and nDCG@10 read a ranking differently?
           - Takeaway: MRR focuses on the first useful hit; nDCG@10 evaluates the ordered top ten.
           - Adjacent prose: `metrics_gentle_intro`.

        4. `beginner-visual-sparse-dense`
           - Question: why keep keyword and meaning retrieval?
           - Takeaway: the systems catch different clues, so fuse ranks without equating raw scores.
           - Adjacent prose: `sparse_dense_intro`.

        5. `beginner-visual-profile-anatomy`
           - Question: what can an expert profile physically contain?
           - Takeaway: text, semantic, behavioral, and topic-local profiles summarize different aspects of one candidate’s stored history.
           - Adjacent prose: `expert_profiles_deep_dive`, `profile_shapes_2025`, and `profile_recommendation_2026`.

        6. `beginner-visual-real-discovery-profile`
           - Question: what does a real candidate behavior profile reveal before embeddings or judges?
           - Takeaway: exact defaults dominate the discovery history, so cleaning, support, and metadata alignment come before semantic profiles.
           - Adjacent prose: `expert_profiles_deep_dive` and the reproducible discovery audit.

        7. `beginner-visual-profile-build-pipeline`
           - Question: how do the large discovery files become one stored row per candidate?
           - Takeaway: stream, flag, group by candidate, aggregate, and save; the output is real behavior data but not a correctness score.
           - Adjacent prose and exact lookup: `profile_build_explained` and `candidate_profiles_table`.

        8. `beginner-visual-single-vs-multi-profile`
           - Question: what does one broad profile lose compared with several supported specialty drawers?
           - Takeaway: compression trades detail for speed; preserve the response bank and shrink low-support local summaries toward a global fallback.
           - Adjacent prose: `profile_shapes_2025` and `profile_recommendation_2026`.

        9. `beginner-visual-profile-to-score`
           - Question: how does stored candidate memory become a different score for each new query?
           - Takeaway: query-to-centroid, query-to-topic, and query-to-response interactions make a fixed profile query-dependent.
           - Adjacent prose: `profile_recommendation_2026` and `ranking_features_deep_dive`.

        10. `beginner-visual-ranking-feature-row`
           - Question: what does one query–candidate feature row contain, and where is the label?
           - Takeaway: query, candidate, interaction, and evidence fields become the input row; the qrel remains a hidden training target.
           - Adjacent prose: `ranking_features_deep_dive`, `ranking_features_2025`, and `ranker_grouping_explained`.

        11. `beginner-visual-srcb-pipeline`
           - Question: how can several imperfect teachers create practice labels for a small neural scorer?
           - Takeaway: preserve and validate each teacher signal before combining it, because the student can learn judge style instead of expertise.
           - Adjacent prose: `srcb`.

        12. `beginner-visual-udinfo-pipeline`
            - Question: how do local neighbors, a global prior, and a domain cluster contribute different evidence?
            - Takeaway: steady specific but noisy local advice with broader candidate estimates; the answer-profile route is a separate method.
            - Adjacent prose: `udinfo`.

        13. `beginner-visual-genaius-pipeline`
            - Question: how does response retrieval become a candidate ranking?
            - Takeaway: filter and retrieve individual work samples, then add their scores to the candidates that produced them.
            - Adjacent prose: `genaius`.

        14. `beginner-visual-uvairlab-pipeline`
            - Question: how do 384 query fields and three candidate fields become a learned ranking?
            - Takeaway: confidence and candidate identity are supporting features, not proof of correctness or transferable identities.
            - Adjacent prose: `uvairlab`.

        15. `beginner-visual-indelab-paulphys-pipelines`
            - Question: how did a dual encoder, an engineered ensemble, and a behavioral profile summarize candidates?
            - Takeaway: model complexity and profile size cannot repair weak supervision or a coarse representation.
            - Adjacent prose: `indelab_paulphys`.

        16. `beginner-visual-router-task-shapes`
            - Question: how does this fixed 150-candidate ranking differ from common model-routing research?
            - Takeaway: borrow query and compatibility evidence, but rebuild methods that depend on live outputs, cost objectives, a smaller pool, or last year's candidates.
            - Adjacent prose: `routing_lit`, `routing_lit_research`, and `routing_lit_transfer`.

        17. `beginner-visual-judge-label-menu`
            - Question: which labels can one structured judge record keep separate?
            - Takeaway: preserve mechanical flags, quality aspects, pairwise outcomes, uncertainty, and audit metadata before deriving one training signal.
            - Adjacent prose: `judge_labels_bridge`.

        18. `beginner-visual-judge-label-funnel`
            - Question: how can the team obtain useful labels without judging every possible response pair?
            - Takeaway: screen exact defaults, pointwise-label a stratified pilot, pairwise-judge ambiguous cases, directly audit a small subset, then validate the router end to end.
            - Adjacent prose: `judge_label_scale`.

        19. `beginner-visual-weak-label-jury`
            - Question: what is a weak label, why use a jury, and what makes the result auditable?
            - Takeaway: hide candidate identity, collect reversible-order votes, preserve raw evidence, and calibrate every teacher before using its practice grades.
            - Adjacent prose: `weak_label_lit`, `weak_label_lit_research`, and `weak_label_lit_audit`.

        20. `beginner-visual-leakage-safe-validation`
            - Question: why is a random pair-row split unsafe?
            - Takeaway: lock away the whole query and all 150 grades, with tuning inside the outer holdout.
            - Adjacent prose: `leakage_preview` and the ten-step protocol.

        21. `beginner-visual-approach-ladder`
            - Question: which experiments should be built first with the data currently local?
            - Takeaway: prove question-only controls, then add discovery evidence, then weak supervision and cross-fitted fusion.
            - Adjacent prose and exact lookup: `recommended_sequence` and `approach_ladder`.

        ## Evidence policy

        Official TREC pages, overview, participant papers, and run pages are primary for 2025. Official aggregate scores override participant-side figures when they conflict. The SRCB paper does not fully map submitted runs to configurations; UDInfo method/run mapping is strongly implied rather than tabulated. UvA paper scores are unreconciled with the official value. indelab and paulphys details are labeled submitter-reported because no reviewed proceedings paper or public reproduction package was found for those runs.

        The broader literature review prioritizes original papers and proceedings. Modern LLM-routing results are treated as conceptual evidence, not score-comparable baselines.

        The focused profile/ranking-feature bridge reuses the already reviewed local companion `reports/trec_2025_mllm_interactive_report.html`. Only unique profile shapes and feature-row mechanics are combined into the canonical guide; the full old leaderboard, team recipes, caveats, and sources are not duplicated. Relative sibling links are unreliable in the enhanced portable reader and would weaken one-file portability, so the companion is linked from the repository README and final handoff instead of from inside the artifact.

        ## Quantitative source inventory

        - `data/dev/llm_dev_data.tsv`
        - `data/dev/llm_dev_qrels.txt`
        - `data/dev/llm_discovery_data.json` (Git-ignored shared local file)
        - `data/dev/llm_discovery_metadata.json` (Git-ignored shared local file)
        - `scripts/audit_discovery_profiles.py`
        - `reports/trec_2026_discovery_profile_audit.json`
        - `scripts/build_llm_profiles.py`
        - `reports/trec_2026_llm_profiles.json`
        - `reports/trec_2026_dense_query_knn_results.json`
        - `trec_mlm/dense_experiment.py`
        - https://trec.nist.gov/pubs/trec34/papers/Overview_mllm.pdf
        - https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main.html

        ## Omitted visuals

        The label distribution uses metric cards and narrative rather than a three-slice chart because one 90.9% zero slice would obscure the two small grades. The discovery-profile audit uses semantic number cards and worked profile cards rather than a two-candidate bar chart because the reader must connect each quantity to a cleaning or modeling action; it is not a leaderboard. The complete candidate-profile dataset remains a table because exact candidate lookup and field definitions matter more than a ranked comparison that could imply expertise. The dense baseline retains one required compact chart, but an exact table follows immediately because final-context QA found that the shared enhanced chart SVG exposes an unnamed keyboard `application` region. The 19-run leaderboard chart was removed entirely; its lookup is split into 10- and 9-row tables so the shared 15-row semantic fallback cannot truncate it. The literature map, file inventory, exact approach ladder, and five-run portfolio remain tables because exact lookup is the user’s task. No trend chart is appropriate: the evidence is cross-sectional, not temporal.

        AI-generated raster illustrations were considered and deliberately omitted. The selected image-generation guidance says to prefer deterministic HTML/CSS/SVG for simple educational diagrams. Code-native figures preserve exact labels, text equivalents, dark mode, phone reflow, and one-file offline delivery more reliably.

        ## Shared-reader accessibility constraints

        The enhanced reader uses each descriptive block ID as the custom-frame title and auto-sizes the frame. The generated semantic/no-script fallback currently gives every custom frame the generic outer title `Custom report content` and a 240-pixel minimum height. Each figure therefore repeats its meaning in an internal heading, semantic lists/cards, a complete audio-summary caption, and adjacent report prose. This keeps the information available even when the fallback frame title or nested scrolling is inconvenient. The limitation belongs to the shared portable renderer rather than the canonical report artifact.

        The reader exposes block IDs as DOM fragment targets, but rendered Markdown links always open a new browsing context and custom HTML frames cannot scroll the parent report. The contents block therefore uses exact visible heading names for screen-reader heading navigation or page Find, while the profile definition, build figure, and table are physically moved directly under the contents. This is the reliable artifact-only navigation path; no unsupported in-place jump behavior is claimed.

        ## Revision-preservation contract

        The earlier revisions added the profile/judge teaching blocks, reproducible discovery audit, 150-row profile data product, and judge-evaluation sources without deleting prior content. This navigation follow-up rewrites only `reader_map`, moves the six existing quickstart blocks from `expert_profiles_deep_dive` through `candidate_profiles_table_block` directly after it, and makes a narrow wording update to `profiles_features_orientation`. Every block ID, card, chart, filter, table, dataset, source, quantitative value, team detail, caveat, and research citation must remain. Relative order must remain unchanged inside the moved quickstart and inside the untouched remainder; the baseline chart still immediately precedes its exact table, and the 2025 run tables remain 10 plus 9 rows.
        """
    ) + "\n"


def main() -> None:
    local = _load_local_data()
    dense = _dense_metrics()
    discovery_audit = _discovery_profile_audit()
    llm_profiles = _llm_profiles()
    artifact = build_artifact(local, dense, discovery_audit, llm_profiles)
    ARTIFACT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    SOURCE_NOTES.write_text(
        build_source_notes(local, discovery_audit, llm_profiles), encoding="utf-8"
    )
    print(ARTIFACT)
    print(SOURCE_NOTES)


if __name__ == "__main__":
    main()
