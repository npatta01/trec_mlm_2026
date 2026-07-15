#!/usr/bin/env python3
"""Build the issue-12 all-pairs semantic-profile query explorer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from math import isfinite
from pathlib import Path
from typing import Any

from trec_mlm.profile_evaluation import validate_qrel_inventories
from trec_mlm.profile_experiment import load_profile_artifacts
from trec_mlm.query_knn import load_complete_qrels, load_queries_tsv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_rows(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "query_id",
            "candidate_id",
            "rank",
            "score",
            "profile_score",
            "winning_centroid_id",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"unexpected prediction columns in {path}")
        for row in reader:
            query_id = row["query_id"]
            candidate_id = row["candidate_id"]
            query_rows = output.setdefault(query_id, {})
            if candidate_id in query_rows:
                raise ValueError(f"duplicate prediction {query_id}/{candidate_id}")
            rank = int(row["rank"])
            score = float(row["score"])
            profile_score = float(row["profile_score"])
            if rank < 1 or not isfinite(score) or not isfinite(profile_score):
                raise ValueError(f"invalid prediction {query_id}/{candidate_id}")
            query_rows[candidate_id] = {
                "rank": rank,
                "score": score,
                "profile_score": profile_score,
                "winning_centroid_id": int(row["winning_centroid_id"]),
            }
    return output


def _cluster_details(
    profiles_path: Path,
    records_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], tuple[str, ...]]:
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    record_lookup = {item["record_id"]: item for item in records}
    if len(record_lookup) != len(records):
        raise ValueError("profile records are not uniquely keyed")
    output: dict[str, list[dict[str, Any]]] = {}
    for candidate_id, profile in profiles.items():
        if profile.get("available") is not True:
            output[candidate_id] = []
            continue
        clusters: list[dict[str, Any]] = []
        for cluster in profile["clusters"]:
            representative_questions = cluster["representative_questions"]
            wanted = set(representative_questions)
            responses: dict[str, list[str]] = {question: [] for question in wanted}
            for record_id in cluster["member_record_ids"]:
                record = record_lookup[record_id]
                question = record["question"]
                if question in wanted:
                    responses[question].append(record["response"])
            unresolved = {
                question: len(matches)
                for question, matches in responses.items()
                if len(matches) != 1
            }
            if unresolved:
                raise ValueError(
                    f"representative questions do not resolve exactly once for "
                    f"{candidate_id}/{cluster['cluster_id']}: {unresolved}"
                )
            representatives = [
                {
                    "question": question,
                    "response": responses[question][0],
                }
                for question in representative_questions
            ]
            annotation = cluster["annotation"]
            clusters.append(
                {
                    "cluster_id": int(cluster["cluster_id"]),
                    "support_count": int(cluster["support_count"]),
                    "dispersion": float(cluster["dispersion"]),
                    "medoid_record_id": cluster["medoid_record_id"],
                    "label_status": annotation["status"],
                    "primary_label": annotation["primary_label"],
                    "secondary_label": annotation["secondary_label"],
                    "label_margin": float(annotation["margin"]),
                    "representatives": representatives,
                }
            )
        if [item["cluster_id"] for item in clusters] != list(range(len(clusters))):
            raise ValueError(f"noncanonical cluster ordering for {candidate_id}")
        output[candidate_id] = clusters
    return output, tuple(profiles)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    construction = args.construction
    validated = load_profile_artifacts(construction)
    manifest = validated.manifest
    primary_path = construction / "predictions" / "dense_profile_equal.tsv"
    profile_path = construction / "predictions" / "profile_only.tsv"
    primary = _prediction_rows(primary_path)
    profile = _prediction_rows(profile_path)
    queries = load_queries_tsv(args.queries)
    qrels = load_complete_qrels(args.qrels, expected_query_ids=queries)
    inputs = json.loads((construction / "inputs.json").read_text(encoding="utf-8"))
    candidate_ids = tuple(inputs["candidate_ids"])
    query_ids = tuple(inputs["query_ids"])
    if candidate_ids != validated.candidate_ids or query_ids != validated.query_ids:
        raise ValueError("construction inputs differ from validated artifact inventory")
    if query_ids != tuple(queries):
        raise ValueError("query order differs from frozen construction")
    validate_qrel_inventories(
        qrels,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )
    dense_rankings = json.loads(
        (construction / "rankings" / "dense_reference.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if (
        evaluation.get("status") != "evaluated"
        or evaluation.get("construction_unchanged") is not True
        or evaluation.get("construction_aggregate_sha256")
        != manifest["aggregate_sha256"]
    ):
        raise ValueError("evaluation is not bound to the validated construction")
    clusters, profile_candidate_ids = _cluster_details(
        construction / "profiles.json",
        construction / "records.json",
    )
    if set(profile_candidate_ids) != set(candidate_ids):
        raise ValueError("profile candidates differ from construction inventory")

    query_payload: list[dict[str, Any]] = []
    pair_count = 0
    winning_counts: dict[str, int] = {}
    if set(primary) != set(query_ids) or set(profile) != set(query_ids):
        raise ValueError("prediction query inventory differs from construction")
    if set(dense_rankings) != set(query_ids):
        raise ValueError("dense ranking query inventory differs from construction")
    for query_id in query_ids:
        if set(primary[query_id]) != set(candidate_ids) or set(profile[query_id]) != set(
            candidate_ids
        ):
            raise ValueError(f"incomplete pair inventory for {query_id}")
        dense_positions = {
            candidate_id: rank
            for rank, candidate_id in enumerate(dense_rankings[query_id], start=1)
        }
        if (
            len(dense_rankings[query_id]) != len(candidate_ids)
            or set(dense_positions) != set(candidate_ids)
        ):
            raise ValueError(f"dense ranking inventory differs for {query_id}")
        expected_ranks = set(range(1, len(candidate_ids) + 1))
        if (
            {row["rank"] for row in primary[query_id].values()} != expected_ranks
            or {row["rank"] for row in profile[query_id].values()} != expected_ranks
        ):
            raise ValueError(f"prediction ranks are not complete for {query_id}")
        rows: list[dict[str, Any]] = []
        for candidate_id in candidate_ids:
            primary_row = primary[query_id][candidate_id]
            profile_row = profile[query_id][candidate_id]
            if (
                primary_row["winning_centroid_id"]
                != profile_row["winning_centroid_id"]
                or primary_row["profile_score"] != profile_row["profile_score"]
            ):
                raise ValueError(f"profile handoff changed for {query_id}/{candidate_id}")
            centroid_id = primary_row["winning_centroid_id"]
            if centroid_id < 0 or centroid_id >= len(clusters[candidate_id]):
                raise ValueError(f"invalid winner for {query_id}/{candidate_id}")
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "primary_rank": primary_row["rank"],
                    "primary_score": primary_row["score"],
                    "profile_rank": profile_row["rank"],
                    "profile_score": profile_row["profile_score"],
                    "dense_rank": dense_positions[candidate_id],
                    "winning_centroid_id": centroid_id,
                    "grade": float(qrels[query_id][candidate_id]),
                }
            )
            pair_count += 1
            key = f"{candidate_id}:{centroid_id}"
            winning_counts[key] = winning_counts.get(key, 0) + 1
        rows.sort(key=lambda item: (item["primary_rank"], item["candidate_id"]))
        dense_metrics = evaluation["per_query"]["dense_reference"][query_id]
        primary_metrics = evaluation["per_query"]["dense_profile_equal"][query_id]
        positive_count = sum(value > 0.0 for value in qrels[query_id].values())
        query_payload.append(
            {
                "query_id": query_id,
                "query_text": queries[query_id],
                "positive_count": positive_count,
                "dense_ndcg_at_10": dense_metrics["ndcg_at_k"],
                "dense_reciprocal_rank": dense_metrics["reciprocal_rank"],
                "primary_ndcg_at_10": primary_metrics["ndcg_at_k"],
                "primary_reciprocal_rank": primary_metrics["reciprocal_rank"],
                "pairs": rows,
            }
        )
    if pair_count != 48_300:
        raise ValueError(f"expected 48,300 pairs, got {pair_count}")

    return {
        "meta": {
            "title": "Candidate semantic profiles — all-pairs query explorer",
            "query_count": len(query_payload),
            "candidate_count": len(candidate_ids),
            "pair_count": pair_count,
            "cluster_count": sum(len(items) for items in clusters.values()),
            "labels_included": True,
            "failure_filters_included": True,
            "construction_aggregate_sha256": manifest["aggregate_sha256"],
            "primary_predictions_sha256": _sha256(primary_path),
            "profile_predictions_sha256": _sha256(profile_path),
            "qrels_usage": "Post-freeze evaluation and failure inspection only",
            "qrel_free_scope": (
                "New profile representations, profile scoring, and fusion choices were qrel-free. "
                "The primary consumes the frozen qrel-derived leave-one-query-out query-kNN baseline."
            ),
        },
        "candidate_ids": list(candidate_ids),
        "clusters": clusters,
        "queries": query_payload,
        "winning_centroid_pair_counts": winning_counts,
    }


STYLE = r"""
:root{color-scheme:light dark;--bg:#f6f8fb;--panel:#fff;--ink:#172033;--muted:#5c687d;--line:#dce2eb;--blue:#0b4f8a;--blue2:#e7f1fb;--gold:#8a5a00;--gold2:#fff5d6;--red:#9d2b2b;--red2:#fdeaea}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}header,main{max-width:1500px;margin:auto;padding:24px}h1{font-size:clamp(1.7rem,4vw,2.6rem);line-height:1.1;margin:.2rem 0}h2{font-size:1.25rem}.lede{color:var(--muted);max-width:1050px}.pills{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}.pill{background:var(--blue2);color:var(--blue);padding:5px 10px;border-radius:999px;font-weight:650}.controls,.layout,.query-metrics{display:grid;gap:12px}.controls{grid-template-columns:2fr 1fr 1fr 1fr;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;position:sticky;top:0;z-index:5}.control label{display:block;font-weight:700;font-size:.82rem;margin-bottom:4px}.control input,.control select{width:100%;min-height:42px;border:1px solid var(--line);border-radius:8px;padding:8px;background:var(--panel);color:var(--ink)}.layout{grid-template-columns:minmax(240px,330px) minmax(0,1fr);margin-top:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px}.query-list{padding:8px;max-height:78vh;overflow:auto}.query-button{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid var(--line);background:transparent;color:var(--ink);padding:10px;border-radius:8px}.query-button:hover,.query-button[aria-current=true]{background:var(--blue2)}.query-button strong,.query-button span{display:block}.query-button span{color:var(--muted);font-size:.8rem}.detail{padding:18px;min-width:0}.query-metrics{grid-template-columns:repeat(5,minmax(110px,1fr));margin:12px 0}.metric{padding:10px;border:1px solid var(--line);border-radius:10px}.metric small{display:block;color:var(--muted)}.metric strong{font-size:1.1rem}.notice{background:var(--gold2);color:var(--gold);padding:10px;border-radius:9px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse;min-width:1050px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}th{position:sticky;top:0;background:var(--panel);z-index:2;font-size:.8rem}td.num{font-variant-numeric:tabular-nums;text-align:right}.grade-positive{font-weight:800;color:var(--blue)}details{min-width:270px}summary{cursor:pointer;font-weight:700;color:var(--blue)}.cluster{padding:10px;background:var(--blue2);border-radius:8px;margin-top:8px}.cluster dl{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;margin:0}.cluster dt{font-weight:700}.cluster dd{margin:0}.representative{margin-top:8px;padding-top:8px;border-top:1px solid color-mix(in srgb,var(--blue) 20%,transparent)}.representative p{margin:4px 0}.empty{padding:24px;color:var(--muted)}@media(max-width:900px){header,main{padding:14px}.controls{position:static;grid-template-columns:1fr 1fr}.layout{grid-template-columns:1fr}.query-list{max-height:260px}.query-metrics{grid-template-columns:1fr 1fr}}@media(max-width:520px){.controls{grid-template-columns:1fr}.query-metrics{grid-template-columns:1fr}.detail{padding:12px}}@media(prefers-color-scheme:dark){:root{--bg:#10141d;--panel:#171d28;--ink:#e8eef9;--muted:#aeb8ca;--line:#30394a;--blue:#8dc8ff;--blue2:#172c40;--gold:#ffd06f;--gold2:#3b2f13;--red:#ffaaaa;--red2:#421f24}}
"""


SCRIPT = r"""
const data=JSON.parse(document.getElementById('explorer-data').textContent);const byId=id=>document.getElementById(id);const state={query:null,querySearch:'',failure:'all',candidateSearch:'',sort:'primary_rank'};const fmt=(v,n=4)=>Number(v).toFixed(n);const isZero=v=>Math.abs(Number(v))<1e-15;function matchesFailure(q){if(state.failure==='all')return true;if(state.failure==='primary_zero')return isZero(q.primary_ndcg_at_10);if(state.failure==='rescue')return isZero(q.dense_ndcg_at_10)&&!isZero(q.primary_ndcg_at_10);if(state.failure==='regression')return !isZero(q.dense_ndcg_at_10)&&isZero(q.primary_ndcg_at_10);if(state.failure==='zero_both')return isZero(q.dense_ndcg_at_10)&&isZero(q.primary_ndcg_at_10);if(state.failure==='mid_density')return q.positive_count>=6&&q.positive_count<=20;return true}function filteredQueries(){const needle=state.querySearch.toLowerCase();return data.queries.filter(q=>matchesFailure(q)&&(!needle||q.query_id.toLowerCase().includes(needle)||q.query_text.toLowerCase().includes(needle))).sort((a,b)=>a.primary_ndcg_at_10-b.primary_ndcg_at_10||a.query_id.localeCompare(b.query_id))}function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node}function renderList(){const root=byId('query-list');root.replaceChildren();const items=filteredQueries();byId('query-count').textContent=`${items.length} matching queries`;if(!items.length){root.append(el('p','No matching queries.','empty'));return}if(!state.query||!items.some(q=>q.query_id===state.query))state.query=items[0].query_id;for(const q of items){const b=el('button',undefined,'query-button');b.type='button';b.setAttribute('aria-current',String(q.query_id===state.query));b.append(el('strong',q.query_text));b.append(el('span',`${q.query_id} · primary nDCG ${fmt(q.primary_ndcg_at_10)} · dense ${fmt(q.dense_ndcg_at_10)}`));b.onclick=()=>{state.query=q.query_id;renderList();renderDetail()};root.append(b)}renderDetail()}function metric(label,value){const box=el('div',undefined,'metric');box.append(el('small',label));box.append(el('strong',value));return box}function clusterDetail(candidateId,centroidId){const c=data.clusters[candidateId][centroidId];const d=el('details');const labels=c.primary_label?(c.primary_label+(c.secondary_label?` / ${c.secondary_label}`:'')):'Mixed / unclear';const s=el('summary',`Centroid ${centroidId}: ${labels}`);d.append(s);const box=el('div',undefined,'cluster');const dl=el('dl');for(const [k,v] of [['Support',c.support_count],['Dispersion',fmt(c.dispersion,6)],['Label status',c.label_status],['Primary label',c.primary_label||'Mixed / unclear'],['Secondary label',c.secondary_label||'—'],['Label margin',fmt(c.label_margin,6)]]){dl.append(el('dt',k));dl.append(el('dd',String(v)))}box.append(dl);for(const r of c.representatives){const rep=el('div',undefined,'representative');const q=el('p');q.append(el('strong','Question: '));q.append(document.createTextNode(r.question));const a=el('p');a.append(el('strong','Response: '));a.append(document.createTextNode(r.response));rep.append(q,a);box.append(rep)}d.append(box);return d}function renderDetail(){const q=data.queries.find(item=>item.query_id===state.query);const root=byId('detail');root.replaceChildren();if(!q){root.append(el('p','Select a query.','empty'));return}root.append(el('h2',q.query_text));root.append(el('p',q.query_id,'lede'));const metrics=el('div',undefined,'query-metrics');metrics.append(metric('Primary nDCG@10',fmt(q.primary_ndcg_at_10)),metric('Query-kNN nDCG@10',fmt(q.dense_ndcg_at_10)),metric('Primary MRR',fmt(q.primary_reciprocal_rank)),metric('Query-kNN MRR',fmt(q.dense_reciprocal_rank)),metric('Positive candidates',String(q.positive_count)));root.append(metrics);const rows=q.pairs.filter(r=>!state.candidateSearch||r.candidate_id.toLowerCase().includes(state.candidateSearch.toLowerCase())).slice();rows.sort((a,b)=>state.sort==='candidate'?a.candidate_id.localeCompare(b.candidate_id):a[state.sort]-b[state.sort]||a.candidate_id.localeCompare(b.candidate_id));root.append(el('p',`${rows.length} of 150 candidates shown. Grades are post-freeze development labels.`,'notice'));const wrap=el('div',undefined,'table-wrap');const table=el('table');const thead=el('thead');const trh=el('tr');for(const h of ['Primary rank','Candidate','Grade','Primary score','Profile rank','Profile score','Query-kNN rank','Winning profile centroid'])trh.append(el('th',h));thead.append(trh);table.append(thead);const tbody=el('tbody');for(const r of rows){const tr=el('tr');for(const [text,cls] of [[r.primary_rank,'num'],[r.candidate_id,''],[r.grade,r.grade>0?'num grade-positive':'num'],[fmt(r.primary_score,8),'num'],[r.profile_rank,'num'],[fmt(r.profile_score,8),'num'],[r.dense_rank,'num']])tr.append(el('td',String(text),cls));const td=el('td');td.append(clusterDetail(r.candidate_id,r.winning_centroid_id));tr.append(td);tbody.append(tr)}table.append(tbody);wrap.append(table);root.append(wrap)}byId('query-search').oninput=e=>{state.querySearch=e.target.value;renderList()};byId('failure-filter').onchange=e=>{state.failure=e.target.value;state.query=null;renderList()};byId('candidate-search').oninput=e=>{state.candidateSearch=e.target.value;renderDetail()};byId('sort-candidates').onchange=e=>{state.sort=e.target.value;renderDetail()};renderList();
"""


def render_html(payload: dict[str, Any]) -> str:
    title = payload["meta"]["title"]
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><title>{html.escape(title)}</title><style>{STYLE}</style></head>
<body><header><h1>{html.escape(title)}</h1><p class="lede">Inspect every frozen query–candidate pair and the new multi-vector centroid that won its profile score. New profile representations, scoring, and fusion choices were qrel-free; the primary consumes the frozen qrel-derived leave-one-query-out query-kNN baseline. Grades appear only for post-freeze evaluation and failure inspection.</p><div class="pills"><span class="pill">322 queries</span><span class="pill">150 candidates</span><span class="pill">48,300 pairs</span><span class="pill">1,500 centroids</span></div></header>
<main><section class="controls" aria-label="Explorer filters"><div class="control"><label for="query-search">Find query</label><input id="query-search" type="search" placeholder="Question text or query ID"></div><div class="control"><label for="failure-filter">Failure slice</label><select id="failure-filter"><option value="all">All queries</option><option value="primary_zero">Primary zero only</option><option value="rescue">Primary rescues</option><option value="regression">Primary regressions</option><option value="zero_both">Zero in both</option><option value="mid_density">6–20 positives</option></select></div><div class="control"><label for="candidate-search">Find candidate</label><input id="candidate-search" type="search" placeholder="Candidate ID"></div><div class="control"><label for="sort-candidates">Sort candidates</label><select id="sort-candidates"><option value="primary_rank">Primary rank</option><option value="profile_rank">Profile rank</option><option value="dense_rank">Query-kNN rank</option><option value="candidate">Candidate ID</option></select></div></section>
<div class="layout"><aside class="panel"><p id="query-count" class="lede" style="padding:0 12px"></p><nav id="query-list" class="query-list" aria-label="Queries"></nav></aside><section id="detail" class="panel detail" aria-live="polite"></section></div></main>
<script id="explorer-data" type="application/json">{embedded}</script><script>{SCRIPT}</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload), encoding="utf-8")
    print(args.output.resolve())
    print(json.dumps(payload["meta"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
