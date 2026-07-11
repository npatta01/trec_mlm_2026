from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from trec_mlm.metrics import QueryMetrics
from trec_mlm.response_retrieval import (
    DEFAULT_RESPONSE_CANDIDATES,
    NORMALIZED_RESPONSE_CANDIDATES,
)
from trec_mlm.runio import validate_trec_run
from trec_mlm.source_query_experiment import (
    SourceQueryExperimentConfig,
    run_source_query_experiment,
    write_source_query_experiment_artifacts,
)
from trec_mlm.source_query_first import DEFAULT_SOURCE_QUERY_CANDIDATES


QUERY_IDS = ("a", "b", "c", "d", "e")
MODEL_IDS = ("m1", "m2", "m3")
GRADES = np.asarray(
    (
        (2.0, 1.0, 0.0),
        (0.0, 2.0, 1.0),
        (1.0, 0.0, 2.0),
        (2.0, 0.0, 1.0),
        (0.0, 1.0, 2.0),
    ),
    dtype=np.float64,
)


def _write_inputs(directory: Path, grades: np.ndarray = GRADES) -> tuple[Path, Path]:
    queries = directory / "queries.tsv"
    qrels = directory / "qrels.txt"
    queries.write_text(
        "".join(f"{query_id}\tquery text {query_id}\n" for query_id in QUERY_IDS),
        encoding="utf-8",
    )
    qrels.write_text(
        "".join(
            f"{query_id} 0 {model_id} {int(grades[q, m])}\n"
            for q, query_id in enumerate(QUERY_IDS)
            for m, model_id in enumerate(MODEL_IDS)
        ),
        encoding="utf-8",
    )
    return queries, qrels


def _source_scores() -> SimpleNamespace:
    candidates = DEFAULT_SOURCE_QUERY_CANDIDATES
    maximum_k = 40
    source_weights = np.zeros((3, len(QUERY_IDS), maximum_k), dtype=np.float64)
    opportunities = np.zeros((3, len(QUERY_IDS), len(MODEL_IDS)), dtype=np.int32)
    for candidate_index, candidate in enumerate(candidates):
        source_weights[candidate_index, :, : candidate.k] = 1.0 / candidate.k
        opportunities[candidate_index] = candidate.k
    selected = tuple(
        tuple(f"source_{position:03d}" for position in range(maximum_k))
        for _query_id in QUERY_IDS
    )
    shape = (len(QUERY_IDS), maximum_k, len(MODEL_IDS))
    answer_normalized = np.zeros(shape, dtype=np.float64)
    target_normalized = np.zeros(shape, dtype=np.float64)
    for query_index in range(len(QUERY_IDS)):
        aligned = (GRADES[query_index] + 0.1) / float(np.sum(GRADES[query_index] + 0.1))
        direct_values = np.roll(GRADES[query_index], 1) + 0.1
        direct = direct_values / float(np.sum(direct_values))
        answer_normalized[query_index, :20] = aligned
        answer_normalized[query_index, 20:] = np.roll(aligned, 1)
        target_normalized[query_index, :20] = direct
        target_normalized[query_index, 20:] = np.roll(direct, 1)
    primary = np.einsum(
        "cqk,qkm->cqm", source_weights, answer_normalized, dtype=np.float64
    )
    diagnostic = np.einsum(
        "cqk,qkm->cqm", source_weights, target_normalized, dtype=np.float64
    )
    return SimpleNamespace(
        candidates=candidates,
        query_ids=QUERY_IDS,
        model_ids=MODEL_IDS,
        selected_source_ids=selected,
        selected_source_keys=selected,
        source_relevance_raw=np.ones((len(QUERY_IDS), maximum_k)),
        source_weights=source_weights,
        answer_alignment_raw=answer_normalized.copy(),
        answer_alignment_normalized=answer_normalized,
        target_response_raw=target_normalized.copy(),
        target_response_normalized=target_normalized,
        model_scores=primary,
        diagnostic_model_scores=diagnostic,
        opportunity_counts=opportunities,
        valid_opportunity_counts=opportunities.copy(),
        validity_codes=np.zeros(shape, dtype=np.uint8),
        validity_reason_names=(
            "valid",
            "null",
            "blank",
            "exact_idk",
            "prefix_refusal",
        ),
        exact_mask_canonical_counts=np.zeros(len(QUERY_IDS), dtype=np.int32),
        exact_mask_source_id_counts=np.zeros(len(QUERY_IDS), dtype=np.int32),
        positive_retrieval_counts=np.full(len(QUERY_IDS), maximum_k, dtype=np.int32),
        source_weight_sums=np.ones((3, len(QUERY_IDS))),
        basic_retained_document_counts=np.asarray((1, 2, 3), dtype=np.int64),
        basic_retained_character_counts=np.asarray((10, 20, 30), dtype=np.int64),
        basic_retained_token_counts=np.asarray((2, 4, 6), dtype=np.int64),
        basic_plus_prefix_retained_document_counts=np.asarray(
            (1, 2, 3), dtype=np.int64
        ),
        basic_plus_prefix_retained_character_counts=np.asarray(
            (10, 20, 30), dtype=np.int64
        ),
        basic_plus_prefix_retained_token_counts=np.asarray((2, 4, 6), dtype=np.int64),
        inventory_filter_names=("basic", "basic_plus_prefix"),
        retained_document_counts=np.asarray(((1, 2, 3), (1, 2, 3))),
        retained_character_counts=np.asarray(((10, 20, 30), (10, 20, 30))),
        retained_token_counts=np.asarray(((2, 4, 6), (2, 4, 6))),
        canonical_audit={"canonical_source_query_count": 100},
        filter_audit={"eligible_response_cells": 15},
        filter_audits=({"eligible_response_cells": 15},),
        index_diagnostics=(),
        analyzer_diagnostics={"analyzer_sha256": "fixture"},
        source_sha256="source",
        corpus_sha256="corpus",
        canonical_query_sha256="canonical",
        response_cell_sha256="cells",
        configuration_sha256="configuration",
        signal_tensor_sha256="signals",
        score_tensor_sha256="scores",
    )


def _response_control_scores() -> SimpleNamespace:
    top10sum = tuple(
        candidate
        for candidate in NORMALIZED_RESPONSE_CANDIDATES
        if candidate.aggregation == "top10_sum"
    )
    candidates = (*DEFAULT_RESPONSE_CANDIDATES, *top10sum)
    values = np.empty((len(candidates), len(QUERY_IDS), len(MODEL_IDS)))
    for index in range(len(candidates)):
        values[index] = np.roll(GRADES, index % len(MODEL_IDS), axis=1) + 0.01
    shape = values.shape
    return SimpleNamespace(
        candidates=candidates,
        query_ids=QUERY_IDS,
        model_ids=MODEL_IDS,
        model_scores=values,
        hit_counts=np.ones(shape, dtype=np.int32),
        mean_hit_scores=values.copy(),
        overlap_document_counts=np.zeros(
            (len(candidates), len(QUERY_IDS)), dtype=np.int32
        ),
        filter_audits=(),
        inventory_filter_names=("basic", "aggressive"),
        retained_document_counts=np.asarray(((1, 2, 3), (1, 2, 3))),
        retained_character_counts=np.asarray(((10, 20, 30), (10, 20, 30))),
        retained_token_counts=np.asarray(((2, 4, 6), (2, 4, 6))),
        index_diagnostics=(),
        analyzer_diagnostics={},
        source_sha256="source",
        corpus_sha256="corpus",
        configuration_sha256="response-controls",
        score_tensor_sha256="response-score-tensor",
    )


def _dense_reference() -> dict[str, QueryMetrics]:
    return {query_id: QueryMetrics(0.5, 0.5) for query_id in QUERY_IDS}


class SourceQueryExperimentTest(unittest.TestCase):
    def test_nested_primary_and_independent_controls_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, qrels = _write_inputs(root)
            result = run_source_query_experiment(
                SourceQueryExperimentConfig(
                    query_path=str(queries),
                    qrels_path=str(qrels),
                    discovery_path=str(root / "unused.json"),
                    dense_per_query_path=None,
                    bootstrap_resamples=50,
                ),
                score_result=_source_scores(),
                response_control_score_result=_response_control_scores(),
                dense_per_query_reference=_dense_reference(),
            )

            self.assertEqual(QUERY_IDS, result.query_ids)
            self.assertEqual(MODEL_IDS, result.model_ids)
            self.assertEqual(set(QUERY_IDS), set(result.selected_config_by_outer))
            self.assertEqual(
                {
                    "source_alignment_nested",
                    "target_response_same_k",
                    "response_sumscore_nested",
                    "response_top10sum_nested",
                    "tfidf_fixed_k10",
                    "global_prior",
                    "dense_nested_reference",
                },
                set(result.evaluations),
            )
            self.assertEqual(5, len(result.comparisons))
            self.assertEqual(4, len(result.decision["gates"]))
            self.assertTrue(
                result.diagnostics["opportunities"]["every_model_receives_same_k"]
            )
            for rankings in result.rankings.values():
                self.assertTrue(
                    all(len(ranking) == len(MODEL_IDS) for ranking in rankings.values())
                )

    def test_writer_emits_six_complete_runs_and_audit_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, qrels = _write_inputs(root)
            result = run_source_query_experiment(
                SourceQueryExperimentConfig(
                    query_path=str(queries),
                    qrels_path=str(qrels),
                    discovery_path=str(root / "unused.json"),
                    dense_per_query_path=None,
                    bootstrap_resamples=20,
                ),
                score_result=_source_scores(),
                response_control_score_result=_response_control_scores(),
                dense_per_query_reference=_dense_reference(),
            )
            paths = write_source_query_experiment_artifacts(result, root / "artifacts")

            for filename in (
                "source_alignment_run.trec",
                "target_response_diagnostic.trec",
                "sumscore_control.trec",
                "top10sum_control.trec",
                "tfidf_control.trec",
                "global_prior_control.trec",
            ):
                receipt = validate_trec_run(
                    root / "artifacts" / filename,
                    expected_query_ids=QUERY_IDS,
                    expected_model_ids=MODEL_IDS,
                )
                self.assertTrue(receipt["valid"])
            for filename in (
                "config.json",
                "source_score_metadata.json",
                "diagnostics.json",
                "score_tensors.npz",
                "selected_sources.tsv",
                "inner_scores.tsv",
                "selections.tsv",
                "predictions.tsv",
                "per_query.tsv",
                "summary.json",
                "validation.json",
            ):
                self.assertIn(filename, paths)
                self.assertTrue((root / "artifacts" / filename).is_file())

    def test_outer_qrel_mutation_cannot_change_that_outer_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, qrels = _write_inputs(root)
            common = dict(
                score_result=_source_scores(),
                response_control_score_result=_response_control_scores(),
                dense_per_query_reference=_dense_reference(),
            )
            original = run_source_query_experiment(
                SourceQueryExperimentConfig(
                    query_path=str(queries),
                    qrels_path=str(qrels),
                    discovery_path=str(root / "unused.json"),
                    dense_per_query_path=None,
                    bootstrap_resamples=10,
                ),
                **common,
            )
            changed_grades = GRADES.copy()
            changed_grades[0] = (0.0, 0.0, 2.0)
            _queries, changed_qrels = _write_inputs(root, changed_grades)
            changed = run_source_query_experiment(
                SourceQueryExperimentConfig(
                    query_path=str(queries),
                    qrels_path=str(changed_qrels),
                    discovery_path=str(root / "unused.json"),
                    dense_per_query_path=None,
                    bootstrap_resamples=10,
                ),
                **common,
            )

            self.assertEqual(
                original.selected_config_by_outer["a"],
                changed.selected_config_by_outer["a"],
            )
            self.assertEqual(
                original.rankings["source_alignment_nested"]["a"],
                changed.rankings["source_alignment_nested"]["a"],
            )
            self.assertEqual(
                original.selected_priors["a"], changed.selected_priors["a"]
            )


if __name__ == "__main__":
    unittest.main()
