from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from trec_mlm.response_retrieval import (
    DEFAULT_RESPONSE_CANDIDATES,
    DiscoveryRecord,
    FrozenAnalyzer,
    ResponseCandidate,
    _build_index,
    audit_response_filters,
    build_index_text,
    load_discovery_records,
    response_filter_reason,
    response_passes_filter,
    score_response_candidates,
)


def _records(
    rows: list[tuple[str, str, str, str | None]],
) -> tuple[DiscoveryRecord, ...]:
    return tuple(
        DiscoveryRecord(query_id, query, model_id, response)
        for query_id, query, model_id, response in sorted(
            rows, key=lambda row: (row[0], row[2])
        )
    )


class CandidateGridTest(unittest.TestCase):
    def test_grid_has_the_frozen_nested_order_and_stable_keys(self) -> None:
        expected = [
            (method, field, filter_name, depth)
            for method in ("bm25", "bm25_rm3")
            for field in ("response_only", "query_response")
            for filter_name in ("basic", "aggressive")
            for depth in (100, 500, 1000)
        ]

        actual = [
            (item.method, item.field, item.filter_name, item.depth)
            for item in DEFAULT_RESPONSE_CANDIDATES
        ]

        self.assertEqual(24, len(DEFAULT_RESPONSE_CANDIDATES))
        self.assertEqual(expected, actual)
        self.assertEqual(
            "bm25__response_only__basic__d100",
            DEFAULT_RESPONSE_CANDIDATES[0].key,
        )
        self.assertEqual(len({item.key for item in DEFAULT_RESPONSE_CANDIDATES}), 24)


class DiscoveryLoadingTest(unittest.TestCase):
    def test_loader_is_strict_and_sorts_query_model_pairs(self) -> None:
        payload = {
            "q2": {
                "query": "second",
                "responses": {"m2": None, "m1": "answer 21"},
            },
            "q1": {
                "query": "first",
                "responses": {"m2": "answer 12", "m1": "answer 11"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "discovery.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            corpus = load_discovery_records(path, ("m2", "m1"))

            self.assertEqual(("m1", "m2"), corpus.model_ids)
            self.assertEqual(2, corpus.source_query_count)
            self.assertEqual(4, corpus.raw_document_count)
            self.assertEqual(
                [("q1", "m1"), ("q1", "m2"), ("q2", "m1"), ("q2", "m2")],
                [(row.source_query_id, row.model_id) for row in corpus],
            )
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), corpus.source_sha256
            )

    def test_loader_rejects_ragged_models_and_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ragged = Path(directory) / "ragged.json"
            ragged.write_text(
                json.dumps({"q": {"query": "text", "responses": {"m1": "x"}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "model mismatch"):
                load_discovery_records(ragged, ("m1", "m2"))

            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text(
                '{"q":{"query":"one","query":"two","responses":{"m":"x"}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                load_discovery_records(duplicate, ("m",))


class FrozenFilterTest(unittest.TestCase):
    def test_basic_normalization_is_exact_and_deliberately_narrow(self) -> None:
        self.assertEqual("null", response_filter_reason(None, "basic"))
        self.assertEqual("blank", response_filter_reason("\u2003\n", "basic"))
        self.assertEqual(
            "exact_idk", response_filter_reason(" Ｉ  DON'T\tKNOW ", "basic")
        )
        self.assertTrue(response_passes_filter("I don:'t know", "basic"))
        self.assertTrue(response_passes_filter("I don’t know", "basic"))

    def test_aggressive_length_boundary_and_prefix_order_are_frozen(self) -> None:
        self.assertEqual("short", response_filter_reason("x" * 49, "aggressive"))
        self.assertIsNone(response_filter_reason("x" * 50, "aggressive"))
        refusal = "There is insufficient information" + " x" * 20
        self.assertGreaterEqual(len(refusal), 50)
        self.assertEqual("prefix", response_filter_reason(refusal, "aggressive"))
        # Length is applied before the prefix list, matching the official audit.
        self.assertEqual(
            "short",
            response_filter_reason("I cannot answer".ljust(49), "aggressive"),
        )

    def test_audit_outcomes_are_mutually_exclusive_and_exhaustive(self) -> None:
        records = _records(
            [
                ("q", "query", "m1", None),
                ("q", "query", "m2", "I don't know"),
                ("q", "query", "m3", "x" * 49),
                ("q", "query", "m4", "y" * 50),
            ]
        )

        basic, aggressive = audit_response_filters(records)

        self.assertEqual(
            (2, 1, 1),
            (
                basic.retained_documents,
                basic.removed_null,
                basic.removed_exact_idk,
            ),
        )
        self.assertEqual(
            (1, 1),
            (
                aggressive.retained_documents,
                aggressive.removed_short,
            ),
        )
        self.assertEqual(
            4, aggressive.retained_documents + aggressive.removed_documents
        )


class AnalyzerAndFieldTest(unittest.TestCase):
    def test_analyzer_nfkc_stopwords_stemming_and_sorted_vocabulary(self) -> None:
        analyzer = FrozenAnalyzer()
        self.assertEqual(("cat", "run"), analyzer.analyze("Ｔｈｅ cats RUNNING"))
        self.assertEqual(
            analyzer.overlap_key("Straße running"),
            analyzer.overlap_key("STRASSE runs"),
        )
        records = _records(
            [
                ("q", "source", "m1", "zebra apple"),
                ("q", "source", "m2", "banana apple"),
            ]
        )

        built = _build_index(
            records=records,
            field="response_only",
            filter_name="basic",
            analyzer=analyzer,
            show_progress=False,
        )

        self.assertEqual(("appl", "banana", "zebra"), built.vocabulary)
        self.assertEqual({"appl": 0, "banana": 1, "zebra": 2}, built.vocabulary_map)

    def test_field_construction_is_exact(self) -> None:
        record = DiscoveryRecord("q", "original query", "m", " response ")
        self.assertEqual(" response ", build_index_text(record, "response_only"))
        self.assertEqual(
            "original query\n response ",
            build_index_text(record, "query_response"),
        )


class RetrievalScoringTest(unittest.TestCase):
    def test_bm25_matches_hand_calculation_and_returns_all_models(self) -> None:
        records = _records(
            [
                ("q1", "fruit history", "mA", "apple apple banana"),
                ("q1", "fruit history", "mB", "apple carrot"),
                ("q1", "fruit history", "mC", "banana carrot"),
            ]
        )
        candidate = ResponseCandidate("bm25", "response_only", "basic", 100)

        result = score_response_candidates(
            records,
            query_ids=("target",),
            query_texts=("apple",),
            expected_model_ids=("mC", "mA", "mB"),
            candidates=(candidate,),
        )

        average_length = 7.0 / 3.0
        inverse_document_frequency = math.log(1.0 + 1.5 / 2.5)

        def score(term_frequency: float, length: float) -> float:
            denominator = 0.9 * (0.6 + 0.4 * length / average_length)
            return (
                inverse_document_frequency
                * term_frequency
                / (denominator + term_frequency)
            )

        np.testing.assert_allclose(
            result.model_scores[0, 0],
            [score(2.0, 3.0), score(1.0, 2.0), 0.0],
            rtol=1e-14,
        )
        np.testing.assert_array_equal(result.hit_counts[0, 0], [1, 1, 0])
        np.testing.assert_allclose(
            result.mean_hit_scores[0, 0], result.model_scores[0, 0]
        )
        self.assertEqual((1, 1, 3), result.model_scores.shape)
        self.assertEqual(("mA", "mB", "mC"), result.model_ids)

    def test_rm3_expansion_matches_hand_calculated_weighted_bm25(self) -> None:
        records = _records(
            [
                ("q1", "fruit one", "mA", "apple apple"),
                ("q1", "fruit one", "mB", "apple banana"),
                ("q2", "fruit two", "mA", "banana banana"),
                ("q2", "fruit two", "mB", "carrot carrot"),
            ]
        )
        rm3 = ResponseCandidate("bm25_rm3", "response_only", "basic", 100)

        result = score_response_candidates(
            records,
            query_ids=("target",),
            query_texts=("apple",),
            expected_model_ids=("mA", "mB"),
            candidates=(rm3,),
        )

        idf = math.log(2.0)
        apple_twice = idf * 2.0 / 2.9
        apple_once = idf / 1.9
        feedback_apple = apple_twice + 0.5 * apple_once
        feedback_banana = 0.5 * apple_once
        feedback_mass = feedback_apple + feedback_banana
        apple_weight = 0.5 + 0.5 * feedback_apple / feedback_mass
        banana_weight = 0.5 * feedback_banana / feedback_mass
        expected_ma = apple_weight * apple_twice + banana_weight * apple_twice
        expected_mb = apple_weight * apple_once + banana_weight * apple_once

        np.testing.assert_allclose(
            result.model_scores[0, 0], [expected_ma, expected_mb], rtol=1e-14
        )
        # RM3 retrieves mA's banana-only second document in addition to its
        # original apple document.
        np.testing.assert_array_equal(result.hit_counts[0, 0], [2, 1])

    def test_document_score_ties_use_stable_ordinal_and_sumscore_uses_fsum(
        self,
    ) -> None:
        records = _records(
            [
                ("q1", "source one", "mA", "alpha"),
                ("q1", "source one", "mB", "alpha"),
                ("q2", "source two", "mA", "alpha"),
                ("q2", "source two", "mB", "alpha"),
            ]
        )
        depth_one = ResponseCandidate("bm25", "response_only", "basic", 1)
        depth_three = ResponseCandidate("bm25", "response_only", "basic", 3)

        result = score_response_candidates(
            records,
            query_ids=("target",),
            query_texts=("alpha",),
            expected_model_ids=("mA", "mB"),
            candidates=(depth_one, depth_three),
        )

        np.testing.assert_array_equal(result.hit_counts[0, 0], [1, 0])
        np.testing.assert_array_equal(result.hit_counts[1, 0], [2, 1])
        self.assertAlmostEqual(
            result.model_scores[1, 0, 0],
            math.fsum([result.mean_hit_scores[1, 0, 0]] * 2),
        )

    def test_exact_query_guard_masks_feedback_and_final_retrieval(self) -> None:
        records = _records(
            [
                ("q1", "APPLE", "mA", "apple poison poison"),
                ("q1", "APPLE", "mB", "apple poison poison"),
                ("q2", "different source", "mA", "apple safe safe"),
                ("q2", "different source", "mB", "poison poison poison"),
            ]
        )
        candidates = (
            ResponseCandidate("bm25", "response_only", "basic", 100),
            ResponseCandidate("bm25_rm3", "response_only", "basic", 100),
        )

        result = score_response_candidates(
            records,
            query_ids=("target",),
            query_texts=("apple",),
            expected_model_ids=("mA", "mB"),
            candidates=candidates,
        )

        np.testing.assert_array_equal(result.overlap_document_counts[:, 0], [2, 2])
        # The two overlap documents never reach SumScore (final guard).
        np.testing.assert_array_equal(result.hit_counts[0, 0], [1, 0])
        # Their poison terms also never enter feedback, so the poison-only
        # non-overlap document remains ineligible (feedback guard).
        np.testing.assert_array_equal(result.hit_counts[1, 0], [1, 0])
        self.assertEqual(0.0, result.model_scores[1, 0, 1])

    def test_score_order_hashes_inventory_and_zero_vectors_are_deterministic(
        self,
    ) -> None:
        records = _records(
            [
                ("q", "source", "mA", "visible answer " + "x" * 50),
                ("q", "source", "mB", "I don't know"),
                ("q", "source", "mC", "unmatched words " + "y" * 50),
            ]
        )
        candidate = ResponseCandidate("bm25", "response_only", "basic", 100)
        kwargs = dict(
            query_ids=("t2", "t1"),
            query_texts=("visible", "outofvocabulary"),
            expected_model_ids=("mC", "mB", "mA"),
            candidates=(candidate,),
        )

        first = score_response_candidates(records, **kwargs)
        second = score_response_candidates(records, **kwargs)

        self.assertEqual(("t2", "t1"), first.query_ids)
        self.assertEqual(("mA", "mB", "mC"), first.model_ids)
        self.assertEqual((1, 2, 3), first.model_scores.shape)
        np.testing.assert_array_equal(first.model_scores[0, 1], np.zeros(3))
        np.testing.assert_array_equal(first.hit_counts[0, 1], np.zeros(3))
        self.assertEqual(first.score_tensor_sha256, second.score_tensor_sha256)
        self.assertEqual(first.configuration_sha256, second.configuration_sha256)
        self.assertEqual(first.index_diagnostics, second.index_diagnostics)
        np.testing.assert_array_equal(first.retained_document_counts[0], [1, 0, 1])
        self.assertFalse(first.model_scores.flags.writeable)


if __name__ == "__main__":
    unittest.main()
