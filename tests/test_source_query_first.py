from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import bm25s
except ImportError:  # pragma: no cover - optional dependency environment
    bm25s = None

from trec_mlm.response_retrieval import DiscoveryRecord
from trec_mlm.source_query_first import (
    VALIDITY_BLANK,
    VALIDITY_EXACT_IDK,
    VALIDITY_NULL,
    VALIDITY_PREFIX_REFUSAL,
    VALIDITY_VALID,
    SourceQueryCandidate,
    _FrozenBM25Index,
    response_validity_code,
    score_source_query_first,
)


def _records() -> tuple[DiscoveryRecord, ...]:
    queries = {
        "s1": "alpha beta",
        "s2": "alpha gamma",
        "s3": "beta gamma",
        "s4": "alpha delta",
        # Same frozen-analyzer key as s1; s1 must be canonical.
        "s5": "ALPHA  beta",
    }
    responses = {
        "m1": lambda text: f"{text} explained",
        "m2": lambda _text: "I don't know",
        "m3": lambda _text: "topic topic topic",
    }
    return tuple(
        DiscoveryRecord(source_id, query, model_id, responses[model_id](query))
        for source_id, query in sorted(queries.items())
        for model_id in sorted(responses)
    )


class FrozenBm25Test(unittest.TestCase):
    def test_lucene_formula_and_all_empty_corpus(self) -> None:
        index = _FrozenBM25Index.build(
            ("d1", "d2", "d3"),
            (("alpha",), ("alpha", "beta"), ()),
        )
        observed = index.score(("alpha",))
        idf = math.log(1.0 + (3 - 2 + 0.5) / (2 + 0.5))
        expected_d1 = idf / (1.0 + 0.9 * (0.6 + 0.4 * 1.0))
        expected_d2 = idf / (1.0 + 0.9 * (0.6 + 0.4 * 2.0))
        np.testing.assert_allclose(observed, (expected_d1, expected_d2, 0.0))

        empty = _FrozenBM25Index.build(("a", "b"), ((), ()))
        np.testing.assert_array_equal(empty.score(("alpha",)), (0.0, 0.0))

    def test_duplicate_query_terms_accumulate_deterministically(self) -> None:
        index = _FrozenBM25Index.build(("a", "b"), (("x",), ("y",)))
        once = index.score(("x",))
        twice = index.score(("x", "x"))
        np.testing.assert_allclose(twice, once * 2.0)

    @unittest.skipUnless(bm25s is not None, "response extra is not installed")
    def test_scores_match_pinned_bm25s_lucene_backend(self) -> None:
        documents = (("alpha",), ("alpha", "beta"), ())
        ours = _FrozenBM25Index.build(("d1", "d2", "d3"), documents)
        vocabulary = tuple(sorted({term for row in documents for term in row}))
        vocabulary_map = {term: index for index, term in enumerate(vocabulary)}
        token_ids = [[vocabulary_map[term] for term in row] for row in documents]
        backend = bm25s.BM25(
            k1=0.9,
            b=0.4,
            method="lucene",
            idf_method="lucene",
            dtype="float64",
            int_dtype="int32",
            backend="numpy",
            csc_backend="numpy",
            auto_compile=False,
        )
        backend.index(
            (token_ids, vocabulary_map),
            create_empty_token=False,
            show_progress=False,
        )
        expected = np.asarray(
            backend.get_scores_from_ids([vocabulary_map["alpha"]]),
            dtype=np.float64,
        )
        np.testing.assert_allclose(ours.score(("alpha",)), expected, atol=1e-15)


class ResponseValidityTest(unittest.TestCase):
    def test_basic_plus_prefix_codes_without_short_cutoff(self) -> None:
        self.assertEqual(VALIDITY_NULL, response_validity_code(None))
        self.assertEqual(VALIDITY_BLANK, response_validity_code(" \n\t"))
        self.assertEqual(VALIDITY_EXACT_IDK, response_validity_code(" I DON'T KNOW "))
        self.assertEqual(
            VALIDITY_PREFIX_REFUSAL,
            response_validity_code(
                "The question cannot be answered from the supplied evidence."
            ),
        )
        self.assertEqual(VALIDITY_VALID, response_validity_code("Paris"))


class SourceQueryFirstScoringTest(unittest.TestCase):
    def test_separate_signals_equal_opportunities_and_canonical_queries(self) -> None:
        candidates = tuple(SourceQueryCandidate(k) for k in (1, 2, 3))
        result = score_source_query_first(
            _records(),
            query_ids=("target",),
            query_texts=("alpha topic",),
            expected_model_ids=("m3", "m1", "m2"),
            candidates=candidates,
        )

        self.assertEqual(("m1", "m2", "m3"), result.model_ids)
        self.assertEqual(5, result.canonical_audit.source_query_id_count)
        self.assertEqual(4, result.canonical_audit.canonical_source_query_count)
        self.assertEqual(1, result.canonical_audit.duplicate_key_group_count)
        self.assertEqual(1, result.canonical_audit.noncanonical_source_id_count)
        self.assertNotIn("s5", result.selected_source_ids[0])
        np.testing.assert_allclose(result.source_weight_sums[:, 0], 1.0)

        for index, k in enumerate((1, 2, 3)):
            np.testing.assert_array_equal(result.opportunity_counts[index, 0], k)
            # m2's exact-IDK cells stay in the opportunity count but never become valid.
            self.assertEqual(0, int(result.valid_opportunity_counts[index, 0, 1]))
        self.assertTrue(np.all(result.validity_codes[0, :, 1] == VALIDITY_EXACT_IDK))

        for source_position in range(3):
            alignment = result.answer_alignment_normalized[0, source_position]
            direct = result.target_response_normalized[0, source_position]
            self.assertTrue(
                math.isclose(float(alignment.sum()), 1.0, abs_tol=1e-12)
                or np.all(alignment == 0.0)
            )
            self.assertTrue(
                math.isclose(float(direct.sum()), 1.0, abs_tol=1e-12)
                or np.all(direct == 0.0)
            )
        self.assertFalse(
            np.array_equal(result.model_scores, result.diagnostic_model_scores)
        )
        self.assertFalse(result.model_scores.flags.writeable)
        self.assertFalse(result.answer_alignment_raw.flags.writeable)

    def test_all_analyzer_equivalent_source_ids_are_masked(self) -> None:
        result = score_source_query_first(
            _records(),
            query_ids=("target",),
            query_texts=("alpha beta",),
            expected_model_ids=("m1", "m2", "m3"),
            candidates=(SourceQueryCandidate(1),),
        )

        self.assertEqual(1, int(result.exact_mask_canonical_counts[0]))
        self.assertEqual(2, int(result.exact_mask_source_id_counts[0]))
        self.assertNotEqual("s1", result.selected_source_ids[0][0])
        self.assertNotEqual("s5", result.selected_source_ids[0][0])

    def test_too_few_positive_sources_fails_instead_of_padding(self) -> None:
        with self.assertRaisesRegex(ValueError, "only .* positive source queries"):
            score_source_query_first(
                _records(),
                query_ids=("target",),
                query_texts=("alpha",),
                expected_model_ids=("m1", "m2", "m3"),
                candidates=(SourceQueryCandidate(4),),
            )


if __name__ == "__main__":
    unittest.main()
