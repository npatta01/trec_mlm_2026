from __future__ import annotations

import unittest

import numpy as np

from trec_mlm.label_retrieval import (
    TASK_INSTRUCTION,
    advancement_gate,
    clean_candidate_label,
    encoder_specs_by_key,
    evaluate_label_rankings,
    format_document,
    format_query,
    normalize_embedding_rows,
    rank_label_scores,
)


class ModelRegistryTest(unittest.TestCase):
    def test_frozen_registry_has_exact_models_revisions_and_stages(self) -> None:
        specs = encoder_specs_by_key()

        self.assertEqual(
            {
                "qwen3_embedding_8b",
                "e5_mistral_7b_instruct",
                "qwen3_embedding_4b",
                "qwen3_embedding_0_6b",
                "jina_v5_small_retrieval",
                "bge_m3",
                "nomic_embed_text_v1_5",
            },
            set(specs),
        )
        self.assertEqual(
            "14492eed37e1dbcfc6613802851c85204297f305",
            specs["qwen3_embedding_8b"].revision,
        )
        self.assertEqual(1, specs["qwen3_embedding_8b"].stage)
        self.assertEqual(1, specs["e5_mistral_7b_instruct"].stage)
        self.assertEqual(2, specs["qwen3_embedding_4b"].stage)
        self.assertEqual("float32", specs["qwen3_embedding_4b"].dtype)
        self.assertEqual("float32", specs["qwen3_embedding_0_6b"].dtype)
        self.assertEqual("CC-BY-NC-4.0", specs["jina_v5_small_retrieval"].license)
        self.assertEqual("float32", specs["jina_v5_small_retrieval"].dtype)
        self.assertEqual(
            "5617a9f61b028005a4858fdac845db406aefb181",
            specs["bge_m3"].revision,
        )
        self.assertEqual("dense_cls_normalized", specs["bge_m3"].pooling)
        self.assertEqual(1024, specs["bge_m3"].embedding_dimension)
        self.assertEqual("float32", specs["bge_m3"].dtype)
        self.assertEqual("MIT", specs["bge_m3"].license)
        self.assertEqual(
            "e9b6763023c676ca8431644204f50c2b100d9aab",
            specs["nomic_embed_text_v1_5"].revision,
        )
        self.assertEqual(
            "mean_layernorm_normalized",
            specs["nomic_embed_text_v1_5"].pooling,
        )
        self.assertEqual(768, specs["nomic_embed_text_v1_5"].embedding_dimension)
        self.assertEqual("float32", specs["nomic_embed_text_v1_5"].dtype)
        self.assertEqual("Apache-2.0", specs["nomic_embed_text_v1_5"].license)

    def test_model_native_query_and_document_formats_are_exact(self) -> None:
        specs = encoder_specs_by_key()
        question = "Who founded autosegmental phonology?"

        self.assertEqual(
            f"Instruct: {TASK_INSTRUCTION}\nQuery:{question}",
            format_query(specs["qwen3_embedding_8b"], question),
        )
        self.assertEqual(
            f"Instruct: {TASK_INSTRUCTION}\nQuery: {question}",
            format_query(specs["e5_mistral_7b_instruct"], question),
        )
        self.assertEqual(
            f"Query: {question}",
            format_query(specs["jina_v5_small_retrieval"], question),
        )
        self.assertEqual(question, format_query(specs["bge_m3"], question))
        self.assertEqual(
            f"search_query: {question}",
            format_query(specs["nomic_embed_text_v1_5"], question),
        )
        self.assertEqual(
            "Linguistics",
            format_document(specs["qwen3_embedding_8b"], "Linguistics"),
        )
        self.assertEqual(
            "Document: Industries economics",
            format_document(
                specs["jina_v5_small_retrieval"],
                "Industries_(economics)",
            ),
        )
        self.assertEqual(
            "Industries economics",
            format_document(specs["bge_m3"], "Industries_(economics)"),
        )
        self.assertEqual(
            "search_document: This LLM is most likely to correctly answer questions "
            "about Industries economics.",
            format_document(
                specs["nomic_embed_text_v1_5"],
                "Industries_(economics)",
                representation="expert_capability",
            ),
        )

    def test_contextual_label_representation_is_fixed_and_qrel_free(self) -> None:
        spec = encoder_specs_by_key()["qwen3_embedding_8b"]

        self.assertEqual(
            "This LLM is most likely to correctly answer questions about "
            "Industries economics.",
            format_document(
                spec,
                "Industries_(economics)",
                representation="expert_capability",
            ),
        )
        with self.assertRaisesRegex(ValueError, "label representation"):
            format_document(spec, "Linguistics", representation="generated")


class LabelScoringTest(unittest.TestCase):
    def test_candidate_label_cleaning_is_frozen(self) -> None:
        self.assertEqual(
            "Industries economics", clean_candidate_label("Industries_(economics)")
        )
        self.assertEqual("Food and drink", clean_candidate_label("Food_and_drink"))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            clean_candidate_label("___")

    def test_normalization_rejects_invalid_rows_and_preserves_unit_rows(self) -> None:
        values = normalize_embedding_rows(np.asarray([[3.0, 4.0], [4.0, -3.0]]))
        np.testing.assert_allclose(values, [[0.6, 0.8], [0.8, -0.6]])
        self.assertEqual(np.float32, values.dtype)

        with self.assertRaisesRegex(ValueError, "zero"):
            normalize_embedding_rows(np.asarray([[0.0, 0.0]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_embedding_rows(np.asarray([[np.nan, 1.0]]))

    def test_cosine_rankings_use_model_id_for_exact_ties(self) -> None:
        query_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        label_embeddings = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        result = rank_label_scores(
            query_embeddings=query_embeddings,
            label_embeddings=label_embeddings,
            query_ids=("q1", "q2"),
            model_ids=("beta", "alpha", "gamma"),
        )

        self.assertEqual(["alpha", "beta", "gamma"], result.rankings["q1"])
        self.assertEqual(["gamma", "alpha", "beta"], result.rankings["q2"])
        np.testing.assert_allclose(result.scores[0], [1.0, 1.0, 0.0])

    def test_qrels_change_metrics_but_cannot_change_frozen_ranking(self) -> None:
        ranking = {"q1": ["a", "b"], "q2": ["b", "a"]}
        original = {
            "q1": {"a": 2.0, "b": 0.0},
            "q2": {"a": 0.0, "b": 2.0},
        }
        mutated = {
            "q1": {"a": 0.0, "b": 2.0},
            "q2": {"a": 2.0, "b": 0.0},
        }

        first = evaluate_label_rankings(ranking, original)
        second = evaluate_label_rankings(ranking, mutated)

        self.assertEqual(ranking, first.rankings)
        self.assertEqual(ranking, second.rankings)
        self.assertNotEqual(
            first.evaluation.macro_ndcg_at_k, second.evaluation.macro_ndcg_at_k
        )


class AdvancementGateTest(unittest.TestCase):
    def test_gate_requires_positive_interval_and_no_extra_zeros(self) -> None:
        baseline = {
            "q1": {"a": 2.0, "b": 0.0},
            "q2": {"a": 2.0, "b": 0.0},
            "q3": {"a": 2.0, "b": 0.0},
        }
        baseline_rankings = {
            "q1": ["b", "a"],
            "q2": ["b", "a"],
            "q3": ["b", "a"],
        }
        candidate_rankings = {
            "q1": ["a", "b"],
            "q2": ["a", "b"],
            "q3": ["a", "b"],
        }
        baseline_result = evaluate_label_rankings(baseline_rankings, baseline)
        candidate_result = evaluate_label_rankings(candidate_rankings, baseline)

        gate = advancement_gate(
            baseline=baseline_result,
            candidate=candidate_result,
            maximum_zero_ndcg=0,
            n_resamples=100,
            seed=7,
        )

        self.assertTrue(gate["passed"])
        self.assertTrue(gate["ndcg_ci_lower_above_zero"])
        self.assertTrue(gate["zero_count_within_limit"])
        self.assertEqual(3, gate["ndcg_win_tie_loss"]["wins"])
        self.assertGreater(gate["mrr_delta"]["lower"], 0.0)
        self.assertEqual(3, gate["mrr_win_tie_loss"]["wins"])
        self.assertEqual(
            {
                "baseline_zero_candidate_zero": 0,
                "baseline_zero_candidate_nonzero": 0,
                "baseline_nonzero_candidate_zero": 0,
                "baseline_nonzero_candidate_nonzero": 3,
            },
            gate["zero_ndcg_transitions"],
        )


if __name__ == "__main__":
    unittest.main()
