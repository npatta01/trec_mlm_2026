from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from trec_mlm.metrics import QueryMetrics
from trec_mlm.response_normalization import (
    ResponseNormalizationConfig,
    run_response_normalization_experiment,
    write_response_normalization_artifacts,
)
from trec_mlm.response_retrieval import (
    ALL_RESPONSE_CANDIDATES,
    DEFAULT_RESPONSE_CANDIDATES,
    NORMALIZED_RESPONSE_CANDIDATES,
)
from trec_mlm.runio import validate_trec_run


QUERY_IDS = ("a", "b", "c", "d", "e")
MODEL_IDS = ("m1", "m2", "m3")
GRADES = np.asarray(
    [
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
RAW_SCORES = np.asarray(
    [
        [3.0, 0.0, 1.0],
        [0.0, 4.0, 1.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass
class FakeCombinedScoreResult:
    candidates: tuple
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    model_scores: np.ndarray
    hit_counts: np.ndarray
    mean_hit_scores: np.ndarray
    overlap_document_counts: np.ndarray
    inventory_filter_names: tuple[str, ...]
    retained_document_counts: np.ndarray
    retained_character_counts: np.ndarray
    retained_token_counts: np.ndarray
    corpus_sha256: str = "corpus"
    configuration_sha256: str = "configuration"
    score_tensor_sha256: str = "score"


def fake_combined_scores() -> FakeCombinedScoreResult:
    scores = np.stack([RAW_SCORES.copy() for _ in ALL_RESPONSE_CANDIDATES])
    hits = (scores > 0.0).astype(np.int32)
    return FakeCombinedScoreResult(
        candidates=ALL_RESPONSE_CANDIDATES,
        query_ids=QUERY_IDS,
        model_ids=MODEL_IDS,
        model_scores=scores,
        hit_counts=hits,
        mean_hit_scores=scores.copy(),
        overlap_document_counts=np.zeros(
            (len(ALL_RESPONSE_CANDIDATES), len(QUERY_IDS)), dtype=np.int32
        ),
        inventory_filter_names=("basic", "aggressive"),
        retained_document_counts=np.asarray([[10, 20, 30], [8, 16, 24]]),
        retained_character_counts=np.asarray(
            [[100, 200, 300], [80, 160, 240]]
        ),
        retained_token_counts=np.asarray([[20, 40, 60], [16, 32, 48]]),
    )


class ResponseNormalizationExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.query_path = root / "queries.tsv"
        self.qrels_path = root / "qrels.txt"
        self.discovery_path = root / "discovery.json"
        self.query_path.write_text(
            "".join(f"{query_id}\tquery {query_id}\n" for query_id in QUERY_IDS),
            encoding="utf-8",
        )
        self.qrels_path.write_text(
            "".join(
                f"{query_id} 0 {model_id} {grade:g}\n"
                for query_id, row in zip(QUERY_IDS, GRADES, strict=True)
                for model_id, grade in zip(MODEL_IDS, row, strict=True)
            ),
            encoding="utf-8",
        )
        self.discovery_path.write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _config(self) -> ResponseNormalizationConfig:
        return ResponseNormalizationConfig(
            query_path=str(self.query_path),
            qrels_path=str(self.qrels_path),
            discovery_path=str(self.discovery_path),
            dense_per_query_path=None,
            bootstrap_resamples=20,
            sumscore_reference_results_path=None,
            sumscore_reference_run_path=None,
        )

    @staticmethod
    def _dense_reference() -> dict[str, QueryMetrics]:
        return {query_id: QueryMetrics(0.25, 0.5) for query_id in QUERY_IDS}

    def _run(self):
        return run_response_normalization_experiment(
            self._config(),
            score_result=fake_combined_scores(),
            dense_per_query_reference=self._dense_reference(),
        )

    def test_sumscore_and_normalized_selectors_are_independent_and_ordered(self) -> None:
        result = self._run()

        self.assertEqual(
            tuple(candidate.key for candidate in NORMALIZED_RESPONSE_CANDIDATES),
            tuple(candidate.key for candidate in result.normalized.candidates),
        )
        self.assertEqual(
            tuple(candidate.key for candidate in DEFAULT_RESPONSE_CANDIDATES),
            tuple(candidate.key for candidate in result.sumscore.candidates),
        )
        self.assertEqual(
            {NORMALIZED_RESPONSE_CANDIDATES[0].key},
            set(result.normalized.selected_config_by_outer.values()),
        )
        self.assertEqual(
            {DEFAULT_RESPONSE_CANDIDATES[0].key},
            set(result.sumscore.selected_config_by_outer.values()),
        )
        self.assertTrue(
            all(
                "__agg_" in key
                for key in result.normalized.selected_config_by_outer.values()
            )
        )
        self.assertNotEqual(
            result.normalized.retrieval_metadata["score_tensor_sha256"],
            result.sumscore.retrieval_metadata["score_tensor_sha256"],
        )
        self.assertEqual(
            "score",
            result.summary["retrieval_reproducibility"]["combined"][
                "score_tensor_sha256"
            ],
        )

    def test_identical_score_fixture_has_exact_zero_sumscore_delta(self) -> None:
        result = self._run()
        comparison = result.comparisons["response_sumscore_nested"]

        self.assertEqual(
            result.rankings["response_normalized_nested"],
            result.rankings["response_sumscore_nested"],
        )
        self.assertEqual(0.0, comparison["ndcg_at_10_delta"]["mean_difference"])
        self.assertEqual(0.0, comparison["mrr_delta"]["mean_difference"])
        self.assertEqual(4, len(result.decision["gates"]))
        self.assertFalse(result.summary["sumscore_reference_validation"]["applicable"])

    def test_locked_candidate_order_cannot_be_overridden(self) -> None:
        config = replace(
            self._config(),
            normalized_candidates=tuple(reversed(NORMALIZED_RESPONSE_CANDIDATES)),
        )
        with self.assertRaisesRegex(ValueError, "frozen 144-candidate grid"):
            run_response_normalization_experiment(
                config,
                score_result=fake_combined_scores(),
                dense_per_query_reference=self._dense_reference(),
            )

    def test_writer_emits_combined_runs_evidence_and_validation(self) -> None:
        result = self._run()
        output = Path(self.directory.name) / "normalization"

        paths = write_response_normalization_artifacts(result, output)

        for name in (
            "normalized_run.trec",
            "sumscore_control.trec",
            "tfidf_control.trec",
            "global_prior_control.trec",
        ):
            receipt = validate_trec_run(
                paths[name],
                expected_query_ids=QUERY_IDS,
                expected_model_ids=MODEL_IDS,
            )
            self.assertEqual(15, receipt["row_count"])
        validation = json.loads(Path(paths["validation.json"]).read_text())
        self.assertEqual(144, validation["inputs"]["normalized_candidate_count"])
        self.assertEqual(24, validation["inputs"]["sumscore_candidate_count"])
        self.assertEqual(168, validation["inputs"]["combined_candidate_count"])
        self.assertTrue(validation["protocol"]["normalized_selector_excludes_sumscore"])
        self.assertIn("code_revision", validation["execution_provenance"])
        self.assertIn(
            "response_normalization.py",
            validation["execution_provenance"]["code_sha256"],
        )
        self.assertEqual(10, len(validation["artifact_sha256"]))
        self.assertTrue((output / "normalized" / "inner_scores.tsv").is_file())
        self.assertTrue((output / "sumscore" / "inner_scores.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
