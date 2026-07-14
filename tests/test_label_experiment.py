from __future__ import annotations

from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from trec_mlm import cli
from trec_mlm.label_experiment import (
    LabelExperimentConfig,
    _load_label_experiment_artifacts,
    _run_label_experiment_with_embeddings,
    load_label_experiment_artifacts,
    run_label_experiment,
    write_label_experiment_artifacts,
)
from trec_mlm.label_inference import ROCM_IMAGE
from trec_mlm.label_retrieval import (
    CONTEXTUAL_E5_SPEC,
    LabelScoreResult,
    format_document,
    format_query,
)


def _metadata(
    input_count: int,
    *,
    batch_size: int = 2,
    seed: int = 7,
    input_sha256: str,
) -> dict[str, object]:
    return {
        "backend": "transformers",
        "model_id": CONTEXTUAL_E5_SPEC.model_id,
        "requested_revision": CONTEXTUAL_E5_SPEC.revision,
        "resolved_model_revision": CONTEXTUAL_E5_SPEC.revision,
        "resolved_tokenizer_revision": CONTEXTUAL_E5_SPEC.revision,
        "license": CONTEXTUAL_E5_SPEC.license,
        "dtype": CONTEXTUAL_E5_SPEC.dtype,
        "pooling": CONTEXTUAL_E5_SPEC.pooling,
        "embedding_dimension": CONTEXTUAL_E5_SPEC.embedding_dimension,
        "input_count": input_count,
        "input_sha256": input_sha256,
        "transformers_version": "fixture",
        "torch_version": "fixture",
        "hip_version": "fixture",
        "device": "rocm_hip",
        "device_properties": "gfx1151 fixture",
        "required_container_image": ROCM_IMAGE,
        "parameter_count": 7_110_000_000,
        "peak_memory_allocated_bytes": 1,
        "peak_memory_reserved_bytes": 2,
        "batch_size": batch_size,
        "seed": seed,
        "padding_side": "left",
        "maximum_observed_tokens": 20,
        "maximum_untruncated_tokens": 20,
        "truncated_input_count": 0,
        "terminal_token_id": 2,
        "terminal_token": "</s>",
        "max_length": 512,
        "encoding_seconds": 1.0,
    }


def _hash_texts(texts: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class QrelFreeLabelExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.queries = self.root / "queries.tsv"
        self.queries.write_text("q1\tQuestion one?\nq2\tQuestion two?\n")
        self.candidates = self.root / "candidates.json"
        self.candidates.write_text(
            json.dumps({"candidate_ids": ["Beta_topic", "Alpha_topic", "Gamma"]})
        )
        self.config = LabelExperimentConfig(
            query_path=str(self.queries),
            candidate_path=str(self.candidates),
            batch_size=2,
            seed=7,
            run_id="contextual_e5_test",
        )
        self.query_embeddings = np.zeros((2, 4096), dtype=np.float32)
        self.query_embeddings[0, 0] = 1.0
        self.query_embeddings[1, 1] = 1.0
        self.label_embeddings = np.zeros((3, 4096), dtype=np.float32)
        self.label_embeddings[0, 0] = 1.0
        self.label_embeddings[1, 0] = 1.0
        self.label_embeddings[2, 1] = 1.0
        self.input_sha256 = _hash_texts(
            (
                format_query("Question one?"),
                format_query("Question two?"),
                format_document("Alpha_topic"),
                format_document("Beta_topic"),
                format_document("Gamma"),
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self):
        return _run_label_experiment_with_embeddings(
            self.config,
            query_embeddings=self.query_embeddings,
            label_embeddings=self.label_embeddings,
            encoder_metadata=_metadata(5, input_sha256=self.input_sha256),
        )

    def test_config_and_public_apis_contain_no_qrel_boundary(self) -> None:
        field_names = {field.name for field in fields(LabelExperimentConfig)}
        self.assertFalse(any("qrel" in name for name in field_names))
        for function in (
            run_label_experiment,
            write_label_experiment_artifacts,
            load_label_experiment_artifacts,
        ):
            self.assertNotIn("qrel", str(inspect.signature(function)).lower())
        self.assertEqual(
            ["config"], list(inspect.signature(run_label_experiment).parameters)
        )
        self.assertEqual(
            "e5_mistral_7b_contextual", self.config.encoder_key
        )
        self.assertEqual("expert_capability", self.config.label_representation)

    def test_ranking_succeeds_without_any_judgment_file(self) -> None:
        result = self._run()

        self.assertEqual(("q1", "q2"), result.query_ids)
        self.assertEqual(("Alpha_topic", "Beta_topic", "Gamma"), result.candidate_ids)
        self.assertEqual(
            ["Alpha_topic", "Beta_topic", "Gamma"], result.rankings["q1"]
        )
        self.assertEqual((2, 3), result.scores.shape)

    def test_fixed_model_and_representation_cannot_be_changed(self) -> None:
        with self.assertRaisesRegex(ValueError, "encoder_key"):
            _run_label_experiment_with_embeddings(
                LabelExperimentConfig(
                    query_path=str(self.queries),
                    candidate_path=str(self.candidates),
                    encoder_key="another-model",
                ),
                query_embeddings=self.query_embeddings,
                label_embeddings=self.label_embeddings,
                encoder_metadata=_metadata(5, input_sha256=self.input_sha256),
            )
        with self.assertRaisesRegex(ValueError, "label_representation"):
            _run_label_experiment_with_embeddings(
                LabelExperimentConfig(
                    query_path=str(self.queries),
                    candidate_path=str(self.candidates),
                    label_representation="bare",
                ),
                query_embeddings=self.query_embeddings,
                label_embeddings=self.label_embeddings,
                encoder_metadata=_metadata(5, input_sha256=self.input_sha256),
            )

    def test_writer_is_atomic_complete_and_loader_recomputes_the_result(self) -> None:
        destination = self.root / "artifact"
        paths = write_label_experiment_artifacts(self._run(), destination)

        self.assertEqual(
            {
                "config.json",
                "encoder_metadata.json",
                "inputs.json",
                "query_embeddings.npy",
                "label_embeddings.npy",
                "scores.npy",
                "predictions.tsv",
                "rankings.json",
                "run.trec",
                "validation.json",
            },
            set(paths),
        )
        loaded = load_label_experiment_artifacts(destination)
        np.testing.assert_array_equal(loaded.scores, self._run().scores)
        self.assertEqual(self._run().rankings, loaded.rankings)
        validation = json.loads((destination / "validation.json").read_text())
        self.assertEqual(6, validation["score_count"])
        self.assertEqual(6, validation["run"]["row_count"])
        self.assertTrue(validation["qrel_free"])
        with self.assertRaisesRegex(FileExistsError, "exists"):
            write_label_experiment_artifacts(self._run(), destination)

    def test_frozen_legacy_provenance_requires_explicit_private_compatibility(self) -> None:
        destination = self.root / "legacy_artifact"
        write_label_experiment_artifacts(self._run(), destination)
        config_path = destination / "config.json"
        config_payload = json.loads(config_path.read_text())
        config_payload["container_image"] = config_payload.pop(
            "required_container_image"
        )
        config_path.write_text(json.dumps(config_payload, sort_keys=True) + "\n")
        metadata_path = destination / "encoder_metadata.json"
        metadata_payload = json.loads(metadata_path.read_text())
        metadata_payload["container_image"] = metadata_payload.pop(
            "required_container_image"
        )
        metadata_payload["device"] = "cuda_rocm"
        metadata_path.write_text(json.dumps(metadata_payload, sort_keys=True) + "\n")
        validation_path = destination / "validation.json"
        validation = json.loads(validation_path.read_text())
        for name in ("config.json", "encoder_metadata.json"):
            validation["artifact_sha256"][name] = hashlib.sha256(
                (destination / name).read_bytes()
            ).hexdigest()
        validation_path.write_text(json.dumps(validation, sort_keys=True) + "\n")

        with self.assertRaisesRegex(ValueError, "resolved config contract"):
            load_label_experiment_artifacts(destination)
        loaded = _load_label_experiment_artifacts(
            destination, allow_frozen_legacy_provenance=True
        )
        self.assertEqual(("q1", "q2"), loaded.query_ids)

    def test_loader_tolerates_cross_blas_noise_but_preserves_stored_ranks(self) -> None:
        result = self._run()
        destination = self.root / "artifact"
        write_label_experiment_artifacts(result, destination)
        perturbed_scores = result.scores.copy()
        perturbed_scores[0, 1] += 5e-7
        perturbed_rankings = {
            **result.rankings,
            "q1": ["Beta_topic", "Alpha_topic", "Gamma"],
        }

        with mock.patch(
            "trec_mlm.label_experiment.rank_label_scores",
            return_value=LabelScoreResult(
                scores=perturbed_scores,
                rankings=perturbed_rankings,
            ),
        ):
            loaded = load_label_experiment_artifacts(destination)

        self.assertEqual(result.rankings, loaded.rankings)
        np.testing.assert_allclose(result.scores, loaded.scores, rtol=0.0, atol=1e-6)

    def test_writer_leaves_no_partial_destination_when_a_write_fails(self) -> None:
        destination = self.root / "broken"
        with mock.patch(
            "trec_mlm.label_experiment.write_trec_run",
            side_effect=RuntimeError("stop"),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                write_label_experiment_artifacts(self._run(), destination)

        self.assertFalse(destination.exists())

    def test_loader_rejects_mutated_artifacts_and_incomplete_provenance(self) -> None:
        destination = self.root / "artifact"
        write_label_experiment_artifacts(self._run(), destination)
        embeddings = np.load(destination / "query_embeddings.npy", allow_pickle=False)
        embeddings[0, 0] = 0.5
        np.save(destination / "query_embeddings.npy", embeddings)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_label_experiment_artifacts(destination)

        metadata = _metadata(5, input_sha256=self.input_sha256)
        del metadata["resolved_model_revision"]
        with self.assertRaisesRegex(ValueError, "provenance"):
            _run_label_experiment_with_embeddings(
                self.config,
                query_embeddings=self.query_embeddings,
                label_embeddings=self.label_embeddings,
                encoder_metadata=metadata,
            )

        metadata = _metadata(5, input_sha256="definitely-wrong")
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            _run_label_experiment_with_embeddings(
                self.config,
                query_embeddings=self.query_embeddings,
                label_embeddings=self.label_embeddings,
                encoder_metadata=metadata,
            )

    def test_loader_validates_human_readable_prediction_ranks(self) -> None:
        destination = self.root / "artifact"
        write_label_experiment_artifacts(self._run(), destination)
        predictions = destination / "predictions.tsv"
        lines = predictions.read_text().splitlines()
        columns = lines[1].split("\t")
        columns[-1] = "2"
        lines[1] = "\t".join(columns)
        predictions.write_text("\n".join(lines) + "\n")
        validation_path = destination / "validation.json"
        validation = json.loads(validation_path.read_text())
        validation["artifact_sha256"]["predictions.tsv"] = hashlib.sha256(
            predictions.read_bytes()
        ).hexdigest()
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n"
        )

        with self.assertRaisesRegex(ValueError, "prediction ranks"):
            load_label_experiment_artifacts(destination)


class FullInventoryContractTest(unittest.TestCase):
    def test_full_shape_contains_48300_scores_and_complete_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path = root / "queries.tsv"
            query_path.write_text(
                "".join(f"q{index:03d}\tQuestion {index}?\n" for index in range(322))
            )
            candidate_path = root / "candidates.json"
            candidate_path.write_text(
                json.dumps(
                    {"candidate_ids": [f"C{index:03d}" for index in range(150)]}
                )
            )
            query_embeddings = np.zeros((322, 4096), dtype=np.float32)
            label_embeddings = np.zeros((150, 4096), dtype=np.float32)
            query_embeddings[np.arange(322), np.arange(322)] = 1.0
            label_embeddings[np.arange(150), np.arange(150)] = 1.0
            formatted_inputs = tuple(
                [format_query(f"Question {index}?") for index in range(322)]
                + [format_document(f"C{index:03d}") for index in range(150)]
            )
            result = _run_label_experiment_with_embeddings(
                LabelExperimentConfig(
                    query_path=str(query_path),
                    candidate_path=str(candidate_path),
                    batch_size=32,
                    seed=2026,
                ),
                query_embeddings=query_embeddings,
                label_embeddings=label_embeddings,
                encoder_metadata=_metadata(
                    472,
                    batch_size=32,
                    seed=2026,
                    input_sha256=_hash_texts(formatted_inputs),
                ),
            )
            paths = write_label_experiment_artifacts(result, root / "artifact")

            validation = json.loads(Path(paths["validation.json"]).read_text())
            self.assertEqual([322, 150], validation["score_shape"])
            self.assertEqual(48_300, validation["score_count"])
            self.assertEqual(322, validation["run"]["query_count"])
            self.assertEqual(150, validation["run"]["candidate_count"])
            self.assertEqual(48_300, validation["run"]["row_count"])


class LabelRetrievalCliTest(unittest.TestCase):
    def test_parser_exposes_only_qrel_free_winner_inputs(self) -> None:
        args = cli.build_parser().parse_args(
            ["label-retrieval", "--output-dir", "artifacts/contextual-e5"]
        )

        self.assertFalse(any("qrel" in key for key in vars(args)))
        self.assertFalse(hasattr(args, "encoder"))
        self.assertFalse(hasattr(args, "label_representation"))
        self.assertEqual(2026, args.seed)
        self.assertEqual(32, args.batch_size)

    def test_main_builds_python_config_and_calls_qrel_free_runner(self) -> None:
        result = mock.Mock(summary={"status": "ranked"})
        with (
            mock.patch(
                "trec_mlm.label_experiment.run_label_experiment",
                return_value=result,
            ) as run,
            mock.patch(
                "trec_mlm.label_experiment.write_label_experiment_artifacts",
                return_value={"run.trec": "out/run.trec"},
            ) as write,
        ):
            exit_code = cli.main(
                [
                    "label-retrieval",
                    "--queries",
                    "queries.tsv",
                    "--candidates",
                    "candidates.json",
                    "--output-dir",
                    "out",
                ]
            )

        self.assertEqual(0, exit_code)
        config = run.call_args.args[0]
        self.assertEqual("queries.tsv", config.query_path)
        self.assertEqual("candidates.json", config.candidate_path)
        write.assert_called_once_with(result, "out")


if __name__ == "__main__":
    unittest.main()
