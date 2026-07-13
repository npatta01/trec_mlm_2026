from __future__ import annotations

from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from trec_mlm.llm_ranker_inference import (
    chat_template_input_ids,
    extract_grade_log_probabilities,
    grade_token_ids,
    listwise_json_schema,
    load_ranker_engine,
    pointwise_sampling_kwargs,
    run_listwise_messages,
    update_inference_stats,
)


MAX_MODEL_LEN = 8192
ATTENTION_BACKEND = "TRITON_ATTN"


class _Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"0": [10], "1": [11], "2": [12]}[text]


@dataclass
class _Logprob:
    logprob: float


@dataclass
class _Completion:
    logprobs: list[dict[int, _Logprob]]


@dataclass
class _Output:
    outputs: list[_Completion]


class PointwiseInferenceAdapterTest(unittest.TestCase):
    def test_engine_loader_honors_and_records_explicit_memory_budget(self) -> None:
        class TorchVersion(str):
            pass

        class Cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_properties(_index):
                return "fixture gfx1151"

        torch = ModuleType("torch")
        torch.__version__ = TorchVersion("2.11.0")
        torch.cuda = Cuda()
        torch.version = SimpleNamespace(hip="7.2")

        class LLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        vllm = ModuleType("vllm")
        vllm.__version__ = "0.24.0"
        vllm.LLM = LLM

        tokenizer = object()

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return tokenizer

        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = AutoTokenizer

        with patch.dict(
            sys.modules,
            {"torch": torch, "vllm": vllm, "transformers": transformers},
        ):
            llm, observed_tokenizer, metadata = load_ranker_engine(
                seed=2026,
                gpu_memory_utilization=0.30,
            )

        self.assertIs(tokenizer, observed_tokenizer)
        self.assertEqual(0.30, llm.kwargs["gpu_memory_utilization"])
        self.assertEqual(MAX_MODEL_LEN, llm.kwargs["max_model_len"])
        self.assertEqual(
            {"backend": ATTENTION_BACKEND}, llm.kwargs["attention_config"]
        )
        self.assertEqual(0.30, metadata["gpu_memory_utilization"])
        self.assertEqual(MAX_MODEL_LEN, metadata["max_model_len"])
        self.assertEqual(ATTENTION_BACKEND, metadata["attention_backend"])
        self.assertIs(str, type(metadata["torch_version"]))

    def test_chat_template_extracts_input_ids_from_batch_encoding(self) -> None:
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": [[1, 2, 3]]}

        tokenizer = Tokenizer()

        ids = chat_template_input_ids(tokenizer, [{"role": "user", "content": "x"}])

        self.assertEqual([[1, 2, 3]], ids)
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])

    def test_grade_tokens_are_distinct_single_tokens(self) -> None:
        self.assertEqual({0: 10, 1: 11, 2: 12}, grade_token_ids(_Tokenizer()))

    def test_sampling_requests_exact_grade_logprobs(self) -> None:
        kwargs = pointwise_sampling_kwargs({0: 10, 1: 11, 2: 12}, seed=2026)

        self.assertEqual([10, 11, 12], kwargs["allowed_token_ids"])
        self.assertEqual([10, 11, 12], kwargs["logprob_token_ids"])
        self.assertEqual(1, kwargs["max_tokens"])
        self.assertEqual(0.0, kwargs["temperature"])

    def test_extracts_all_requested_grade_logprobs(self) -> None:
        output = _Output(
            [_Completion([{10: _Logprob(-2.0), 11: _Logprob(-1.0), 12: _Logprob(-0.5)}])]
        )

        values = extract_grade_log_probabilities(
            output,
            {0: 10, 1: 11, 2: 12},
        )

        self.assertEqual({0: -2.0, 1: -1.0, 2: -0.5}, values)

    def test_accumulates_request_prompt_and_output_token_counts(self) -> None:
        class Completion:
            token_ids = [4, 5]

        class Output:
            prompt_token_ids = [1, 2, 3]
            outputs = [Completion()]

        stats = {"request_count": 2, "prompt_token_count": 7, "output_token_count": 1}
        update_inference_stats(stats, [Output()])
        self.assertEqual(
            {"request_count": 3, "prompt_token_count": 10, "output_token_count": 3},
            stats,
        )


class ListwiseInferenceAdapterTest(unittest.TestCase):
    def test_schema_requires_twenty_known_handles_and_defers_uniqueness(self) -> None:
        handles = [f"C{index:03d}" for index in range(150)]

        schema = listwise_json_schema(handles)

        self.assertEqual(20, schema["minItems"])
        self.assertEqual(20, schema["maxItems"])
        self.assertNotIn("uniqueItems", schema)
        self.assertEqual(handles, schema["items"]["enum"])

    def test_schema_supports_complete_twelve_handle_pilot_rankings(self) -> None:
        handles = [f"C{index:02d}" for index in range(12)]

        schema = listwise_json_schema(handles, selection_count=len(handles))

        self.assertEqual(12, schema["minItems"])
        self.assertEqual(12, schema["maxItems"])
        self.assertEqual(handles, schema["items"]["enum"])

    def test_schema_rejects_unsupported_complete_thirteen_handle_ranking(self) -> None:
        handles = [f"C{index:02d}" for index in range(13)]

        with self.assertRaisesRegex(ValueError, "supports only"):
            listwise_json_schema(handles, selection_count=len(handles))

    def test_schema_rejects_noncanonical_default_handle_counts(self) -> None:
        for handle_count in range(21, 150):
            with self.subTest(handle_count=handle_count):
                handles = [f"C{index:03d}" for index in range(handle_count)]

                with self.assertRaisesRegex(ValueError, "supports only"):
                    listwise_json_schema(handles)

    def test_schema_rejects_other_handle_and_selection_count_mismatches(self) -> None:
        for handle_count, selection_count in ((12, 20), (20, 12), (150, 150)):
            with self.subTest(
                handle_count=handle_count,
                selection_count=selection_count,
            ):
                handles = [f"C{index:03d}" for index in range(handle_count)]

                with self.assertRaisesRegex(ValueError, "supports only"):
                    listwise_json_schema(
                        handles,
                        selection_count=selection_count,
                    )

    def test_runner_preserves_canonical_default_and_accepts_pilot_count(self) -> None:
        class StructuredOutputsParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class SamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        vllm = ModuleType("vllm")
        vllm.SamplingParams = SamplingParams
        sampling_params = ModuleType("vllm.sampling_params")
        sampling_params.StructuredOutputsParams = StructuredOutputsParams

        class LLM:
            def chat(self, messages, *, sampling_params, use_tqdm):
                self.messages = messages
                self.sampling_params = sampling_params
                self.use_tqdm = use_tqdm
                return [SimpleNamespace(outputs=[SimpleNamespace(text='["C00"]')])]

        messages = [[{"role": "user", "content": "prompt"}]]
        with patch.dict(
            sys.modules,
            {"vllm": vllm, "vllm.sampling_params": sampling_params},
        ):
            canonical_llm = LLM()
            canonical_handles = [f"C{index:03d}" for index in range(150)]
            self.assertEqual(
                ['["C00"]'],
                run_listwise_messages(
                    canonical_llm,
                    messages,
                    handles=canonical_handles,
                    use_tqdm=False,
                ),
            )
            canonical_schema = canonical_llm.sampling_params.kwargs[
                "structured_outputs"
            ].kwargs["json"]
            self.assertEqual(20, canonical_schema["minItems"])
            self.assertEqual(20, canonical_schema["maxItems"])

            pilot_llm = LLM()
            pilot_handles = [f"C{index:02d}" for index in range(12)]
            run_listwise_messages(
                pilot_llm,
                messages,
                handles=pilot_handles,
                selection_count=len(pilot_handles),
                use_tqdm=False,
            )
            pilot_schema = pilot_llm.sampling_params.kwargs[
                "structured_outputs"
            ].kwargs["json"]
            self.assertEqual(12, pilot_schema["minItems"])
            self.assertEqual(12, pilot_schema["maxItems"])


if __name__ == "__main__":
    unittest.main()
