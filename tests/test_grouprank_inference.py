from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from trec_mlm.grouprank import GroupRankRequest
from trec_mlm.grouprank_inference import (
    ATTENTION_BACKEND,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    ROCM_CONTAINER_IMAGE,
    finalize_grouprank_runtime,
    load_grouprank_engine,
    run_grouprank_requests,
)


def _request(*, group: int, seed: int) -> GroupRankRequest:
    candidates = tuple(f"candidate-{group}-{index}" for index in range(10))
    handles = {
        f"C{index:02d}": candidate for index, candidate in enumerate(candidates)
    }
    return GroupRankRequest(
        query_id=f"query-{group}",
        repetition=666,
        group=group,
        ordered_candidate_ids=candidates,
        handle_to_candidate=handles,
        messages=(
            {"role": "system", "content": "compare experts"},
            {"role": "user", "content": f"request {group}"},
        ),
        sampling_seed=seed,
    )


class GroupRankLoaderTest(unittest.TestCase):
    def test_loader_pins_model_runtime_and_provenance(self) -> None:
        class Cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_properties(index):
                self.assertEqual(0, index)
                return "fixture gfx1151"

        torch = ModuleType("torch")
        torch.__version__ = "2.11.0"
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
            calls = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return tokenizer

        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = AutoTokenizer

        with patch.dict(
            sys.modules,
            {"torch": torch, "vllm": vllm, "transformers": transformers},
        ):
            llm, observed_tokenizer, metadata = load_grouprank_engine()

        self.assertIs(tokenizer, observed_tokenizer)
        self.assertEqual(
            [
                (
                    (MODEL_ID,),
                    {
                        "revision": MODEL_REVISION,
                        "trust_remote_code": False,
                    },
                )
            ],
            AutoTokenizer.calls,
        )
        self.assertEqual(
            {
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "dtype": MODEL_DTYPE,
                "trust_remote_code": False,
                "max_model_len": MAX_MODEL_LEN,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                "attention_config": {"backend": ATTENTION_BACKEND},
            },
            llm.kwargs,
        )
        self.assertEqual(MODEL_ID, metadata["model_id"])
        self.assertEqual(MODEL_REVISION, metadata["revision"])
        self.assertEqual(MODEL_REVISION, metadata["resolved_revision"])
        self.assertEqual(MODEL_DTYPE, metadata["dtype"])
        self.assertEqual(MODEL_LICENSE, metadata["license"])
        self.assertEqual(7_615_616_512, metadata["parameter_count"])
        self.assertEqual(MAX_MODEL_LEN, metadata["max_model_len"])
        self.assertEqual(
            GPU_MEMORY_UTILIZATION, metadata["gpu_memory_utilization"]
        )
        self.assertEqual(ATTENTION_BACKEND, metadata["attention_backend"])
        self.assertEqual(ROCM_CONTAINER_IMAGE, metadata["container_image"])
        self.assertEqual("0.24.0", metadata["vllm_version"])
        self.assertEqual("2.11.0", metadata["torch_version"])
        self.assertEqual("7.2", metadata["hip_version"])
        self.assertEqual("fixture gfx1151", metadata["device_properties"])

    def test_runtime_is_reconciled_from_raw_records_not_caller_stats(self) -> None:
        caller_stats = {
            "request_count": 999,
            "prompt_token_count": 999,
            "output_token_count": 999,
        }
        metadata = {
            "container_image": ROCM_CONTAINER_IMAGE,
            "inference_stats": caller_stats,
        }
        raw_outputs = (
            {"prompt_token_count": 7, "output_token_count": 3},
            {"prompt_token_count": 11, "output_token_count": 5},
        )

        runtime = finalize_grouprank_runtime(
            metadata,
            raw_outputs,
            elapsed_seconds=1.25,
            expected_request_count=2,
        )

        self.assertEqual(
            {
                "request_count": 2,
                "prompt_token_count": 18,
                "output_token_count": 8,
            },
            runtime["inference_stats"],
        )
        self.assertEqual(1.25, runtime["elapsed_seconds"])
        self.assertIsNot(caller_stats, runtime["inference_stats"])

    def test_runtime_requires_exact_container_positive_time_and_request_count(
        self,
    ) -> None:
        record = {"prompt_token_count": 7, "output_token_count": 3}
        cases = (
            (
                {"container_image": "wrong/image"},
                (record,),
                1.0,
                1,
                "container image",
            ),
            (
                {"container_image": ROCM_CONTAINER_IMAGE},
                (record,),
                0.0,
                1,
                "positive finite",
            ),
            (
                {"container_image": ROCM_CONTAINER_IMAGE},
                (record,),
                1.0,
                2,
                "exactly 2 requests",
            ),
        )
        for metadata, outputs, elapsed, count, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    finalize_grouprank_runtime(
                        metadata,
                        outputs,
                        elapsed_seconds=elapsed,
                        expected_request_count=count,
                    )


class GroupRankRunnerTest(unittest.TestCase):
    @staticmethod
    def _vllm_double():
        class SamplingParams:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.__class__.instances.append(self)

        vllm = ModuleType("vllm")
        vllm.SamplingParams = SamplingParams
        return vllm, SamplingParams

    def test_runner_uses_chat_and_one_frozen_sampling_seed_per_request(self) -> None:
        vllm, SamplingParams = self._vllm_double()
        requests = [_request(group=0, seed=101), _request(group=1, seed=202)]

        class LLM:
            def __init__(self):
                self.calls = []

            def chat(self, messages, *, sampling_params, use_tqdm):
                self.calls.append((messages, sampling_params, use_tqdm))
                index = len(self.calls)
                return [
                    SimpleNamespace(
                        prompt_token_ids=list(range(index + 1)),
                        outputs=[
                            SimpleNamespace(
                                text=f"raw response {index}",
                                token_ids=list(range(index + 2)),
                                finish_reason="stop",
                            )
                        ],
                    )
                ]

        llm = LLM()
        stats = {"request_count": 3, "prompt_token_count": 5}
        with patch.dict(sys.modules, {"vllm": vllm}):
            records = run_grouprank_requests(llm, requests, stats=stats)

        self.assertEqual(2, len(llm.calls))
        self.assertEqual(
            [[dict(message) for message in request.messages] for request in requests],
            [call[0][0] for call in llm.calls],
        )
        self.assertTrue(all(call[2] is False for call in llm.calls))
        self.assertEqual(
            [
                {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "max_tokens": 2048,
                    "seed": 101,
                },
                {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "max_tokens": 2048,
                    "seed": 202,
                },
            ],
            [sampling.kwargs for sampling in SamplingParams.instances],
        )
        self.assertEqual(
            [sampling for sampling in SamplingParams.instances],
            [call[1] for call in llm.calls],
        )
        self.assertEqual(
            {
                "query_id": "query-0",
                "repetition": 666,
                "group": 0,
                "ordered_candidate_ids": list(requests[0].ordered_candidate_ids),
                "handle_to_candidate": dict(requests[0].handle_to_candidate),
                "raw_text": "raw response 1",
                "prompt_token_count": 2,
                "output_token_count": 3,
                "finish_reason": "stop",
                "sampling_seed": 101,
            },
            records[0],
        )
        self.assertEqual(
            {
                "request_count": 5,
                "prompt_token_count": 10,
                "output_token_count": 7,
            },
            stats,
        )

    def test_runner_preserves_length_finish_reason_without_parsing(self) -> None:
        vllm, _ = self._vllm_double()

        class LLM:
            @staticmethod
            def chat(_messages, *, sampling_params, use_tqdm):
                return [
                    SimpleNamespace(
                        prompt_token_ids=[1],
                        outputs=[
                            SimpleNamespace(
                                text="not parseable",
                                token_ids=[2] * 2048,
                                finish_reason="length",
                            )
                        ],
                    )
                ]

        with patch.dict(sys.modules, {"vllm": vllm}):
            records = run_grouprank_requests(
                LLM(), [_request(group=0, seed=101)], stats={}
            )

        self.assertEqual("not parseable", records[0]["raw_text"])
        self.assertEqual("length", records[0]["finish_reason"])

    def test_runner_rejects_missing_token_accounting(self) -> None:
        vllm, _ = self._vllm_double()

        class LLM:
            @staticmethod
            def chat(_messages, *, sampling_params, use_tqdm):
                return [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                text="raw", token_ids=[1], finish_reason="stop"
                            )
                        ]
                    )
                ]

        with patch.dict(sys.modules, {"vllm": vllm}):
            with self.assertRaisesRegex(ValueError, "token accounting"):
                run_grouprank_requests(
                    LLM(), [_request(group=0, seed=101)], stats={}
                )

    def test_runner_rejects_missing_finish_reason(self) -> None:
        vllm, _ = self._vllm_double()

        class LLM:
            @staticmethod
            def chat(_messages, *, sampling_params, use_tqdm):
                return [
                    SimpleNamespace(
                        prompt_token_ids=[1],
                        outputs=[SimpleNamespace(text="raw", token_ids=[1])],
                    )
                ]

        with patch.dict(sys.modules, {"vllm": vllm}):
            with self.assertRaisesRegex(ValueError, "finish reason"):
                run_grouprank_requests(
                    LLM(), [_request(group=0, seed=101)], stats={}
                )


if __name__ == "__main__":
    unittest.main()
