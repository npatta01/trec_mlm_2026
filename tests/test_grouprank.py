from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import unittest

from trec_mlm.grouprank import (
    aggregate_grouprank_scores,
    build_group_requests,
    format_grouprank_messages,
    grouprank_score_schema,
    parse_grouprank_scores,
    request_seed,
)


def _candidates() -> list[str]:
    return [
        "Arts",
        "Biology",
        "Economy",
        "Food_and_drink",
        *[f"Topic_{index:02d}" for index in range(4, 20)],
    ]


def _selection() -> dict[str, object]:
    return {
        "stage": 2,
        "seed": 2026,
        "input_sha256": {"selection": "frozen-fixture"},
        "questions": [
            {
                "query_id": f"q{index:02d}",
                "question": (
                    "Which category covers a Paloma cocktail?"
                    if index == 0
                    else (
                        "Who sets individual degree requirements?"
                        if index == 1
                        else f"Frozen question {index:02d}?"
                    )
                ),
                "candidate_ids": list(reversed(_candidates())),
            }
            for index in range(12)
        ],
    }


def _request_bytes(requests: object) -> bytes:
    return json.dumps(
        [
            {
                "query_id": request.query_id,
                "repetition": request.repetition,
                "group": request.group,
                "ordered_candidate_ids": request.ordered_candidate_ids,
                "handle_to_candidate": request.handle_to_candidate,
                "messages": request.messages,
                "sampling_seed": request.sampling_seed,
            }
            for request in requests
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


class GroupRankGroupingTest(unittest.TestCase):
    def test_group_requests_are_deterministic_complete_and_immutable(self) -> None:
        selection = _selection()

        first = build_group_requests(selection)
        second = build_group_requests(selection)

        self.assertEqual(48, len(first))
        self.assertEqual(_request_bytes(first), _request_bytes(second))
        with self.assertRaises(FrozenInstanceError):
            first[0].group = 9  # type: ignore[misc]

        expected_inventory = set(_candidates())
        for query_index in range(12):
            query_id = f"q{query_index:02d}"
            requests = [request for request in first if request.query_id == query_id]
            self.assertEqual(4, len(requests))
            self.assertEqual({666, 667}, {request.repetition for request in requests})
            self.assertEqual({0, 1}, {request.group for request in requests})
            self.assertTrue(
                all(len(request.ordered_candidate_ids) == 10 for request in requests)
            )
            for request in requests:
                self.assertEqual(
                    request_seed(query_id, request.repetition, request.group),
                    request.sampling_seed,
                )
                self.assertEqual(
                    set(request.ordered_candidate_ids),
                    set(request.handle_to_candidate.values()),
                )
            for grouping_seed in (666, 667):
                repetition = [
                    request
                    for request in requests
                    if request.repetition == grouping_seed
                ]
                flattened = tuple(
                    candidate
                    for request in sorted(repetition, key=lambda item: item.group)
                    for candidate in request.ordered_candidate_ids
                )
                expected_order = tuple(
                    sorted(
                        expected_inventory,
                        key=lambda candidate: hashlib.sha256(
                            f"{grouping_seed}|{query_id}|{candidate}".encode()
                        ).digest(),
                    )
                )
                self.assertEqual(expected_order, flattened)
                self.assertEqual(expected_inventory, set(flattened))

        handles = {
            handle
            for request in first
            if request.query_id == "q00"
            for handle in request.handle_to_candidate
        }
        self.assertEqual({f"C{index:02d}" for index in range(20)}, handles)
        self.assertTrue(
            all(
                request.handle_to_candidate.get("C03") in (None, "Food_and_drink")
                for request in first
            )
        )

    def test_group_requests_require_the_exact_stage_two_inventory(self) -> None:
        selection = _selection()

        invalid_selections = []
        two_questions = dict(selection)
        two_questions["questions"] = selection["questions"][:2]  # type: ignore[index]
        invalid_selections.append(two_questions)

        wrong_stage = dict(selection)
        wrong_stage["stage"] = 1
        invalid_selections.append(wrong_stage)

        nineteen_candidates = json.loads(json.dumps(selection))
        nineteen_candidates["questions"][0]["candidate_ids"].pop()
        invalid_selections.append(nineteen_candidates)

        duplicate_candidate = json.loads(json.dumps(selection))
        duplicate_candidate["questions"][0]["candidate_ids"][-1] = (
            duplicate_candidate["questions"][0]["candidate_ids"][0]
        )
        invalid_selections.append(duplicate_candidate)

        duplicate_query = json.loads(json.dumps(selection))
        duplicate_query["questions"][1]["query_id"] = "q00"
        invalid_selections.append(duplicate_query)

        for invalid in invalid_selections:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "Stage 2|12|20|unique"):
                    build_group_requests(invalid)

        with self.assertRaisesRegex(ValueError, "Stage 2"):
            build_group_requests(None)  # type: ignore[arg-type]


class GroupRankPromptTest(unittest.TestCase):
    def test_prompt_contains_only_delimited_question_and_expert_cards(self) -> None:
        candidates = _candidates()[:10]
        handles = {f"C{index:02d}": candidate for index, candidate in enumerate(candidates)}

        messages = format_grouprank_messages(
            "Which category covers a Paloma cocktail?",
            ordered_candidate_ids=tuple(reversed(candidates)),
            handle_to_candidate=handles,
        )

        self.assertEqual(("system", "user"), tuple(item["role"] for item in messages))
        user = messages[1]["content"]
        self.assertIn(
            "<question>\nWhich category covers a Paloma cocktail?\n</question>",
            user,
        )
        self.assertIn("<experts>\n", user)
        self.assertIn("\n</experts>", user)
        self.assertIn(
            "C03 | Food_and_drink | This LLM is most likely to correctly answer "
            "questions about Food and drink.",
            user,
        )
        self.assertEqual(10, sum(user.count(handle) for handle in handles))
        prompt_content = "\n".join(item["content"] for item in messages).lower()
        for prohibited in ("grade", "relevance", "qrel", "response", "exemplar"):
            with self.subTest(prohibited=prohibited):
                self.assertNotIn(prohibited, prompt_content)


class GroupRankParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.handles = [f"C{index:02d}" for index in range(10)]

    def _answer(self, payload: str) -> str:
        return f"I compared every expert.\n<answer>\n{payload}\n</answer>\n"

    def test_parser_accepts_exact_integer_scores_from_final_answer_block(self) -> None:
        expected = {handle: index for index, handle in enumerate(self.handles)}

        parsed = parse_grouprank_scores(
            self._answer(json.dumps(expected)),
            expected_handles=self.handles,
        )

        self.assertEqual(expected, parsed)

    def test_parser_rejects_missing_extra_and_duplicate_handles(self) -> None:
        base = {handle: 5 for handle in self.handles}
        invalid_payloads = {
            "missing": json.dumps(dict(list(base.items())[:-1])),
            "extra": json.dumps({**base, "C99": 5}),
            "duplicate": "{" + ",".join(
                [f'"{handle}":5' for handle in self.handles]
                + ['"C00":6']
            ) + "}",
        }

        for kind, payload in invalid_payloads.items():
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "duplicate|missing or extra"):
                    parse_grouprank_scores(
                        self._answer(payload),
                        expected_handles=self.handles,
                    )

    def test_parser_rejects_non_integer_and_out_of_range_scores(self) -> None:
        invalid_values = (True, 1.0, "1", -1, 11)

        for invalid_value in invalid_values:
            payload = {handle: 5 for handle in self.handles}
            payload["C00"] = invalid_value
            with self.subTest(value=invalid_value):
                with self.assertRaisesRegex(ValueError, "integers from 0 through 10"):
                    parse_grouprank_scores(
                        self._answer(json.dumps(payload)),
                        expected_handles=self.handles,
                    )

    def test_parser_requires_an_object_inside_one_final_answer_block(self) -> None:
        payload = json.dumps({handle: 5 for handle in self.handles})
        invalid_outputs = (
            payload,
            f"<answer>{payload}</answer>\n<answer>{payload}</answer>",
            f"<answer>{payload}</answer>\nmore reasoning",
        )

        for raw in invalid_outputs:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "one final <answer> block"):
                    parse_grouprank_scores(raw, expected_handles=self.handles)

        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_grouprank_scores(
                self._answer(json.dumps([5] * 10)),
                expected_handles=self.handles,
            )

    def test_schema_is_deterministic_and_requires_exact_fixed_keys(self) -> None:
        expected = {
            "type": "object",
            "properties": {
                handle: {"type": "integer", "minimum": 0, "maximum": 10}
                for handle in self.handles
            },
            "required": self.handles,
            "additionalProperties": False,
        }

        first = grouprank_score_schema(self.handles)
        second = grouprank_score_schema(tuple(self.handles))

        self.assertEqual(expected, first)
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )


class GroupRankAggregationTest(unittest.TestCase):
    def test_aggregation_averages_two_observations_and_breaks_ties_lexically(
        self,
    ) -> None:
        requests = build_group_requests(_selection())
        parsed_scores = []
        for request in requests:
            result = {}
            for handle, candidate in request.handle_to_candidate.items():
                if candidate == "Arts":
                    result[handle] = 8 if request.repetition == 666 else 10
                elif candidate == "Biology":
                    result[handle] = 9
                else:
                    result[handle] = 1
            parsed_scores.append(result)

        candidate_scores, rankings = aggregate_grouprank_scores(
            requests,
            parsed_scores,
        )

        self.assertEqual(9.0, candidate_scores["q00"]["Arts"])
        self.assertEqual(9.0, candidate_scores["q00"]["Biology"])
        self.assertEqual(["Arts", "Biology"], rankings["q00"][:2])
        self.assertEqual(sorted(_candidates()[2:]), rankings["q00"][2:])
        self.assertEqual(12, len(rankings))
        self.assertTrue(all(len(ranking) == 20 for ranking in rankings.values()))

    def test_aggregation_rejects_incomplete_observations(self) -> None:
        requests = build_group_requests(_selection())
        parsed_scores = [
            {handle: 5 for handle in request.handle_to_candidate}
            for request in requests
        ]

        with self.assertRaisesRegex(ValueError, "one score set per request"):
            aggregate_grouprank_scores(requests, parsed_scores[:-1])

    def test_aggregation_rejects_cross_repetition_candidate_swaps(self) -> None:
        requests = list(build_group_requests(_selection()))
        for index, request in enumerate(requests):
            mapping = dict(request.handle_to_candidate)
            if (
                request.query_id == "q00"
                and request.repetition == 666
                and "Biology" in mapping.values()
            ):
                handle = next(
                    handle
                    for handle, candidate in mapping.items()
                    if candidate == "Biology"
                )
                mapping[handle] = "Arts"
                requests[index] = replace(request, handle_to_candidate=mapping)
            elif (
                request.query_id == "q00"
                and request.repetition == 667
                and "Arts" in mapping.values()
            ):
                handle = next(
                    handle
                    for handle, candidate in mapping.items()
                    if candidate == "Arts"
                )
                mapping[handle] = "Biology"
                requests[index] = replace(request, handle_to_candidate=mapping)
        parsed_scores = [
            {handle: 5 for handle in request.handle_to_candidate}
            for request in requests
        ]

        with self.assertRaisesRegex(ValueError, "exactly once per repetition"):
            aggregate_grouprank_scores(requests, parsed_scores)


if __name__ == "__main__":
    unittest.main()
