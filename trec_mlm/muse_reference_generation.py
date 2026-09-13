"""Direct Meta Muse Responses API adapter for grounded reference answers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping
import urllib.error

from trec_mlm.reference_generation import (
    ProviderRequestError,
    RequestFunction,
    RETRIABLE_STATUS_CODES,
    SleepFunction,
    _post_json,
    _request_with_retries,
)


MUSE_ENDPOINT = "https://api.meta.ai/v1/responses"
MUSE_MODEL = "muse-spark-1.2-contributor"

MUSE_INITIAL_PROMPT = (
    "Answer the user's question using web search grounding. Search as many times "
    "as useful to establish the facts; do not impose an arbitrary word or character "
    "limit. Return a self-contained answer that is concise without omitting needed "
    "context, dates, comparisons, or caveats. Prefer primary, official, academic, "
    "medical, legal, institutional, or reputable news sources. Use weaker sources "
    "only when better sources are unavailable. Do not invent facts, citations, "
    "quotes, numbers, or dates. If evidence is insufficient, conflicting, or the "
    "question is ambiguous, say so clearly instead of guessing."
)
MUSE_FOLLOWUP_PROMPT = (
    "Review the answer and web evidence above with a fresh grounding pass. Search "
    "again wherever useful, resolve important conflicts or missing details, and "
    "return the final self-contained answer with citations. Prefer the highest "
    "quality sources available. Keep the answer as long as needed for correctness; "
    "do not impose an arbitrary word or character limit. Do not invent facts or "
    "citations, and state any remaining uncertainty."
)
MUSE_PROMPT_SHA256 = hashlib.sha256(
    f"{MUSE_INITIAL_PROMPT}\n\n{MUSE_FOLLOWUP_PROMPT}".encode("utf-8")
).hexdigest()

MUSE_REFERENCE_INITIAL_PROMPT = (
    "You are generating a factual reference record for evaluating other LLM answers. "
    "Answer the question independently using web search grounding; do not assume an "
    "answer key or try to match a presumed answer. Search as many times as useful; "
    "do not impose an arbitrary search or output-length limit. Use any accessible, "
    "relevant, non-fabricated source. Prefer primary, official, academic, medical, "
    "legal, institutional, or reputable news sources when available, but Wikipedia "
    "is acceptable when it is relevant and supports the claim. Never invent facts, "
    "citations, quotes, numbers, or dates, and do not cite a source that does not "
    "support the claim.\n\n"
    "Return a self-contained reference record with these clearly labeled sections:\n"
    "1. Canonical answer: the direct answer first.\n"
    "2. Acceptable variants: aliases, equivalent wording, or date variants that "
    "should also be accepted.\n"
    "3. Necessary caveats: include only qualifications whose omission could make "
    "an answer misleading or unfair to mark.\n"
    "4. Evidence: the retrieved URLs and what each supports.\n"
    "5. Confidence: high, medium, or low, with a brief reason.\n\n"
    "Long evidence is fine when needed. Do not add unrelated background. If the "
    "evidence genuinely conflicts or the question is ambiguous, preserve that "
    "uncertainty and explain which interpretation is most likely intended."
)
MUSE_REFERENCE_FOLLOWUP_PROMPT = (
    "Re-check the draft as a factual reference record for evaluating other LLM "
    "answers. Independently verify the central answer and any dates, names, numbers, "
    "or classifications with fresh web search grounding. Search again wherever "
    "useful; there is no fixed search or output-length limit. Keep Wikipedia or "
    "another secondary source when it is relevant and non-fabricated, but prefer a "
    "stronger accessible source when one is available. Preserve only necessary "
    "caveats—those that affect whether a candidate answer should be accepted or "
    "rejected—and do not add unrelated background. Return the same five labeled "
    "sections: Canonical answer, Acceptable variants, Necessary caveats, Evidence, "
    "and Confidence. Do not invent facts or citations."
)
MUSE_REFERENCE_PROMPT_SHA256 = hashlib.sha256(
    f"{MUSE_REFERENCE_INITIAL_PROMPT}\n\n{MUSE_REFERENCE_FOLLOWUP_PROMPT}".encode("utf-8")
).hexdigest()

MUSE_REFERENCE_GENERIC_INITIAL_PROMPT = (
    "You are generating a grounded reference record for evaluating other LLM "
    "answers. Answer the question independently; do not assume an answer key. "
    "First identify exactly what the question asks: the entities, relationship, "
    "attribute, comparison, scope, and requested answer type. Answer that same "
    "question directly. Do not substitute a related fact merely because it is "
    "easier to verify or has stronger search results.\n\n"
    "Use web search as many times as needed. Do not impose an arbitrary search or "
    "output-length limit. Prefer primary, official, academic, institutional, or "
    "reputable sources when available. Wikipedia and other secondary sources are "
    "acceptable when relevant and factually supportive. Never invent facts, "
    "citations, quotes, dates, or numbers. Do not generalize from one example to "
    "an entire group unless the evidence supports that scope.\n\n"
    "Return a self-contained reference record with these clearly labeled sections:\n"
    "1. Canonical answer: the direct answer to the question as written.\n"
    "2. Acceptable variants: equivalent wording or genuinely acceptable alternatives "
    "answering the same question; do not include merely related facts.\n"
    "3. Necessary caveats: only caveats that affect correctness or fair grading.\n"
    "4. Evidence: sources and the specific claims they support.\n"
    "5. Confidence: high, medium, or low, with a brief reason.\n\n"
    "Before finalizing, check whether the canonical answer addresses the exact "
    "attribute or relationship requested, whether its scope is supported, whether "
    "the variants answer the same question, and whether any material ambiguity "
    "should be preserved rather than forced into one answer."
)
MUSE_REFERENCE_GENERIC_FOLLOWUP_PROMPT = (
    "Treat the previous reference record as a hypothesis, not an answer key. "
    "Re-check it with fresh web grounding. Look specifically for semantic mismatch, "
    "scope overreach, unsupported generalization, and alternatives that answer a "
    "different question. Preserve genuine ambiguity instead of forcing a single "
    "answer. Return the same five labeled sections: Canonical answer, Acceptable "
    "variants, Necessary caveats, Evidence, and Confidence. Search as many times "
    "as needed; do not impose an arbitrary search or output-length limit. Do not "
    "invent facts or citations."
)
MUSE_REFERENCE_GENERIC_PROMPT_SHA256 = hashlib.sha256(
    f"{MUSE_REFERENCE_GENERIC_INITIAL_PROMPT}\n\n{MUSE_REFERENCE_GENERIC_FOLLOWUP_PROMPT}".encode(
        "utf-8"
    )
).hexdigest()

MUSE_REFERENCE_GENERIC_CITATIONS_INITIAL_PROMPT = (
    MUSE_REFERENCE_GENERIC_INITIAL_PROMPT
    + "\n\nWhenever you use web search, attach inline URL citations to the central "
    "answer and the evidence claims that support it. The Evidence section must "
    "also identify the retrieved URLs and what each URL supports."
)
MUSE_REFERENCE_GENERIC_CITATIONS_FOLLOWUP_PROMPT = (
    MUSE_REFERENCE_GENERIC_FOLLOWUP_PROMPT
    + " Whenever you use web search, attach inline URL citations to the central "
    "answer and supporting evidence claims."
)
MUSE_REFERENCE_GENERIC_CITATIONS_PROMPT_SHA256 = hashlib.sha256(
    f"{MUSE_REFERENCE_GENERIC_CITATIONS_INITIAL_PROMPT}\n\n{MUSE_REFERENCE_GENERIC_CITATIONS_FOLLOWUP_PROMPT}".encode(
        "utf-8"
    )
).hexdigest()

MUSE_REFERENCE_GENERIC_UNCERTAINTY_INITIAL_PROMPT = (
    "You are generating a grounded reference record for evaluating other LLM "
    "answers. Answer the question independently; do not assume an answer key. "
    "Use web search as many times as useful. Do not impose an arbitrary search or "
    "output-length limit. Prefer primary, official, academic, institutional, or "
    "reputable sources when available. Wikipedia and other secondary sources are "
    "acceptable when relevant and factually supportive. Never invent facts, "
    "citations, quotes, dates, or numbers.\n\n"
    "Begin with exactly these two lines:\n"
    "Reference status: ANSWERED or UNCERTAIN\n"
    "Uncertainty reason: none, or one short reason\n\n"
    "Use ANSWERED only when the question has a sufficiently clear reading and the "
    "central answer is supported by the evidence. Use UNCERTAIN when the wording "
    "has a materially plausible alternative reading, the evidence is insufficient, "
    "or credible sources conflict. Do not silently choose one interpretation when "
    "the ambiguity could change what a fair grader should accept. When UNCERTAIN, "
    "give the best-supported answer, name the other plausible interpretation or "
    "evidence problem, and explain what a grader should accept.\n\n"
    "After the two-line header, return a self-contained record with these sections:\n"
    "1. Canonical answer: the direct answer or best-supported answer.\n"
    "2. Acceptable variants: equivalent answers, plus any supported alternative "
    "interpretation that should be accepted when status is UNCERTAIN.\n"
    "3. Necessary caveats: only qualifications that affect correctness or fair "
    "grading.\n"
    "4. Evidence: retrieved URLs and the specific claims each supports.\n"
    "5. Grading guidance: how to evaluate answers given the status and evidence.\n"
    "6. Confidence: high, medium, or low, with a brief reason. Attach inline URL "
    "citations to the answer and supporting evidence claims."
)
MUSE_REFERENCE_GENERIC_UNCERTAINTY_FOLLOWUP_PROMPT = (
    "Treat the previous reference record as a hypothesis, not an answer key. "
    "Perform a fresh grounding pass and re-check the exact wording, intended "
    "attribute or relationship, scope, and answer type. Search again wherever "
    "useful; there is no arbitrary search or output-length limit. Check whether "
    "the question has a materially plausible alternative reading, whether the "
    "central claims are supported, and whether credible sources conflict. Do not "
    "silently choose an interpretation that could change fair grading. Return the "
    "final self-contained record beginning with exactly these two lines:\n"
    "Reference status: ANSWERED or UNCERTAIN\n"
    "Uncertainty reason: none, or one short reason\n\n"
    "Use UNCERTAIN when ambiguity, insufficient evidence, or source conflict remains; "
    "otherwise use ANSWERED. When UNCERTAIN, include the best-supported answer, the "
    "plausible alternative or evidence problem, and grading guidance. Return the "
    "same sections: Canonical answer, Acceptable variants, Necessary caveats, "
    "Evidence, Grading guidance, and Confidence. Attach inline URL citations and "
    "do not invent facts or citations."
)
MUSE_REFERENCE_GENERIC_UNCERTAINTY_PROMPT_SHA256 = hashlib.sha256(
    f"{MUSE_REFERENCE_GENERIC_UNCERTAINTY_INITIAL_PROMPT}\n\n{MUSE_REFERENCE_GENERIC_UNCERTAINTY_FOLLOWUP_PROMPT}".encode(
        "utf-8"
    )
).hexdigest()

MUSE_REFERENCE_GENERIC_BRANCHES_INITIAL_PROMPT = (
    "You are generating a grounded reference record for evaluating other LLM "
    "answers. Answer the question independently; do not assume an answer key. "
    "Before searching, inspect the wording and consider whether it has more than "
    "one materially different reading. Use multiple interpretation branches only "
    "when each reading honors all explicit constraints in the question, has "
    "supporting evidence, and could change the grading outcome. Do not create "
    "branches merely because evidence is incomplete, sources disagree, or the "
    "answer can be phrased at different levels of detail. Distinguish: "
    "interpretation ambiguity, conflicting evidence, and scope or granularity. "
    "Rank interpretations by fit to the question wording, not by the number of "
    "search results.\n\n"
    "Use web search as many times as needed. Do not impose an arbitrary search or "
    "output-length limit. Prefer primary, official, academic, institutional, or "
    "reputable sources when available. Wikipedia and other secondary sources are "
    "acceptable when relevant and factually supportive. Never invent facts, "
    "citations, quotes, dates, or numbers. Do not generalize beyond the scope "
    "supported by the evidence.\n\n"
    "For a clear question, return one direct canonical answer. For a materially "
    "ambiguous question, return `Status: AMBIGUOUS` and two or more structured "
    "interpretation-answer pairs. Each pair must include the interpretation, "
    "answer, scope, supporting evidence, confidence, and grading guidance. Then "
    "include an `Acceptance set` that flattens the supported answers a grader may "
    "accept. Do not list unrelated facts as interpretations, and do not avoid "
    "committing by listing every remotely possible reading. If evidence conflicts "
    "but the wording has one clear meaning, return one answer and describe the "
    "conflict and confidence instead.\n\n"
    "Return a self-contained record with clearly labeled sections: Status; "
    "Canonical answer or Interpretation-answer pairs; Acceptable variants or "
    "Acceptance set; Necessary caveats; Evidence; Confidence; and Grading "
    "guidance. Attach inline URL citations to the central answer and supporting "
    "evidence claims, and identify what each URL supports."
)
MUSE_REFERENCE_GENERIC_BRANCHES_FOLLOWUP_PROMPT = (
    "Treat the previous reference record as a hypothesis, not an answer key. "
    "Perform a fresh grounding pass and check the exact wording, entities, "
    "relationship, requested attribute, scope, and answer type. Check whether "
    "each proposed interpretation really satisfies all question constraints, is "
    "supported by evidence, and would produce a materially different grading "
    "outcome. Remove branches that are merely related facts, evidence conflicts, "
    "or nested levels of detail. If the question has one clear reading, collapse "
    "to one canonical answer; if genuine interpretation ambiguity remains, retain "
    "the structured branches and a flattened Acceptance set. Keep evidence "
    "conflicts as caveats unless they create a genuine interpretation ambiguity. "
    "Search as many times as needed; do not impose an arbitrary search or "
    "output-length limit. Attach inline URL citations to the answer and supporting "
    "evidence claims. Return the same labeled sections and do not invent facts or "
    "citations."
)
MUSE_REFERENCE_GENERIC_BRANCHES_PROMPT_SHA256 = hashlib.sha256(
    f"{MUSE_REFERENCE_GENERIC_BRANCHES_INITIAL_PROMPT}\n\n{MUSE_REFERENCE_GENERIC_BRANCHES_FOLLOWUP_PROMPT}".encode(
        "utf-8"
    )
).hexdigest()

MUSE_REFERENCE_GENERIC_BRANCHES_V2_INITIAL_PROMPT = (
    MUSE_REFERENCE_GENERIC_BRANCHES_INITIAL_PROMPT
    + "\n\nUse this exact ambiguity contract. Before searching, privately inventory any "
    "plausible semantic dimensions of the requested attribute. Terms such as "
    "type, kind, form, class, or role can refer to different dimensions, such as "
    "physical form, habitat, taxonomy, or function. If the wording does not select "
    "one dimension, at least two dimensions remain plausible, and each dimension "
    "has evidence that yields a materially different answer, that is material "
    "interpretation ambiguity. Do not collapse it merely because one dimension is "
    "more common, one source is more prominent, or one search path returned more "
    "results. Do not relabel that situation as scope or granularity. If a fresh "
    "search pass finds a supported answer in a different semantic dimension from "
    "the draft, compare the dimensions and preserve both rather than silently "
    "replacing the draft.\n\n"
    "Begin the final record with exactly one line: `Interpretation decision: "
    "SINGLE` or `Interpretation decision: MULTIPLE`. If the decision is MULTIPLE, "
    "immediately provide `Interpretation 1` and `Interpretation 2` (and any other "
    "supported reading), each with its answer, scope, evidence, confidence, and "
    "grading guidance, followed by an `Acceptance set`. Do not use a long list of "
    "acceptable variants as a substitute for separate interpretation branches. If "
    "the decision is SINGLE, include `Reason no branch:` and explain why every "
    "plausible alternative is ruled out by the wording or evidence."
)
MUSE_REFERENCE_GENERIC_BRANCHES_V2_FOLLOWUP_PROMPT = (
    MUSE_REFERENCE_GENERIC_BRANCHES_FOLLOWUP_PROMPT
    + "\n\nApply the exact ambiguity contract again. Do not silently switch between "
    "semantic dimensions. Begin the final record with exactly one line, "
    "`Interpretation decision: SINGLE` or `Interpretation decision: MULTIPLE`. "
    "For MULTIPLE, provide at least `Interpretation 1` and `Interpretation 2` "
    "with separate answers and grading guidance. For SINGLE, include `Reason no "
    "branch:`."
)
MUSE_REFERENCE_GENERIC_BRANCHES_V2_PROMPT_SHA256 = hashlib.sha256(
    f"{MUSE_REFERENCE_GENERIC_BRANCHES_V2_INITIAL_PROMPT}\n\n{MUSE_REFERENCE_GENERIC_BRANCHES_V2_FOLLOWUP_PROMPT}".encode(
        "utf-8"
    )
).hexdigest()

MUSE_REFERENCE_GENERIC_BRANCHES_V3_INITIAL_PROMPT = (
    MUSE_REFERENCE_GENERIC_BRANCHES_V2_INITIAL_PROMPT
    + "\n\nBefore choosing SINGLE for an unqualified attribute term, perform at least "
    "one focused grounding search for every plausible semantic dimension you "
    "identified. A dimension that was never tested is not ruled out by the "
    "wording. If two tested dimensions remain evidence-supported and produce "
    "different grading answers, the decision must be MULTIPLE. Choose SINGLE only "
    "when the wording rules out the alternatives or the focused grounding checks "
    "fail to support them."
)
MUSE_REFERENCE_GENERIC_BRANCHES_V3_FOLLOWUP_PROMPT = (
    MUSE_REFERENCE_GENERIC_BRANCHES_V2_FOLLOWUP_PROMPT
    + "\n\nBefore choosing SINGLE, verify that every plausible semantic dimension was "
    "tested with focused grounding. Do not treat an unsearched dimension as ruled "
    "out."
)
MUSE_REFERENCE_GENERIC_BRANCHES_V3_PROMPT_SHA256 = hashlib.sha256(
    f"{MUSE_REFERENCE_GENERIC_BRANCHES_V3_INITIAL_PROMPT}\n\n{MUSE_REFERENCE_GENERIC_BRANCHES_V3_FOLLOWUP_PROMPT}".encode(
        "utf-8"
    )
).hexdigest()


def _prompt_pair(prompt_variant: str) -> tuple[str, str, str]:
    if prompt_variant == "answer":
        return MUSE_INITIAL_PROMPT, MUSE_FOLLOWUP_PROMPT, MUSE_PROMPT_SHA256
    if prompt_variant == "reference":
        return (
            MUSE_REFERENCE_INITIAL_PROMPT,
            MUSE_REFERENCE_FOLLOWUP_PROMPT,
            MUSE_REFERENCE_PROMPT_SHA256,
        )
    if prompt_variant == "reference_generic":
        return (
            MUSE_REFERENCE_GENERIC_INITIAL_PROMPT,
            MUSE_REFERENCE_GENERIC_FOLLOWUP_PROMPT,
            MUSE_REFERENCE_GENERIC_PROMPT_SHA256,
        )
    if prompt_variant == "reference_generic_citations":
        return (
            MUSE_REFERENCE_GENERIC_CITATIONS_INITIAL_PROMPT,
            MUSE_REFERENCE_GENERIC_CITATIONS_FOLLOWUP_PROMPT,
            MUSE_REFERENCE_GENERIC_CITATIONS_PROMPT_SHA256,
        )
    if prompt_variant == "reference_generic_uncertainty":
        return (
            MUSE_REFERENCE_GENERIC_UNCERTAINTY_INITIAL_PROMPT,
            MUSE_REFERENCE_GENERIC_UNCERTAINTY_FOLLOWUP_PROMPT,
            MUSE_REFERENCE_GENERIC_UNCERTAINTY_PROMPT_SHA256,
        )
    if prompt_variant == "reference_generic_branches":
        return (
            MUSE_REFERENCE_GENERIC_BRANCHES_INITIAL_PROMPT,
            MUSE_REFERENCE_GENERIC_BRANCHES_FOLLOWUP_PROMPT,
            MUSE_REFERENCE_GENERIC_BRANCHES_PROMPT_SHA256,
        )
    if prompt_variant == "reference_generic_branches_v2":
        return (
            MUSE_REFERENCE_GENERIC_BRANCHES_V2_INITIAL_PROMPT,
            MUSE_REFERENCE_GENERIC_BRANCHES_V2_FOLLOWUP_PROMPT,
            MUSE_REFERENCE_GENERIC_BRANCHES_V2_PROMPT_SHA256,
        )
    if prompt_variant == "reference_generic_branches_v3":
        return (
            MUSE_REFERENCE_GENERIC_BRANCHES_V3_INITIAL_PROMPT,
            MUSE_REFERENCE_GENERIC_BRANCHES_V3_FOLLOWUP_PROMPT,
            MUSE_REFERENCE_GENERIC_BRANCHES_V3_PROMPT_SHA256,
        )
    raise ValueError(
        "prompt_variant must be 'answer', 'reference', 'reference_generic', "
        "'reference_generic_citations', 'reference_generic_uncertainty', "
        "'reference_generic_branches', or "
        "'reference_generic_branches_v2', or 'reference_generic_branches_v3'"
    )


def muse_prompt_sha256(prompt_variant: str) -> str:
    return _prompt_pair(prompt_variant)[2]


class MuseRunError(RuntimeError):
    """A first-stage Muse failure with non-secret diagnostics for the cache."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__("Muse run failed")
        self.payload = dict(payload)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_muse_initial_request(
    question: str, *, model: str = MUSE_MODEL, prompt_variant: str = "answer"
) -> dict[str, Any]:
    """Build the first grounded Responses request without output/search caps."""

    if not question.strip():
        raise ValueError("question must be non-empty")
    initial_prompt, _, _ = _prompt_pair(prompt_variant)
    return {
        "model": model,
        "input": f"{initial_prompt}\n\nUser question:\n{question}",
        "tools": [{"type": "web_search"}],
        "store": True,
    }


def build_muse_followup_request(
    previous_response_id: str,
    *,
    model: str = MUSE_MODEL,
    prompt_variant: str = "answer",
) -> dict[str, Any]:
    """Build the generic second-pass request using the prior response context."""

    if not previous_response_id.strip():
        raise ValueError("previous_response_id must be non-empty")
    _, followup_prompt, _ = _prompt_pair(prompt_variant)
    return {
        "model": model,
        "previous_response_id": previous_response_id,
        "input": followup_prompt,
        "tools": [{"type": "web_search"}],
        "store": True,
    }


def _numeric_tree(value: object) -> object:
    if isinstance(value, Mapping):
        result = {
            str(key): child
            for key, item in value.items()
            if (child := _numeric_tree(item)) is not None
        }
        return result or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _annotation_payload(annotation: Mapping[str, Any]) -> dict[str, Any] | None:
    if annotation.get("type") != "url_citation":
        return None
    url = annotation.get("url")
    if not isinstance(url, str) or not url:
        return None
    payload: dict[str, Any] = {"url": url}
    title = annotation.get("title")
    if isinstance(title, str):
        payload["title"] = title
    for field in ("start_index", "end_index"):
        value = annotation.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            payload[field] = value
    return payload


def parse_muse_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the final answer and all structured web provenance from a response."""

    if not isinstance(response, Mapping):
        raise ValueError("Muse response must be an object")
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("Muse response contains no output list")

    assistant_messages: list[Mapping[str, Any]] = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "message" and item.get("role") == "assistant":
            assistant_messages.append(item)
    final_message: Mapping[str, Any] | None = None
    final_text = ""
    for message in reversed(assistant_messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text_parts = [
            str(part["text"])
            for part in content
            if isinstance(part, Mapping)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ]
        if text_parts:
            final_message = message
            final_text = "".join(text_parts).strip()
            break
    if final_message is None or not final_text:
        raise ValueError("Muse response contains no final assistant text")

    search_actions: list[dict[str, Any]] = []
    search_queries: list[str] = []
    opened_urls: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "web_search_call":
            continue
        action = item.get("action")
        if not isinstance(action, Mapping):
            continue
        action_type = action.get("type")
        payload: dict[str, Any] = {"type": str(action_type or "unknown")}
        item_id = item.get("id")
        if isinstance(item_id, str):
            payload["id"] = item_id
        if action_type == "search" and isinstance(action.get("query"), str):
            payload["query"] = action["query"]
            search_queries.append(action["query"])
        elif action_type == "open_page" and isinstance(action.get("url"), str):
            payload["url"] = action["url"]
            opened_urls.append(action["url"])
        elif action_type == "find_in_page":
            url = action.get("url")
            pattern = action.get("pattern")
            if isinstance(url, str):
                payload["url"] = url
                opened_urls.append(url)
            if isinstance(pattern, str):
                payload["pattern"] = pattern
        search_actions.append(payload)

    citations: list[dict[str, Any]] = []
    content = final_message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, Mapping):
                continue
            annotations = part.get("annotations")
            if not isinstance(annotations, list):
                continue
            for annotation in annotations:
                if isinstance(annotation, Mapping):
                    parsed = _annotation_payload(annotation)
                    if parsed is not None:
                        citations.append(parsed)

    status = response.get("status")
    status_value = str(status) if isinstance(status, str) else "unknown"
    incomplete_details = response.get("incomplete_details")
    safe_incomplete = (
        dict(incomplete_details)
        if isinstance(incomplete_details, Mapping)
        else None
    )
    usage = response.get("usage")
    safe_usage = _numeric_tree(usage) if isinstance(usage, Mapping) else {}
    if not isinstance(safe_usage, Mapping):
        safe_usage = {}
    response_id = response.get("id")
    return {
        "response_id": response_id if isinstance(response_id, str) else None,
        "model": response.get("model") if isinstance(response.get("model"), str) else None,
        "answer": final_text,
        "status": status_value,
        "incomplete_details": safe_incomplete,
        "usage": dict(safe_usage),
        "citations": citations,
        "search_actions": search_actions,
        "search_queries": search_queries,
        "opened_urls": opened_urls,
        "search_call_count": len(search_actions),
    }


def _failure_payload(exc: Exception) -> dict[str, Any]:
    status: int | None = None
    category = "provider"
    retryable = False
    if isinstance(exc, ProviderRequestError):
        category = exc.category
        status = exc.http_status
        retryable = exc.retryable
    elif isinstance(exc, urllib.error.HTTPError):
        category = "http"
        status = exc.code
        retryable = exc.code in RETRIABLE_STATUS_CODES
    elif isinstance(exc, (TimeoutError, urllib.error.URLError)):
        category = "transport"
        retryable = True
    return {
        "category": category,
        "error_type": type(exc).__name__,
        "http_status": status,
        "retryable": retryable,
    }


def _incomplete_payload(parsed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "category": "incomplete",
        "error_type": "IncompleteResponse",
        "http_status": None,
        "retryable": False,
        "status": parsed.get("status"),
        "incomplete_details": parsed.get("incomplete_details"),
    }


def _stage_is_usable(parsed: Mapping[str, Any]) -> bool:
    return (
        parsed.get("status") == "completed"
        and isinstance(parsed.get("answer"), str)
        and bool(parsed["answer"].strip())
    )


def _call_muse(
    body: dict[str, Any],
    *,
    api_key: str,
    request_fn: RequestFunction,
    sleep_fn: SleepFunction,
) -> tuple[Mapping[str, Any], int]:
    if not api_key:
        raise ValueError("MODEL_API_KEY is required")
    return _request_with_retries(
        MUSE_ENDPOINT,
        body,
        {"Authorization": f"Bearer {api_key}"},
        request_fn=request_fn,
        sleep_fn=sleep_fn,
    )


def _combine_numeric_usage(*usages: Mapping[str, Any]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for usage in usages:
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[field] = totals.get(field, 0) + value
    return totals


def _ordered_union(*values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value_list in values:
        for value in value_list:
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


def _stage_record(
    request: dict[str, Any],
    response: Mapping[str, Any] | None,
    parsed: Mapping[str, Any] | None,
    *,
    elapsed_seconds: float,
    attempts: int | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "request": request,
        "elapsed_seconds": elapsed_seconds,
    }
    if attempts is not None:
        record["attempts"] = attempts
    if response is not None:
        record["raw_response"] = dict(response)
    if parsed is not None:
        record["parsed"] = dict(parsed)
    if error is not None:
        record["error"] = dict(error)
    return record


def run_muse_two_call(
    question: str,
    *,
    api_key: str,
    model: str = MUSE_MODEL,
    prompt_variant: str = "answer",
    request_fn: RequestFunction = _post_json,
    sleep_fn: SleepFunction = time.sleep,
) -> dict[str, Any]:
    """Run the initial and generic follow-up calls, with explicit stage fallback."""

    if not api_key:
        raise ValueError("MODEL_API_KEY is required")
    _, _, prompt_sha256 = _prompt_pair(prompt_variant)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    initial_request = build_muse_initial_request(
        question, model=model, prompt_variant=prompt_variant
    )
    try:
        initial_raw, initial_attempts = _call_muse(
            initial_request,
            api_key=api_key,
            request_fn=request_fn,
            sleep_fn=sleep_fn,
        )
    except Exception as exc:
        raise MuseRunError(_failure_payload(exc)) from exc
    initial_elapsed = time.monotonic() - started_monotonic
    try:
        initial_parsed = parse_muse_response(initial_raw)
    except Exception as exc:
        failure = _failure_payload(exc)
        failure["call_1"] = _stage_record(
            initial_request,
            initial_raw,
            None,
            elapsed_seconds=initial_elapsed,
            attempts=initial_attempts,
            error=failure,
        )
        raise MuseRunError(failure) from exc
    initial_stage = _stage_record(
        initial_request,
        initial_raw,
        initial_parsed,
        elapsed_seconds=initial_elapsed,
        attempts=initial_attempts,
    )
    if not _stage_is_usable(initial_parsed):
        failure = _incomplete_payload(initial_parsed)
        failure["call_1"] = _stage_record(
            initial_request,
            initial_raw,
            initial_parsed,
            elapsed_seconds=initial_elapsed,
            attempts=initial_attempts,
            error=failure,
        )
        raise MuseRunError(failure)

    previous_response_id = initial_parsed.get("response_id")
    if not isinstance(previous_response_id, str) or not previous_response_id:
        followup_failure = {
            "category": "missing_response_id",
            "error_type": "MissingResponseId",
            "http_status": None,
            "retryable": False,
        }
        followup_stage = {
            "request": None,
            "elapsed_seconds": 0.0,
            "error": followup_failure,
        }
        final_stage = 1
        final_parsed = initial_parsed
    else:
        followup_request = build_muse_followup_request(
            previous_response_id, model=model, prompt_variant=prompt_variant
        )
        followup_started = time.monotonic()
        followup_raw: Mapping[str, Any] | None = None
        followup_parsed: dict[str, Any] | None = None
        followup_attempts: int | None = None
        followup_failure: dict[str, Any] | None = None
        try:
            followup_raw, followup_attempts = _call_muse(
                followup_request,
                api_key=api_key,
                request_fn=request_fn,
                sleep_fn=sleep_fn,
            )
            followup_parsed = parse_muse_response(followup_raw)
            if not _stage_is_usable(followup_parsed):
                followup_failure = _incomplete_payload(followup_parsed)
        except Exception as exc:
            followup_failure = _failure_payload(exc)
        followup_stage = _stage_record(
            followup_request,
            followup_raw,
            followup_parsed,
            elapsed_seconds=time.monotonic() - followup_started,
            attempts=followup_attempts,
            error=followup_failure,
        )
        if followup_failure is None and followup_parsed is not None:
            final_stage = 2
            final_parsed = followup_parsed
        else:
            final_stage = 1
            final_parsed = initial_parsed

    usages = [initial_parsed.get("usage", {})]
    if isinstance(followup_stage.get("parsed"), Mapping):
        usages.append(followup_stage["parsed"].get("usage", {}))
    final_record: dict[str, Any] = {
        "schema_version": 1,
        "provider": "meta-muse",
        "model": model,
        "question": question,
        "prompt_variant": prompt_variant,
        "prompt_sha256": prompt_sha256,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "call_1": initial_stage,
        "call_2": followup_stage,
        "final_stage": final_stage,
        "answer": final_parsed["answer"],
        "status": final_parsed["status"],
        "response_id": final_parsed.get("response_id"),
        "citations": final_parsed.get("citations", []),
        "search_actions": [
            *initial_parsed.get("search_actions", []),
            *(
                followup_stage.get("parsed", {}).get("search_actions", [])
                if isinstance(followup_stage.get("parsed"), Mapping)
                else []
            ),
        ],
        "search_queries": _ordered_union(
            initial_parsed.get("search_queries", []),
            followup_stage.get("parsed", {}).get("search_queries", [])
            if isinstance(followup_stage.get("parsed"), Mapping)
            else [],
        ),
        "opened_urls": _ordered_union(
            initial_parsed.get("opened_urls", []),
            followup_stage.get("parsed", {}).get("opened_urls", [])
            if isinstance(followup_stage.get("parsed"), Mapping)
            else [],
        ),
        "usage": {
            "call_1": initial_parsed.get("usage", {}),
            "call_2": followup_stage.get("parsed", {}).get("usage", {})
            if isinstance(followup_stage.get("parsed"), Mapping)
            else {},
            "combined": _combine_numeric_usage(*usages),
        },
        "answer_characters": len(final_parsed["answer"]),
        "answer_words": len(final_parsed["answer"].split()),
    }
    if final_stage == 1:
        final_record["fallback_reason"] = "followup_failed"
        final_record["followup_failure"] = followup_stage["error"]
    return final_record


def load_muse_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "records": {}, "failures": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("records"), dict)
        or not isinstance(payload.get("failures"), dict)
    ):
        raise ValueError("Muse cache must contain records and failures objects")
    return payload


def save_muse_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache["updated_at"] = utc_now()
    serialized = json.dumps(cache, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def is_complete_muse_record(
    record: object, *, model: str = MUSE_MODEL, prompt_variant: str | None = None
) -> bool:
    variant_matches = (
        prompt_variant is None
        or (
            isinstance(record, Mapping)
            and record.get("prompt_variant", "answer") == prompt_variant
        )
    )
    prompt_matches = (
        prompt_variant is None
        or (
            isinstance(record, Mapping)
            and record.get("prompt_sha256") == muse_prompt_sha256(prompt_variant)
        )
    )
    return (
        isinstance(record, Mapping)
        and record.get("provider") == "meta-muse"
        and record.get("model") == model
        and variant_matches
        and prompt_matches
        and isinstance(record.get("answer"), str)
        and bool(record.get("answer", "").strip())
        and record.get("final_stage") in {1, 2}
        and isinstance(record.get("call_1"), Mapping)
        and isinstance(record.get("call_2"), Mapping)
    )


def select_canary_questions(
    questions: list[tuple[str, str]], *, size: int, seed: int
) -> list[tuple[str, str]]:
    """Select a deterministic stable-hash sample without using answer labels."""

    if size < 1 or size > len(questions):
        raise ValueError("size must be between one and the number of questions")
    if len({question_id for question_id, _ in questions}) != len(questions):
        raise ValueError("question IDs must be unique")
    ranked = sorted(
        questions,
        key=lambda item: hashlib.sha256(
            f"{seed}\0{item[0]}\0{item[1]}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:size]
