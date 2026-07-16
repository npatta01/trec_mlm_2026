from __future__ import annotations

import ast
import inspect

import pytest

from trec_mlm import (
    label_experiment,
    label_inference,
    profile_experiment,
    profile_inference_artifact,
    profile_multiresolution,
)


@pytest.mark.parametrize(
    ("module", "public_name", "private_name"),
    (
        (profile_experiment, "profile_to_json", "_profile_to_json"),
        (
            profile_experiment,
            "recompute_profile_centroid",
            "_recompute_profile_centroid",
        ),
        (
            profile_experiment,
            "validate_complete_rankings",
            "_validate_complete_rankings",
        ),
        (label_inference, "encode_contextual_e5", "_encode_contextual_e5"),
        (
            label_experiment,
            "load_label_experiment_artifacts",
            "_load_label_experiment_artifacts",
        ),
    ),
)
def test_cross_module_helpers_have_public_exports_with_private_aliases(
    module: object,
    public_name: str,
    private_name: str,
) -> None:
    public = getattr(module, public_name)
    private = getattr(module, private_name)

    assert public is private
    assert public_name in module.__all__


@pytest.mark.parametrize(
    ("consumer", "source_module", "expected_aliases"),
    (
        (
            profile_multiresolution,
            "trec_mlm.profile_experiment",
            {
                "profile_to_json": "_profile_to_json",
                "recompute_profile_centroid": "_recompute_profile_centroid",
                "validate_complete_rankings": "_validate_complete_rankings",
            },
        ),
        (
            profile_inference_artifact,
            "trec_mlm.label_experiment",
            {"load_label_experiment_artifacts": "_load_label_experiment_artifacts"},
        ),
        (
            profile_inference_artifact,
            "trec_mlm.label_inference",
            {"encode_contextual_e5": "_encode_contextual_e5"},
        ),
    ),
)
def test_consumers_import_public_boundaries_under_compatible_local_names(
    consumer: object,
    source_module: str,
    expected_aliases: dict[str, str],
) -> None:
    tree = ast.parse(inspect.getsource(consumer))
    imported_aliases = {
        alias.name: alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == source_module
        for alias in node.names
    }

    assert expected_aliases.items() <= imported_aliases.items()
