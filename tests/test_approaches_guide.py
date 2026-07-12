from scripts.build_approaches_guide import (
    _dense_metrics,
    _discovery_profile_audit,
    _llm_profiles,
    _load_local_data,
    build_artifact,
)


def test_profile_quickstart_is_immediately_after_contents() -> None:
    artifact = build_artifact(
        _load_local_data(),
        _dense_metrics(),
        _discovery_profile_audit(),
        _llm_profiles(),
    )
    block_ids = [block["id"] for block in artifact["manifest"]["blocks"]]
    assert block_ids[:9] == [
        "title",
        "technical_summary",
        "reader_map",
        "expert_profiles_deep_dive",
        "beginner-visual-profile-anatomy",
        "beginner-visual-real-discovery-profile",
        "profile_build_explained",
        "beginner-visual-profile-build-pipeline",
        "candidate_profiles_table_block",
    ]
    assert len(block_ids) == len(set(block_ids)) == 84
