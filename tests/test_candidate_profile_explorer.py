import json

import pytest

from scripts.build_candidate_profile_explorer import (
    _cluster_details,
    _prediction_rows,
    render_html,
)


def test_prediction_rows_rejects_nonfinite_scores(tmp_path):
    path = tmp_path / "predictions.tsv"
    path.write_text(
        "query_id\tcandidate_id\trank\tscore\tprofile_score\twinning_centroid_id\n"
        "q1\tc1\t1\tnan\t0.5\t0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid prediction q1/c1"):
        _prediction_rows(path)


def test_cluster_details_fails_closed_when_representative_is_unresolved(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    records_path = tmp_path / "records.json"
    profiles_path.write_text(
        json.dumps(
            {
                "c1": {
                    "available": True,
                    "clusters": [
                        {
                            "cluster_id": 0,
                            "representative_questions": ["missing"],
                            "member_record_ids": ["r1"],
                            "support_count": 1,
                            "dispersion": 0.0,
                            "medoid_record_id": "r1",
                            "annotation": {
                                "status": "mixed",
                                "primary_label": None,
                                "secondary_label": None,
                                "margin": 0.0,
                            },
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    records_path.write_text(
        json.dumps([{"record_id": "r1", "question": "different", "response": "a"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not resolve exactly once"):
        _cluster_details(profiles_path, records_path)


def test_render_html_names_unlabeled_centroids_mixed_or_unclear():
    output = render_html({"meta": {"title": "Explorer"}})

    assert "'Mixed / unclear'" in output
    assert "c.primary_label||'Mixed / unclear'" in output
