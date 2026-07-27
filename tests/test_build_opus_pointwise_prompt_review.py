import hashlib
import json
import stat
from pathlib import Path

from scripts.build_opus_pointwise_prompt_review import main
from trec_mlm.pointwise_production import build_opus_prompt_review_packet


def test_review_packet_binds_exact_v2_contract_and_blinds_judge_identity(
    tmp_path: Path,
) -> None:
    protocol_dir = Path(__file__).parents[1] / "protocols" / "pointwise_judge_bakeoff"
    packet_path = tmp_path / "opus-review.json"

    packet = build_opus_prompt_review_packet(
        protocol_dir=protocol_dir,
        output_path=packet_path,
    )

    prompt = (protocol_dir / "pointwise-v2.txt").read_text(encoding="utf-8")
    schema_text = (protocol_dir / "decision-v2.schema.json").read_text(encoding="utf-8")
    serialized = packet_path.read_text(encoding="utf-8")
    assert (
        packet["contract"]["prompt_sha256"]
        == hashlib.sha256(prompt.encode()).hexdigest()
    )
    assert (
        packet["contract"]["schema_sha256"]
        == hashlib.sha256(schema_text.encode()).hexdigest()
    )
    assert packet["contract"]["prompt"] == prompt
    assert packet["contract"]["schema"] == json.loads(schema_text)
    assert {"ambiguity", "reference evidence", "invariants", "repair behavior"} <= set(
        packet["adjudication_questions"]
    )
    assert "deepseek" not in serialized.casefold()
    assert "openrouter" not in serialized.casefold()
    assert "api key" not in serialized.casefold()
    assert stat.S_IMODE(packet_path.stat().st_mode) == 0o600


def test_review_packet_cli_creates_private_parent_and_is_idempotent(
    tmp_path: Path,
) -> None:
    protocol_dir = Path(__file__).parents[1] / "protocols" / "pointwise_judge_bakeoff"
    output_dir = tmp_path / "nested" / "private"
    packet_path = output_dir / "opus-review.json"

    assert (
        main(["--protocol-dir", str(protocol_dir), "--output", str(packet_path)]) == 0
    )
    first = packet_path.read_bytes()
    assert (
        main(["--protocol-dir", str(protocol_dir), "--output", str(packet_path)]) == 0
    )

    assert packet_path.read_bytes() == first
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(packet_path.stat().st_mode) == 0o600
