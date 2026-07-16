import hashlib

from trec_mlm.artifact_io import (
    length_framed_sha256,
    manifest_aggregate_sha256,
    manifest_entries,
    sha256_file,
    source_tree_sha256,
    write_pretty_json,
)


def test_write_pretty_json_has_frozen_utf8_bytes(tmp_path) -> None:
    path = tmp_path / "nested" / "payload.json"

    write_pretty_json(path, {"z": "café", "a": [1, True, None]})

    payload = path.read_bytes()
    assert len(payload) == 59
    assert payload.endswith(b"\n")
    assert payload == (
        b'{\n  "a": [\n    1,\n    true,\n    null\n  ],\n'
        b'  "z": "caf\xc3\xa9"\n}\n'
    )
    assert hashlib.sha256(payload).hexdigest() == (
        "7b494852f8424e42f70ebb61d358c329c048ceaacffed872df5e764c451d673e"
    )


def test_write_pretty_json_preserves_default_nan_behavior(tmp_path) -> None:
    path = tmp_path / "payload.json"

    write_pretty_json(path, {"value": float("nan")})

    assert path.read_text(encoding="utf-8") == '{\n  "value": NaN\n}\n'


def test_length_framed_sha256_has_frozen_utf8_hash() -> None:
    assert length_framed_sha256(("a", "β", "")) == (
        "0741cb3a91300dca6785fe5931bd1bfe28a1f620834ac49737d55205af638212"
    )


def test_sha256_file_matches_standard_abc_vector(tmp_path) -> None:
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")

    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_file_streams_file_larger_than_one_mib(tmp_path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"x" * (1024 * 1024 + 17))

    assert sha256_file(path) == (
        "baf900d02555b923124540fd1a01ef1b3f033da00244ce97a188d7108d4fa6d3"
    )


def test_manifest_entries_excludes_all_manifests_and_sorts_posix_keys(
    tmp_path,
) -> None:
    (tmp_path / "z.txt").write_bytes(b"z")
    (tmp_path / "manifest.json").write_text("excluded", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.txt").write_bytes(b"alpha")
    (nested / "manifest.json").write_text("also excluded", encoding="utf-8")

    entries = manifest_entries(tmp_path)

    assert list(entries) == ["nested/a.txt", "z.txt"]
    assert entries == {
        "nested/a.txt": {
            "byte_size": 5,
            "sha256": "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8",
        },
        "z.txt": {
            "byte_size": 1,
            "sha256": "594e519ae499312b29433b7dd8a97ff068defcba9755b6d5d00e84c524d67b06",
        },
    }


def test_manifest_aggregate_sha256_has_frozen_escaped_unicode_hash() -> None:
    entries = {
        "a.txt": {"byte_size": 1, "sha256": "a" * 64},
        "β.txt": {"byte_size": 2, "sha256": "b" * 64},
    }

    assert manifest_aggregate_sha256(entries) == (
        "5e33ce02b94bd9d959981775654cf52f7882221cef483246524e81a76eb6b0db"
    )


def test_source_tree_sha256_has_frozen_path_and_payload_framing(tmp_path) -> None:
    (tmp_path / "a.py").write_bytes(b"print(1)\n")
    nested = tmp_path / "b"
    nested.mkdir()
    (nested / "c.py").write_text("π = 3\n", encoding="utf-8")
    (nested / "ignored.txt").write_text("not Python", encoding="utf-8")

    assert source_tree_sha256(tmp_path) == (
        "82cc50ade6b1554b9a32c98f61e9ee3eb5c9aad7df0b347f111bbc0d3d56f3c8"
    )
