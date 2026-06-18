import json
from pathlib import Path

import pytest

from dataset.split_manifest import split_manifest


def test_split_manifest_keeps_speaker_rows_together(tmp_path: Path) -> None:
    manifest_path = tmp_path / "brain_all.jsonl"
    train_path = tmp_path / "brain_train.jsonl"
    validation_path = tmp_path / "brain_validation.jsonl"
    rows = [
        _row("sample_000001", "ID-02-25-N-KSM-02-01-M-45-JL.wav"),
        _row("sample_000002", "ID-02-25-N-KSM-02-02-M-45-JL.wav"),
        _row("sample_000003", "ID-02-25-N-PYG-01-01-M-47-SU.wav"),
        _row("sample_000004", "ID-02-25-N-LMB-06-01-F-72-GS.wav"),
    ]
    manifest_path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows),
        encoding="utf-8",
    )

    summary = split_manifest(
        input_path=manifest_path,
        train_output_path=train_path,
        validation_output_path=validation_path,
        validation_ratio=0.25,
    )

    train_rows = _read_jsonl(train_path)
    validation_rows = _read_jsonl(validation_path)
    assert summary.total_rows == 4
    assert summary.speaker_groups == 3
    assert len(train_rows) + len(validation_rows) == 4
    assert _speaker_ids(train_rows).isdisjoint(_speaker_ids(validation_rows))


def test_split_manifest_rejects_invalid_ratio(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="validation_ratio"):
        split_manifest(
            input_path=tmp_path / "missing.jsonl",
            train_output_path=tmp_path / "train.jsonl",
            validation_output_path=tmp_path / "validation.jsonl",
            validation_ratio=1.0,
        )


def _row(sample_id: str, file_name: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "audio_path": str(Path("audio") / file_name),
        "text": "테스트 문장",
        "disorder_type": "brain_neurologic",
        "split": "train",
        "duration_seconds": 60.0,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _speaker_ids(rows: list[dict[str, object]]) -> set[str]:
    return {"-".join(Path(str(row["audio_path"])).stem.split("-")[:5]) for row in rows}
