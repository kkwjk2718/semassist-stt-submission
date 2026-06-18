import json
from pathlib import Path

import pytest

from training.audit_lora_alignment import audit_manifest_alignment
from training.finetune_whisper_lora import main, run_lora_training
from training.lora_alignment import duration_seconds


def test_audit_manifest_alignment_blocks_long_rows(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                _row(tmp_path, "short.wav", 25.0),
                _row(tmp_path, "long.wav", 31.0),
                _row(tmp_path, "missing.wav", None),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_manifest_alignment(manifest_path, max_audio_seconds=30.0)

    assert audit.safe_to_train is False
    assert audit.total_rows == 3
    assert audit.aligned_rows == 1
    assert audit.window_mismatch_rows == 1
    assert audit.missing_duration_rows == 1


def test_lora_training_refuses_unaligned_window_transcripts(tmp_path: Path) -> None:
    audio_path = tmp_path / "long.wav"
    audio_path.write_bytes(b"fake audio")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "sample_000001",
                "audio_path": str(audio_path),
                "text": "긴 전체 전사",
                "duration_seconds": 60.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="mismatched_rows=1"):
        run_lora_training(
            train_manifest=manifest_path,
            model_name="openai/whisper-small",
            output_dir=tmp_path / "adapter",
            max_steps=1,
            num_train_epochs=1.0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            logging_steps=1,
            save_steps=None,
            dataloader_num_workers=0,
            max_audio_seconds=30.0,
            allow_window_transcript_mismatch=False,
        )


def test_lora_dry_run_reports_unaligned_window_transcripts(tmp_path: Path) -> None:
    audio_path = tmp_path / "long.wav"
    audio_path.write_bytes(b"fake audio")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "sample_000001",
                "audio_path": str(audio_path),
                "text": "긴 전체 전사",
                "duration_seconds": 60.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--train-manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "adapter"),
            "--dry-run",
        ]
    )

    assert exit_code == 2


def test_duration_seconds_parses_string_values() -> None:
    assert duration_seconds({"duration_seconds": "12.5"}) == 12.5


def _row(tmp_path: Path, file_name: str, duration: float | None) -> dict[str, object]:
    audio_path = tmp_path / file_name
    audio_path.write_bytes(b"fake audio")
    row: dict[str, object] = {
        "id": file_name,
        "audio_path": str(audio_path),
        "text": "테스트",
    }
    if duration is not None:
        row["duration_seconds"] = duration
    return row
