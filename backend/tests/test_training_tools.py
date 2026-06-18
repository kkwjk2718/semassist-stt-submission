import csv
import json
from pathlib import Path

import pytest

from app.audio import AudioTranscription
from training import compare_asr_outputs
from training.compare_asr_outputs import compare_manifest
from training.finetune_whisper_lora import inspect_training_manifest, read_training_rows


def test_compare_manifest_writes_asr_comparison_csv(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "sample_000001",
                "audio_path": str(audio_path),
                "text": "학교 가요",
                "disorder_type": "brain_neurologic",
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "asr_comparison.csv"

    rows = compare_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        transcribe_audio=lambda path: "학교 가나요",
    )

    assert rows[0].base_asr == "학교 가나요"
    assert rows[0].error_type == "statement_to_question"
    with output_path.open(encoding="utf-8-sig", newline="") as handle:
        persisted_rows = list(csv.DictReader(handle))
    assert persisted_rows == [
        {
            "id": "sample_000001",
            "reference": "학교 가요",
            "base_asr": "학교 가나요",
            "error_type": "statement_to_question",
            "note": "Reference is a statement but ASR looks like a question.",
        }
    ]


def test_compare_manifest_reuses_default_asr_provider(monkeypatch, tmp_path: Path) -> None:
    first_audio_path = tmp_path / "first.wav"
    second_audio_path = tmp_path / "second.wav"
    first_audio_path.write_bytes(b"fake audio")
    second_audio_path.write_bytes(b"fake audio")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows = [
        {
            "id": "sample_000001",
            "audio_path": str(first_audio_path),
            "text": "학교 가요",
            "disorder_type": "brain_neurologic",
            "split": "train",
        },
        {
            "id": "sample_000002",
            "audio_path": str(second_audio_path),
            "text": "학교 가요",
            "disorder_type": "brain_neurologic",
            "split": "train",
        },
    ]
    manifest_path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in manifest_rows),
        encoding="utf-8",
    )
    FakeComparisonProvider.init_count = 0
    monkeypatch.setattr(compare_asr_outputs, "FasterWhisperAsrProvider", FakeComparisonProvider)

    rows = compare_asr_outputs.compare_manifest(
        manifest_path=manifest_path,
        output_path=tmp_path / "asr_comparison.csv",
    )

    assert FakeComparisonProvider.init_count == 1
    assert [row.base_asr for row in rows] == ["학교 가요", "학교 가요"]


def test_compare_manifest_raises_when_default_asr_warns(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "sample_000001",
                "audio_path": str(audio_path),
                "text": "학교 가요",
                "disorder_type": "brain_neurologic",
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(compare_asr_outputs, "FasterWhisperAsrProvider", FakeWarningProvider)

    with pytest.raises(RuntimeError, match="ASR benchmark failed"):
        compare_asr_outputs.compare_manifest(
            manifest_path=manifest_path,
            output_path=tmp_path / "asr_comparison.csv",
        )


def test_compare_manifest_resumes_existing_csv(tmp_path: Path) -> None:
    first_audio_path = tmp_path / "first.wav"
    second_audio_path = tmp_path / "second.wav"
    first_audio_path.write_bytes(b"fake audio")
    second_audio_path.write_bytes(b"fake audio")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows = [
        {
            "id": "sample_000001",
            "audio_path": str(first_audio_path),
            "text": "학교 가요",
            "disorder_type": "brain_neurologic",
            "split": "train",
        },
        {
            "id": "sample_000002",
            "audio_path": str(second_audio_path),
            "text": "학교 가요",
            "disorder_type": "brain_neurologic",
            "split": "train",
        },
    ]
    manifest_path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in manifest_rows),
        encoding="utf-8",
    )
    output_path = tmp_path / "asr_comparison.csv"
    output_path.write_text(
        "\ufeffid,reference,base_asr,error_type,note\r\n"
        "sample_000001,학교 가요,학교 가요,exact_match,ASR matched the reference.\r\n",
        encoding="utf-8",
    )
    transcribed_paths: list[str] = []

    def transcribe_audio(path: Path) -> str:
        transcribed_paths.append(path.name)
        return "학교 가요"

    rows = compare_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        transcribe_audio=transcribe_audio,
        resume=True,
    )

    assert transcribed_paths == ["second.wav"]
    assert [row.id for row in rows] == ["sample_000001", "sample_000002"]
    with output_path.open(encoding="utf-8-sig", newline="") as handle:
        persisted_rows = list(csv.DictReader(handle))
    assert [row["id"] for row in persisted_rows] == ["sample_000001", "sample_000002"]


def test_compare_manifest_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "sample_000001",
                "audio_path": str(audio_path),
                "text": "학교 가요",
                "disorder_type": "brain_neurologic",
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        compare_manifest(
            manifest_path=manifest_path,
            output_path=tmp_path / "asr_comparison.csv",
            batch_size=0,
        )


def test_inspect_training_manifest_reports_ready_samples(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "sample_000001",
                "audio_path": str(audio_path),
                "text": "학교 가요",
                "disorder_type": "brain_neurologic",
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    inspection = inspect_training_manifest(manifest_path)

    assert inspection.total_rows == 1
    assert inspection.existing_audio_rows == 1
    assert inspection.empty_text_rows == 0
    assert inspection.ready is True


def test_read_training_rows_returns_manifest_audio_and_text(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "sample_000001",
                "audio_path": str(audio_path),
                "text": " 학교 가요 ",
                "disorder_type": "brain_neurologic",
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_training_rows(manifest_path)

    assert len(rows) == 1
    assert rows[0].id == "sample_000001"
    assert rows[0].audio_path == audio_path
    assert rows[0].text == "학교 가요"


class FakeComparisonProvider:
    init_count = 0

    def __init__(self) -> None:
        self.__class__.init_count += 1

    async def transcribe(self, audio_path: Path) -> AudioTranscription:
        if audio_path.name == "first.wav":
            return AudioTranscription(asr_text="학교 가요", warnings=())
        return AudioTranscription(asr_text="학교 가요", warnings=())


class FakeWarningProvider:
    async def transcribe(self, audio_path: Path) -> AudioTranscription:
        return AudioTranscription(asr_text="fallback", warnings=("not configured",))
