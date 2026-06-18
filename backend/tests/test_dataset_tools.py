import json
from pathlib import Path

from dataset.build_manifest import build_manifest
from dataset.common import find_duration_seconds
from dataset.inspect_labels import inspect_label_root
from dataset.sample_manifest import sample_manifest


def test_dataset_tools_inspect_build_and_sample_fixture(tmp_path: Path) -> None:
    label_root = tmp_path / "labels"
    audio_root = tmp_path / "audio"
    output_root = tmp_path / "outputs"

    _write_json(
        label_root / "1.Training" / "TL01" / "sample_a.json",
        {
            "metadata": {"fileName": "brain/sample_a.wav"},
            "transcription": "학교 가요",
            "duration": 1.25,
        },
    )
    _write_json(
        label_root / "2.Validation" / "TL01" / "sample_b.json",
        {
            "audio": {"path": "brain/sample_b.wav"},
            "sentence": {"text": "도와주세요"},
            "duration_seconds": 2.5,
        },
    )
    (audio_root / "brain").mkdir(parents=True)
    (audio_root / "brain" / "sample_a.wav").write_bytes(b"audio-a")
    (audio_root / "brain" / "sample_b.wav").write_bytes(b"audio-b")

    summary_path = output_root / "label_structure_summary.md"
    summary = inspect_label_root(label_root, summary_path, max_files=10)

    assert summary.json_files == 2
    assert summary_path.exists()
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "metadata.fileName" in summary_text
    assert "transcription" in summary_text
    assert "sentence.text" in summary_text

    manifest_path = output_root / "brain_all.jsonl"
    manifest_rows = build_manifest(
        label_root=label_root,
        audio_root=audio_root,
        output_path=manifest_path,
        disorder_type="brain_neurologic",
    )

    assert len(manifest_rows) == 2
    persisted_rows = _read_jsonl(manifest_path)
    assert persisted_rows[0] == {
        "id": "sample_000001",
        "audio_path": str(audio_root / "brain" / "sample_a.wav"),
        "text": "학교 가요",
        "disorder_type": "brain_neurologic",
        "split": "train",
        "duration_seconds": 1.25,
    }
    assert persisted_rows[1]["text"] == "도와주세요"
    assert persisted_rows[1]["split"] == "validation"

    sample_path = output_root / "brain_demo_2.jsonl"
    sampled_rows = sample_manifest(
        input_path=manifest_path,
        output_path=sample_path,
        max_samples=2,
    )

    assert sampled_rows == persisted_rows
    assert _read_jsonl(sample_path) == persisted_rows


def test_build_manifest_matches_aihub_filename_variants(tmp_path: Path) -> None:
    label_root = tmp_path / "labels"
    audio_root = tmp_path / "audio"
    output_path = tmp_path / "outputs" / "brain_all.jsonl"
    cases = [
        ("duplicate_one.json", "sample중복1.wav", "sample.wav"),
        ("duplicate_two.json", "sample중복2.wav", "sample2.wav"),
        ("upper_ext.json", "case.wav", "case.WAV"),
        ("wrong_disorder_letter.json", "ID-02-26-M-YHY-04-F-28-SU.wav", "ID-02-26-N-YHY-04-F-28-SU.wav"),
    ]
    for index, (label_name, file_id, audio_name) in enumerate(cases, start=1):
        _write_json(
            label_root / "1.Training" / "TL01" / label_name,
            {"File_id": file_id, "Transcript": f"문장 {index}"},
        )
        audio_path = audio_root / "nested" / audio_name
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"audio")

    rows = build_manifest(
        label_root=label_root,
        audio_root=audio_root,
        output_path=output_path,
    )

    assert len(rows) == 4
    assert [Path(str(row["audio_path"])).name for row in rows] == [
        "sample.wav",
        "sample2.wav",
        "case.WAV",
        "ID-02-26-N-YHY-04-F-28-SU.wav",
    ]


def test_find_duration_seconds_reads_aihub_play_time() -> None:
    payload = {"Meta_info": {"PlayTime": 36.5}}

    duration_seconds = find_duration_seconds(payload)

    assert duration_seconds == 36.5


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
