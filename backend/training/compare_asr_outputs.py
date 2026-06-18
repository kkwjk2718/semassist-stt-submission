import argparse
import csv
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import anyio

if __package__:
    from app.audio import FasterWhisperAsrProvider
    from dataset.common import read_jsonl
else:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app.audio import FasterWhisperAsrProvider
    from dataset.common import read_jsonl

CSV_FIELDNAMES: Final = ("id", "reference", "base_asr", "error_type", "note")


@dataclass(frozen=True, slots=True)
class AsrComparisonRow:
    id: str
    reference: str
    base_asr: str
    error_type: str
    note: str


def compare_manifest(
    manifest_path: Path,
    output_path: Path,
    transcribe_audio: Callable[[Path], str] | None = None,
    max_samples: int | None = None,
    resume: bool = False,
    batch_size: int | None = None,
) -> list[AsrComparisonRow]:
    manifest_rows = read_jsonl(manifest_path)
    selected_rows = manifest_rows[:max_samples] if max_samples is not None else manifest_rows
    selected_ids = {str(row["id"]) for row in selected_rows}
    resume_rows = [row for row in read_comparison_csv(output_path) if row.id in selected_ids] if resume else []
    completed_ids = {row.id for row in resume_rows}
    comparison_rows: list[AsrComparisonRow] = list(resume_rows)
    if not resume or not output_path.exists():
        initialize_comparison_csv(output_path)

    transcriber = transcribe_audio if transcribe_audio is not None else build_default_transcriber(batch_size)

    for index, row in enumerate(selected_rows, start=1):
        row_id = str(row["id"])
        if row_id in completed_ids:
            print(f"progress={index}/{len(selected_rows)} id={row_id} skipped=resume")
            continue
        audio_path = Path(str(row["audio_path"]))
        reference = str(row["text"])
        base_asr = transcriber(audio_path)
        error_type, note = classify_error(reference, base_asr)
        comparison_row = AsrComparisonRow(
            id=row_id,
            reference=reference,
            base_asr=base_asr,
            error_type=error_type,
            note=note,
        )
        comparison_rows.append(comparison_row)
        append_comparison_csv_row(output_path, comparison_row)
        print(f"progress={index}/{len(selected_rows)} id={row_id}")

    return comparison_rows


def transcribe_with_faster_whisper(audio_path: Path) -> str:
    provider = FasterWhisperAsrProvider()
    transcription = anyio.run(provider.transcribe, audio_path)
    return transcription.asr_text


def build_default_transcriber(batch_size: int | None) -> Callable[[Path], str]:
    if batch_size is not None:
        return build_batched_faster_whisper_transcriber(batch_size)

    provider = FasterWhisperAsrProvider()

    def transcriber(audio_path: Path) -> str:
        transcription = anyio.run(provider.transcribe, audio_path)
        if transcription.warnings:
            joined_warnings = "; ".join(transcription.warnings)
            raise RuntimeError(f"ASR benchmark failed for {audio_path}: {joined_warnings}")
        return transcription.asr_text

    return transcriber


def build_batched_faster_whisper_transcriber(batch_size: int) -> Callable[[Path], str]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive: {batch_size}")

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model = WhisperModel(
        _env_text("WHISPER_MODEL_SIZE", "small"),
        device=_env_text("WHISPER_DEVICE", "cpu"),
        compute_type=_env_text("WHISPER_COMPUTE_TYPE", "int8"),
    )
    pipeline = BatchedInferencePipeline(model=model)

    def transcriber(audio_path: Path) -> str:
        try:
            segments, _info = pipeline.transcribe(
                str(audio_path),
                beam_size=5,
                language="ko",
                vad_filter=True,
                batch_size=batch_size,
            )
            text = "".join(segment.text for segment in segments).strip()
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"ASR benchmark failed for {audio_path}: {exc}") from exc

        if not text:
            raise RuntimeError(f"ASR benchmark failed for {audio_path}: empty transcription")
        return text

    return transcriber


def classify_error(reference: str, hypothesis: str) -> tuple[str, str]:
    normalized_reference = reference.strip()
    normalized_hypothesis = hypothesis.strip()
    if not normalized_hypothesis:
        return ("empty_asr", "ASR returned an empty transcription.")
    if normalized_reference == normalized_hypothesis:
        return ("exact_match", "ASR matched the reference.")
    if not normalized_reference.endswith("?") and normalized_hypothesis.endswith(("?", "요")):
        return ("statement_to_question", "Reference is a statement but ASR looks like a question.")
    if "안" in normalized_reference and "안" not in normalized_hypothesis:
        return ("negation_dropped", "Reference includes negation that ASR may have dropped.")
    return ("text_mismatch", "ASR differs from the reference.")


def write_comparison_csv(output_path: Path, rows: list[AsrComparisonRow]) -> None:
    initialize_comparison_csv(output_path)
    for row in rows:
        append_comparison_csv_row(output_path, row)


def initialize_comparison_csv(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()


def append_comparison_csv_row(output_path: Path, row: AsrComparisonRow) -> None:
    with output_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row_to_csv_dict(row))


def read_comparison_csv(output_path: Path) -> list[AsrComparisonRow]:
    if not output_path.exists():
        return []
    with output_path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            AsrComparisonRow(
                id=_csv_value(row, "id"),
                reference=_csv_value(row, "reference"),
                base_asr=_csv_value(row, "base_asr"),
                error_type=_csv_value(row, "error_type"),
                note=_csv_value(row, "note"),
            )
            for row in csv.DictReader(handle)
        ]


def row_to_csv_dict(row: AsrComparisonRow) -> dict[str, str]:
    return {
        "id": row.id,
        "reference": row.reference,
        "base_asr": row.base_asr,
        "error_type": row.error_type,
        "note": row.note,
    }


def _csv_value(row: dict[str, str | None], field_name: str) -> str:
    value = row.get(field_name)
    if value is None:
        raise ValueError(f"Missing {field_name} in ASR comparison CSV")
    return value


def _env_text(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value if value else default


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)

    rows = compare_manifest(
        args.manifest,
        args.output,
        max_samples=args.max_samples,
        resume=args.resume,
        batch_size=args.batch_size,
    )
    print(f"Wrote {len(rows)} ASR comparison rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
