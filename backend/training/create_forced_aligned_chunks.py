from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf

if __package__:
    from dataset.common import read_jsonl, write_jsonl
    from training.forced_alignment_chunks import AlignedWord, chunk_aligned_words, sanitize_romanized
else:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dataset.common import read_jsonl, write_jsonl
    from training.forced_alignment_chunks import AlignedWord, chunk_aligned_words, sanitize_romanized


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    rows: tuple[dict[str, str | float], ...]
    source_id: str
    chunks: int
    skipped_reason: str | None = None


def create_aligned_chunk_manifest(
    input_manifest: Path,
    output_manifest: Path,
    chunk_audio_dir: Path,
    limit: int | None,
    max_chunk_seconds: float,
    max_source_seconds: float | None,
    device: str,
    sort_by_duration: bool,
) -> tuple[AlignmentResult, ...]:
    rows = read_jsonl(input_manifest)
    if sort_by_duration:
        rows = sorted(rows, key=lambda row: _float_value(row.get("duration_seconds")) or float("inf"))
    selected_rows = rows if limit is None else rows[:limit]
    context = _load_alignment_context(device)
    results: list[AlignmentResult] = []
    output_rows: list[dict[str, str | float]] = []
    for row in selected_rows:
        result = _align_manifest_row(row, chunk_audio_dir, max_chunk_seconds, max_source_seconds, context)
        results.append(result)
        output_rows.extend(result.rows)
    write_jsonl(output_manifest, output_rows)
    return tuple(results)


def _load_alignment_context(device: str) -> dict[str, object]:
    import torch
    import torchaudio
    from uroman import Uroman

    bundle = torchaudio.pipelines.MMS_FA
    actual_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if actual_device == "auto":
        actual_device = "cpu"
    return {
        "torch": torch,
        "torchaudio": torchaudio,
        "bundle": bundle,
        "model": bundle.get_model().to(actual_device),
        "tokenizer": bundle.get_tokenizer(),
        "aligner": bundle.get_aligner(),
        "romanizer": Uroman(),
        "device": actual_device,
    }


def _align_manifest_row(
    row: dict[str, object],
    chunk_audio_dir: Path,
    max_chunk_seconds: float,
    max_source_seconds: float | None,
    context: dict[str, object],
) -> AlignmentResult:
    source_id = str(row.get("id", ""))
    duration = _float_value(row.get("duration_seconds"))
    if max_source_seconds is not None and duration is not None and duration > max_source_seconds:
        return AlignmentResult(rows=(), source_id=source_id, chunks=0, skipped_reason="source_too_long")
    words = _romanized_words(str(row.get("text", "")), context["romanizer"])
    if not words:
        return AlignmentResult(rows=(), source_id=source_id, chunks=0, skipped_reason="empty_romanized_text")
    waveform, sample_rate = _load_waveform(Path(str(row.get("audio_path", ""))), context)
    spans = _align_words(waveform, sample_rate, words, context)
    chunks = chunk_aligned_words(spans, max_chunk_seconds)
    chunk_rows = _write_chunk_rows(row, chunks, waveform, sample_rate, chunk_audio_dir)
    return AlignmentResult(rows=tuple(chunk_rows), source_id=source_id, chunks=len(chunk_rows))


def _romanized_words(text: str, romanizer) -> tuple[tuple[str, str], ...]:
    words: list[tuple[str, str]] = []
    for word in text.split():
        romanized = sanitize_romanized(romanizer.romanize_string(word, lcode="kor"))
        if romanized:
            words.append((word, romanized))
    return tuple(words)


def _load_waveform(audio_path: Path, context: dict[str, object]):
    torch = context["torch"]
    torchaudio = context["torchaudio"]
    bundle = context["bundle"]
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        sample_rate = bundle.sample_rate
    return waveform, int(sample_rate)


def _align_words(
    waveform,
    sample_rate: int,
    words: tuple[tuple[str, str], ...],
    context: dict[str, object],
) -> tuple[AlignedWord, ...]:
    torch = context["torch"]
    model = context["model"]
    tokenizer = context["tokenizer"]
    aligner = context["aligner"]
    device = str(context["device"])
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    tokens = tokenizer([romanized for _, romanized in words])
    token_spans = aligner(emission[0], tokens)
    ratio = waveform.shape[1] / sample_rate / emission.shape[1]
    aligned_words: list[AlignedWord] = []
    for (text, romanized), spans in zip(words, token_spans, strict=True):
        aligned_words.append(
            AlignedWord(
                text=text,
                romanized=romanized,
                start_seconds=float(spans[0].start * ratio),
                end_seconds=float(spans[-1].end * ratio),
                score=float(sum(span.score for span in spans) / len(spans)),
            )
        )
    return tuple(aligned_words)


def _write_chunk_rows(
    source_row: dict[str, object],
    chunks: tuple[object, ...],
    waveform,
    sample_rate: int,
    chunk_audio_dir: Path,
) -> list[dict[str, str | float]]:
    chunk_audio_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str | float]] = []
    source_id = str(source_row.get("id", "sample"))
    for index, chunk in enumerate(chunks, start=1):
        chunk_path = chunk_audio_dir / f"{source_id}_chunk_{index:04d}.wav"
        _write_audio_slice(waveform, sample_rate, chunk.start_seconds, chunk.end_seconds, chunk_path)
        output_rows.append(
            {
                "id": f"{source_id}_chunk_{index:04d}",
                "audio_path": str(chunk_path),
                "text": chunk.text,
                "disorder_type": str(source_row.get("disorder_type", "")),
                "split": str(source_row.get("split", "")),
                "duration_seconds": chunk.duration_seconds,
                "source_id": source_id,
                "source_audio_path": str(source_row.get("audio_path", "")),
                "start_seconds": chunk.start_seconds,
                "end_seconds": chunk.end_seconds,
                "alignment_score": chunk.mean_score,
                "alignment_method": "torchaudio_mms_fa_uroman",
            }
        )
    return output_rows


def _write_audio_slice(waveform, sample_rate: int, start_seconds: float, end_seconds: float, path: Path) -> None:
    start_frame = max(0, int(start_seconds * sample_rate))
    end_frame = min(waveform.shape[1], max(start_frame + 1, int(end_seconds * sample_rate)))
    sf.write(str(path), waveform[:, start_frame:end_frame].squeeze(0).cpu().numpy(), sample_rate)


def _float_value(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--chunk-audio-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-chunk-seconds", default=30.0, type=float)
    parser.add_argument("--max-source-seconds", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--sort-by-duration", action="store_true")
    args = parser.parse_args(argv)

    results = create_aligned_chunk_manifest(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        chunk_audio_dir=args.chunk_audio_dir,
        limit=args.limit,
        max_chunk_seconds=args.max_chunk_seconds,
        max_source_seconds=args.max_source_seconds,
        device=args.device,
        sort_by_duration=args.sort_by_duration,
    )
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"aligned_sources={len(results)} chunks={sum(result.chunks for result in results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
