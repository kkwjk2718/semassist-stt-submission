import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from rapidfuzz.distance import Levenshtein

if __package__:
    from dataset.common import read_jsonl
else:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dataset.common import read_jsonl


@dataclass(frozen=True, slots=True)
class AsrMetricSummary:
    label: str
    rows: int
    cer: float
    wer: float
    character_errors: int
    reference_characters: int
    word_errors: int
    reference_words: int


def evaluate_comparison_csv(
    comparison_csv_path: Path,
    label: str = "all",
    filter_manifest_path: Path | None = None,
) -> AsrMetricSummary:
    allowed_ids = _manifest_ids(filter_manifest_path) if filter_manifest_path is not None else None
    with comparison_csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if allowed_ids is None or row["id"] in allowed_ids
        ]
    return summarize_asr_metrics(label, rows)


def summarize_asr_metrics(label: str, rows: list[dict[str, str]]) -> AsrMetricSummary:
    character_errors = 0
    reference_characters = 0
    word_errors = 0
    reference_words = 0
    for row in rows:
        reference = row["reference"]
        hypothesis = row["base_asr"]
        reference_chars = tuple("".join(reference.split()))
        hypothesis_chars = tuple("".join(hypothesis.split()))
        reference_tokens = tuple(reference.split())
        hypothesis_tokens = tuple(hypothesis.split())
        character_errors += edit_distance(reference_chars, hypothesis_chars)
        reference_characters += len(reference_chars)
        word_errors += edit_distance(reference_tokens, hypothesis_tokens)
        reference_words += len(reference_tokens)

    return AsrMetricSummary(
        label=label,
        rows=len(rows),
        cer=_rate(character_errors, reference_characters),
        wer=_rate(word_errors, reference_words),
        character_errors=character_errors,
        reference_characters=reference_characters,
        word_errors=word_errors,
        reference_words=reference_words,
    )


def write_metric_summary(summary: AsrMetricSummary, json_output_path: Path, markdown_output_path: Path) -> None:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_output_path.write_text(
        "# ASR Metric Summary\n\n"
        f"- Label: `{summary.label}`\n"
        f"- Rows: {summary.rows}\n"
        f"- CER: {summary.cer:.6f}\n"
        f"- WER: {summary.wer:.6f}\n"
        f"- Character errors/reference chars: {summary.character_errors}/{summary.reference_characters}\n"
        f"- Word errors/reference words: {summary.word_errors}/{summary.reference_words}\n",
        encoding="utf-8",
    )


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    return int(Levenshtein.distance(left, right))


def _manifest_ids(manifest_path: Path) -> set[str]:
    return {str(row["id"]) for row in read_jsonl(manifest_path)}


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    parser.add_argument("--label", default="all")
    parser.add_argument("--filter-manifest", type=Path)
    args = parser.parse_args(argv)

    summary = evaluate_comparison_csv(
        comparison_csv_path=args.comparison_csv,
        label=args.label,
        filter_manifest_path=args.filter_manifest,
    )
    write_metric_summary(summary, args.json_output, args.markdown_output)
    print(f"metrics label={summary.label} rows={summary.rows} cer={summary.cer:.6f} wer={summary.wer:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
