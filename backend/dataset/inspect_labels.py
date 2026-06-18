import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__:
    from dataset.common import (
        find_audio_path_candidates,
        find_transcript_candidates,
        iter_json_files,
        read_json,
        top_level_keys,
    )
else:
    from common import (
        find_audio_path_candidates,
        find_transcript_candidates,
        iter_json_files,
        read_json,
        top_level_keys,
    )


@dataclass(frozen=True)
class LabelInspectionSummary:
    json_files: int
    top_level_key_counts: dict[str, int]
    transcript_candidate_counts: dict[str, int]
    audio_candidate_counts: dict[str, int]


def inspect_label_root(
    label_root: Path,
    report_path: Path = Path("results/label_structure_summary.md"),
    max_files: int = 50,
) -> LabelInspectionSummary:
    json_files = iter_json_files(label_root)
    inspected_files = json_files[:max_files]
    top_keys: Counter[str] = Counter()
    transcript_candidates: Counter[str] = Counter()
    audio_candidates: Counter[str] = Counter()

    for json_file in inspected_files:
        payload = read_json(json_file)
        top_keys.update(top_level_keys(payload))
        transcript_candidates.update(candidate.field_path for candidate in find_transcript_candidates(payload))
        audio_candidates.update(candidate.field_path for candidate in find_audio_path_candidates(payload))

    summary = LabelInspectionSummary(
        json_files=len(json_files),
        top_level_key_counts=dict(sorted(top_keys.items())),
        transcript_candidate_counts=dict(sorted(transcript_candidates.items())),
        audio_candidate_counts=dict(sorted(audio_candidates.items())),
    )
    write_summary(report_path, summary)
    return summary


def write_summary(report_path: Path, summary: LabelInspectionSummary) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_summary_markdown(summary), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("results/label_structure_summary.md"), type=Path)
    parser.add_argument("--max-files", default=50, type=int)
    args = parser.parse_args(argv)

    summary = inspect_label_root(args.label_root, args.output, args.max_files)
    print(f"Inspected {summary.json_files} JSON files")
    print(f"Wrote {args.output}")
    return 0


def _summary_markdown(summary: LabelInspectionSummary) -> str:
    return "\n".join(
        [
            "# Label Structure Summary",
            "",
            f"- JSON files found: {summary.json_files}",
            "",
            "## Top-Level Keys",
            _counter_markdown(summary.top_level_key_counts),
            "",
            "## Transcript Candidate Fields",
            _counter_markdown(summary.transcript_candidate_counts),
            "",
            "## Audio Path Candidate Fields",
            _counter_markdown(summary.audio_candidate_counts),
            "",
        ]
    )


def _counter_markdown(counter: dict[str, int]) -> str:
    if not counter:
        return "- None found"
    return "\n".join(f"- `{key}`: {count}" for key, count in counter.items())


if __name__ == "__main__":
    raise SystemExit(main())
