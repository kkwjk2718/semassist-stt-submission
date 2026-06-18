import argparse
from pathlib import Path
from typing import Sequence

if __package__:
    from dataset.common import read_jsonl, write_jsonl
else:
    from common import read_jsonl, write_jsonl


def sample_manifest(
    input_path: Path,
    output_path: Path,
    max_samples: int | None = None,
    max_hours: float | None = None,
) -> list[dict[str, object]]:
    if max_samples is None and max_hours is None:
        raise ValueError("Set --max-samples, --max-hours, or both")

    rows = read_jsonl(input_path)
    selected_rows: list[dict[str, object]] = []
    max_seconds = max_hours * 3600 if max_hours is not None else None
    total_seconds = 0.0

    for row in rows:
        if max_samples is not None and len(selected_rows) >= max_samples:
            break

        duration_seconds = _duration_seconds(row)
        if max_seconds is not None and duration_seconds is not None:
            if total_seconds + duration_seconds > max_seconds and selected_rows:
                break
            total_seconds += duration_seconds

        selected_rows.append(row)

    write_jsonl(output_path, selected_rows)
    return selected_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args(argv)

    rows = sample_manifest(
        input_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples,
        max_hours=args.max_hours,
    )
    print(f"Wrote {len(rows)} sampled rows to {args.output}")
    return 0


def _duration_seconds(row: dict[str, object]) -> float | None:
    raw_value = row.get("duration_seconds")
    if isinstance(raw_value, int | float):
        return float(raw_value)
    if isinstance(raw_value, str):
        try:
            return float(raw_value)
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
