import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

if __package__:
    from dataset.common import read_jsonl
else:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dataset.common import read_jsonl


@dataclass(frozen=True, slots=True)
class AlignmentAudit:
    total_rows: int
    aligned_rows: int
    window_mismatch_rows: int
    missing_duration_rows: int
    max_audio_seconds: float

    @property
    def safe_to_train(self) -> bool:
        return self.total_rows > 0 and self.window_mismatch_rows == 0 and self.missing_duration_rows == 0


def audit_manifest_alignment(manifest_path: Path, max_audio_seconds: float = 30.0) -> AlignmentAudit:
    if max_audio_seconds <= 0:
        raise ValueError(f"max_audio_seconds must be positive: {max_audio_seconds}")

    rows = read_jsonl(manifest_path)
    aligned_rows = 0
    window_mismatch_rows = 0
    missing_duration_rows = 0
    for row in rows:
        duration_seconds = _duration_seconds(row)
        if duration_seconds is None:
            missing_duration_rows += 1
            continue
        if duration_seconds <= max_audio_seconds:
            aligned_rows += 1
        else:
            window_mismatch_rows += 1

    return AlignmentAudit(
        total_rows=len(rows),
        aligned_rows=aligned_rows,
        window_mismatch_rows=window_mismatch_rows,
        missing_duration_rows=missing_duration_rows,
        max_audio_seconds=max_audio_seconds,
    )


def write_alignment_audit(audit: AlignmentAudit, json_output_path: Path, markdown_output_path: Path) -> None:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(audit), "safe_to_train": audit.safe_to_train}
    json_output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_output_path.write_text(_markdown_report(audit), encoding="utf-8")


def _duration_seconds(row: dict[str, object]) -> float | None:
    raw_value = row.get("duration_seconds")
    if isinstance(raw_value, int | float):
        return float(raw_value)
    if isinstance(raw_value, str) and _is_float(raw_value):
        return float(raw_value)
    return None


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _markdown_report(audit: AlignmentAudit) -> str:
    status = "PASS" if audit.safe_to_train else "BLOCKED"
    return (
        "# LoRA Alignment Audit\n\n"
        f"- Status: `{status}`\n"
        f"- Rows: {audit.total_rows}\n"
        f"- Max audio seconds per training sample: {audit.max_audio_seconds:g}\n"
        f"- Rows within the window: {audit.aligned_rows}\n"
        f"- Rows longer than the window: {audit.window_mismatch_rows}\n"
        f"- Rows missing duration: {audit.missing_duration_rows}\n\n"
        "Rows longer than the configured window need segment-level transcripts before multi-epoch LoRA training.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    parser.add_argument("--max-audio-seconds", default=30.0, type=float)
    args = parser.parse_args(argv)

    audit = audit_manifest_alignment(args.manifest, args.max_audio_seconds)
    write_alignment_audit(audit, args.json_output, args.markdown_output)
    print(
        "alignment "
        f"safe_to_train={audit.safe_to_train} "
        f"total={audit.total_rows} "
        f"aligned={audit.aligned_rows} "
        f"window_mismatch={audit.window_mismatch_rows} "
        f"missing_duration={audit.missing_duration_rows}"
    )
    return 0 if audit.safe_to_train else 2


if __name__ == "__main__":
    raise SystemExit(main())
