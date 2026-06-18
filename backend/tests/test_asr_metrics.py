import csv
import json
from pathlib import Path

from training.evaluate_asr_csv import edit_distance, evaluate_comparison_csv


def test_edit_distance_counts_insertions_deletions_and_substitutions() -> None:
    assert edit_distance(("a", "b", "c"), ("a", "x", "c", "d")) == 2


def test_evaluate_comparison_csv_filters_manifest_ids(tmp_path: Path) -> None:
    comparison_path = tmp_path / "comparison.csv"
    manifest_path = tmp_path / "validation.jsonl"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "reference", "base_asr"))
        writer.writeheader()
        writer.writerow({"id": "sample_000001", "reference": "학교 가요", "base_asr": "학교 가요"})
        writer.writerow({"id": "sample_000002", "reference": "학교 가요", "base_asr": "학교"})
    manifest_path.write_text(
        json.dumps({"id": "sample_000002"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = evaluate_comparison_csv(
        comparison_csv_path=comparison_path,
        label="validation",
        filter_manifest_path=manifest_path,
    )

    assert summary.label == "validation"
    assert summary.rows == 1
    assert summary.cer > 0
    assert summary.wer == 0.5
