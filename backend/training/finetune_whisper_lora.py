import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__:
    from dataset.common import read_jsonl
    from training.lora_alignment import count_window_transcript_mismatches, duration_seconds
else:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dataset.common import read_jsonl
    from training.lora_alignment import count_window_transcript_mismatches, duration_seconds


@dataclass(frozen=True, slots=True)
class TrainingManifestInspection:
    total_rows: int
    existing_audio_rows: int
    empty_text_rows: int

    @property
    def ready(self) -> bool:
        return self.total_rows > 0 and self.total_rows == self.existing_audio_rows and self.empty_text_rows == 0


@dataclass(frozen=True, slots=True)
class TrainingRow:
    id: str
    audio_path: Path
    text: str
    duration_seconds: float | None


def inspect_training_manifest(manifest_path: Path) -> TrainingManifestInspection:
    rows = read_jsonl(manifest_path)
    existing_audio_rows = 0
    empty_text_rows = 0
    for row in rows:
        audio_path = Path(str(row.get("audio_path", "")))
        text = str(row.get("text", "")).strip()
        if audio_path.is_file():
            existing_audio_rows += 1
        if not text:
            empty_text_rows += 1
    return TrainingManifestInspection(
        total_rows=len(rows),
        existing_audio_rows=existing_audio_rows,
        empty_text_rows=empty_text_rows,
    )


def read_training_rows(manifest_path: Path) -> list[TrainingRow]:
    rows = []
    for row in read_jsonl(manifest_path):
        rows.append(
            TrainingRow(
                id=str(row.get("id", "")),
                audio_path=Path(str(row.get("audio_path", ""))),
                text=str(row.get("text", "")).strip(),
                duration_seconds=duration_seconds(row),
            )
        )
    return rows


def run_lora_training(
    train_manifest: Path,
    model_name: str,
    output_dir: Path,
    max_steps: int,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    logging_steps: int,
    save_steps: int | None,
    dataloader_num_workers: int,
    max_audio_seconds: float,
    allow_window_transcript_mismatch: bool,
) -> None:
    inspection = inspect_training_manifest(train_manifest)
    if not inspection.ready:
        raise RuntimeError(
            "Training manifest is not ready: "
            f"total={inspection.total_rows}, "
            f"existing_audio={inspection.existing_audio_rows}, "
            f"empty_text={inspection.empty_text_rows}"
        )

    rows = read_training_rows(train_manifest)
    mismatch_rows = count_window_transcript_mismatches(rows, max_audio_seconds)
    if mismatch_rows and not allow_window_transcript_mismatch:
        raise RuntimeError(
            "Training rows are longer than the audio window. "
            f"mismatched_rows={mismatch_rows}, max_audio_seconds={max_audio_seconds}. "
            "Create segment-level transcripts or pass --allow-window-transcript-mismatch for an explicit smoke run."
        )

    import torch
    import librosa
    import soundfile
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    processor = WhisperProcessor.from_pretrained(model_name, language="Korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    max_label_length = int(model.config.max_target_positions)

    if max_audio_seconds <= 0:
        raise RuntimeError(f"max_audio_seconds must be positive: {max_audio_seconds}")

    class ManifestSpeechDataset(TorchDataset[dict[str, object]]):
        def __init__(
            self,
            training_rows: Sequence[TrainingRow],
            whisper_processor: WhisperProcessor,
            audio_seconds: float,
            label_length: int,
        ) -> None:
            self._rows = tuple(training_rows)
            self._processor = whisper_processor
            self._audio_seconds = audio_seconds
            self._label_length = label_length

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, index: int) -> dict[str, object]:
            row = self._rows[index]
            audio_info = soundfile.info(str(row.audio_path))
            frames = int(audio_info.samplerate * self._audio_seconds)
            audio_array, sampling_rate = soundfile.read(
                str(row.audio_path),
                frames=frames,
                dtype="float32",
                always_2d=False,
            )
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)
            if sampling_rate != 16_000:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16_000)
                sampling_rate = 16_000
            input_features = self._processor.feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
            ).input_features[0]
            labels = self._processor.tokenizer(
                row.text,
                truncation=True,
                max_length=self._label_length,
            ).input_ids
            return {"input_features": input_features, "labels": labels}

    @dataclass(frozen=True, slots=True)
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: WhisperProcessor

        def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    resolved_save_steps = save_steps if save_steps is not None else max(max_steps, 1)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=0,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=logging_steps,
        save_steps=resolved_save_steps,
        dataloader_num_workers=dataloader_num_workers,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ManifestSpeechDataset(rows, processor, max_audio_seconds, max_label_length),
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        processing_class=processor.feature_extractor,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-steps", default=1, type=int)
    parser.add_argument("--num-train-epochs", default=1.0, type=float)
    parser.add_argument("--per-device-train-batch-size", default=1, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--logging-steps", default=1, type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--dataloader-num-workers", default=0, type=int)
    parser.add_argument("--max-audio-seconds", default=30.0, type=float)
    parser.add_argument("--allow-window-transcript-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    inspection = inspect_training_manifest(args.train_manifest)
    print(
        "manifest "
        f"total={inspection.total_rows} "
        f"existing_audio={inspection.existing_audio_rows} "
        f"empty_text={inspection.empty_text_rows} "
        f"ready={inspection.ready}"
    )
    if args.dry_run:
        rows = read_training_rows(args.train_manifest)
        mismatch_rows = count_window_transcript_mismatches(rows, args.max_audio_seconds)
        print(f"window_mismatch_rows={mismatch_rows} max_audio_seconds={args.max_audio_seconds}")
        if mismatch_rows and not args.allow_window_transcript_mismatch:
            return 2
        return 0 if inspection.ready else 1

    run_lora_training(
        args.train_manifest,
        args.model_name,
        args.output_dir,
        args.max_steps,
        args.num_train_epochs,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.logging_steps,
        args.save_steps,
        args.dataloader_num_workers,
        args.max_audio_seconds,
        args.allow_window_transcript_mismatch,
    )
    print(f"Saved LoRA adapter and processor to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
