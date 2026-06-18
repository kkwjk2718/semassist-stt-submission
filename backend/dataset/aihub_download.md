# AI-Hub Data Notes

AI-Hub 원천 데이터와 개인 API key는 GitHub에 올리지 않습니다. 데이터가 필요한 경우 AI-Hub에서 사용 권한을 받은 뒤, 로컬 `.env` 또는 현재 shell session에만 key를 설정해 다운로드합니다.

## 권장 절차

1. AI-Hub에서 구음장애 음성 데이터 사용 권한을 확인합니다.
2. 로컬 환경 변수에 개인 API key와 dataset key를 설정합니다.
3. `aihubshell`로 label zip과 source audio zip을 내려받습니다.
4. 압축을 푼 뒤 label 구조를 먼저 검사합니다.
5. manifest를 생성하고 train/validation split을 만듭니다.

## Manifest 생성 예시

```bash
python backend/dataset/inspect_labels.py --label-root data/aihub --output results/label_structure_summary.md
python backend/dataset/build_manifest.py --label-root data/aihub --audio-root data/aihub --output data/manifests/brain_all.jsonl
python backend/dataset/sample_manifest.py --input data/manifests/brain_all.jsonl --output data/manifests/brain_demo_100.jsonl --max-samples 100
python backend/dataset/split_manifest.py --input data/manifests/brain_all.jsonl --output-dir data/manifests/splits --validation-ratio 0.2
```

## 주의

- API key를 command, log, screenshot, committed file에 남기지 않습니다.
- 원천 WAV/JSON과 manifest는 `.gitignore`에 의해 제외됩니다.
- 대용량 Whisper weight와 LoRA checkpoint도 repo에 포함하지 않습니다.
