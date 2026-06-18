# Presentation Script

## Opening

SemAssist is an accessibility AI MVP for people whose speech can be unclear because of dysarthria or related conditions. The goal is to help the user express short, important everyday sentences more clearly.

## Demo Flow

1. Select `일상 대화`.
2. Enter `오널 만나는 곳 지하철역 맞나여` in the recognized speech input.
3. Press `의도 확인`.
4. Show that the app asks: `오늘 만나는 곳이 지하철역 맞나요?`
5. Select `오늘 만나는 곳이 지하철역 맞나요?`
6. Show the final large card: `오늘 만나는 곳이 지하철역 맞나요?`

## Technical Points

- FastAPI receives recognized speech text or uploaded audio.
- Audio upload uses temporary files and deletes them by default.
- The ASR module supports local Whisper/faster-whisper integration when installed.
- A deterministic interpreter normalizes unclear utterances into safer candidate sentences.
- Risk rules prevent questions, negation, symptoms, and emergency expressions from being silently changed.
- AI-Hub tooling prepares manifest creation, train/validation split, ASR comparison, and LoRA fine-tuning workflows.
