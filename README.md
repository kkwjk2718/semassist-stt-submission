# SemAssist

SemAssist는 구음장애인의 발화를 인식하고, 의미가 더 자연스럽고 명확하게 전달되도록 보정 후보를 제시하는 웹 기반 의사소통 보조 프로젝트입니다.

## 주요 기능

- FastAPI 기반 백엔드 API
- React, Vite, TypeScript 기반 프론트엔드
- 텍스트 입력 및 브라우저 음성 녹음 UI
- 구음장애 발화를 자연스러운 한국어 문장 후보로 보정
- 사용자가 보정 후보를 선택하면 전달용 큰 문장 카드로 표시
- AI-Hub 구음장애 음성 데이터 분석, Whisper 기반 STT benchmark, LoRA fine-tuning 실험 코드 포함

## 폴더 구조

```text
backend/   FastAPI API, ASR module, dataset/training utility scripts, tests
frontend/  React/Vite web UI and tests
examples/  demo cases and sample ASR error examples
```

## 실행 방법

Backend:

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev -- --port 5173
```

브라우저에서 `http://127.0.0.1:5173`을 열면 됩니다.

## 검증

Backend:

```bash
cd backend
python -m pytest tests -q
```

Frontend:

```bash
cd frontend
npm test -- --run
npm run build
```

## 데이터와 모델 파일 안내

GitHub 제출본에는 API key, AI-Hub 원천 WAV/JSON, transcript manifest, benchmark CSV, Whisper weight, LoRA checkpoint를 포함하지 않았습니다.

- AI-Hub 데이터는 이용 조건에 따라 각자 발급받은 권한으로 다운로드해야 합니다.
- Whisper 원본 weight는 실행 환경에서 별도로 내려받아 사용합니다.
- 이 repo에는 재현 가능한 코드와 실행에 필요한 최소 예시 파일만 포함했습니다.
