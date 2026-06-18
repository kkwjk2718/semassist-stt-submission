import { CheckCircle2, Clipboard, Keyboard, Mic, RotateCcw, ShieldCheck, Square, Volume2 } from "lucide-react";
import { type FormEvent, useEffect, useRef, useState } from "react";

import { ApiError, type AssistResponse, type ConfirmResponse, type Domain, assistAudio, assistText, confirmChoice } from "./api";
import "./styles.css";

const DOMAINS: readonly { readonly id: Domain; readonly label: string; readonly detail: string }[] = [
  { id: "daily", label: "일상 대화", detail: "가족, 식사, 이동" },
  { id: "transit", label: "이동/장소", detail: "지하철, 약속, 길 안내" },
  { id: "emergency", label: "응급 도움", detail: "119, 호흡, 즉시 도움" },
  { id: "civil", label: "민원/상담", detail: "창구, 안내, 요청" }
];

type Phase = "input" | "result" | "final";

export function App() {
  const [domain, setDomain] = useState<Domain>("daily");
  const [mockText, setMockText] = useState("오널 만나는 곳 지하철역 맞나여.");
  const [phase, setPhase] = useState<Phase>("input");
  const [assistResult, setAssistResult] = useState<AssistResponse | null>(null);
  const [finalCard, setFinalCard] = useState<ConfirmResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioStatus, setAudioStatus] = useState("마이크 대기");
  const [errorMessage, setErrorMessage] = useState("");
  const audioChunksRef = useRef<Blob[]>([]);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const stopActiveStream = (): void => {
    audioStreamRef.current?.getTracks().forEach((track) => track.stop());
    audioStreamRef.current = null;
  };

  useEffect(() => {
    return () => {
      stopActiveStream();
    };
  }, []);

  const submitInterpretation = async (): Promise<void> => {
    setIsLoading(true);
    setErrorMessage("");
    setFinalCard(null);
    try {
      const result = await assistText(domain, mockText);
      setAssistResult(result);
      setPhase("result");
    } catch (error) {
      setErrorMessage(messageFromError(error));
    } finally {
      setIsLoading(false);
    }
  };

  const submitAudioInterpretation = async (audio: Blob): Promise<void> => {
    setIsLoading(true);
    setErrorMessage("");
    setFinalCard(null);
    try {
      const result = await assistAudio(domain, audio);
      setAssistResult(result);
      setAudioStatus(result.warnings.length > 0 ? "mock 입력 권장" : "해석 완료");
      setPhase("result");
    } catch (error) {
      setAudioStatus("해석 실패");
      setErrorMessage(messageFromError(error));
    } finally {
      setIsLoading(false);
    }
  };

  const submitConfirmation = async (choice: string): Promise<void> => {
    if (assistResult === null) {
      return;
    }
    setIsLoading(true);
    setErrorMessage("");
    try {
      const result = await confirmChoice(assistResult.session_id, choice);
      setFinalCard(result);
      setPhase("final");
    } catch (error) {
      setErrorMessage(messageFromError(error));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    void submitInterpretation();
  };

  const resetFlow = (): void => {
    setPhase("input");
    setAssistResult(null);
    setFinalCard(null);
    setAudioStatus("마이크 대기");
    setErrorMessage("");
  };

  const startRecording = async (): Promise<void> => {
    if (navigator.mediaDevices?.getUserMedia === undefined || typeof MediaRecorder === "undefined") {
      setErrorMessage("브라우저에서 녹음을 지원하지 않습니다. 빠른 입력을 사용해주세요.");
      return;
    }

    setErrorMessage("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];
      audioStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        stopActiveStream();
        mediaRecorderRef.current = null;
        if (audioBlob.size === 0) {
          setAudioStatus("녹음 없음");
          setErrorMessage("녹음된 음성이 없습니다.");
          return;
        }
        void submitAudioInterpretation(audioBlob);
      };
      recorder.start();
      setIsRecording(true);
      setAudioStatus("녹음 중");
    } catch (error) {
      stopActiveStream();
      setAudioStatus("마이크 대기");
      setErrorMessage(messageFromError(error));
    }
  };

  const stopRecording = (): void => {
    if (mediaRecorderRef.current === null) {
      return;
    }
    setIsRecording(false);
    setAudioStatus("해석 중");
    mediaRecorderRef.current.stop();
  };

  const copyFinalText = async (): Promise<void> => {
    if (finalCard !== null && navigator.clipboard !== undefined) {
      await navigator.clipboard.writeText(finalCard.final_text);
    }
  };

  const speakFinalText = (): void => {
    if (finalCard === null || window.speechSynthesis === undefined) {
      return;
    }
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(finalCard.tts_text));
  };

  return (
    <main className="app-shell">
      <section className="top-band" aria-labelledby="app-title">
        <div>
          <p className="eyebrow">SemAssist</p>
          <h1 id="app-title">의미 보존형 음성 보조</h1>
        </div>
        <p className="safety-note">이 도구는 구음장애 발화를 자연스러운 문장으로 정리하는 의사소통 보조용입니다.</p>
      </section>

      <section className="domain-band" aria-label="상황 선택">
        {DOMAINS.map((option) => (
          <button
            className={option.id === domain ? "domain-button is-active" : "domain-button"}
            key={option.id}
            onClick={() => setDomain(option.id)}
            type="button"
          >
            <span>{option.label}</span>
            <small>{option.detail}</small>
          </button>
        ))}
      </section>

      <section className="tool-band" aria-live="polite">
        <form className="input-panel" onSubmit={handleSubmit}>
          <label className="field-label" htmlFor="mock-asr">
            <Keyboard aria-hidden="true" size={24} />
            빠른 입력
          </label>
          <textarea
            aria-label="Mock ASR input"
            id="mock-asr"
            onChange={(event) => setMockText(event.target.value)}
            rows={4}
            value={mockText}
          />
          <button className="primary-action" disabled={isLoading || mockText.trim().length === 0} type="submit">
            <ShieldCheck aria-hidden="true" size={24} />
            의미 확인
          </button>
          <div className="audio-panel" aria-label="음성 녹음">
            <div className="audio-title">
              <Mic aria-hidden="true" size={24} />
              <span>음성 녹음</span>
            </div>
            <p className="audio-status">{audioStatus}</p>
            {isRecording ? (
              <button className="secondary-action is-recording" onClick={stopRecording} type="button">
                <Square aria-hidden="true" size={22} />
                녹음 중지
              </button>
            ) : (
              <button className="secondary-action" disabled={isLoading} onClick={() => void startRecording()} type="button">
                <Mic aria-hidden="true" size={22} />
                녹음 시작
              </button>
            )}
          </div>
        </form>

        {phase === "result" && assistResult !== null ? (
          <div className="result-panel">
            <div className="result-copy">
              <span className={`risk-badge risk-${assistResult.risk_level}`}>{riskLabel(assistResult.risk_level)}</span>
              <p>{assistResult.possible_meaning}</p>
              {assistResult.warnings.length > 0 ? (
                <ul className="warning-list" aria-label="설정 안내">
                  {assistResult.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              ) : null}
              <h2>{assistResult.clarification_question}</h2>
            </div>
            <div className="choice-list">
              {assistResult.choices.map((choice) => (
                <button className="choice-button" key={choice} onClick={() => void submitConfirmation(choice)} type="button">
                  <CheckCircle2 aria-hidden="true" size={24} />
                  {choice}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {phase === "final" && finalCard !== null ? (
          <div className="final-panel">
            <p>상대방에게 보여주세요</p>
            <strong>{finalCard.final_text}</strong>
            <div className="final-actions">
              <button onClick={() => void copyFinalText()} type="button">
                <Clipboard aria-hidden="true" size={22} />
                복사하기
              </button>
              <button onClick={speakFinalText} type="button">
                <Volume2 aria-hidden="true" size={22} />
                소리내어 읽기
              </button>
              <button onClick={resetFlow} type="button">
                <RotateCcw aria-hidden="true" size={22} />
                다시 말하기
              </button>
            </div>
          </div>
        ) : null}

        {errorMessage.length > 0 ? <p className="error-text">{errorMessage}</p> : null}
      </section>
    </main>
  );
}

function riskLabel(level: AssistResponse["risk_level"]): string {
  switch (level) {
    case "low":
      return "낮은 위험";
    case "medium":
      return "확인 필요";
    case "high":
      return "긴급 확인";
    default:
      return assertNever(level);
  }
}

function assertNever(value: never): never {
  throw new ApiError(500, `처리할 수 없는 상태입니다: ${value}`);
}

function messageFromError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "알 수 없는 오류가 발생했습니다.";
}
