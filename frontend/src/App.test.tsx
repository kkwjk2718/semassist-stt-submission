import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders recording controls for audio mode", () => {
    render(<App />);

    expect(screen.getByRole("button", { name: "녹음 시작" })).toBeInTheDocument();
  });

  it("shows the final corrected card when the user confirms the mock interpretation", async () => {
    const fetchStub: typeof fetch = async (input, init) => {
      const requestUrl = String(input);
      if (requestUrl.endsWith("/api/assist/text")) {
        expect(init?.method).toBe("POST");
        return new Response(
          JSON.stringify({
            session_id: "session-1",
            domain: "daily",
            asr_text: "오널 만나는 곳 지하철역 맞나여.",
            possible_meaning: "만나는 장소를 확인하는 질문으로 보입니다.",
            corrected_candidate: "오늘 만나는 곳이 지하철역 맞나요?",
            intent: "meeting_place_confirmation",
            risk_level: "medium",
            critical_uncertainty: true,
            clarification_needed: true,
            clarification_question: "만나는 장소를 확인하는 뜻인가요?",
            choices: [
              "오늘 만나는 곳이 지하철역 맞나요?",
              "오늘 지하철역에서 만나나요?",
              "다시 말할게요."
            ],
            warnings: []
          }),
          { headers: { "Content-Type": "application/json" }, status: 200 }
        );
      }
      if (requestUrl.endsWith("/api/confirm")) {
        expect(init?.method).toBe("POST");
        return new Response(
          JSON.stringify({
            final_text: "오늘 만나는 곳이 지하철역 맞나요?",
            display_mode: "large_card",
            tts_text: "오늘 만나는 곳이 지하철역 맞나요?"
          }),
          { headers: { "Content-Type": "application/json" }, status: 200 }
        );
      }
      return new Response("not found", { status: 404 });
    };
    vi.stubGlobal("fetch", fetchStub);

    render(<App />);

    fireEvent.change(screen.getByLabelText("Mock ASR input"), {
      target: { value: "오널 만나는 곳 지하철역 맞나여." }
    });
    fireEvent.click(screen.getByRole("button", { name: "의미 확인" }));
    fireEvent.click(await screen.findByRole("button", { name: "오늘 만나는 곳이 지하철역 맞나요?" }));

    await waitFor(() => {
      expect(screen.getByText("상대방에게 보여주세요")).toBeInTheDocument();
    });
    expect(screen.getByText("오늘 만나는 곳이 지하철역 맞나요?")).toBeInTheDocument();
  });
});
