import { z } from "zod";

export const DomainSchema = z.enum(["daily", "transit", "emergency", "civil"]);
export type Domain = z.infer<typeof DomainSchema>;

export const AssistResponseSchema = z.object({
  session_id: z.string().min(1),
  domain: DomainSchema,
  asr_text: z.string(),
  possible_meaning: z.string(),
  corrected_candidate: z.string(),
  intent: z.string(),
  risk_level: z.enum(["low", "medium", "high"]),
  critical_uncertainty: z.boolean(),
  clarification_needed: z.boolean(),
  clarification_question: z.string(),
  choices: z.array(z.string()),
  warnings: z.array(z.string())
});
export type AssistResponse = z.infer<typeof AssistResponseSchema>;

export const ConfirmResponseSchema = z.object({
  final_text: z.string(),
  display_mode: z.literal("large_card"),
  tts_text: z.string()
});
export type ConfirmResponse = z.infer<typeof ConfirmResponseSchema>;

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function assistText(domain: Domain, asrText: string): Promise<AssistResponse> {
  const response = await fetch("/api/assist/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ domain, asr_text: asrText })
  });
  return parseResponse(response, AssistResponseSchema);
}

export async function assistAudio(domain: Domain, audio: Blob): Promise<AssistResponse> {
  const body = new FormData();
  body.append("domain", domain);
  body.append("audio", audio, "recording.webm");

  const response = await fetch("/api/assist/audio", {
    method: "POST",
    body
  });
  return parseResponse(response, AssistResponseSchema);
}

export async function confirmChoice(sessionId: string, selectedChoice: string): Promise<ConfirmResponse> {
  const response = await fetch("/api/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, selected_choice: selectedChoice })
  });
  return parseResponse(response, ConfirmResponseSchema);
}

async function parseResponse<T>(response: Response, schema: z.ZodType<T>): Promise<T> {
  const payload: unknown = await response.json();
  if (!response.ok) {
    throw new ApiError(response.status, "요청을 처리하지 못했습니다.");
  }
  return schema.parse(payload);
}
