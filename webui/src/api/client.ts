import type {
  CitationContent,
  DatasetInfo,
  OfflinePairsPage,
  SupportPair,
  TraceRecord,
} from "../types";

export interface StreamHandlers {
  onStep?: (record: TraceRecord) => void;
  onTranscript?: (text: string) => void;
  onAnswer?: (pair: SupportPair) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
}

async function detail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") return body.detail;
    return JSON.stringify(body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

/**
 * POST a question (text) or a recording (Blob) and dispatch the SSE frames.
 *
 * EventSource cannot POST, so this is fetch + a ReadableStream reader with a
 * small frame parser. Frames are separated by a blank line; a line starting
 * with `:` is a heartbeat comment and carries nothing.
 */
export async function streamQuery(
  input: string | Blob,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  const request: RequestInit = { method: "POST", signal };

  if (typeof input === "string") {
    request.headers = { "Content-Type": "application/json" };
    request.body = JSON.stringify({ question: input });
  } else {
    const form = new FormData();
    form.append("audio", input, "recording.webm");
    request.body = form; // no Content-Type: the browser sets the boundary
  }

  try {
    response = await fetch("/api/query", request);
  } catch (error) {
    handlers.onError?.(
      `לא ניתן להגיע לשרת הגישור: ${error instanceof Error ? error.message : String(error)}`,
    );
    return;
  }

  if (!response.ok || !response.body) {
    // This is how the 501 "STT backend not configured" text reaches the user,
    // verbatim and untranslated.
    handlers.onError?.(await detail(response));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let split = buffer.indexOf("\n\n");
      while (split !== -1) {
        dispatch(buffer.slice(0, split), handlers);
        buffer = buffer.slice(split + 2);
        split = buffer.indexOf("\n\n");
      }
    }
    if (buffer.trim()) dispatch(buffer, handlers);
  } catch (error) {
    if ((error as Error)?.name === "AbortError") return;
    handlers.onError?.(
      `השידור נקטע: ${error instanceof Error ? error.message : String(error)}`,
    );
  } finally {
    reader.releaseLock();
  }
}

function dispatch(frame: string, handlers: StreamHandlers): void {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (!line || line.startsWith(":")) continue; // heartbeat / padding
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return;

  let data: unknown;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    return; // a partial frame is not worth crashing a live answer over
  }

  switch (event) {
    case "step":
      handlers.onStep?.(data as TraceRecord);
      break;
    case "transcript":
      handlers.onTranscript?.((data as { text: string }).text);
      break;
    case "answer":
      handlers.onAnswer?.(data as SupportPair);
      break;
    case "error":
      handlers.onError?.((data as { message: string }).message);
      break;
    case "done":
      handlers.onDone?.();
      break;
  }
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await detail(response));
  return (await response.json()) as T;
}

export function getDatasets(): Promise<DatasetInfo[]> {
  return getJson<DatasetInfo[]>("/api/datasets");
}

export function getOfflinePairs(
  datasetId: string,
  { limit = 200, offset = 0 }: { limit?: number; offset?: number } = {},
): Promise<OfflinePairsPage> {
  const params = new URLSearchParams({
    dataset: datasetId,
    limit: String(limit),
    offset: String(offset),
  });
  return getJson<OfflinePairsPage>(`/api/offline-pairs?${params}`);
}

export function getCitationContent(
  fileName: string,
  page: number | null,
): Promise<CitationContent> {
  const params = new URLSearchParams({ file: fileName });
  if (page !== null) params.set("page", String(page));
  return getJson<CitationContent>(`/api/citation/content?${params}`);
}
