// Hand-written mirror of webapi/schema.py and webapi/datasets.py.
// THOSE are the source of truth — if a field changes there, change it here.

export interface SupportCitation {
  id: string;
  file_name: string;
  page_number: number | null;
  content_preview: string | null;
  thumbnail_url: string | null;
}

export interface JudgeGrades {
  correctness: number | null;
  completeness: number | null;
  conversational_quality: number | null;
  verdict: string | null;
  hallucination: boolean | null;
  reasoning: Record<string, string> | null;
}

/** One agent pipeline step. `step` names the kind; the rest varies by kind
 *  (see rag/agent/engine.py's _emit call sites). */
export interface TraceRecord {
  step: string;
  ms?: number;
  // hint
  top_category?: string | null;
  top_share?: number | null;
  n_hits?: number;
  // classify
  mode?: string;
  categories?: string[];
  sub_questions?: string[];
  needs_calculation?: boolean;
  dependent?: boolean;
  difficulty?: string;
  // retrieve (prefetch and loop)
  phase?: string;
  query?: string;
  category?: string | string[] | null;
  n_gated?: number;
  retried_unfiltered?: boolean;
  // orchestrator
  hop?: number;
  n_tool_calls?: number;
  content?: string;
  error?: string;
  // calculate
  expression?: string;
  value?: number;
  // synthesize
  model?: string;
  fast_synthesis?: boolean;
}

export interface SupportPair {
  id: string;
  question: string | null;
  answer: string | null;
  citations: SupportCitation[];
  domain: string | null;
  confidence: number | null;
  latency_ms: number | null;
  cost_usd: number | null;
  trace: TraceRecord[] | null;
  classification: Record<string, unknown> | null;
  difficulty: string | null;
  reference_answer: string | null;
  judgment: JudgeGrades | null;
}

export interface DatasetInfo {
  id: string;
  label: string;
  kind: "answers" | "judged";
  n_pairs: number;
  has_trace: boolean;
  has_judgment: boolean;
  questions_file: string | null;
}

export interface OfflinePairsPage {
  dataset: string;
  total: number;
  pairs: SupportPair[];
}

/** The SSE events webapi/agent_app.py emits, plus the bridge's `transcript`. */
export type SseEvent =
  | { event: "step"; data: TraceRecord }
  | { event: "transcript"; data: { text: string } }
  | { event: "answer"; data: SupportPair }
  | { event: "error"; data: { message: string } }
  | { event: "done"; data: Record<string, never> };

export interface CitationContent {
  kind: "text" | "pdf";
  text: string | null;
  source_url: string | null;
  file_name: string;
  page_number: number | null;
}
