import { detectDir } from "../lib/direction";
import type { TraceRecord } from "../types";

/** The pipeline, as it happens.
 *
 * This is the thing a generic chat UI cannot show: the agent publishes a real
 * trace — retrieval hint, classification tags, one record per sub-question
 * retrieval, every orchestrator hop and tool call, then synthesis — and every
 * record arrives before the next step starts (rag/agent/engine.py `_emit`).
 *
 * One component, two uses: live in the chat view, replayed from `pair.trace`
 * in the history view.
 */

function Chip({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "accent" | "warn" | "danger";
}) {
  const tones = {
    neutral: "bg-slate-100 text-slate-600",
    accent: "bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
    warn: "bg-[var(--color-warn-soft)] text-[var(--color-warn)]",
    danger: "bg-[var(--color-danger-soft)] text-[var(--color-danger)]",
  };
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

const STEP_LABEL: Record<string, string> = {
  hint: "רמז אחזור",
  classify: "סיווג",
  retrieve: "אחזור",
  orchestrator: "מתזמר",
  calculate: "חישוב",
  synthesize: "ניסוח",
};

function StepBody({ record }: { record: TraceRecord }) {
  switch (record.step) {
    case "hint":
      return (
        <div className="text-xs text-[var(--color-muted)]">
          {record.top_category ? (
            <>
              קטגוריה מובילה <Chip>{record.top_category}</Chip>{" "}
              {typeof record.top_share === "number" &&
                `${Math.round(record.top_share * 100)}%`}{" "}
              מתוך {record.n_hits} תוצאות
            </>
          ) : (
            "החיפוש באינדקס לא החזיר תוצאות"
          )}
        </div>
      );

    case "classify":
      return (
        <div className="space-y-1.5">
          <div className="flex flex-wrap gap-1">
            {(record.categories ?? []).map((category) => (
              <Chip key={category} tone="accent">
                {category}
              </Chip>
            ))}
            {record.difficulty && <Chip>רמת קושי: {record.difficulty}</Chip>}
            {record.mode && <Chip>{record.mode}</Chip>}
            {record.needs_calculation && <Chip tone="warn">דורש חישוב</Chip>}
            {record.dependent && <Chip tone="warn">תת-שאלות תלויות</Chip>}
          </div>
          {(record.sub_questions ?? []).length > 0 && (
            <ol className="list-decimal space-y-0.5 ps-5 text-xs text-[var(--color-muted)]">
              {(record.sub_questions ?? []).map((question, i) => (
                <li key={i} dir={detectDir(question)}>
                  {question}
                </li>
              ))}
            </ol>
          )}
        </div>
      );

    case "retrieve":
      return (
        <div className="space-y-1 text-xs text-[var(--color-muted)]">
          {record.query && (
            <div dir={detectDir(record.query)} className="text-[var(--color-ink)]">
              {record.query}
            </div>
          )}
          <div className="flex flex-wrap items-center gap-1">
            <Chip>{record.n_gated ?? 0} קטעים</Chip>
            {record.category && (
              <Chip tone="accent">
                {Array.isArray(record.category)
                  ? record.category.join(" · ")
                  : record.category}
              </Chip>
            )}
            {record.phase === "loop" && <Chip>בלולאת הסוכן</Chip>}
            {record.retried_unfiltered && <Chip tone="warn">ניסיון חוזר ללא סינון</Chip>}
          </div>
        </div>
      );

    case "orchestrator":
      if (record.error) {
        return (
          <div className="text-xs text-[var(--color-danger)]">
            הלולאה נכשלה ({record.error}) — ממשיך עם הראיות שכבר נאספו
          </div>
        );
      }
      return (
        <div className="flex flex-wrap items-center gap-1 text-xs text-[var(--color-muted)]">
          <Chip>סבב {(record.hop ?? 0) + 1}</Chip>
          <Chip>{record.n_tool_calls ?? 0} קריאות כלי</Chip>
          {record.content && (
            <span dir={detectDir(record.content)} className="truncate">
              {record.content}
            </span>
          )}
        </div>
      );

    case "calculate":
      return (
        <div dir="ltr" className="text-xs font-mono text-[var(--color-ink)]">
          {record.error ? (
            <span className="text-[var(--color-danger)]">
              {record.expression} — {record.error}
            </span>
          ) : (
            <>
              {record.expression} = <strong>{record.value}</strong>
            </>
          )}
        </div>
      );

    case "synthesize":
      return (
        <div className="flex flex-wrap items-center gap-1 text-xs text-[var(--color-muted)]">
          <span dir="ltr" className="font-mono">
            {record.model}
          </span>
          {record.fast_synthesis && <Chip>מודל מהיר</Chip>}
        </div>
      );

    default:
      return (
        <div dir="ltr" className="text-xs font-mono text-[var(--color-muted)]">
          {JSON.stringify(record)}
        </div>
      );
  }
}

export function TracePanel({
  records,
  live = false,
}: {
  records: TraceRecord[];
  live?: boolean;
}) {
  if (records.length === 0) return null;

  return (
    <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] p-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-[var(--color-muted)]">
        <span>מסלול העיבוד</span>
        {live && (
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--color-accent)]" />
        )}
      </div>
      <ol className="space-y-2">
        {records.map((record, i) => (
          <li key={i} className="flex gap-2">
            <div className="flex flex-col items-center pt-1">
              <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-accent)]" />
              {i < records.length - 1 && (
                <span className="mt-1 w-px flex-1 bg-[var(--color-line)]" />
              )}
            </div>
            <div className="flex-1 pb-1">
              <div className="flex items-baseline gap-2">
                <span className="text-xs font-semibold">
                  {STEP_LABEL[record.step] ?? record.step}
                </span>
                {typeof record.ms === "number" && (
                  <span dir="ltr" className="text-[11px] text-[var(--color-muted)]">
                    {record.ms} ms
                  </span>
                )}
              </div>
              <StepBody record={record} />
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
