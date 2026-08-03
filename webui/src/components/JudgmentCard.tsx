import { detectDir } from "../lib/direction";
import type { JudgeGrades } from "../types";

/** The evaluation committee's verdict on one answer — the reason the History
 *  view exists at all: read the grade next to the answer that earned it. */

const VERDICT_TONE: Record<string, string> = {
  correct: "bg-emerald-50 text-emerald-700",
  partial: "bg-[var(--color-warn-soft)] text-[var(--color-warn)]",
  incorrect: "bg-[var(--color-danger-soft)] text-[var(--color-danger)]",
};

function GradeBar({ label, score }: { label: string; score: number | null }) {
  if (score === null) return null;
  const percent = Math.max(0, Math.min(100, score * 10));
  return (
    <div className="flex items-center gap-2">
      <span className="w-24 shrink-0 text-[11px] text-[var(--color-muted)]">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-line)]">
        <div
          className="h-full rounded-full bg-[var(--color-accent)]"
          style={{ width: `${percent}%` }}
        />
      </div>
      <span dir="ltr" className="w-8 shrink-0 text-[11px] font-medium">
        {score}
      </span>
    </div>
  );
}

export function JudgmentCard({ judgment }: { judgment: JudgeGrades }) {
  const reasoning = Object.entries(judgment.reasoning ?? {});

  return (
    <div className="space-y-3 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-[var(--color-muted)]">
          שיפוט הוועדה
        </span>
        {judgment.verdict && (
          <span
            className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${
              VERDICT_TONE[judgment.verdict] ?? "bg-slate-100 text-slate-600"
            }`}
          >
            {judgment.verdict}
          </span>
        )}
        {judgment.hallucination && (
          <span className="rounded bg-[var(--color-danger-soft)] px-1.5 py-0.5 text-[11px] font-medium text-[var(--color-danger)]">
            הזיה
          </span>
        )}
      </div>

      <div className="space-y-1.5">
        <GradeBar label="נכונות" score={judgment.correctness} />
        <GradeBar label="שלמות" score={judgment.completeness} />
        <GradeBar label="איכות שיחה" score={judgment.conversational_quality} />
      </div>

      {reasoning.length > 0 && (
        <details className="text-[11px]">
          <summary className="cursor-pointer text-[var(--color-muted)]">
            נימוקי השופטים ({reasoning.length})
          </summary>
          <div className="mt-2 space-y-2">
            {reasoning.map(([judge, text]) => (
              <div key={judge}>
                <div dir="ltr" className="font-mono text-[10px] text-[var(--color-muted)]">
                  {judge}
                </div>
                <p dir={detectDir(text)} className="leading-relaxed">
                  {text}
                </p>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
