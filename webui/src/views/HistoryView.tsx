import { useEffect, useMemo, useState } from "react";

import { getDatasets, getOfflinePairs } from "../api/client";
import { CitationSidebar } from "../components/CitationSidebar";
import { DatasetPicker } from "../components/DatasetPicker";
import { JudgmentCard } from "../components/JudgmentCard";
import { SkeletonBlock, SkeletonList } from "../components/Skeleton";
import { TracePanel } from "../components/TracePanel";
import { detectDir } from "../lib/direction";
import { useSelection } from "../state/SelectionContext";
import type { DatasetInfo, SupportPair } from "../types";

const ANY = "";

function Filters({
  pairs,
  verdict,
  difficulty,
  domain,
  hallucinationOnly,
  set,
}: {
  pairs: SupportPair[];
  verdict: string;
  difficulty: string;
  domain: string;
  hallucinationOnly: boolean;
  set: (patch: Partial<FilterState>) => void;
}) {
  const options = (values: (string | null)[]) =>
    Array.from(new Set(values.filter((v): v is string => Boolean(v)))).sort();

  const selectClass =
    "rounded border border-[var(--color-line)] bg-[var(--color-panel)] p-1 text-[11px] outline-none";

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap gap-1.5">
        <select
          value={verdict}
          onChange={(e) => set({ verdict: e.target.value })}
          className={selectClass}
          aria-label="פסיקה"
        >
          <option value={ANY}>כל הפסיקות</option>
          {options(pairs.map((p) => p.judgment?.verdict ?? null)).map((value) => (
            <option key={value}>{value}</option>
          ))}
        </select>
        <select
          value={difficulty}
          onChange={(e) => set({ difficulty: e.target.value })}
          className={selectClass}
          aria-label="רמת קושי"
        >
          <option value={ANY}>כל רמות הקושי</option>
          {options(pairs.map((p) => p.difficulty)).map((value) => (
            <option key={value}>{value}</option>
          ))}
        </select>
        <select
          value={domain}
          onChange={(e) => set({ domain: e.target.value })}
          className={selectClass}
          aria-label="תחום"
        >
          <option value={ANY}>כל התחומים</option>
          {options(pairs.map((p) => p.domain)).map((value) => (
            <option key={value}>{value}</option>
          ))}
        </select>
      </div>
      <label className="flex items-center gap-1.5 text-[11px] text-[var(--color-muted)]">
        <input
          type="checkbox"
          checked={hallucinationOnly}
          onChange={(e) => set({ hallucinationOnly: e.target.checked })}
        />
        הזיות בלבד
      </label>
    </div>
  );
}

interface FilterState {
  verdict: string;
  difficulty: string;
  domain: string;
  hallucinationOnly: boolean;
}

function PairDetail({ pair }: { pair: SupportPair }) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-1 text-xs font-semibold text-[var(--color-muted)]">השאלה</h3>
        {pair.question ? (
          <p dir={detectDir(pair.question)} className="text-sm leading-relaxed">
            {pair.question}
          </p>
        ) : (
          // Reachable: an arbitrary answers JSONL can carry ids that no
          // reference set has. Never render this as an empty box.
          <div className="space-y-1">
            <p dir="ltr" className="font-mono text-xs">
              {pair.id}
            </p>
            <p className="text-[11px] text-[var(--color-muted)]">
              השאלה לא נמצאה באף מערך ייחוס
            </p>
          </div>
        )}
        <div className="mt-1.5 flex flex-wrap gap-1.5 text-[11px] text-[var(--color-muted)]">
          {pair.domain && <span>תחום: {pair.domain}</span>}
          {pair.difficulty && <span>קושי: {pair.difficulty}</span>}
          {pair.latency_ms !== null && (
            <span dir="ltr">{(pair.latency_ms / 1000).toFixed(1)}s</span>
          )}
          {pair.cost_usd !== null && <span dir="ltr">${pair.cost_usd.toFixed(4)}</span>}
        </div>
      </div>

      <div>
        <h3 className="mb-1 text-xs font-semibold text-[var(--color-muted)]">
          תשובת המערכת
        </h3>
        {pair.answer ? (
          <p
            dir={detectDir(pair.answer)}
            className="whitespace-pre-wrap rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] p-3 text-sm leading-relaxed"
          >
            {pair.answer}
          </p>
        ) : (
          <p className="rounded-lg bg-[var(--color-warn-soft)] p-2 text-xs text-[var(--color-warn)]">
            אין קובץ תשובות מקושר לריצה הזו — מוצגים השיפוטים בלבד
          </p>
        )}
      </div>

      {pair.reference_answer && (
        <div>
          <h3 className="mb-1 text-xs font-semibold text-[var(--color-muted)]">
            תשובת הייחוס
          </h3>
          <p
            dir={detectDir(pair.reference_answer)}
            className="whitespace-pre-wrap rounded-lg border border-dashed border-[var(--color-line)] p-3 text-sm leading-relaxed text-[var(--color-muted)]"
          >
            {pair.reference_answer}
          </p>
        </div>
      )}

      {pair.judgment && <JudgmentCard judgment={pair.judgment} />}
      {pair.trace && pair.trace.length > 0 && <TracePanel records={pair.trace} />}
    </div>
  );
}

export function HistoryView() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [pairs, setPairs] = useState<SupportPair[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    verdict: ANY,
    difficulty: ANY,
    domain: ANY,
    hallucinationOnly: false,
  });
  const { pair: selectedPair, select } = useSelection();

  useEffect(() => {
    getDatasets()
      .then((found) => {
        setDatasets(found);
        setLoading(false);
        if (found.length > 0) setSelectedDataset(found[0].id);
      })
      .catch((exc: Error) => {
        setError(exc.message);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!selectedDataset) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    select(null);
    getOfflinePairs(selectedDataset)
      .then((page) => {
        if (cancelled) return;
        setPairs(page.pairs);
        setTotal(page.total);
        setLoading(false);
      })
      .catch((exc: Error) => {
        if (cancelled) return;
        setError(exc.message);
        setPairs([]);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedDataset, select]);

  const visible = useMemo(
    () =>
      pairs.filter((pair) => {
        if (filters.verdict && pair.judgment?.verdict !== filters.verdict) return false;
        if (filters.difficulty && pair.difficulty !== filters.difficulty) return false;
        if (filters.domain && pair.domain !== filters.domain) return false;
        if (filters.hallucinationOnly && !pair.judgment?.hallucination) return false;
        return true;
      }),
    [pairs, filters],
  );

  return (
    // RTL: picker + list on the RIGHT, detail in the CENTRE, sources LEFT.
    <div className="grid h-full grid-cols-[25%_50%_25%]">
      <section className="flex min-h-0 flex-col gap-3 overflow-y-auto border-s border-[var(--color-line)] p-3">
        <DatasetPicker
          datasets={datasets}
          selected={selectedDataset}
          onSelect={setSelectedDataset}
        />
        <Filters
          pairs={pairs}
          {...filters}
          set={(patch) => setFilters((current) => ({ ...current, ...patch }))}
        />

        <div className="text-[11px] text-[var(--color-muted)]">
          {visible.length} מתוך {total}
        </div>

        {loading ? (
          <SkeletonList rows={6} />
        ) : (
          <ul className="space-y-1.5">
            {visible.map((pair) => (
              <li key={pair.id}>
                <button
                  type="button"
                  onClick={() => select(pair)}
                  className={`w-full rounded-lg border p-2 text-start transition-colors ${
                    selectedPair?.id === pair.id
                      ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]"
                      : "border-[var(--color-line)] bg-[var(--color-panel)] hover:border-slate-300"
                  }`}
                >
                  <span
                    dir={detectDir(pair.question ?? pair.id)}
                    className="line-clamp-2 block text-xs"
                  >
                    {pair.question ?? pair.id}
                  </span>
                  <span className="mt-1 flex flex-wrap gap-1 text-[10px] text-[var(--color-muted)]">
                    {pair.judgment?.verdict && <span>{pair.judgment.verdict}</span>}
                    {pair.judgment?.hallucination && (
                      <span className="text-[var(--color-danger)]">הזיה</span>
                    )}
                    {pair.difficulty && <span>{pair.difficulty}</span>}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="min-h-0 overflow-y-auto p-4">
        {error && (
          <p className="rounded-lg border border-[var(--color-danger)] bg-[var(--color-danger-soft)] p-2.5 text-xs text-[var(--color-danger)]">
            {error}
          </p>
        )}
        {!error && loading && <SkeletonBlock lines={5} />}
        {!error && !loading && !selectedPair && (
          <p className="pt-12 text-center text-sm text-[var(--color-muted)]">
            בחרו שאלה מהרשימה כדי לראות את התשובה, השיפוט ומסלול העיבוד.
          </p>
        )}
        {!error && selectedPair && <PairDetail pair={selectedPair} />}
      </section>

      <CitationSidebar />
    </div>
  );
}
