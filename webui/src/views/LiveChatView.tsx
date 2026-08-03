import { useCallback, useEffect, useRef, useState } from "react";

import { streamQuery } from "../api/client";
import { CitationSidebar } from "../components/CitationSidebar";
import { Composer } from "../components/Composer";
import { TracePanel } from "../components/TracePanel";
import { detectDir } from "../lib/direction";
import { useSelection } from "../state/SelectionContext";
import type { SupportPair, TraceRecord } from "../types";

type Message =
  | { kind: "user"; text: string }
  | { kind: "answer"; pair: SupportPair }
  | { kind: "error"; text: string };

function MetricStrip({ pair }: { pair: SupportPair }) {
  const metrics: [string, string][] = [];
  if (pair.domain) metrics.push(["תחום", pair.domain]);
  if (pair.confidence !== null)
    metrics.push(["ביטחון", `${Math.round(pair.confidence * 100)}%`]);
  if (pair.latency_ms !== null)
    metrics.push(["זמן", `${(pair.latency_ms / 1000).toFixed(1)} ש'`]);
  if (pair.cost_usd !== null) metrics.push(["עלות", `$${pair.cost_usd.toFixed(4)}`]);
  if (metrics.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-3 border-t border-[var(--color-line)] pt-2 text-[11px] text-[var(--color-muted)]">
      {metrics.map(([label, value]) => (
        <span key={label}>
          {label}: <strong className="font-medium text-[var(--color-ink)]">{value}</strong>
        </span>
      ))}
    </div>
  );
}

function AnswerBubble({ pair }: { pair: SupportPair }) {
  const { pair: selected, select, selectCitation } = useSelection();
  const isSelected = selected?.id === pair.id;

  return (
    <div className="space-y-2">
      {pair.trace && pair.trace.length > 0 && <TracePanel records={pair.trace} />}

      <div
        onClick={() => select(pair)}
        className={`cursor-pointer rounded-lg border p-3 transition-colors ${
          isSelected
            ? "border-[var(--color-accent)] bg-[var(--color-panel)]"
            : "border-[var(--color-line)] bg-[var(--color-panel)] hover:border-slate-300"
        }`}
      >
        <p
          dir={detectDir(pair.answer)}
          className="whitespace-pre-wrap text-sm leading-relaxed"
        >
          {pair.answer}
        </p>

        {pair.citations.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {pair.citations.map((citation) => (
              <button
                key={citation.id}
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  select(pair);
                  selectCitation(citation);
                }}
                className="max-w-[15rem] truncate rounded-md border border-[var(--color-line)] bg-[var(--color-surface)] px-2 py-1 text-[11px] hover:border-[var(--color-accent)]"
                title={citation.file_name}
              >
                {citation.file_name.split("/").pop()}
                {citation.page_number !== null && ` · עמ' ${citation.page_number}`}
              </button>
            ))}
          </div>
        )}

        <MetricStrip pair={pair} />
      </div>
    </div>
  );
}

export function LiveChatView() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [liveTrace, setLiveTrace] = useState<TraceRecord[]>([]);
  const [busy, setBusy] = useState(false);
  const { select } = useSelection();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, liveTrace]);

  const send = useCallback(
    async (input: string | Blob) => {
      setBusy(true);
      setLiveTrace([]);
      if (typeof input === "string") {
        setMessages((current) => [...current, { kind: "user", text: input }]);
      }

      await streamQuery(input, {
        // Audio: the bridge tells us what it heard before the answer starts.
        onTranscript: (text) =>
          setMessages((current) => [...current, { kind: "user", text }]),
        onStep: (record) => setLiveTrace((current) => [...current, record]),
        onAnswer: (pair) => {
          setMessages((current) => [...current, { kind: "answer", pair }]);
          select(pair);
        },
        onError: (message) =>
          setMessages((current) => [...current, { kind: "error", text: message }]),
      });

      setLiveTrace([]);
      setBusy(false);
    },
    [select],
  );

  return (
    // RTL: the first grid column is the RIGHT one — conversation right (65%),
    // sources left (35%).
    <div className="grid h-full grid-cols-[65%_35%]">
      <section className="flex min-h-0 flex-col">
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 && !busy && (
            <div className="mx-auto max-w-md pt-12 text-center text-sm text-[var(--color-muted)]">
              <p className="mb-1 font-medium text-[var(--color-ink)]">
                שאלו על פוליסות הראל
              </p>
              <p>
                המערכת מציגה את מסלול העיבוד — סיווג, תת-שאלות, אחזור וכלים —
                בזמן אמת, לפני שהתשובה מתחילה להיכתב.
              </p>
            </div>
          )}

          {messages.map((message, i) => {
            if (message.kind === "user") {
              return (
                <div key={i} className="flex justify-start">
                  <p
                    dir={detectDir(message.text)}
                    className="max-w-[80%] whitespace-pre-wrap rounded-lg bg-[var(--color-accent-soft)] p-2.5 text-sm text-[var(--color-ink)]"
                  >
                    {message.text}
                  </p>
                </div>
              );
            }
            if (message.kind === "error") {
              return (
                <div
                  key={i}
                  dir={detectDir(message.text)}
                  className="rounded-lg border border-[var(--color-danger)] bg-[var(--color-danger-soft)] p-2.5 text-xs text-[var(--color-danger)]"
                >
                  {message.text}
                </div>
              );
            }
            return <AnswerBubble key={i} pair={message.pair} />;
          })}

          {busy && (
            <div className="space-y-2">
              <TracePanel records={liveTrace} live />
              {liveTrace.length === 0 && (
                <p className="text-xs text-[var(--color-muted)]">ממתין לסוכן…</p>
              )}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <Composer
          busy={busy}
          onSend={(question) => void send(question)}
          onSendAudio={(audio) => void send(audio)}
        />
      </section>

      <CitationSidebar />
    </div>
  );
}
