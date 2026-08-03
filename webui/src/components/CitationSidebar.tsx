import { useEffect, useState } from "react";

import { getCitationContent } from "../api/client";
import { detectDir } from "../lib/direction";
import { useSelection } from "../state/SelectionContext";
import type { CitationContent, SupportCitation } from "../types";

/** The source panel, shared by both views (that is the whole point of
 *  SelectionContext). RTL-mirrored: it sits on the LEFT.
 *
 *  `corpus/` is gitignored and frequently absent on the machine running the
 *  bridge, so a missing thumbnail is a placeholder, not an error. */

function CitationCard({
  citation,
  active,
  onSelect,
}: {
  citation: SupportCitation;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full rounded-lg border p-2.5 text-start transition-colors ${
        active
          ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]"
          : "border-[var(--color-line)] bg-[var(--color-panel)] hover:border-slate-300"
      }`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-xs font-medium" title={citation.file_name}>
          {citation.file_name.split("/").pop()}
        </span>
        {citation.page_number !== null && (
          <span dir="ltr" className="shrink-0 text-[11px] text-[var(--color-muted)]">
            עמ' {citation.page_number}
          </span>
        )}
      </div>
      {citation.content_preview && (
        <p
          dir={detectDir(citation.content_preview)}
          className="mt-1 line-clamp-2 text-[11px] text-[var(--color-muted)]"
        >
          {citation.content_preview}
        </p>
      )}
    </button>
  );
}

function Thumbnail({ url }: { url: string }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [url]);

  if (failed) {
    return (
      <div className="flex h-40 items-center justify-center rounded-lg border border-dashed border-[var(--color-line)] p-4 text-center text-[11px] text-[var(--color-muted)]">
        אין תצוגה מקדימה — התיקייה <code dir="ltr">corpus/</code> אינה זמינה במכונה
        הזו
      </div>
    );
  }
  return (
    <img
      src={url}
      alt="עמוד המקור"
      onError={() => setFailed(true)}
      className="w-full rounded-lg border border-[var(--color-line)] bg-white"
    />
  );
}

function CitationDetail({ citation }: { citation: SupportCitation }) {
  const [content, setContent] = useState<CitationContent | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setContent(null);
    setError(null);
    getCitationContent(citation.file_name, citation.page_number)
      .then((body) => !cancelled && setContent(body))
      .catch((exc: Error) => !cancelled && setError(exc.message));
    return () => {
      cancelled = true;
    };
  }, [citation.file_name, citation.page_number]);

  const text = content?.text ?? citation.content_preview;

  return (
    <div className="space-y-3 border-t border-[var(--color-line)] pt-3">
      <div className="text-xs font-medium" dir={detectDir(citation.file_name)}>
        {citation.file_name}
      </div>

      {citation.thumbnail_url && <Thumbnail url={citation.thumbnail_url} />}

      {/* The cited page's text, from the parse cache — shown alongside the
          page image, since it is the text the retriever actually indexed. */}
      {text && (
        <p
          dir={detectDir(text)}
          className="max-h-64 overflow-y-auto whitespace-pre-wrap rounded-lg border border-[var(--color-line)] bg-white p-2.5 text-xs leading-relaxed"
        >
          {text}
        </p>
      )}

      {error && (
        <p className="rounded bg-[var(--color-warn-soft)] p-2 text-[11px] text-[var(--color-warn)]">
          {error}
        </p>
      )}

      {content?.source_url && (
        <a
          href={content.source_url}
          target="_blank"
          rel="noreferrer"
          dir="ltr"
          className="block truncate text-[11px] text-[var(--color-accent)] underline"
        >
          {content.source_url}
        </a>
      )}
    </div>
  );
}

export function CitationSidebar() {
  const { pair, citation, selectCitation } = useSelection();
  const citations = pair?.citations ?? [];

  return (
    <aside className="flex h-full flex-col gap-3 overflow-y-auto border-e border-[var(--color-line)] bg-[var(--color-surface)] p-3">
      <h2 className="text-xs font-semibold text-[var(--color-muted)]">
        מקורות {citations.length > 0 && `(${citations.length})`}
      </h2>

      {citations.length === 0 ? (
        <p className="text-xs text-[var(--color-muted)]">
          בחרו תשובה כדי לראות את המקורות שעליהם היא נשענת.
        </p>
      ) : (
        <div className="space-y-2">
          {citations.map((item) => (
            <CitationCard
              key={item.id}
              citation={item}
              active={citation?.id === item.id}
              onSelect={() => selectCitation(citation?.id === item.id ? null : item)}
            />
          ))}
        </div>
      )}

      <div
        className={`transition-transform duration-200 ${
          citation ? "translate-x-0" : "translate-x-4 opacity-0"
        }`}
      >
        {citation && <CitationDetail citation={citation} />}
      </div>
    </aside>
  );
}
