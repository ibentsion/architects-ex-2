import { useState, type ReactElement } from "react";

import { SelectionProvider } from "./state/SelectionContext";
import { HistoryView } from "./views/HistoryView";
import { LiveChatView } from "./views/LiveChatView";

type Tab = "live" | "history";

function ChatIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      className="h-5 w-5"
    >
      <path
        d="M21 12a8 8 0 0 1-8 8H7l-4 3v-5.5A8 8 0 0 1 11 4h2a8 8 0 0 1 8 8Z"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      className="h-5 w-5"
    >
      <rect x="4" y="3" width="16" height="18" rx="2" />
      <path d="M8 8h8M8 12h8M8 16h5" strokeLinecap="round" />
    </svg>
  );
}

const TABS: { id: Tab; label: string; Icon: () => ReactElement }[] = [
  { id: "live", label: "תמיכה חיה", Icon: ChatIcon },
  { id: "history", label: "היסטוריית שאלות", Icon: HistoryIcon },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("live");

  return (
    <SelectionProvider>
      {/* RTL: the first flex child is the RIGHT one, so this persistent nav
          rail sits on the inline-start edge and `border-e` faces the content.
          The product mark lives in the rail — there is no header bar. */}
      <div className="flex h-full">
        <nav
          aria-label="ניווט ראשי"
          className="flex w-20 shrink-0 flex-col items-center gap-1 border-e border-[var(--color-line)] bg-[var(--color-panel)] py-3"
        >
          <div className="mb-3 flex flex-col items-center gap-1">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--color-accent)] text-sm font-bold text-white">
              ה
            </span>
            <span className="text-[10px] font-semibold leading-tight text-[var(--color-muted)]">
              סוכן תמיכה
            </span>
          </div>

          {TABS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              aria-current={tab === id ? "page" : undefined}
              className={`flex w-16 flex-col items-center gap-1 rounded-lg px-1 py-2 text-[10px] font-medium leading-tight transition-colors ${
                tab === id
                  ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                  : "text-[var(--color-muted)] hover:bg-slate-100"
              }`}
            >
              <Icon />
              <span className="text-center">{label}</span>
            </button>
          ))}
        </nav>

        <main className="min-h-0 min-w-0 flex-1">
          {/* Each view owns its own split, because the proportions differ:
              65/35 for the conversation, 25/50/25 for the history triptych. */}
          {tab === "live" ? <LiveChatView /> : <HistoryView />}
        </main>
      </div>
    </SelectionProvider>
  );
}
