import { useState } from "react";

import { CitationSidebar } from "./components/CitationSidebar";
import { SelectionProvider } from "./state/SelectionContext";

type Tab = "live" | "history";

const TABS: { id: Tab; label: string }[] = [
  { id: "live", label: "תמיכה חיה" },
  { id: "history", label: "היסטוריית שאלות" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("live");

  return (
    <SelectionProvider>
      <div className="flex h-full flex-col">
        <header className="flex items-center gap-4 border-b border-[var(--color-line)] bg-[var(--color-panel)] px-4 py-2.5">
          <span className="text-sm font-semibold">הראל — סוכן תמיכה</span>
          <nav className="flex gap-1">
            {TABS.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                onClick={() => setTab(id)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  tab === id
                    ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                    : "text-[var(--color-muted)] hover:bg-slate-100"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </header>

        <main className="min-h-0 flex-1">
          {/* RTL-mirrored split: content on the right, sources on the left. */}
          <div className="grid h-full grid-cols-[35%_65%]">
            <CitationSidebar />
            <section className="overflow-y-auto p-4 text-sm text-[var(--color-muted)]">
              {tab === "live" ? "תמיכה חיה" : "היסטוריית שאלות"}
            </section>
          </div>
        </main>
      </div>
    </SelectionProvider>
  );
}
