/** Loading placeholders. A dataset of 121 pairs takes a beat to join against
 *  the question index; an empty pane while it does looks like a bug. */
export function SkeletonLine({ width = "100%" }: { width?: string }) {
  return (
    <div
      className="h-3 rounded bg-[var(--color-line)] animate-pulse"
      style={{ width }}
    />
  );
}

export function SkeletonBlock({ lines = 3 }: { lines?: number }) {
  const widths = ["100%", "92%", "78%", "85%", "60%"];
  return (
    <div className="space-y-2 py-2">
      {Array.from({ length: lines }, (_, i) => (
        <SkeletonLine key={i} width={widths[i % widths.length]} />
      ))}
    </div>
  );
}

export function SkeletonList({ rows = 6 }: { rows?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="rounded-lg border border-[var(--color-line)] p-3">
          <SkeletonBlock lines={2} />
        </div>
      ))}
    </div>
  );
}
