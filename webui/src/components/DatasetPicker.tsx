import type { DatasetInfo } from "../types";

/** Every answer/judgment file discovered in the repo, newest first. The list
 *  comes from the server (webapi/datasets.py) — the browser never names a
 *  path of its own. */
export function DatasetPicker({
  datasets,
  selected,
  onSelect,
}: {
  datasets: DatasetInfo[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="space-y-1">
      <label
        htmlFor="dataset"
        className="block text-xs font-semibold text-[var(--color-muted)]"
      >
        מערך נתונים ({datasets.length})
      </label>
      <select
        id="dataset"
        value={selected ?? ""}
        onChange={(event) => onSelect(event.target.value)}
        className="w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] p-2 text-xs outline-none focus:border-[var(--color-accent)]"
      >
        <option value="" disabled>
          בחרו קובץ…
        </option>
        {datasets.map((dataset) => (
          <option key={dataset.id} value={dataset.id}>
            {dataset.label} · {dataset.n_pairs}
            {dataset.has_judgment ? " · שיפוט" : ""}
            {dataset.has_trace ? " · מסלול" : ""}
          </option>
        ))}
      </select>
    </div>
  );
}
