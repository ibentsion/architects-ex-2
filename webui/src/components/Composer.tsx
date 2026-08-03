import { useCallback, useMemo, useState } from "react";

import { useVoiceInput } from "../hooks/useVoiceInput";
import { detectDir } from "../lib/direction";
import { Waveform } from "./Waveform";

export function Composer({
  busy,
  onSend,
  onSendAudio,
}: {
  busy: boolean;
  onSend: (question: string) => void;
  onSendAudio: (audio: Blob) => void;
}) {
  const [text, setText] = useState("");

  const handlers = useMemo(
    () => ({
      onInterim: (partial: string) => setText(partial),
      onTranscript: (final: string) => setText(final),
      onRecording: onSendAudio,
    }),
    [onSendAudio],
  );
  const voice = useVoiceInput(handlers);

  const submit = useCallback(() => {
    const question = text.trim();
    if (!question || busy) return;
    onSend(question);
    setText("");
  }, [text, busy, onSend]);

  return (
    <div className="space-y-2 border-t border-[var(--color-line)] bg-[var(--color-panel)] p-3">
      {voice.error && (
        <div className="flex items-start justify-between gap-2 rounded-lg bg-[var(--color-danger-soft)] p-2 text-xs text-[var(--color-danger)]">
          <span>{voice.error}</span>
          <button type="button" onClick={voice.clearError} className="shrink-0 underline">
            סגירה
          </button>
        </div>
      )}

      <div className="flex items-end gap-2">
        <textarea
          value={text}
          dir={detectDir(text)}
          rows={2}
          disabled={busy}
          placeholder="מה תרצו לשאול? (Enter לשליחה, Shift+Enter לשורה חדשה)"
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          className="min-h-[3rem] flex-1 resize-y rounded-lg border border-[var(--color-line)] p-2 text-sm outline-none focus:border-[var(--color-accent)] disabled:bg-slate-50"
        />

        <div className="flex flex-col items-stretch gap-1.5">
          <button
            type="button"
            onClick={submit}
            disabled={busy || !text.trim()}
            className="rounded-lg bg-[var(--color-accent)] px-4 py-2 text-xs font-medium text-white disabled:opacity-40"
          >
            {busy ? "מעבד…" : "שליחה"}
          </button>
          <button
            type="button"
            disabled={busy || voice.mode === "unsupported"}
            onClick={() => (voice.active ? voice.stop() : void voice.start())}
            title={
              voice.mode === "speech-api"
                ? "זיהוי דיבור בדפדפן (he-IL)"
                : "הקלטה ושליחה לתמלול בשרת"
            }
            className={`rounded-lg border px-4 py-2 text-xs font-medium transition-colors disabled:opacity-40 ${
              voice.active
                ? "border-[var(--color-danger)] bg-[var(--color-danger-soft)] text-[var(--color-danger)]"
                : "border-[var(--color-line)] text-[var(--color-muted)] hover:border-slate-300"
            }`}
          >
            {voice.active ? "עצירה ●" : "דיבור 🎤"}
          </button>
        </div>
      </div>

      {voice.active && voice.analyser && (
        <div className="flex items-center gap-2">
          <Waveform analyser={voice.analyser} />
          <span className="text-[11px] text-[var(--color-muted)]">מקליט…</span>
        </div>
      )}
    </div>
  );
}
