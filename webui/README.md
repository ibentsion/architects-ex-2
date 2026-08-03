# Support web UI

Stage 3's optional bonus ("voice interface, simple UI"). **Not** part of the
graded contract — `contract.py` is, and nothing here touches it.

What makes it worth looking at: this agent publishes a real trace, so the UI
shows the pipeline while it runs — retrieval hint, classification tags and
sub-questions, each sub-question's retrieval count, every orchestrator hop and
tool call, then synthesis — all before the answer text arrives.

Two views:

- **תמיכה חיה** — ask a question, watch the pipeline, read the answer with
  citation cards that open the cited PDF page.
- **היסטוריית שאלות** — browse every answer/judgment JSONL in the repo next to
  the reference answer and the evaluation committee's grades. Read-only.

## Running it (three processes)

**1. The agent, on the GPU node.** The cross-encoder is 9 ms/pair there against
~1.7 s/pair on a laptop CPU, so the engine runs where the GPU is:

```bash
cloud/submit_job.sh run 'bash cloud/serve_ui.sh' --timeout 3h
```

**2. The tunnel and the bridge, on the laptop:**

```bash
ssh -N -L 8000:localhost:8000 <node>              # AGENT_BASE_URL points here
./venv/bin/python -m uvicorn webapi.bridge_app:app --port 8080
```

**3. The frontend, on the laptop:**

```bash
npm --prefix webui install
npm --prefix webui run dev                        # http://localhost:5173
```

Vite proxies `/api` to the bridge on 8080, so the browser is always
same-origin — which is why the bridge ships no CORS middleware. Port 8080 keeps
8000 free for the tunnel.

## Environment

| Variable | Default | Where | What |
|---|---|---|---|
| `AGENT_BASE_URL` | `http://localhost:8000` | bridge | The agent app. Read per request, so pointing it at a public node URL needs no code change. |
| `RAG_CONFIG` | `configs/ship.yaml` | agent app | Same config the graded endpoint uses. Read, never written. |
| `STT_MODEL` | *(unset — STT off)* | bridge | Enables backend transcription, e.g. `base` on CPU or an `ivrit-ai` model on the node. |
| `STT_DEVICE` | `cpu` | bridge | `cuda` on the node. |
| `STT_COMPUTE_TYPE` | `int8` | bridge | `float16` on a GPU. |

## Voice input

Browser first: where the Web Speech API exists (Chrome/Edge) it transcribes
`he-IL` locally and no audio leaves the machine. Elsewhere the clip is recorded
and POSTed to the bridge.

The backend leg needs a local model, because Token Factory has **no ASR** at all
(verified 2026-08-03: 27 models, none audio; `/v1/audio/transcriptions` 404s).
So it is off by default and `faster-whisper` is deliberately **not** in
`requirements.txt`:

```bash
pip install faster-whisper
export STT_MODEL=base
```

Without it, recording returns HTTP 501 and the UI shows the remediation text
verbatim. It never invents or empties a transcript.

## Caveats

- **Citation previews need a local `corpus/` and `cache/parsed/`**, both
  gitignored and often absent. Page images are rendered with pypdfium2; page
  *text* is resolved through `evalharness.pages.PageStore` (corpus walk →
  sha256 → Docling parse cache), so nothing is ever parsed on a request. With
  either missing, cards fall back to "no preview available" and the endpoints
  answer 404 — the app keeps working.
- The first citation preview after startup walks the corpus once (~3 s for 571
  files) and is memoized for the life of the process; every later lookup is
  instant. Thumbnails are cached on disk under `cache/webui_thumbs/`.
- **No auth, by decision.** This is a localhost tool. Do not expose the bridge
  on a public interface: it reads repo files by design.
- No multi-turn memory — the engine is stateless per query, and the UI's
  message list is for display only. History is never sent back.

## Layout

```
src/api/client.ts            fetch + ReadableStream SSE parser (EventSource cannot POST)
src/types.ts                 mirror of webapi/schema.py — that is the source of truth
src/state/SelectionContext   the selected pair/citation, shared by both views
src/components/TracePanel    the pipeline trace, live or replayed
src/components/CitationSidebar  one sidebar, both views
src/views/                   LiveChatView, HistoryView
```
