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

## Running it

The engine can run locally, but the cross-encoder is ~1.7 s/pair on a laptop CPU
against 9 ms/pair on the node's L40S — measured end to end, that is **~30 s per
question locally vs ~3 s on the node**.

### Local only (no cloud, ~30 s/question)

```bash
./venv/bin/python -m uvicorn webapi.agent_app:app --port 8000    # the engine
./venv/bin/python -m uvicorn webapi.bridge_app:app --port 8080   # the bridge
npm --prefix webui install && npm --prefix webui run dev         # localhost:5173
```

The history view needs none of this — only the bridge.

### Agent on the GPU node (~3 s/question)

**There is no SSH tunnel to a job.** A `nebius ai job` gets a private VPC IP
only: no public IP, no SSH authorized keys, and `nebius ai job ssh` has no `-L`.
`--container-port` on a job yields a private endpoint and nothing more. Anything
that must be *reachable* is a `nebius ai endpoint`, which publishes each HTTP
port on a managed `https://` URL with no public IP required.

```bash
cloud/serve_endpoint.sh create      # ~4 min: clone, setup, serve, warm
cloud/serve_endpoint.sh status      # URL + state

export AGENT_BASE_URL=$(cloud/serve_endpoint.sh status | grep -o 'https://[^ ]*')
export AGENT_TOKEN=$(cat .agent_token)
./venv/bin/python -m uvicorn webapi.bridge_app:app --port 8080
npm --prefix webui run dev

cloud/serve_endpoint.sh stop        # it holds a GPU until you do this
```

### Everything on the node (one URL to share)

Agent, bridge and the built frontend all run in the endpoint container, so
anyone you give the URL to can open it in a browser with nothing installed:

```bash
npm --prefix webui run build
cloud/upload_artifacts.sh webui-dist      # ships dist/ like corpus/cache/rag_index
cloud/serve_endpoint.sh create --full     # prints the password
cloud/serve_endpoint.sh status            # prints the URL
```

The UI bundle travels through the artifacts dataset rather than being built on
the node, so the GPU image needs no Node toolchain.

`status` prints the URL to hand out — use the `https:` line.

Getting https took a Cloudflare quick tunnel, and the reason is the microphone:
browsers refuse `getUserMedia` and the Web Speech API outside a **secure
context** (https, or localhost). The platform's own routes cannot provide one
here. It publishes a managed `https://` tunnel and the instance's raw
`IP:port`, but the managed tunnel terminates TLS itself and returns its own 404
unless the endpoint was created with `--auth token` — which `--full` cannot use,
because a browser sends no `Authorization` header when you navigate to a URL.
Verified on both `/http` and `/tcp` port modes. That leaves the raw IP, which is
plain HTTP, where the mic is blocked and the password would cross the network in
the clear.

So `--full` starts `cloudflared` against the bridge and prints the
`https://….trycloudflare.com` URL. The binary is pinned by version and verified
by sha256 before it is executed — it fronts an app that spends the shared key.
The password gate applies to the tunnel exactly as to the raw IP, and uvicorn
runs with `--proxy-headers` (trusted from loopback only, where cloudflared
connects) so the session cookie is marked `Secure` on an https visit.

The raw `http://IP:port` stays reachable as a fallback; everything works there
except voice.

**This mode is internet-facing**, and the bridge reads repo files by design —
the corpus, `eval_results/`, the graded submission — while every question spends
the *shared* course Token Factory key. So access is gated by a password
(`webapi/auth.py`): the platform's `--auth token` is useless here because a
browser navigating to a URL sends no `Authorization` header, so the endpoint is
open at the platform layer and the **bridge** is the gate. `serve_ui.sh` refuses
to start in this mode without `UI_PASSWORD`. The password is generated at create
time and written to `.ui_password` (gitignored).

Sessions are HMAC-signed cookies keyed off the password, so changing the
password logs everyone out and restarting the bridge does not.

The endpoint is created with `--auth token` and a generated bearer token: the
URL is public, and every question spends the **shared** course Token Factory
key. `AGENT_TOKEN` is what the bridge presents; without it the endpoint answers
401. Leave both unset for a local agent.

Vite proxies `/api` to the bridge on 8080, so the browser is always
same-origin — which is why the bridge ships no CORS middleware.

## Environment

| Variable | Default | Where | What |
|---|---|---|---|
| `AGENT_BASE_URL` | `http://localhost:8000` | bridge | The agent app. Read per request, so pointing it at a public node URL needs no code change. |
| `AGENT_TOKEN` | *(unset)* | bridge | Bearer token sent to the agent. Required when it is a `nebius ai endpoint` created with `--auth token`; unset for a local agent. |
| `UI_PASSWORD` | *(unset — gate off)* | bridge | Shared password. Unset means no gate, which is the localhost default. Set it and every path except `/login` and `/healthz` needs a session. |
| `WEBUI_DIST` | `webui/dist` | bridge | Built frontend to serve. Ignored when the bundle isn't there, which is the dev case (Vite serves it and proxies `/api`). |
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
