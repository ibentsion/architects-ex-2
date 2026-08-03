"""FastAPI layer for the Stage-3 bonus web UI (webui/).

Two apps, deliberately separate processes:
  * ``webapi.agent_app`` runs on the GPU node and streams the agentic RAG
    pipeline's own trace records as SSE.
  * ``webapi.bridge_app`` runs on the laptop, relays that stream to the
    browser, and serves the offline QA-history and citation endpoints.

Nothing here is part of the graded contract — ``contract.py`` is.
"""
