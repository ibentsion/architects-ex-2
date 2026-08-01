"""Query-CLI stage-tool tests: subcommand dispatch, parameter plumbing to the
Retriever stages, JSON output shape, and rerank candidate loading/hydration.
Backends all faked (via tests.test_retrieve fakes) — no index, no models.
"""
from __future__ import annotations

import json

import pytest

from rag.cli import query as query_cli
from tests.test_retrieve import (
    APT1,
    APT2,
    make_chunk,
    make_retriever,
)


@pytest.fixture
def config_path(repo_root):
    return str(repo_root / "configs" / "default.yaml")


@pytest.fixture
def fake_retriever(monkeypatch):
    """Patch config/retriever loading so tools_main runs on the fakes."""
    apt1, apt2 = make_chunk(APT1), make_chunk(APT2)
    retriever = make_retriever(
        dense_hits=[(apt1, 0.9), (apt2, 0.8)],
        sparse_hits=[(APT2, 7.5), (APT1, 3.0)],
        store={APT1: apt1, APT2: apt2},
        rerank_scores={APT1: 0.8, APT2: 0.6},
    )
    monkeypatch.setattr(query_cli, "load_config", lambda path: {"path": path})
    monkeypatch.setattr(query_cli, "load_retriever", lambda config: retriever)
    return retriever


def run_tool(capsys, argv) -> dict:
    assert query_cli.main(argv) == query_cli.EXIT_OK
    return json.loads(capsys.readouterr().out)


def test_dense_command_with_top_k_override(fake_retriever, config_path, capsys):
    out = run_tool(capsys, ["dense", "שאלה", "--config", config_path, "--top-k", "1"])
    assert len(out["results"]) == 1
    top = out["results"][0]
    assert top["chunk"]["chunk_id"] == APT1
    assert top["dense_score"] == 0.9
    assert top["rerank_score"] is None


def test_sparse_command_hydrates_chunks(fake_retriever, config_path, capsys):
    out = run_tool(capsys, ["sparse", "שאלה", "--config", config_path])
    assert [r["chunk"]["chunk_id"] for r in out["results"]] == [APT2, APT1]
    assert out["results"][0]["sparse_score"] == 7.5
    assert out["results"][0]["chunk"]["text"]  # hydrated from the payload store


def test_fuse_command_emits_stats_without_rerank(fake_retriever, config_path, capsys):
    out = run_tool(capsys, ["fuse", "שאלה", "--config", config_path, "--rrf-k", "1"])
    assert all(r["rerank_score"] is None for r in out["results"])
    assert all(r["rrf_score"] is not None for r in out["results"])
    assert set(out["stats"]) == {"dense", "sparse", "fused"}


def test_retrieve_command_full_pipeline_with_overrides(fake_retriever, config_path, capsys):
    out = run_tool(
        capsys,
        ["retrieve", "שאלה", "--config", config_path, "--gate-threshold", "0.7", "--top-n", "1"],
    )
    assert [r["chunk"]["chunk_id"] for r in out["results"]] == [APT1]  # 0.6 gated out
    assert out["results"][0]["rerank_score"] == 0.8
    assert set(out["stats"]) == {"dense", "sparse", "fused", "gated"}


def test_rerank_command_from_chunk_id_file(fake_retriever, config_path, capsys, tmp_path):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps([APT1, APT2, "apartment/files/gone.pdf#p9#c0"]))
    argv = ["rerank", "שאלה", "--config", config_path, "--candidates", str(candidates), "--gate-threshold", "0"]
    assert query_cli.main(argv) == query_cli.EXIT_OK
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert [r["chunk"]["chunk_id"] for r in out["results"]] == [APT1, APT2]
    assert "gone.pdf" in captured.err  # skew warning for the missing id


def test_rerank_command_accepts_fuse_envelope(fake_retriever, config_path, capsys, tmp_path):
    envelope = {
        "results": [
            {"chunk_id": APT1, "rrf_score": 0.03},
            make_chunk(APT2).model_dump(mode="json") | {},
        ]
    }
    # Full RetrievedChunk-style entry alongside a bare {chunk_id, score} one.
    envelope["results"][1] = {"chunk": make_chunk(APT2).model_dump(mode="json")}
    candidates = tmp_path / "fused.json"
    candidates.write_text(json.dumps(envelope, ensure_ascii=False))
    out = run_tool(
        capsys,
        ["rerank", "שאלה", "--config", config_path, "--candidates", str(candidates)],
    )
    assert sorted(r["chunk"]["chunk_id"] for r in out["results"]) == [APT1, APT2]


def test_legacy_invocation_still_dispatches_to_answer_flow(config_path, capsys):
    # No tool keyword -> classic parser; missing question -> config error exit
    # (before any config/index loading).
    assert query_cli.main(["--config", config_path]) == query_cli.EXIT_CONFIG_ERROR
    assert "Provide a question" in capsys.readouterr().err


def test_batch_records_include_agent_diagnostics(monkeypatch, config_path, tmp_path, capsys):
    from rag.types import Answer

    class FakeEngine:
        def _answer(self, question, category=None):
            answer = Answer(
                text="תשובה",
                citations=[],
                category="apartment",
                confidence=0.8,
                latency_ms=1.0,
                trace=[{"step": "classify", "ms": 5}],
            )
            return answer, {"prompt": 3, "completion": 2}

        def close(self):
            pass

    monkeypatch.setattr(query_cli, "load_config", lambda path: {"path": path})
    monkeypatch.setattr(query_cli, "AgentEngine", lambda config: FakeEngine())
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([{"id": "q1", "question": "ש"}], ensure_ascii=False))
    out = tmp_path / "a.jsonl"
    argv = ["--config", config_path, "--engine", "agent", "--questions", str(questions), "--out", str(out)]
    assert query_cli.main(argv) == query_cli.EXIT_OK
    record = json.loads(out.read_text().strip())
    assert record["category"] == "apartment"
    assert record["confidence"] == 0.8
    assert record["trace"] == [{"step": "classify", "ms": 5}]


def test_engine_flag_selects_agent_engine(monkeypatch, config_path, capsys):
    from rag.types import Answer

    constructed = []

    class FakeAgentEngine:
        def __init__(self, config):
            constructed.append(config)

        def answer(self, question, category=None):
            return Answer(text=f"agent:{question}", citations=[])

        def close(self):
            pass

    monkeypatch.setattr(query_cli, "load_config", lambda path: {"path": path})
    monkeypatch.setattr(query_cli, "AgentEngine", FakeAgentEngine)
    assert query_cli.main(["--config", config_path, "--engine", "agent", "שאלה"]) == query_cli.EXIT_OK
    assert constructed  # AgentEngine (not QueryEngine) was built
    assert "agent:שאלה" in capsys.readouterr().out
