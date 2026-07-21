# Evaluation Report — rag-default

- **Date:** 2026-07-21
- **Answers file:** `rag_answers.jsonl`
- **Questions file:** `reference_questions.json` (48 questions evaluated)
- **Judge(s):** deepseek-ai/DeepSeek-V4-Pro (prompt variant: `rubric`, temperature 0)
- **Estimated judge cost:** ~$0.07

## Executive summary

Mean correctness **2.17/10**, completeness **2.10/10**, conversational quality **6.21/10**. Hallucination rate **0**, refusal rate **0.75**, citation recall **0.10** (full-credit rate 0.10). Latency p50 42901 ms / p95 83519 ms.

Verdicts: correct = 9, partially correct = 2, incorrect = 1, refusal = 36.

## Overall metrics

| Slice | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| all | 48 | 2.17 | 2.10 | 6.21 | 0 | 0.75 | 0.10 | 42901 |

## By difficulty

| Difficulty | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| easy | 16 | 3.38 | 3.19 | 6.75 | 0 | 0.62 | 0.12 | 44292 |
| medium | 16 | 1.88 | 1.88 | 6.06 | 0 | 0.75 | 0.19 | 41271 |
| hard | 16 | 1.25 | 1.25 | 5.81 | 0 | 0.88 | 0.00 | 42901 |

## By domain

| Domain | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| apartment | 6 | 8.33 | 8.33 | 9.67 | 0 | 0.17 | 0.67 | 52119 |
| business | 6 | 0 | 0 | 5 | 0 | 1 | 0.00 | 37574 |
| car | 6 | 0 | 0 | 5 | 0 | 1 | 0.00 | 40694 |
| dental | 6 | 0 | 0 | 5 | 0 | 1 | 0.00 | 39677 |
| health | 6 | 0 | 0 | 5 | 0 | 1 | 0.00 | 39926 |
| life | 6 | 0 | 0 | 5 | 0 | 1 | 0.00 | 40303 |
| mortgage | 6 | 4.50 | 4.17 | 7.33 | 0 | 0.50 | 0.00 | 47759 |
| travel | 6 | 4.50 | 4.33 | 7.67 | 0 | 0.33 | 0.17 | 49312 |

## By number of required source groups

| Sources | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 group(s) | 38 | 2.74 | 2.66 | 6.45 | 0 | 0.68 | 0.13 | 41271 |
| 2 group(s) | 10 | 0 | 0 | 5.30 | 0 | 1 | 0.00 | 38891 |

## Lowest-scoring questions

- **`dev-05-apartment-hard`** (apartment, hard) — correctness 0, verdict refusal, hallucination false. Judge: The system explicitly states it lacks sufficient information to answer the question, which is a refusal. It does not contradict the ground truth, so no hallucination. The reply is polite, well-structured, and clear, earning a good conversational quality score despite the refusal.
- **`dev-07-business-easy`** (business, easy) — correctness 0, verdict refusal, hallucination false. Judge: The system explicitly states it lacks sufficient information to answer, which is a clear refusal. It does not contradict the ground truth, so no hallucination. Correctness and completeness are 0 per refusal rules. The reply is polite and understandable but very brief, earning a mid-range conversational quality score.
- **`dev-08-business-easy`** (business, easy) — correctness 0, verdict refusal, hallucination false. Judge: The system refused to answer, stating it lacks sufficient information. This is a clear refusal, so correctness and completeness are 0. The reply is polite and understandable but very brief, earning a mid-range conversational quality score.
- **`dev-09-business-medium`** (business, medium) — correctness 0, verdict refusal, hallucination false. Judge: The system explicitly states it lacks sufficient information to answer, which is a refusal. Therefore correctness and completeness are 0. The reply is polite and clear but very brief, so conversational quality is average.
- **`dev-10-business-medium`** (business, medium) — correctness 0, verdict refusal, hallucination false. Judge: The system explicitly states it lacks sufficient information to answer, which is a clear refusal. Therefore correctness and completeness are 0. The reply is polite and understandable but very brief, so conversational quality is average.

## Improvement suggestions (prioritized)

1. **Grounding is absent or broken** (citation recall 0.10). Highest-leverage fix: build retrieval that returns file + page metadata and require the generator to cite every factual claim. Citation accuracy is a graded criterion and currently earns ~nothing.
2. **High refusal rate** (0.75). The system abstains often — if retrieval exists, raise recall (more candidates, query rewriting); if this is the bare baseline, retrieval will convert refusals to grounded answers.
3. **Hard questions lag easy ones** (correctness 3.4 easy vs 1.2 hard). Hard questions combine several documents and/or a calculation — add multi-hop retrieval (retrieve per sub-question) and let the model do explicit arithmetic on retrieved numbers.
4. **Cross-document questions underperform** (correctness 2.7 with one required source vs 0.0 with two). Retrieval must surface passages from multiple documents per question — decompose the question or retrieve per detected sub-topic.
5. **Weak domains** (mean correctness 2.2 overall): business (0.0), car (0.0), dental (0.0), health (0.0), life (0.0). Prioritize corpus parsing/coverage checks for these domains — inspect whether their documents parse cleanly (tables, scanned PDFs) before tuning prompts.
6. **Latency tail is heavy** (p95 83.5s). Efficiency is graded — cap generation length, and check whether slow answers correlate with long ramble rather than genuine retrieval work.

---
*Generated by `evalharness` — numeric grades per category are in `judgments.jsonl` (per question) and `metrics.json` (aggregates) for automated comparison across runs.*
