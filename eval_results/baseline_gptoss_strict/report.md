# Evaluation Report — baseline_gptoss_strict

- **Date:** 2026-07-15
- **Answers file:** `baseline_answers.jsonl`
- **Questions file:** `reference_questions.json` (48 questions evaluated)
- **Judge(s):** openai/gpt-oss-120b (prompt variant: `strict`, temperature 0)
- **Estimated judge cost:** ~$0.07

## Executive summary

Mean correctness **4.21/10**, completeness **4.25/10**, conversational quality **5.98/10**. Hallucination rate **0.54**, refusal rate **0.02**, citation recall **0.00** (full-credit rate 0). Latency p50 5169 ms / p95 15495 ms.

Verdicts: correct = 16, partially correct = 5, incorrect = 26, refusal = 1.

## Overall metrics

| Slice | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| all | 48 | 4.21 | 4.25 | 5.98 | 0.54 | 0.02 | 0.00 | 5169 |

## By difficulty

| Difficulty | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| easy | 16 | 5.25 | 5.12 | 6.69 | 0.50 | 0 | 0.00 | 4395 |
| medium | 16 | 3.38 | 3.50 | 5.44 | 0.56 | 0 | 0.00 | 6333 |
| hard | 16 | 4 | 4.12 | 5.81 | 0.56 | 0.06 | 0.00 | 7363 |

## By domain

| Domain | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| apartment | 6 | 5 | 5.17 | 6.67 | 0.50 | 0 | 0.00 | 6467 |
| business | 6 | 4.83 | 5 | 6 | 0.50 | 0 | 0.00 | 3891 |
| car | 6 | 2.17 | 2.17 | 5 | 0.83 | 0 | 0.00 | 4024 |
| dental | 6 | 3.33 | 3.33 | 5.50 | 0.50 | 0 | 0.00 | 3072 |
| health | 6 | 6.50 | 6.33 | 7.17 | 0 | 0.17 | 0.00 | 2917 |
| life | 6 | 5.33 | 4.83 | 6.33 | 0.50 | 0 | 0.00 | 2865 |
| mortgage | 6 | 4.50 | 4.17 | 5.83 | 0.67 | 0 | 0.00 | 6760 |
| travel | 6 | 2 | 3 | 5.33 | 0.83 | 0 | 0.00 | 6503 |

## By number of required source groups

| Sources | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 group(s) | 38 | 4.16 | 4.13 | 5.89 | 0.53 | 0.03 | 0.00 | 4395 |
| 2 group(s) | 10 | 4.40 | 4.70 | 6.30 | 0.60 | 0 | 0.00 | 10907 |

## Lowest-scoring questions

- **`dev-01-apartment-easy`** (apartment, easy) — correctness 0, verdict incorrect, hallucination true. Judge: The system answer claims that filing a claim with the insurance company stops the prescription period, directly contradicting the ground‑truth answer which states it does not. This is a clear hallucination and provides no correct information.
- **`dev-04-apartment-medium`** (apartment, medium) — correctness 0, verdict incorrect, hallucination true. Judge: The ground-truth answer limits reimbursement to 6,000 ₪, while the system answer claims a limit of 70,000 ₪, directly contradicting the authoritative fact. Therefore the answer is incorrect and constitutes a hallucination.
- **`dev-09-business-medium`** (business, medium) — correctness 0, verdict incorrect, hallucination true. Judge: The system answer states a maximum compensation period of 60 days, which directly contradicts the ground‑truth answer that the period ends no later than 100 days (unless otherwise specified). This is a factual error, not a refusal.
- **`dev-11-business-hard`** (business, hard) — correctness 0, verdict incorrect, hallucination true. Judge: The system answer states a specific 25%/30,000 NIS ceiling for Shir and claims Harel has no separate ceiling, which directly contradicts the ground‑truth that both policies have ceilings defined by loss of income and by gross profit rate times lost turnover. Therefore it is a hallucination and provides no correct information.
- **`dev-14-car-easy`** (car, easy) — correctness 0, verdict incorrect, hallucination true. Judge: The system answer states the service is free except for the spare wheel cost, contradicting the ground truth which says a 59 ₪ co‑payment is required. This is a direct conflict, so it is incorrect and a hallucination.

## Improvement suggestions (prioritized)

1. **Grounding is absent or broken** (citation recall 0.00). Highest-leverage fix: build retrieval that returns file + page metadata and require the generator to cite every factual claim. Citation accuracy is a graded criterion and currently earns ~nothing.
2. **High hallucination rate** (0.54 of answers confidently contradict the ground truth). Add an evidence-or-abstain policy: answer only from retrieved context and fall back to "I don't have enough information" when evidence is missing. Worst domains: car, mortgage, travel.
3. **Weak domains** (mean correctness 4.2 overall): car (2.2), travel (2.0). Prioritize corpus parsing/coverage checks for these domains — inspect whether their documents parse cleanly (tables, scanned PDFs) before tuning prompts.
4. **Latency tail is heavy** (p95 15.5s). Efficiency is graded — cap generation length, and check whether slow answers correlate with long ramble rather than genuine retrieval work.

---
*Generated by `evalharness` — numeric grades per category are in `judgments.jsonl` (per question) and `metrics.json` (aggregates) for automated comparison across runs.*
