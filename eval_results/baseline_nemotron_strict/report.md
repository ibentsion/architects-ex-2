# Evaluation Report — baseline_nemotron_strict

- **Date:** 2026-07-15
- **Answers file:** `baseline_answers.jsonl`
- **Questions file:** `reference_questions.json` (48 questions evaluated)
- **Judge(s):** nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (prompt variant: `strict`, temperature 0)
- **Estimated judge cost:** ~$0.07

## Executive summary

Mean correctness **4.81/10**, completeness **5.02/10**, conversational quality **6.27/10**. Hallucination rate **0.35**, refusal rate **0.02**, citation recall **0.00** (full-credit rate 0). Latency p50 5169 ms / p95 15495 ms.

Verdicts: correct = 21, partially correct = 5, incorrect = 21, refusal = 1.

## Overall metrics

| Slice | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| all | 48 | 4.81 | 5.02 | 6.27 | 0.35 | 0.02 | 0.00 | 5169 |

## By difficulty

| Difficulty | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| easy | 16 | 6.25 | 5.69 | 6.69 | 0.38 | 0 | 0.00 | 4395 |
| medium | 16 | 3.56 | 4 | 6.06 | 0.44 | 0 | 0.00 | 6333 |
| hard | 16 | 4.62 | 5.38 | 6.06 | 0.25 | 0.06 | 0.00 | 7363 |

## By domain

| Domain | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| apartment | 6 | 5 | 6.17 | 7.17 | 0.50 | 0 | 0.00 | 6467 |
| business | 6 | 6.67 | 6.33 | 7 | 0.33 | 0 | 0.00 | 3891 |
| car | 6 | 3.33 | 4.50 | 6 | 0.50 | 0 | 0.00 | 4024 |
| dental | 6 | 2.50 | 3.17 | 5.50 | 0.33 | 0 | 0.00 | 3072 |
| health | 6 | 6.67 | 6.33 | 6.83 | 0 | 0.17 | 0.00 | 2917 |
| life | 6 | 5.33 | 5.50 | 6.83 | 0.33 | 0 | 0.00 | 2865 |
| mortgage | 6 | 5.33 | 4.33 | 5.83 | 0.33 | 0 | 0.00 | 6760 |
| travel | 6 | 3.67 | 3.83 | 5 | 0.50 | 0 | 0.00 | 6503 |

## By number of required source groups

| Sources | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 group(s) | 38 | 4.66 | 4.76 | 6.08 | 0.37 | 0.03 | 0.00 | 4395 |
| 2 group(s) | 10 | 5.40 | 6 | 7 | 0.30 | 0 | 0.00 | 10907 |

## Lowest-scoring questions

- **`dev-09-business-medium`** (business, medium) — correctness 0, verdict incorrect, hallucination true. Judge: The system answer states a maximum payout period of 60 days, which directly contradicts the ground‑truth limit of 100 days. Because it asserts a fact that opposes the authoritative answer, it is judged incorrect and constitutes a hallucination. Scores for correctness and completeness are set to 0, while conversational quality receives a modest rating.
- **`dev-15-car-medium`** (car, medium) — correctness 0, verdict incorrect, hallucination true. Judge: The system claims third‑party insurance does not cover lost keys, directly contradicting the ground‑truth which states it does cover such loss without extra premium.
- **`dev-16-car-medium`** (car, medium) — correctness 0, verdict incorrect, hallucination true. Judge: The system claims the insurance can be bought online without visiting a branch, which directly contradicts the ground‑truth statement that it must be purchased at the Milamag sales point at the border crossing. Therefore the answer is factually wrong and constitutes a hallucination.
- **`dev-22-dental-medium`** (dental, medium) — correctness 0, verdict incorrect, hallucination false. Judge: The system avoids directly answering whether a minor's Bיט can receive a refund, instead asking for more details and giving a general rule without confirming or denying applicability to minors. It does not contradict the ground truth but fails to answer the specific question, making it incomplete and effectively incorrect for the asked query.
- **`dev-23-dental-hard`** (dental, hard) — correctness 0, verdict incorrect, hallucination true. Judge: The system supplies an email address and phone number that differ from the ground‑truth contact details, directly contradicting the authoritative answer, which constitutes a hallucination and results in an incorrect verdict. Correctness is 0 because the facts contradict the ground truth; completeness is 0 as no correct information is provided; conversational quality is moderate, rated 5.

## Improvement suggestions (prioritized)

1. **Grounding is absent or broken** (citation recall 0.00). Highest-leverage fix: build retrieval that returns file + page metadata and require the generator to cite every factual claim. Citation accuracy is a graded criterion and currently earns ~nothing.
2. **High hallucination rate** (0.35 of answers confidently contradict the ground truth). Add an evidence-or-abstain policy: answer only from retrieved context and fall back to "I don't have enough information" when evidence is missing. Worst domains: apartment, car, travel.
3. **Weak domains** (mean correctness 4.8 overall): dental (2.5). Prioritize corpus parsing/coverage checks for these domains — inspect whether their documents parse cleanly (tables, scanned PDFs) before tuning prompts.
4. **Latency tail is heavy** (p95 15.5s). Efficiency is graded — cap generation length, and check whether slow answers correlate with long ramble rather than genuine retrieval work.

---
*Generated by `evalharness` — numeric grades per category are in `judgments.jsonl` (per question) and `metrics.json` (aggregates) for automated comparison across runs.*
