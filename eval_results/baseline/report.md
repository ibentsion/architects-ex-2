# Evaluation Report — baseline

- **Date:** 2026-07-15
- **Answers file:** `baseline_answers.jsonl`
- **Questions file:** `reference_questions.json` (48 questions evaluated)
- **Judge(s):** deepseek-ai/DeepSeek-V4-Pro (prompt variant: `rubric`, temperature 0)
- **Estimated judge cost:** ~$0.07

## Executive summary

Mean correctness **4.29/10**, completeness **3.71/10**, conversational quality **8.58/10**. Hallucination rate **0.40**, refusal rate **0.02**, citation recall **0.00** (full-credit rate 0). Latency p50 5169 ms / p95 15495 ms.

Verdicts: correct = 8, partially correct = 19, incorrect = 20, refusal = 1.

## Overall metrics

| Slice | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| all | 48 | 4.29 | 3.71 | 8.58 | 0.40 | 0.02 | 0.00 | 5169 |

## By difficulty

| Difficulty | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| easy | 16 | 5.75 | 5.12 | 8.88 | 0.25 | 0 | 0.00 | 4395 |
| medium | 16 | 3.31 | 2.56 | 8.38 | 0.56 | 0 | 0.00 | 6333 |
| hard | 16 | 3.81 | 3.44 | 8.50 | 0.38 | 0.06 | 0.00 | 7363 |

## By domain

| Domain | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| apartment | 6 | 4.17 | 3.83 | 8.83 | 0.50 | 0 | 0.00 | 6467 |
| business | 6 | 5 | 4.33 | 8.67 | 0.50 | 0 | 0.00 | 3891 |
| car | 6 | 2.50 | 2.33 | 8.33 | 0.67 | 0 | 0.00 | 4024 |
| dental | 6 | 5.33 | 4.33 | 8.67 | 0.17 | 0 | 0.00 | 3072 |
| health | 6 | 6.33 | 6 | 9 | 0 | 0.17 | 0.00 | 2917 |
| life | 6 | 5.33 | 4.17 | 8.67 | 0.33 | 0 | 0.00 | 2865 |
| mortgage | 6 | 4.83 | 4 | 8.50 | 0.33 | 0 | 0.00 | 6760 |
| travel | 6 | 0.83 | 0.67 | 8 | 0.67 | 0 | 0.00 | 6503 |

## By number of required source groups

| Sources | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 group(s) | 38 | 4.29 | 3.68 | 8.53 | 0.39 | 0.03 | 0.00 | 4395 |
| 2 group(s) | 10 | 4.30 | 3.80 | 8.80 | 0.40 | 0 | 0.00 | 10907 |

## Lowest-scoring questions

- **`dev-01-apartment-easy`** (apartment, easy) — correctness 0, verdict incorrect, hallucination true. Judge: The ground truth states that filing a claim with an institutional body does NOT stop the statute of limitations; only filing in court does. The system answer confidently asserts the opposite—that filing with the insurance company does stop the clock—and cites specific legal sections. This directly contradicts the ground truth on the main fact, making it incorrect and a hallucination. Completeness is 0 because it covers none of the ground-truth components. Conversational quality is good: clear, structured, polite Hebrew.
- **`dev-03-apartment-medium`** (apartment, medium) — correctness 0, verdict incorrect, hallucination true. Judge: The ground truth states that the state law covers only agricultural infrastructure and does not compensate residential buildings at all. The system answer contradicts this by claiming the state provides partial compensation for the residential structure itself, which is a direct hallucination. The answer is well-structured and polite, hence high conversational quality, but factually incorrect and incomplete relative to the authoritative ground truth.
- **`dev-04-apartment-medium`** (apartment, medium) — correctness 0, verdict incorrect, hallucination true. Judge: The ground truth states the reimbursement limit is 6,000 ILS. The system answer confidently asserts a limit of 70,000 ILS, which directly contradicts the ground truth on the main asked fact. This is a clear hallucination. The answer is well-structured and polite, hence the high conversational quality score.
- **`dev-07-business-easy`** (business, easy) — correctness 0, verdict incorrect, hallucination true. Judge: The ground truth states unequivocally that the policy excludes liability toward subcontractor employees. The system answer contradicts this by claiming coverage exists under an extension, which is a direct hallucination. The answer is therefore incorrect and receives 0 for correctness and completeness, though it is well-structured and polite.
- **`dev-09-business-medium`** (business, medium) — correctness 0, verdict incorrect, hallucination true. Judge: The ground truth states the maximum indemnity period is 100 days (unless otherwise specified), while the system answer confidently asserts it is 60 days. This directly contradicts the ground truth on the main asked fact, making it incorrect and a hallucination. The answer covers none of the ground-truth components (100 days, end of shutdown period, exception clause), so completeness is 0. The reply is clear and polite, earning a decent conversational quality score.

## Improvement suggestions (prioritized)

1. **Grounding is absent or broken** (citation recall 0.00). Highest-leverage fix: build retrieval that returns file + page metadata and require the generator to cite every factual claim. Citation accuracy is a graded criterion and currently earns ~nothing.
2. **High hallucination rate** (0.40 of answers confidently contradict the ground truth). Add an evidence-or-abstain policy: answer only from retrieved context and fall back to "I don't have enough information" when evidence is missing. Worst domains: apartment, business, car, travel.
3. **Weak domains** (mean correctness 4.3 overall): car (2.5), travel (0.8). Prioritize corpus parsing/coverage checks for these domains — inspect whether their documents parse cleanly (tables, scanned PDFs) before tuning prompts.
4. **Latency tail is heavy** (p95 15.5s). Efficiency is graded — cap generation length, and check whether slow answers correlate with long ramble rather than genuine retrieval work.

---
*Generated by `evalharness` — numeric grades per category are in `judgments.jsonl` (per question) and `metrics.json` (aggregates) for automated comparison across runs.*
