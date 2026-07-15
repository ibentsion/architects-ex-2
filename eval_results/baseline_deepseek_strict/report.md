# Evaluation Report — baseline_deepseek_strict

- **Date:** 2026-07-15
- **Answers file:** `baseline_answers.jsonl`
- **Questions file:** `reference_questions.json` (48 questions evaluated)
- **Judge(s):** deepseek-ai/DeepSeek-V4-Pro (prompt variant: `strict`, temperature 0)
- **Estimated judge cost:** ~$0.07

## Executive summary

Mean correctness **4.88/10**, completeness **6.79/10**, conversational quality **8.12/10**. Hallucination rate **0.40**, refusal rate **0.02**, citation recall **0.00** (full-credit rate 0). Latency p50 5169 ms / p95 15495 ms.

Verdicts: correct = 13, partially correct = 16, incorrect = 18, refusal = 1.

## Overall metrics

| Slice | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| all | 48 | 4.88 | 6.79 | 8.12 | 0.40 | 0.02 | 0.00 | 5169 |

## By difficulty

| Difficulty | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| easy | 16 | 6.50 | 7.81 | 8.56 | 0.25 | 0 | 0.00 | 4395 |
| medium | 16 | 3.56 | 6.12 | 7.94 | 0.56 | 0 | 0.00 | 6333 |
| hard | 16 | 4.56 | 6.44 | 7.88 | 0.38 | 0.06 | 0.00 | 7363 |

## By domain

| Domain | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| apartment | 6 | 4.67 | 7.67 | 8.50 | 0.50 | 0 | 0.00 | 6467 |
| business | 6 | 5 | 7 | 8.17 | 0.50 | 0 | 0.00 | 3891 |
| car | 6 | 2.33 | 6.17 | 7.67 | 0.67 | 0 | 0.00 | 4024 |
| dental | 6 | 5.33 | 6.33 | 8 | 0.17 | 0 | 0.00 | 3072 |
| health | 6 | 7.17 | 7.50 | 8.67 | 0 | 0.17 | 0.00 | 2917 |
| life | 6 | 7.17 | 7 | 8.17 | 0.33 | 0 | 0.00 | 2865 |
| mortgage | 6 | 4.83 | 6.83 | 8.17 | 0.33 | 0 | 0.00 | 6760 |
| travel | 6 | 2.50 | 5.83 | 7.67 | 0.67 | 0 | 0.00 | 6503 |

## By number of required source groups

| Sources | n | Correctness | Completeness | Conv. quality | Halluc. rate | Refusal rate | Cite recall | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 group(s) | 38 | 4.76 | 6.68 | 8.11 | 0.39 | 0.03 | 0.00 | 4395 |
| 2 group(s) | 10 | 5.30 | 7.20 | 8.20 | 0.40 | 0 | 0.00 | 10907 |

## Lowest-scoring questions

- **`dev-30-health-hard`** (health, hard) — correctness 0, verdict refusal, hallucination false. Judge: The system answer declines to provide the requested price and instead directs the user to contact customer service, which constitutes a refusal. It does not contradict the ground truth, so there is no hallucination. Correctness and completeness are 0 because the factual price was not given, while conversational quality is decent as it offers helpful next steps.
- **`dev-33-life-medium`** (life, medium) — correctness 0, verdict incorrect, hallucination true. Judge: The ground truth explicitly states the policy has no surrender value (ערכי פדיון). The system answer contradicts this by claiming it does accumulate surrender value after two years, which is a direct hallucination. The answer is completely wrong on the asked fact.
- **`dev-24-dental-hard`** (dental, hard) — correctness 0, verdict incorrect, hallucination false. Judge: The ground truth states that 3 bite-wing X-rays yield 6 total images (2 per X-ray). The system answer does not answer the asked fact at all; instead it discusses insurance coverage limits. It neither confirms nor contradicts the ground truth, so it is incorrect relative to the question. No hallucination because it does not assert a contradictory fact.
- **`dev-01-apartment-easy`** (apartment, easy) — correctness 0, verdict incorrect, hallucination true. Judge: The ground truth states that filing a claim with an institutional body does NOT stop the statute of limitations; only filing a lawsuit in court does. The system answer confidently asserts the opposite—that filing with the insurance company does stop the clock—citing specific legal sections. This directly contradicts the authoritative ground truth, making it a clear hallucination and incorrect.
- **`dev-04-apartment-medium`** (apartment, medium) — correctness 0, verdict incorrect, hallucination true. Judge: The ground truth states the indemnity limit is 6,000 ILS at the date of the insured event. The system answer asserts a limit of 70,000 ILS, which directly contradicts the authoritative figure. This is a clear hallucination.

## Improvement suggestions (prioritized)

1. **Grounding is absent or broken** (citation recall 0.00). Highest-leverage fix: build retrieval that returns file + page metadata and require the generator to cite every factual claim. Citation accuracy is a graded criterion and currently earns ~nothing.
2. **High hallucination rate** (0.40 of answers confidently contradict the ground truth). Add an evidence-or-abstain policy: answer only from retrieved context and fall back to "I don't have enough information" when evidence is missing. Worst domains: apartment, business, car, travel.
3. **Weak domains** (mean correctness 4.9 overall): car (2.3), travel (2.5). Prioritize corpus parsing/coverage checks for these domains — inspect whether their documents parse cleanly (tables, scanned PDFs) before tuning prompts.
4. **Latency tail is heavy** (p95 15.5s). Efficiency is graded — cap generation length, and check whether slow answers correlate with long ramble rather than genuine retrieval work.

---
*Generated by `evalharness` — numeric grades per category are in `judgments.jsonl` (per question) and `metrics.json` (aggregates) for automated comparison across runs.*
