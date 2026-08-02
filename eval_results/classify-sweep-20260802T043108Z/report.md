# Classifier arm sweep — classify-sweep-20260802T043108Z

- **Date:** 2026-08-02
- **Question sets:** reference_questions.json, reference_questions_v2.json (169 questions)
- **Arms:** `baseline`, `abstain`, `examples`, `rich-desc`, `decision-rules`, `hint-sparse`, `hint-vote`, `model-qwen`, `model-deepseek`, `effort-medium`, `selfcons-3`, `verify-2stage`
- **Total cost:** ~$1.69

The agent filters retrieval for a sub-question only when it carries exactly one category, so `harmful` = filtered to the wrong domain, `correct` = filtered to the gold domain, `no filter` = zero or 2+ tags. **Harmful is the primary metric** (lower is better); a wrong filter costs a retrieval round, a missing one only costs precision.

## Pooled (v1 + v2), ranked by harmful_rate

| Arm | n | Harmful | Correct | No filter | Recall any | Tags/sub | Subs/q | Parse fail | Trunc. | p50 ms | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `hint-vote` | 169 | 0.077 | 0.497 | 0.426 | 0.497 | 0.574 | 1 | 0 | 0 | 0 | 0 |
| `model-deepseek` | 169 | 0.124 | 0.657 | 0.219 | 0.805 | 1.106 | 1.503 | 0 | 0 | 2571 | 0.0006 |
| `abstain` | 169 | 0.142 | 0.651 | 0.207 | 0.675 | 0.843 | 1.284 | 0.095 | 0 | 643 | 0.0006 |
| `verify-2stage` | 169 | 0.142 | 0.675 | 0.183 | 0.698 | 0.847 | 1.272 | 0.035 | 0 | 1028 | 0.001 |
| `selfcons-3` | 169 | 0.148 | 0.686 | 0.166 | 0.704 | 0.873 | 1.260 | 0.118 | 0 | 2119 | 0.0018 |
| `decision-rules` | 169 | 0.154 | 0.734 | 0.112 | 0.751 | 0.917 | 1.207 | 0.018 | 0 | 727 | 0.0008 |
| `effort-medium` | 169 | 0.160 | 0.675 | 0.166 | 0.704 | 0.887 | 1.308 | 0.059 | 0.024 | 1320 | 0.0011 |
| `hint-sparse` | 169 | 0.166 | 0.769 | 0.065 | 0.793 | 0.972 | 1.248 | 0.018 | 0 | 699 | 0.0008 |
| `baseline` | 169 | 0.172 | 0.680 | 0.148 | 0.704 | 0.888 | 1.272 | 0.041 | 0 | 720 | 0.0006 |
| `rich-desc` | 169 | 0.177 | 0.698 | 0.124 | 0.710 | 0.894 | 1.231 | 0.047 | 0 | 616 | 0.0013 |
| `model-qwen` | 169 | 0.189 | 0.722 | 0.089 | 0.757 | 0.991 | 1.331 | 0 | 0 | 1699 | 0.0005 |
| `examples` | 169 | 0.201 | 0.698 | 0.101 | 0.722 | 0.940 | 1.278 | 0.018 | 0 | 654 | 0.0009 |

## v1

| Arm | n | Harmful | Correct | No filter | Recall any | Tags/sub | Subs/q | Parse fail | Trunc. | p50 ms | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `decision-rules` | 48 | 0.083 | 0.750 | 0.167 | 0.771 | 0.883 | 1.250 | 0.062 | 0 | 729 | 0.0008 |
| `baseline` | 48 | 0.104 | 0.729 | 0.167 | 0.750 | 0.869 | 1.271 | 0.062 | 0 | 694 | 0.0006 |
| `abstain` | 48 | 0.104 | 0.750 | 0.146 | 0.750 | 0.871 | 1.292 | 0.062 | 0 | 618 | 0.0006 |
| `model-deepseek` | 48 | 0.104 | 0.708 | 0.188 | 0.812 | 1.076 | 1.375 | 0 | 0 | 2332 | 0.0005 |
| `rich-desc` | 48 | 0.125 | 0.708 | 0.167 | 0.708 | 0.850 | 1.250 | 0.083 | 0 | 647 | 0.0013 |
| `hint-sparse` | 48 | 0.125 | 0.792 | 0.083 | 0.812 | 0.950 | 1.250 | 0.021 | 0 | 618 | 0.0008 |
| `hint-vote` | 48 | 0.125 | 0.562 | 0.312 | 0.562 | 0.688 | 1 | 0 | 0 | 0 | 0 |
| `effort-medium` | 48 | 0.125 | 0.729 | 0.146 | 0.750 | 0.887 | 1.292 | 0.042 | 0 | 1258 | 0.001 |
| `selfcons-3` | 48 | 0.125 | 0.708 | 0.167 | 0.729 | 0.869 | 1.271 | 0.188 | 0 | 1862 | 0.0018 |
| `verify-2stage` | 48 | 0.125 | 0.708 | 0.167 | 0.729 | 0.869 | 1.271 | 0.062 | 0 | 1076 | 0.001 |
| `examples` | 48 | 0.167 | 0.729 | 0.104 | 0.750 | 0.934 | 1.271 | 0.021 | 0 | 682 | 0.0009 |
| `model-qwen` | 48 | 0.188 | 0.729 | 0.083 | 0.771 | 1 | 1.333 | 0 | 0 | 1821 | 0.0005 |

## v2

| Arm | n | Harmful | Correct | No filter | Recall any | Tags/sub | Subs/q | Parse fail | Trunc. | p50 ms | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `hint-vote` | 121 | 0.058 | 0.471 | 0.471 | 0.471 | 0.529 | 1 | 0 | 0 | 0 | 0 |
| `model-deepseek` | 121 | 0.132 | 0.636 | 0.231 | 0.802 | 1.117 | 1.554 | 0 | 0 | 2708 | 0.0006 |
| `verify-2stage` | 121 | 0.149 | 0.661 | 0.190 | 0.686 | 0.838 | 1.273 | 0.025 | 0 | 1000 | 0.001 |
| `abstain` | 121 | 0.157 | 0.612 | 0.231 | 0.645 | 0.832 | 1.281 | 0.107 | 0 | 659 | 0.0006 |
| `selfcons-3` | 121 | 0.157 | 0.678 | 0.165 | 0.694 | 0.875 | 1.256 | 0.091 | 0 | 2232 | 0.0018 |
| `effort-medium` | 121 | 0.174 | 0.653 | 0.174 | 0.686 | 0.887 | 1.314 | 0.066 | 0.033 | 1340 | 0.0011 |
| `decision-rules` | 121 | 0.182 | 0.727 | 0.091 | 0.744 | 0.931 | 1.190 | 0 | 0 | 727 | 0.0008 |
| `hint-sparse` | 121 | 0.182 | 0.760 | 0.058 | 0.785 | 0.980 | 1.248 | 0.017 | 0 | 725 | 0.0008 |
| `model-qwen` | 121 | 0.190 | 0.719 | 0.091 | 0.752 | 0.988 | 1.331 | 0 | 0 | 1676 | 0.0005 |
| `baseline` | 121 | 0.198 | 0.661 | 0.141 | 0.686 | 0.896 | 1.273 | 0.033 | 0 | 732 | 0.0006 |
| `rich-desc` | 121 | 0.198 | 0.694 | 0.107 | 0.711 | 0.912 | 1.223 | 0.033 | 0 | 596 | 0.0013 |
| `examples` | 121 | 0.215 | 0.686 | 0.099 | 0.711 | 0.942 | 1.281 | 0.017 | 0 | 651 | 0.0009 |

## Paired against baseline (same question ids)

`fixed` = baseline was harmful and this arm is not; `broken` = the reverse. p is an exact two-sided McNemar over those two counts — at n=169 an arm that is not significant here has not been shown to differ from baseline at all.

| Arm | pooled harmful | fixed | broken | McNemar p | verdict |
|---|---|---|---|---|---|
| `hint-vote` | 0.077 | 23 | 7 | 0.0052 | **better** |
| `model-deepseek` | 0.124 | 11 | 3 | 0.0574 | tied with baseline |
| `abstain` | 0.142 | 10 | 5 | 0.3018 | tied with baseline |
| `verify-2stage` | 0.142 | 9 | 4 | 0.2668 | tied with baseline |
| `selfcons-3` | 0.148 | 8 | 4 | 0.3877 | tied with baseline |
| `decision-rules` | 0.154 | 5 | 2 | 0.4531 | tied with baseline |
| `effort-medium` | 0.160 | 6 | 4 | 0.7539 | tied with baseline |
| `hint-sparse` | 0.166 | 11 | 10 | 1.0000 | tied with baseline |
| `rich-desc` | 0.177 | 7 | 8 | 1.0000 | tied with baseline |
| `model-qwen` | 0.189 | 5 | 8 | 0.5811 | tied with baseline |
| `examples` | 0.201 | 3 | 8 | 0.2266 | tied with baseline |

## Confusion — baseline

| gold \ predicted | apartment | business | car | dental | diseases-disabilities | health | life | long-term-care | loss-of-working-ability | mortgage | personal-accident | travel | (none) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apartment | **12** |  |  |  |  |  |  |  |  |  | 2 |  | 2 |
| business | 1 | **12** |  |  |  | 1 |  |  |  |  |  |  | 3 |
| car |  |  | **15** |  |  |  |  |  |  |  | 1 |  | 1 |
| dental |  |  |  | **10** |  | 1 |  |  |  |  |  |  |  |
| diseases-disabilities |  |  |  |  | **7** | 1 | 1 |  |  |  |  |  | 2 |
| health |  |  |  |  |  | **14** |  |  |  |  |  |  | 3 |
| life |  |  |  |  |  |  | **10** |  |  |  | 3 |  | 2 |
| long-term-care |  |  |  |  |  | 1 |  | **6** | 1 |  |  |  | 1 |
| loss-of-working-ability |  |  |  |  | 1 | 1 | 1 |  | **6** |  | 1 |  | 1 |
| mortgage | 7 |  |  |  |  |  | 1 |  |  | **2** |  |  | 7 |
| personal-accident |  |  |  |  |  | 1 |  |  |  |  | **8** | 2 |  |
| travel |  |  |  |  |  |  |  |  |  |  | 1 | **13** | 3 |

Most frequent wrong filters: `mortgage` tagged `apartment` — 7x; `life` tagged `personal-accident` — 3x; `personal-accident` tagged `travel` — 2x; `apartment` tagged `personal-accident` — 2x; `travel` tagged `personal-accident` — 1x; `personal-accident` tagged `health` — 1x; `mortgage` tagged `life` — 1x; `loss-of-working-ability` tagged `personal-accident` — 1x.

## Confusion — best arm (`hint-vote`)

| gold \ predicted | apartment | business | car | dental | diseases-disabilities | health | life | long-term-care | loss-of-working-ability | mortgage | personal-accident | travel | (none) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apartment | **5** | 2 | 1 |  |  |  |  |  |  |  |  |  | 8 |
| business |  | **10** |  |  |  | 1 |  |  |  |  |  |  | 6 |
| car |  |  | **12** |  |  |  |  |  |  |  |  | 1 | 4 |
| dental |  |  |  | **5** |  | 1 |  |  |  |  |  |  | 5 |
| diseases-disabilities |  |  |  |  | **5** |  |  |  |  |  |  |  | 6 |
| health |  |  |  |  |  | **12** |  |  |  |  |  |  | 5 |
| life |  |  |  |  |  |  | **3** |  |  |  |  |  | 12 |
| long-term-care |  |  |  |  |  | 1 |  | **6** |  |  |  |  | 2 |
| loss-of-working-ability |  |  |  |  |  |  |  |  | **6** |  |  |  | 5 |
| mortgage | 3 | 1 |  |  |  |  |  |  |  | **2** |  |  | 11 |
| personal-accident |  | 1 |  |  |  |  |  |  |  |  | **4** | 1 | 5 |
| travel |  |  |  |  |  |  |  |  |  |  |  | **14** | 3 |

Most frequent wrong filters: `mortgage` tagged `apartment` — 3x; `apartment` tagged `business` — 2x; `personal-accident` tagged `travel` — 1x; `personal-accident` tagged `business` — 1x; `mortgage` tagged `business` — 1x; `long-term-care` tagged `health` — 1x; `dental` tagged `health` — 1x; `car` tagged `travel` — 1x.

## Arms

- `baseline` — today's production prompt, gpt-oss-120b, reasoning low, temp 0
- `abstain` — + a wrong tag is worse than none — abstain when undetermined
- `examples` — + 2 invented example questions per category
- `rich-desc` — category descriptions with belongs / does-NOT-belong + overlap disambiguation
- `decision-rules` — + ordered decision rules with a stated precedence for overlaps
- `hint-sparse` — baseline prompt + retrieval evidence block
- `hint-vote` — no LLM — filter = top hint category when its share is high enough
- `model-qwen` — baseline prompt on Qwen3-235B-A22B-Instruct
- `model-deepseek` — baseline prompt on DeepSeek-V4-Pro
- `effort-medium` — gpt-oss-120b at reasoning_effort medium
- `selfcons-3` — baseline prompt, temp 0.7, 3 samples, keep categories with >=2 votes
- `verify-2stage` — baseline tags, then a cheap call that sees the evidence and may retract a tag

---
*Generated by `evalharness.classify_eval`; per-question records are in `<arm>/predictions.jsonl` and every metric in `summary.json`.*
