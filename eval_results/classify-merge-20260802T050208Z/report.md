# Classifier arm sweep — classify-merge-20260802T050208Z

- **Date:** 2026-08-02
- **Question sets:** reference_questions.json, reference_questions_v2.json (169 questions)
- **Arms:** `baseline`, `abstain`, `decision-rules`, `hint-sparse`, `model-deepseek`, `merged-A`, `merged-B`, `merged-C`, `abl-B-no-hint`, `abl-B-no-rules`, `abl-C-no-hint`, `abl-C-no-rules`, `abl-C-no-abstain`
- **Total cost:** ~$1.82

The agent filters retrieval for a sub-question only when it carries exactly one category, so `harmful` = filtered to the wrong domain, `correct` = filtered to the gold domain, `no filter` = zero or 2+ tags. A wrong filter costs a retrieval round; a missing one only costs precision.

**`net` = correct − harmful is the decision metric** and arms are ranked by it. Minimising `harmful` on its own is degenerate: an arm that tags nothing scores a perfect 0 and is worthless, which is exactly how `hint-vote` topped the T2 harmful table while abstaining on 43% of the questions. `harmful` is still reported next to it, because the two errors are not interchangeable.

## Pooled (v1 + v2), ranked by net

| Arm | n | Net | Harmful | Correct | No filter | Recall any | Tags/sub | Subs/q | Parse fail | Trunc. | p50 ms | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `merged-B` | 169 | 0.669 | 0.136 | 0.805 | 0.059 | 0.823 | 0.966 | 1.201 | 0 | 0 | 719 | 0.001 |
| `abl-C-no-abstain` | 169 | 0.657 | 0.118 | 0.775 | 0.106 | 0.828 | 0.996 | 1.319 | 0 | 0 | 1941 | 0.001 |
| `merged-C` | 169 | 0.639 | 0.136 | 0.775 | 0.089 | 0.799 | 0.949 | 1.278 | 0 | 0 | 2234 | 0.001 |
| `abl-C-no-rules` | 169 | 0.627 | 0.101 | 0.728 | 0.172 | 0.828 | 1.064 | 1.379 | 0 | 0 | 2444 | 0.0009 |
| `abl-B-no-rules` | 169 | 0.621 | 0.160 | 0.781 | 0.059 | 0.805 | 0.971 | 1.207 | 0 | 0 | 647 | 0.0008 |
| `abl-C-no-hint` | 169 | 0.609 | 0.130 | 0.740 | 0.130 | 0.769 | 0.940 | 1.284 | 0 | 0 | 2471 | 0.0008 |
| `hint-sparse` | 169 | 0.598 | 0.183 | 0.781 | 0.035 | 0.805 | 0.995 | 1.237 | 0 | 0 | 744 | 0.0008 |
| `merged-A` | 169 | 0.598 | 0.177 | 0.775 | 0.047 | 0.793 | 0.980 | 1.207 | 0 | 0 | 725 | 0.001 |
| `abl-B-no-hint` | 169 | 0.568 | 0.160 | 0.728 | 0.112 | 0.746 | 0.914 | 1.237 | 0 | 0 | 641 | 0.0008 |
| `decision-rules` | 169 | 0.562 | 0.166 | 0.728 | 0.106 | 0.740 | 0.913 | 1.219 | 0 | 0 | 741 | 0.0008 |
| `baseline` | 169 | 0.544 | 0.183 | 0.728 | 0.089 | 0.746 | 0.938 | 1.248 | 0 | 0 | 715 | 0.0006 |
| `abstain` | 169 | 0.544 | 0.172 | 0.716 | 0.112 | 0.740 | 0.915 | 1.260 | 0 | 0 | 726 | 0.0006 |
| `model-deepseek` | 169 | 0.491 | 0.130 | 0.621 | 0.248 | 0.793 | 1.167 | 1.420 | 0 | 0 | 2469 | 0.0006 |

## v1

| Arm | n | Net | Harmful | Correct | No filter | Recall any | Tags/sub | Subs/q | Parse fail | Trunc. | p50 ms | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `merged-C` | 48 | 0.708 | 0.104 | 0.812 | 0.083 | 0.812 | 0.938 | 1.333 | 0 | 0 | 1757 | 0.001 |
| `abl-C-no-abstain` | 48 | 0.708 | 0.104 | 0.812 | 0.083 | 0.854 | 1 | 1.333 | 0 | 0 | 1998 | 0.001 |
| `merged-B` | 48 | 0.688 | 0.125 | 0.812 | 0.062 | 0.833 | 0.966 | 1.229 | 0 | 0 | 688 | 0.001 |
| `hint-sparse` | 48 | 0.667 | 0.146 | 0.812 | 0.042 | 0.833 | 0.983 | 1.250 | 0 | 0 | 774 | 0.0008 |
| `merged-A` | 48 | 0.667 | 0.146 | 0.812 | 0.042 | 0.833 | 0.983 | 1.250 | 0 | 0 | 723 | 0.001 |
| `abl-B-no-rules` | 48 | 0.667 | 0.125 | 0.792 | 0.083 | 0.812 | 0.931 | 1.208 | 0 | 0 | 607 | 0.0008 |
| `abl-B-no-hint` | 48 | 0.667 | 0.104 | 0.771 | 0.125 | 0.792 | 0.902 | 1.271 | 0 | 0 | 686 | 0.0008 |
| `baseline` | 48 | 0.646 | 0.146 | 0.792 | 0.062 | 0.792 | 0.950 | 1.250 | 0 | 0 | 706 | 0.0006 |
| `abl-C-no-hint` | 48 | 0.646 | 0.104 | 0.750 | 0.146 | 0.771 | 0.922 | 1.333 | 0 | 0 | 2702 | 0.0008 |
| `abl-C-no-rules` | 48 | 0.646 | 0.104 | 0.750 | 0.146 | 0.833 | 1.046 | 1.375 | 0 | 0 | 2606 | 0.0008 |
| `decision-rules` | 48 | 0.625 | 0.125 | 0.750 | 0.125 | 0.771 | 0.900 | 1.250 | 0 | 0 | 740 | 0.0008 |
| `abstain` | 48 | 0.583 | 0.146 | 0.729 | 0.125 | 0.750 | 0.902 | 1.271 | 0 | 0 | 685 | 0.0006 |
| `model-deepseek` | 48 | 0.562 | 0.104 | 0.667 | 0.229 | 0.812 | 1.092 | 1.354 | 0 | 0 | 2398 | 0.0005 |

## v2

| Arm | n | Net | Harmful | Correct | No filter | Recall any | Tags/sub | Subs/q | Parse fail | Trunc. | p50 ms | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `merged-B` | 121 | 0.661 | 0.141 | 0.802 | 0.058 | 0.818 | 0.965 | 1.190 | 0 | 0 | 721 | 0.001 |
| `abl-C-no-abstain` | 121 | 0.636 | 0.124 | 0.760 | 0.116 | 0.818 | 0.994 | 1.314 | 0 | 0 | 1894 | 0.001 |
| `abl-C-no-rules` | 121 | 0.620 | 0.099 | 0.719 | 0.182 | 0.826 | 1.072 | 1.380 | 0 | 0 | 2410 | 0.0009 |
| `merged-C` | 121 | 0.612 | 0.149 | 0.760 | 0.091 | 0.793 | 0.954 | 1.256 | 0 | 0 | 2477 | 0.0011 |
| `abl-B-no-rules` | 121 | 0.603 | 0.174 | 0.777 | 0.050 | 0.802 | 0.986 | 1.207 | 0 | 0 | 689 | 0.0009 |
| `abl-C-no-hint` | 121 | 0.595 | 0.141 | 0.736 | 0.124 | 0.769 | 0.948 | 1.264 | 0 | 0 | 2315 | 0.0008 |
| `hint-sparse` | 121 | 0.570 | 0.198 | 0.769 | 0.033 | 0.793 | 1 | 1.231 | 0 | 0 | 727 | 0.0008 |
| `merged-A` | 121 | 0.570 | 0.190 | 0.760 | 0.050 | 0.777 | 0.979 | 1.190 | 0 | 0 | 725 | 0.001 |
| `decision-rules` | 121 | 0.537 | 0.182 | 0.719 | 0.099 | 0.727 | 0.918 | 1.207 | 0 | 0 | 742 | 0.0008 |
| `abstain` | 121 | 0.529 | 0.182 | 0.711 | 0.107 | 0.736 | 0.921 | 1.256 | 0 | 0 | 752 | 0.0006 |
| `abl-B-no-hint` | 121 | 0.529 | 0.182 | 0.711 | 0.107 | 0.727 | 0.919 | 1.223 | 0 | 0 | 632 | 0.0008 |
| `baseline` | 121 | 0.504 | 0.198 | 0.703 | 0.099 | 0.727 | 0.934 | 1.248 | 0 | 0 | 716 | 0.0006 |
| `model-deepseek` | 121 | 0.463 | 0.141 | 0.603 | 0.256 | 0.785 | 1.194 | 1.446 | 0 | 0 | 2610 | 0.0006 |

## Paired against baseline (same question ids)

`fixed` = the reference arm was harmful and this arm is not; `broken` = the reverse; `+correct`/`-correct` are the same two counts on the useful verdict. Each p is an exact two-sided McNemar over its own pair of counts. At n=169 an arm significant on neither has not been shown to differ at all — and the two halves of `net` move independently, so an arm can remove wrong filters and lose right ones in the same breath.

| Arm | vs | net | fixed | broken | p(harmful) | +correct | -correct | p(correct) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `merged-B` | `baseline` | 0.669 | 12 | 4 | 0.0768 | 16 | 3 | 0.0044 | **better** |
| `abl-C-no-abstain` | `baseline` | 0.657 | 13 | 2 | 0.0074 | 11 | 3 | 0.0574 | **better** |
| `merged-C` | `baseline` | 0.639 | 12 | 4 | 0.0768 | 14 | 6 | 0.1153 | tied with `baseline` |
| `abl-C-no-rules` | `baseline` | 0.627 | 16 | 2 | 0.0013 | 11 | 11 | 1.0000 | **better** |
| `abl-B-no-rules` | `baseline` | 0.621 | 10 | 6 | 0.4545 | 14 | 5 | 0.0636 | tied with `baseline` |
| `abl-C-no-hint` | `baseline` | 0.609 | 12 | 3 | 0.0352 | 8 | 6 | 0.7905 | **better** |
| `hint-sparse` | `baseline` | 0.598 | 9 | 9 | 1.0000 | 14 | 5 | 0.0636 | tied with `baseline` |
| `merged-A` | `baseline` | 0.598 | 8 | 7 | 1.0000 | 11 | 3 | 0.0574 | tied with `baseline` |
| `abl-B-no-hint` | `baseline` | 0.568 | 5 | 1 | 0.2188 | 3 | 3 | 1.0000 | tied with `baseline` |
| `decision-rules` | `baseline` | 0.562 | 6 | 3 | 0.5078 | 5 | 5 | 1.0000 | tied with `baseline` |
| `abstain` | `baseline` | 0.544 | 7 | 5 | 0.7744 | 5 | 7 | 0.7744 | tied with `baseline` |
| `model-deepseek` | `baseline` | 0.491 | 12 | 3 | 0.0352 | 5 | 23 | 0.0009 | worse |

## Paired against components and merges

A merged arm scored against each isolated arm whose treatment it combines, and each ablation against the merge it drops a component from. This is what settles whether merging bought anything the best single treatment did not.

`fixed` = the reference arm was harmful and this arm is not; `broken` = the reverse; `+correct`/`-correct` are the same two counts on the useful verdict. Each p is an exact two-sided McNemar over its own pair of counts. At n=169 an arm significant on neither has not been shown to differ at all — and the two halves of `net` move independently, so an arm can remove wrong filters and lose right ones in the same breath.

| Arm | vs | net | fixed | broken | p(harmful) | +correct | -correct | p(correct) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `merged-B` | `abstain` | 0.669 | 10 | 4 | 0.1796 | 17 | 2 | 0.0007 | **better** |
| `merged-B` | `decision-rules` | 0.669 | 9 | 4 | 0.2668 | 14 | 1 | 0.0010 | **better** |
| `merged-B` | `hint-sparse` | 0.669 | 8 | 0 | 0.0078 | 5 | 1 | 0.2188 | **better** |
| `merged-B` | `merged-A` | 0.669 | 8 | 1 | 0.0391 | 6 | 1 | 0.1250 | **better** |
| `abl-C-no-abstain` | `merged-C` | 0.657 | 4 | 1 | 0.3750 | 3 | 3 | 1.0000 | tied with `merged-C` |
| `merged-C` | `abstain` | 0.639 | 12 | 6 | 0.2379 | 17 | 7 | 0.0639 | tied with `abstain` |
| `merged-C` | `decision-rules` | 0.639 | 9 | 4 | 0.2668 | 13 | 5 | 0.0963 | tied with `decision-rules` |
| `merged-C` | `hint-sparse` | 0.639 | 10 | 2 | 0.0386 | 5 | 6 | 1.0000 | **better** |
| `merged-C` | `model-deepseek` | 0.639 | 4 | 5 | 1.0000 | 28 | 2 | 0.0000 | **better** |
| `merged-C` | `merged-B` | 0.639 | 4 | 4 | 1.0000 | 3 | 8 | 0.2266 | tied with `merged-B` |
| `abl-C-no-rules` | `merged-C` | 0.627 | 7 | 1 | 0.0703 | 2 | 10 | 0.0386 | worse |
| `abl-B-no-rules` | `merged-B` | 0.621 | 0 | 4 | 0.1250 | 0 | 4 | 0.1250 | tied with `merged-B` |
| `abl-C-no-hint` | `merged-C` | 0.609 | 6 | 5 | 1.0000 | 4 | 10 | 0.1796 | tied with `merged-C` |
| `merged-A` | `decision-rules` | 0.598 | 6 | 8 | 0.7905 | 10 | 2 | 0.0386 | **better** |
| `merged-A` | `hint-sparse` | 0.598 | 4 | 3 | 1.0000 | 3 | 4 | 1.0000 | tied with `hint-sparse` |
| `abl-B-no-hint` | `merged-B` | 0.568 | 6 | 10 | 0.4545 | 2 | 15 | 0.0024 | worse |

## Confusion — baseline

| gold \ predicted | apartment | business | car | dental | diseases-disabilities | health | life | long-term-care | loss-of-working-ability | mortgage | personal-accident | travel | (none) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apartment | **12** |  |  |  |  |  |  |  |  |  | 1 |  | 3 |
| business | 1 | **13** |  |  |  | 1 |  |  |  |  |  |  | 2 |
| car |  |  | **15** |  |  |  |  |  |  |  | 1 |  | 1 |
| dental |  |  |  | **10** |  | 1 |  |  |  |  |  |  |  |
| diseases-disabilities |  |  |  |  | **8** | 1 | 1 |  |  |  |  |  | 1 |
| health |  |  |  |  |  | **14** |  |  |  |  |  |  | 3 |
| life |  |  |  |  |  | 1 | **11** |  |  |  | 3 |  |  |
| long-term-care |  |  |  |  |  | 2 |  | **7** |  |  |  |  |  |
| loss-of-working-ability |  |  |  |  | 1 |  | 1 |  | **7** |  | 1 |  | 1 |
| mortgage | 9 |  |  |  |  |  | 1 |  |  | **4** |  |  | 3 |
| personal-accident |  |  |  |  |  | 1 |  |  |  |  | **8** | 2 |  |
| travel |  |  |  |  |  |  |  |  |  |  | 2 | **14** | 1 |

Most frequent wrong filters: `mortgage` tagged `apartment` — 9x; `life` tagged `personal-accident` — 3x; `travel` tagged `personal-accident` — 2x; `personal-accident` tagged `travel` — 2x; `long-term-care` tagged `health` — 2x; `personal-accident` tagged `health` — 1x; `mortgage` tagged `life` — 1x; `loss-of-working-ability` tagged `personal-accident` — 1x.

## Confusion — best arm (`merged-B`)

| gold \ predicted | apartment | business | car | dental | diseases-disabilities | health | life | long-term-care | loss-of-working-ability | mortgage | personal-accident | travel | (none) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apartment | **13** |  |  |  |  |  |  |  |  |  | 1 |  | 2 |
| business | 1 | **16** |  |  |  |  |  |  |  |  |  |  |  |
| car |  |  | **16** |  |  |  |  |  |  |  |  |  | 1 |
| dental |  |  |  | **9** |  | 2 |  |  |  |  |  |  |  |
| diseases-disabilities |  |  |  |  | **8** | 1 | 1 |  |  | 1 |  |  |  |
| health |  |  |  |  |  | **15** |  |  |  |  |  |  | 2 |
| life |  |  |  |  |  |  | **13** |  |  |  | 2 |  |  |
| long-term-care |  |  |  |  |  | 1 |  | **8** |  |  |  |  |  |
| loss-of-working-ability |  |  |  |  |  |  | 1 |  | **8** |  | 1 |  | 1 |
| mortgage | 8 |  |  |  |  |  |  |  |  | **5** |  |  | 4 |
| personal-accident |  | 1 |  |  |  |  |  |  |  |  | **9** | 1 |  |
| travel |  |  |  |  |  |  |  |  |  |  | 1 | **16** |  |

Most frequent wrong filters: `mortgage` tagged `apartment` — 8x; `life` tagged `personal-accident` — 2x; `dental` tagged `health` — 2x; `travel` tagged `personal-accident` — 1x; `personal-accident` tagged `travel` — 1x; `personal-accident` tagged `business` — 1x; `loss-of-working-ability` tagged `personal-accident` — 1x; `loss-of-working-ability` tagged `life` — 1x.

## Arms

- `baseline` — today's production prompt, gpt-oss-120b, reasoning low, temp 0
- `abstain` — + a wrong tag is worse than none — abstain when undetermined
- `decision-rules` — + ordered decision rules with a stated precedence for overlaps
- `hint-sparse` — baseline prompt + retrieval evidence block
- `model-deepseek` — baseline prompt on DeepSeek-V4-Pro
- `merged-A` — decision rules + retrieval evidence
- `merged-B` — decision rules + abstain + retrieval evidence
- `merged-C` — merged-B on DeepSeek-V4-Pro
- `abl-B-no-hint` — merged-B minus the retrieval evidence
- `abl-B-no-rules` — merged-B minus the decision rules
- `abl-C-no-hint` — merged-C minus the retrieval evidence
- `abl-C-no-rules` — merged-C minus the decision rules
- `abl-C-no-abstain` — merged-C minus the abstain rule

---
*Generated by `evalharness.classify_eval`; per-question records are in `<arm>/predictions.jsonl` and every metric in `summary.json`.*
