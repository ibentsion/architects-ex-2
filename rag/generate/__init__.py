"""Generation phase: context assembly, citation-mandating prompts, tf_client
LLM call, citation parsing + validation (rag_plan.md §6 stages 5-7).

The prompt-variant registry lives in ``rag.generate.prompts.PROMPT_REGISTRY``
(the generation config block is ``{model, prompt, …}``, not ``{impl, params}``).
"""
