# Offline VM rationale generation

Set LOCAL_FINETUNING_VM_HOME to an external bundle containing `data/{train,valid}.jsonl`, `notes/*.md`, `PROMPT_A2_merge.md`, and the curated `gold.json` expected by the code. Copy the generic merge prompt from `prompts/`; supply your own authorized data separately.

`vm_reason.py cards` merges notes. `vm_reason.py reason --limit 600` runs a pilot with generate/verify/retry behavior. Install vLLM and compatible model support separately. The script does not provision compute or authorize uploading data. Outputs remain private.
