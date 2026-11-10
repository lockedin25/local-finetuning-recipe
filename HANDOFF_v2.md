# V2 technical handoff

## State and purpose

V1 established parsing, redaction, assistant-only targets, token budgeting, base-model scoring, local LoRA training, and reply/style evaluation. V2 adds media tags and long context. The public release provides source and fictional inputs, not the original dataset or completed annotations.

## Data interfaces

- `v2/final/{train,valid,test}.jsonl`: chat examples with stable reply IDs in `meta.rid` for a fixed build.
- `v2/reasoning/months/`: monthly transcript inputs.
- `v2/reasoning/chunks/`: numbered chunks and their index; reply IDs are annotation keys.
- `v2/reasoning/notes/`: A1 output.
- `v2/reasoning/memory/after_<month>.md`: A2 memory and card snapshots.
- `v2/reasoning/out/<tag>/`: B outputs and missing-ID markers.

All are private workspace paths. Rebuilding examples requires checking ID compatibility before reusing annotations.

## Memory and rationale passes

A1 splits long month inputs into parts and writes notes. A2 consumes those parts sequentially, preserving state and producing `=== MEMORY ===` and `=== CARD ===` sections. B reads chunks with the previous month's memory and produces keyed annotations. Finished files are reused; incomplete chunks are resumed and resolved missing markers removed. Verify coverage rather than trusting file existence alone.

Use a small blind pilot before generating all rationales. Compare speaker attribution, grounding, unsupported memories, and explanation quality. A short reply can legitimately have an empty rationale. Never treat annotations as recovered human thoughts.

## Training continuation

Verify the exact chat template and target mask. The thought-plus-reply variant enables thinking; reply-only mode disables it explicitly. Use a 20-step/400-example smoke test, watch memory and speed, then choose the full run. For the historical MLX approach, rank 16, batch 4, and learning rate 2e-5 were used as starting settings. Pass data and adapter paths outside the checkout.

## Validation and failure cases

Check edited-marker stripping after multiline merges, captions, empty turns, speaker alternation, truncation, memory chronology, missing rationale IDs, and target coverage. Compare baseline BPC with the adapter on the same held-out sessions. Keep local review pages private. A hosted backend is optional and requires the operator's own authorization for its inputs.
