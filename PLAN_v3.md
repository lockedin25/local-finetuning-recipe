# V3 technical plan

Build matched datasets from V2 examples, previous-month cards, and validated rationale annotations. Include a card in both variants. Add `reasoning_content` only to the final assistant turn of the thinking variant. Drop failed/missing annotations from both training variants to preserve matched samples. Keep test inputs matched and omit reference rationales from test prompts.

Train A on replies, B on rationale plus reply, C on replies with a supplied rationale, and D on rationales alone. Keep tokenizer, masking, split selection, and budgets explicit. Verify both the self-contained B path and the two-model B-to-C/D-to-C path with self-generated plans rather than reference annotations.

Use fixed seeds and consistent evaluation contexts. Track reply BPC, style, empty outputs, incomplete thought blocks, repetition, speaker ownership, and semantic fit. Have a human review multi-turn behavior. A lower target loss does not establish a stronger conversational model.

Weight-edit experiments interpolate the fine-tuning delta or restore module families from the base. Verify tensor alignment and test on untouched held-out prompts. Save experimental checkpoints separately; never overwrite the original fine-tune or treat an edit as proven privacy removal.
