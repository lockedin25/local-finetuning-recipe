# Pipeline and experiment design

The pipeline turns an authorized export into typed messages, redacts direct identifiers, groups messages into sessions, constructs assistant-target examples, optionally adds past-only memory and rationale annotations, and trains a base model with LoRA. Data and learned artifacts remain in an external workspace.

## Message and target boundaries

Message records carry `ts`, `sender`, `text`, and `kind`. Speaker roles map `her` to user/context and `me` to assistant/target for compatibility with the original scripts; configure the actual identities locally. The labels do not require a particular relationship or gender.

Examples carry a `messages` array and `meta` identifiers. A completed assistant turn ends every example. Adjacent examples from the same session must remain within one split. Test evaluation uses held-out sessions, not randomly sampled messages from training conversations.

Chat templates can add tokens to the generation prompt that do not appear at the same position in the completed rendering. The custom encoder finds the common prefix, takes the target suffix from the completed conversation, and attaches it to the actual generation prompt. Its offset marks the target boundary. Only the reply and end marker contribute to reply-only training. Padding must be excluded with a strict length comparison.

## Media and memory

V2 preserves captions and replaces attachments with typed tags: photo, video, GIF, sticker, voice note, document, contact, location, missing media, and view-once. Local VLM descriptions and human sticker labels are private derived data. Unknown media remain unknown. Sensitive descriptions are reduced to bare tags before the reasoning-input path that can use hosted services.

Monthly notes feed sequential memory snapshots. Each example receives the card from the end of the preceding month. Rationale generation uses an indexed reply ID and past context. Rebuilding data can change IDs; do not combine annotation runs from different builds without checking alignment.

## Four targets

| Variant | Input | Trained output |
|---|---|---|
| A | Conversation plus memory | Reply |
| B | Conversation plus memory | Rationale, then reply |
| C | Conversation, memory, supplied rationale | Reply |
| D | Conversation plus memory | Rationale only |

B-to-C and D-to-C pipelines first generate a rationale, then pass it to the writer. A hosted planner can instead produce a short response plan. The planner requests public response content and deliberately refuses to substitute the provider's internal `reasoning_content` as the plan.

## Weight experiments

Interpolation computes `W_candidate = W_base + alpha * (W_B - W_base)` after verifying tensor names and shapes. Module restoration swaps selected attention, MLP, or convolution tensors back to the base model. Separate development and holdout fixtures test narrow preferences and constrained answers. These are experimental interventions, not established privacy removal or quality improvements.
