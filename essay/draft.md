# A recipe for local conversational fine-tuning

A conversational fine-tune can learn visible style while still mishandling context. This source release separates the implementation from the private inputs used during development. The small example conversation uses fictional participants; the learned checkpoints and their outputs are absent.

## Start with a defensible target

Parse messages, merge multiline text, classify media, redact direct identifiers, and group turns into sessions. Split by session so adjacent messages do not leak between train and test. Each example ends with the assistant reply to learn. Trim old context before cutting the target.

## Match inference when constructing training tokens

Chat templates can render completed conversations differently from generation prompts. Compute the divergence, preserve the actual generation prefix, and train on the reply suffix plus end marker. Exclude padding with a strict sequence-length bound. Otherwise a superficially successful run can learn an incomplete target.

## Add context in stages

The media-aware pipeline keeps captions and uses local descriptions for visible attachments. Unknown media remain unknown. A monthly memory pass can supply a short card containing only past information. Generated rationales are annotations conditioned on observed context and replies; they are not recovered human thoughts.

## Separate four objectives

A predicts replies. B predicts a rationale and a reply. C predicts a reply given a rationale. D predicts only a rationale. Evaluate pipelines with their own generated plans, not just reference annotations. Different prediction targets make raw validation losses unsuitable as a cross-variant quality ranking.

## Measure behavior as well as loss

Reply BPC supports tokenizer comparisons. Length, emoji, multiple-message turns, exact-copy frequency, repetition, speaker ownership, factual grounding, and multi-turn coherence answer different questions. Use human review where narrow proxies miss the behavior you care about.

## Keep the release boundary explicit

Publish source, generic prompts, configuration templates, fictional fixtures, and reviewed aggregate metrics. Keep conversations, media, annotations, memories, generated outputs, credentials, and learned weights private. Merging an adapter or interpolating weights does not remove the information learned from private inputs.

The complete procedure is in [the recipe](../docs/recipe.md), with [all scripts](../docs/scripts.md) and [architecture notes](../docs/architecture.md).
