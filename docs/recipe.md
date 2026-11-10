# Reproduce the method

## 1. Install and create an external workspace

Use Python 3.12. Run `uv sync --locked` from the repository. The lockfile records resolved package versions. Optional extras are `essay`, `vision`, and `gpu`; GPU packages alone do not provision CUDA or a remote machine.

```sh
uv run python tools/recipe.py init --workspace "$HOME/LocalData/my-finetune"
export LOCAL_FINETUNING_HOME="$HOME/LocalData/my-finetune"
export LOCAL_FINETUNING_CONFIG="$LOCAL_FINETUNING_HOME/config.local.yaml"
export LOCAL_FINETUNING_DATA="$LOCAL_FINETUNING_HOME/data"
```

The workspace must resolve outside this repository. Initialization does not overwrite an existing local configuration. Edit its sender mapping, date order, chronological test cutoff, system prompt, tokenizer, model, and output paths. Alex and Sam are fictional defaults. The config's test date is an example, not a recommended split for every dataset.

The original low-level scripts retain some relative paths for historical compatibility. Use the recipe wrapper for the initial data stages. For later stages, export the environment variables above and pass absolute data/output paths. Do not put real data inside the source checkout merely because a historical default names `data/`.

## 2. Supply an authorized export, parse, and redact

Put an Android-format WhatsApp export at the `raw_chat` path in your private config. The parser expects `DD/MM/YYYY, HH:MM - Sender: text`, with MDY available through config. Check your export format first.

```sh
uv run python tools/recipe.py parse --workspace "$LOCAL_FINETUNING_HOME"
uv run python tools/recipe.py clean --workspace "$LOCAL_FINETUNING_HOME"
```

The parser merges continuation lines before stripping edited markers. It classifies media placeholders, deleted messages, view-once placeholders, polls, and locations. The redactor masks links, emails, UPI handles, phone numbers, long numeric identifiers, PAN-style identifiers, social handles, OTPs, and password-like strings. Deleted messages are dropped. Markdown/list paste detection differs from a simple length cutoff; review `keep_long` overrides locally.

This redactor is a training-data preparation tool. It is not proof that a real dataset is anonymous or safe to publish. Do not publish its output.

## 3. Choose a base and build examples

`src/m0_tokens.py` probes tokenizer behavior using fictional text. `src/m0_chat_template.py` inspects an invented conversation template. `src/download_models.sh <repo-id>` downloads explicitly selected upstream weights; omitting the argument downloads the historical candidate set. `src/dl_status.py --once` reports cache progress.

```sh
zsh src/download_models.sh LiquidAI/LFM3-1.7B-MLX-bf16
uv run python tools/recipe.py build --workspace "$LOCAL_FINETUNING_HOME"
```

Set `build.budget_tokenizer` and `train.model` consistently. Base-model access and licensing are your responsibility. Downloads stay in the Hugging Face cache, never in a Git commit.

V1 examples end with one target assistant reply. Sessions split at three-hour gaps by default; repeated same-sender messages become turns. The initial budget is 12 context turns, 512 context tokens, and 1,536 total tokens. The builder trims old context to retain the target, caps filler-heavy replies, and splits by session rather than randomly separating adjacent messages.

For media-aware V2, run the local vision descriptor on your own media archive, review sticker labels locally, set `WHATSAPP_EXPORT_DIR` to the external folder, then run `src/build_v2.py`. Its default budget is 40 turns and 2,048 context tokens, with 3,072 total tokens. Missing or hidden media stay unknown. Never infer the contents of view-once messages.

## 4. Verify reply masking and run a smoke test

`src/encode.py` builds the exact generation prompt, finds where the completed rendering diverges, and appends the target plus end-of-turn marker. Training scores only target tokens. `src/train.py` uses strict sequence lengths so padding is not included in loss.

```sh
export LOCAL_FINETUNING_CONFIG="$LOCAL_FINETUNING_HOME/config.local.yaml"
uv run python src/train.py --data "$LOCAL_FINETUNING_DATA/final" --iters 20 --limit 400 --adapter-path "$LOCAL_FINETUNING_HOME/adapters/smoke"
```

For the historical LFM3 implementation, replace `uv run python` with `zsh src/lfm3.sh`. That wrapper uses the recorded mlx-lm implementation commit. Confirm that your chosen model loads before launching a long run.

Check target offsets, nonempty targets, end-of-turn handling, peak memory, speed, and validation behavior. The historical LoRA recipe uses rank 16, scale 20, dropout 0.05, learning rate 2e-5, and batch 4. Treat these as starting settings, not universal optima. Use `--resume`, `--lr`, `--max-seq`, `--steps-per-eval`, `--save-every`, and `--cosine` deliberately.

## 5. Add memory or rationales only after the baseline

Workspace initialization installs generic prompts under `v2/reasoning/`. `src/reasoning_inputs.py` prepares monthly transcripts and indexed chunks. Pass A1 writes monthly notes; A2 merges them sequentially into dated memory and a short card; B writes a rationale per reply using only context available before that reply.

`src/reasoning_run.py` supports hosted Claude or OpenRouter backends. These send text outside your machine and may cost money. No hosted call is needed for the public demo. Choose and authorize your backend before using private material. Its OpenRouter request asks for zero retention and denied data collection; this is request configuration, not an independently verified guarantee of provider policy.

`src/reasoning_local.py` and `src/reasoning_local_pilot.py` provide local alternatives. `experiments/vm_bundle/vm_reason.py` runs vLLM in a separately prepared environment. Review its required inputs in the script reference; VM helpers are not unattended provisioning scripts.

Review pilot outputs with the local review-page builders. Previous-month cards prevent future information from entering earlier examples. Keep test rationales out of training. Generated rationales are annotations, not evidence of a person's actual thoughts.

`src/merge_reasoning.py` expects V2 data plus `api_out/cards_clean.json`, `api_out/cards.json`, and `api_out/reasoning.jsonl`, with optional locally curated annotations. It builds matched `think` and `nothink` splits, dropping failed/missing rationale examples from both. Adapt its recorded cutoff to your dataset. The local-generation path has its own output layout; reconcile schemas before merging. Do not assume every experimental branch plugs into every other branch unchanged.

## 6. Compare four objectives on GPU, if needed

Prepare an authorized compute environment and install the `gpu` extra there. Copy only inputs you intend that environment to receive. Point `LOCAL_FINETUNING_GPU_HOME` at an external bundle with `data/think` and `data/nothink` split files.

```sh
uv run --extra gpu python experiments/gpu_bundle/gpu_train.py --variant nothink --out /external/workspace/runs/A --steps 20 --limit 400
```

Variants are `nothink` (A, reply), `think` (B, rationale plus reply), `texter` (C, reply conditioned on a supplied rationale), and `thinker` (D, rationale only). Historical settings: batch 16, rank 16, scale 20, learning rate 4e-5, warmup 50 steps, cosine schedule, one epoch, and a 2,560-token cap. Inspect truncation before trusting loss.

The trainer saves merged bf16 weights under its output directory. Evaluation helpers separately test self-generated thoughts, thinker/writer pipelines, and fixed-context generations. No trained weights are included here.

## 7. Evaluate and chat locally

`src/evaluate.py` compares reply BPC, style proxies, and exact training-copy frequency. Set `LOCAL_FINETUNING_DATA` to the external data directory containing `final/train.jsonl` and `final/test.jsonl`. `src/chat_v3.py` supports direct, thought-plus-reply, local thinker/writer, and hosted-planner modes. Its aliases A/B/C/D expect models under the external workspace.

Review meaning, speaker ownership, factual grounding, coherence, repetition, and incomplete thought blocks. A lower validation loss does not establish better conversation. Keep generated evaluations and review pages private, even for fictional prompts against a private model.

## 8. Merge adapters without publishing them

The GPU trainer already exports merged models. For an MLX adapter, use the installed `mlx_lm.fuse --help` to verify options, then an external destination, for example:

```sh
uv run python -m mlx_lm.fuse --model LiquidAI/LFM3-1.7B-MLX-bf16 --adapter-path "$LOCAL_FINETUNING_HOME/adapters/run" --save-path "$LOCAL_FINETUNING_HOME/fused/run"
```

Merging changes the packaging of learned weights; it does not remove private training information. Keep adapters and fused checkpoints local. A later model release needs its own privacy assessment and upstream license review. This repository contains no automatic Hugging Face upload command.
