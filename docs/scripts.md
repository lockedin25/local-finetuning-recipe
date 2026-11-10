# Complete script reference

All 59 imported scripts are listed below. CLI declarations come from syntax trees without importing modules, reading keys, or loading models. Follow the recipe for supported stage order and inspect branch prerequisites before execution.

## src/bakeoff.py

Module 4: score candidate base models on how well they ALREADY predict Alex's replies.

Metric: bits per character (BPC) of his reply given the context, on 500 test examples.
Per-character (not per-token) so models with different tokenizers are compared fairly.
Lower = the model is less surprised by how he texts = better starting point.
Runs one model at a time, writes results to bakeoff_results.json. No text is printed.

Entry points: `encode_plain`, `reply_token_count`, `score`, `main`.

## src/build_dataset.py

Module 3: clean.jsonl -> data/final/{train,valid,test}.jsonl (mlx-lm chat format)

One example per Alex reply:
  {"messages": [system, (context turns...), assistant=<his reply>], "meta": {...}}
mlx-lm's mask_prompt trains only on the LAST message, so every example ends on his reply.
mlx-lm truncates over-long examples from the END, so we enforce the token budget here:
his reply is always kept whole; the oldest context is trimmed instead.
Prints statistics only, never message text.

Entry points: `load_messages`, `sessionize`, `is_filler`, `build_examples`, `split`, `downsample_filler`, `main`.

```python
ap.add_argument('--ctx-turns', type=int, default=B['max_context_turns'])
ap.add_argument('--ctx-tokens', type=int, default=B['max_context_tokens'])
ap.add_argument('--out', default='data/final')
```

## src/build_sticker_page.py

Build a LOCAL labelling page for the most-sent stickers (opened from disk, never published).

For each sticker (ranked by how often it was sent): the sticker itself, who sends it, the vision model's
guess, and the messages that most often come right before it, so Alex can write what it MEANS
between them (e.g. 'kiss', 'sorry baby', 'goodnight'). Labels are saved with a button that downloads
sticker_labels.json (and kept in the browser while editing).

Prints counts only.
  python3 -I src/build_sticker_page.py --export "<export folder>" --desc <media_descriptions.jsonl> --out <page.html> [--top 100]

Entry points: `main`.

```python
ap.add_argument('--export', required=True)
ap.add_argument('--desc', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--top', type=int, default=100)
```

## src/build_v2.py

Dataset v2: the with-media export, every attachment replaced by a text tag, long context.

  uv run python -I src/build_v2.py            (all outputs go to ~/LocalData/hinglish-clone/v2, outside iCloud)

What changes vs v1:
  * new export (with media); captions on photos/videos kept (v1 dropped them)
  * photo / video / gif  -> [photo: <local LFM3-VL description>] etc. (intimate ones: [intimate photo: ...])
  * stickers             -> [sticker: core | context | emotion] from Alex's labels, else the model's guess
  * voice notes, docs, contacts, view-once, unsaved media -> short tags
  * '<This message was edited>' stripped after multi-line merge
  * long context: up to 40 turns / 2,048 tokens (v1: 12 / 512), budgeted with the LFM3 tokenizer
Outputs:
  v2/final/{train,valid,test}.jsonl          mlx-lm chat format (same shape as v1)
  v2/reasoning/  chunks for an outside model to write the <think> reasoning (see README there)
Prints counts only.

Entry points: `parse`, `load_media`, `media_tag`, `to_text`, `sessionize`, `is_filler`, `build_examples`, `split`, `cap_filler`, `reasoning_record`, `main`.

```python
ap.add_argument('--ctx-turns', type=int, default=40)
ap.add_argument('--ctx-tokens', type=int, default=2048)
ap.add_argument('--max-seq', type=int, default=3072)
ap.add_argument('--chunk', type=int, default=500, help='examples per reasoning chunk file')
ap.add_argument('--keep-intimate-in-reasoning', action='store_true', help='default: intimate photo/video descriptions become a bare [intimate photo] tag in the reasoning chunks (they go to an outside model); training data keeps them in full')
```

## src/chat.py

Chat with the clone (or the untrained base model) in your terminal.

  uv run python src/chat.py --adapter adapters/run1      # the clone
  uv run python src/chat.py                               # untrained Gemma-4, for comparison
  zsh src/lfm3.sh src/chat.py --adapter adapters/lfm3-run1 --ctx-turns 12   # LFM3 clone
You type as her; the model replies as Alex. Commands: /reset (new conversation), /quit.

Matches training exactly: same system prompt, thinking OFF, last N turns as context.
Multi-line replies are shown as separate texts, like WhatsApp double-texting.

Entry points: `clean`, `main`.

```python
ap.add_argument('--adapter', default=None, help='adapter folder, e.g. adapters/run1')
ap.add_argument('--model', default=None, help="base model (default: the adapter's own model, else config.yaml)")
ap.add_argument('--temp', type=float, default=0.7)
ap.add_argument('--top-p', type=float, default=0.9)
ap.add_argument('--rep-penalty', type=float, default=1.0, help='1.0 = off. >1 penalises words already in the chat, which pushes replies OFF topic')
ap.add_argument('--ctx-turns', type=int, default=12, help='previous turns kept as context (training used up to 11)')
ap.add_argument('--no-emoji', action='store_true', help='strip emoji from replies')
ap.add_argument('--think', action='store_true', help='let the model reason before replying (LFM3/Gemma/Qwen)')
ap.add_argument('--show-thoughts', action='store_true', help='with --think: print the reasoning too')
```

## src/chat_v3.py

Chat with the v3 clones on the Mac, set up exactly like training (system prompt + memory card + long context).

  zsh src/lfm3.sh src/chat_v3.py --model A                      # A: plain clone (no thinking)
  zsh src/lfm3.sh src/chat_v3.py --model B --mode think         # B: thinks, then replies (thought shown)
  zsh src/lfm3.sh src/chat_v3.py --model C --mode pipeline --thinker D   # D thinks, C texts (or --thinker B)
You type as Sam; the model replies as Alex. Commands: /reset, /quit, /card (show the memory card).
Models are looked up in ~/LocalData/hinglish-clone/models/<name>/mlx (A=A_nothink, B=B_think, C=C_texter, D=D_thinker).
Tip: you can type media the way the training data shows it, e.g. [sticker: kiss | her giving me a kiss | love, affection]

Entry points: `model_path`, `main`.

```python
ap.add_argument('--model', required=True, help='A, B, C, D or a path to an MLX model')
ap.add_argument('--mode', choices=['plain', 'think', 'pipeline', 'jarvis'], default='plain')
ap.add_argument('--thinker', default='D', help='pipeline mode: the model that writes the thought (D or B)')
ap.add_argument('--jarvis-model', default='deepseek-v4.1-flash-1017')
ap.add_argument('--jarvis-effort', choices=['low', 'high', 'max'], default='high')
ap.add_argument('--jarvis-max-tokens', type=int, default=4096, help='Jarvis output budget, including provider thinking tokens')
ap.add_argument('--card', default='latest', help="memory card month (YYYY-MM) or 'latest' or 'none'")
ap.add_argument('--ctx-turns', type=int, default=40, help='previous turns kept (training used up to 40)')
ap.add_argument('--temp', type=float, default=0.7)
ap.add_argument('--top-p', type=float, default=0.9)
ap.add_argument('--hide-thoughts', action='store_true')
ap.add_argument('--personal-style', action='store_true', help='Experimental personal texting and speaker-grounding instructions')
```

## src/clean_cards.py

Clean the monthly memory cards for the system prompt (Jarvis API, DeepSeek V4.1 Flash, thinking off).

Removes exact dates, day numbers, percentiles and markdown (the summaries got dates wrong), keeps durable facts,
writes first person from Alex's side, <= 120 words. Counts only are printed.
  python3 -I src/clean_cards.py   ->  ~/LocalData/hinglish-clone/api_out/cards_clean.json

Entry points: `main`.

## src/clean_redact.py

Module 2: messages.jsonl -> clean.jsonl with sensitive strings replaced by tags.

Keeps conversation structure (media becomes <media>), drops deleted messages,
and redacts links, emails, UPI IDs, phones, long numbers, OTPs, PANs and passwords.
Prints counts only, never message text.

Entry points: `redact`, `is_paste`, `main`.

## src/describe_media.py

Describe the photos and stickers inside a WhatsApp 'Export chat -> Include media' zip with a LOCAL
vision model (LFM3-VL-2B on MLX). Nothing leaves the Mac: images are read from the zip straight into
memory (never unpacked to disk) and the descriptions go to one JSONL file.

Prints COUNTS ONLY. Never prints a description (Claude reads this terminal, and that would leave the Mac).

  uv run --no-project --python 3.12 --with "mlx-vlm>=0.7.11" python -I src/describe_media.py       --zip "~/Downloads/WhatsApp Chat with X.zip" --out <non-iCloud path>/media_descriptions.jsonl [--limit 10]

Output rows: {"file": "IMG-...jpg", "kind": "photo"|"sticker", "sha1": ..., "category": ..., "desc": ...}
Stickers are de-duplicated by content hash (the same sticker is sent many times; described once).
Safe to stop and restart: files already in --out are skipped.

Entry points: `shrink`, `video_frames`, `kind_of`, `two_lines`, `one_line`, `main`.

```python
ap.add_argument('--zip', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0, help='stop after N new descriptions (for a test run)')
ap.add_argument('--files', nargs='*', default=None, help='only these file names (e.g. IMG-20261106-WA0000.jpg)')
ap.add_argument('--only', choices=['photo', 'sticker', 'gif', 'video', 'all'], default='all', help='gif = silent videos <= 15 s (WhatsApp sends GIFs as mp4); video = the rest')
```

## src/dl_status.py

Live download tracker for the bake-off models.

Usage:  uv run python src/dl_status.py          (refreshes every 5 s, Ctrl+C to quit)
        uv run python src/dl_status.py --once   (print once and exit)
Reads only file sizes on disk; expected sizes come from the Hugging Face API (cached).

Entry points: `expected_sizes`, `on_disk`, `fmt`, `main`.

## src/download_models.sh

Module 4: download bake-off candidates one after another (Xet backend off: it hangs on this network).
Usage: zsh src/download_models.sh            (all, in order)
zsh src/download_models.sh <repo> ...  (only these)

## src/encode.py

Turn one chat example into (tokens, offset) for training, matching chat time exactly.

Why not mlx-lm's built-in ChatDataset?  It tokenizes the *finished* conversation but
computes the mask offset from the *generation prompt*. For templates that add extra
tokens to the generation prompt (Gemma-4's empty thought channel, Qwen's <think> block),
the offset lands inside the reply and the reply's first tokens are never trained.

Here the training sequence is built as:
    generation prompt (exactly what the model sees at chat time)   -> masked
  + reply + end-of-turn marker (taken from the finished rendering)  -> trained

Entry points: `encode_example`.

## src/evaluate.py

Module 7: evaluate a base model or a trained adapter on the TEST set (sessions from 2026-09-18 on,
never seen in training). Prints statistics only, never chat text.

  1. Bits per character (BPC) of the real replies, on the SAME 500 test examples as bakeoff.py,
     so it compares directly with the untrained baselines (Gemma-4 2.114, LFM3 1.858).
  2. Style: generate one reply for each of 100 test contexts and compare with the real replies:
     length, double-texting, emoji, formal/assistant words, Devanagari, empty or overlong replies.
  3. Memorisation: share of generated replies (>= 25 chars) that appear verbatim in train.

  zsh src/lfm3.sh src/evaluate.py --model LiquidAI/LFM3-1.7B-MLX-bf16 --adapter adapters/lfm3-best
  zsh src/lfm3.sh src/evaluate.py --model LiquidAI/LFM3-1.7B-MLX-bf16            # untrained base
Results are appended to eval_results.json under --name.

Entry points: `style`, `bpc`, `main`.

```python
ap.add_argument('--model', required=True)
ap.add_argument('--adapter', default=None)
ap.add_argument('--name', default=None)
ap.add_argument('--n-gen', type=int, default=100)
ap.add_argument('--temp', type=float, default=0.7)
```

## src/health_watch.py

Watch a training run and the Mac's health. Prints one line per EVENT (for Claude's Monitor):
  - each validation result (every --steps-per-eval steps) with temps / memory since the last one
  - health alerts, printed once when a condition starts and once when it clears
  - DONE / CRASHED when the training process ends
Every 60 s it also appends a row to health.log (CSV) so the whole night can be reviewed later.

Protective action: if macOS reports a CRITICAL thermal state, any temperature reaches 105 C,
or memory pressure is critical with heavy swap, the training process is PAUSED (SIGSTOP), never killed.
Resume with:  kill -CONT <pid>   (the pid is printed in the alert)

  /usr/bin/python3 -I src/health_watch.py lfm3_run1.log
Safe to restart at any time: progress is kept in .health_state.json.

Entry points: `say`, `thermal_state`, `sample`, `train_pid`, `parse_log`, `main`.

## src/jarvis_planner.py

Jarvis response planning for the local texter. Never log request/response bodies.

Entry points: `JarvisError`, `redact`, `build_messages`, `extract_plan`, `plan_reply`.

## src/jarvis_reason.py

Runs ON THE MAC. Writes Alex's reasoning with DeepSeek V4.1 Flash via the Jarvis Token Factory API (thinking off).

  python3 vm_reason.py cards                 # monthly memory cards from notes (sequential, small)
  python3 vm_reason.py reason --limit 600    # pilot: first N selected replies
  python3 vm_reason.py reason                # all selected replies (train/valid, from 2025-11-08)
Each reason run = generate -> verify (same model) -> regenerate failures once -> verify.
Outputs: out/cards.json, out/reasoning.jsonl ({id, reasoning, verdict, why}).
Prints counts only.

Entry points: `cost`, `SamplingParams`, `_O`, `_R`, `LLM`, `render`, `load`, `prev_month`, `cards`, `fewshot`, `reason`, `main`.

```python
ap.add_argument('step', choices=['cards', 'reason'])
ap.add_argument('--limit', type=int, default=0)
```

## src/judge_eval.py

Meaning-based evaluation: DeepSeek (Jarvis API, approved) judges each setup's reply BLIND against the conversation
and Alex's real reply. Prints scores only.
  python3 -I src/judge_eval.py   (reads ~/LocalData/hinglish-clone/eval/generations.jsonl)
Scores per reply (1-5): sense (responds to what she said, uses context), intent (same intent as the real reply),
style (sounds like Alex), plus which candidate is best overall.

Entry points: `main`.

## src/lfm3.sh

Run any project script with an mlx-lm that can load LFM3.
No mlx-lm release supports LFM3 yet; this pins the open PR ml-explore/mlx-lm#2016
(one new file, mlx_lm/models/lfm3.py, reviewed 2026-11-08) as a temporary overlay.
The project's own environment (used for Gemma) is untouched.
Usage: zsh src/lfm3.sh src/train.py --model LiquidAI/LFM3-1.7B-MLX-bf16 ...

## src/m0_chat_template.py

Module 0.2: what a chat conversation actually looks like to the model.

## src/m0_tokens.py

Module 0: how do different model families tokenize Hinglish vs English?

Downloads only tokenizer files (a few MB each), not model weights.
Sentences are generic examples, not from the real chats.

Entry points: `main`.

## src/merge_reasoning.py

Build dataset v3 (two variants) from v2 + memory cards + reasoning. Local only; prints counts.

  python3 -I src/merge_reasoning.py
Writes ~/LocalData/hinglish-clone/v3/{think,nothink}/{train,valid,test}.jsonl
  * scope: train/valid replies from 2025-11-08 (last 12 months); test unchanged (all test examples)
  * system prompt = base prompt + "What I know right now:" + cleaned card from the END OF THE PREVIOUS MONTH
  * think:   final assistant message gets "reasoning_content" (Claude's hand-written ones win, else DeepSeek PASS;
             EMPTY -> "" which renders as an empty think block). Replies whose reasoning FAILED are dropped
             from BOTH variants so the comparison stays fair.
  * nothink: identical examples, no reasoning (template renders an empty think block)
  * test:    card in the system prompt, no reasoning, identical in both variants
Uses the full local v2 data (intimate descriptions kept, they never left the Mac).

Entry points: `ngrams`, `prev_month`, `main`.

## src/parse_whatsapp.py

Module 1: WhatsApp (Android export) -> data/interim/messages.jsonl

One JSON object per message: {"ts", "sender", "text", "kind"}.
Nothing is dropped here; every message gets a `kind` label so Module 2
can decide what to filter. Prints counts only, never message text.

Entry points: `parse_dt`, `classify`, `parse`, `main`.

## src/postmortem_v3.py

Offline v3 audit. Private outputs stay in LocalData; stdout contains statistics only.

Entry points: `save`, `structural`, `record`, `render_review`, `benchmark`.

```python
ap.add_argument('phase', choices=['structural', 'benchmark'])
ap.add_argument('--models', nargs='+', default=['base', 'A', 'B', 'C'])
```

## src/reasoning_inputs.py

Build the inputs for writing Alex's reasoning (two passes, see v2/reasoning/PLAN.md).

  uv run python -I src/reasoning_inputs.py

Writes to ~/LocalData/hinglish-clone/v2/reasoning/:
  months/YYYY-MM.md    every session of that month, readable transcript (pass A: memory). Includes the
                       test period, so the memory stays current; those months get no reasoning.
  chunks/chunk_NNN.md  consecutive train/valid sessions with an ID on each of Alex's turns (pass B)
  chunks/index.json    chunk -> month, memory snapshot to use, IDs (full / short-optional)
Intimate photo/video descriptions become a bare tag here (these files go to an outside model).
Prints counts only.

Entry points: `sessions_with_messages`, `render_session`, `fmt_gap`, `main`.

## src/reasoning_local.py

Resumable offline reasoning. Private data stays under ~/LocalData; stdout is counts only.

Entry points: `say`, `atomic_json`, `previous_month`, `select_examples`, `clean`, `read_completed`, `review`, `main`.

```python
ap.add_argument('--limit', type=int, default=0, help='Optional limit for verification; 0 means full run')
```

## src/reasoning_local_pilot.py

Offline local rationale pilot. Content is written only to local files, never stdout.

Entry points: `main`.

```python
ap.add_argument('--n', type=int, default=4)
```

## src/reasoning_review.py

Check reasoning outputs and build the blind pilot comparison page. Prints counts only.

python3 -I src/reasoning_review.py check pilot_deepseek
python3 -I src/reasoning_review.py compare pilot_deepseek pilot_claude --n 60
    -> ~/LocalData/hinglish-clone/v2/reasoning/pilot_compare.html  (local file; writers shown as A/B)
       the A/B key is saved to pilot_key.json; your votes download as pilot_votes.json

Entry points: `load_examples`, `load_out`, `ngrams`, `check`, `compare`, `main`.

```python
ap.add_argument('cmd', choices=['check', 'compare'])
ap.add_argument('tags', nargs='+')
ap.add_argument('--n', type=int, default=60)
```

## src/reasoning_run.py

Run the reasoning passes (see ~/LocalData/hinglish-clone/v2/reasoning/PLAN.md). Standard library only.

  a1  month notes, all months in parallel          python3 -I src/reasoning_run.py a1 --backend claude --workers 6
  a2  merge notes into dated memory + card          python3 -I src/reasoning_run.py a2 --backend claude
  b   reasoning for chunks, in parallel             python3 -I src/reasoning_run.py b --backend openrouter --chunks 117,118 --tag pilot_deepseek
                                                    python3 -I src/reasoning_run.py b --backend claude --from-month 2025-09 --workers 6

Backends
  claude      headless `claude -p` on Alex's subscription, no tools, no MCP, run from an empty folder
  openrouter  DeepSeek V4 Pro, ONLY zero-data-retention providers that do not collect data
              (key file: ~/LocalData/hinglish-clone/.openrouter_key)
Everything is resumable: finished outputs are skipped. Prints progress counts only, never chat text.

Entry points: `say`, `call_claude`, `call_openrouter`, `call`, `month_units`, `run_a1`, `run_a2`, `parse_out`, `run_b`, `main`.

```python
ap.add_argument('pass_', choices=['a1', 'a2', 'b'])
ap.add_argument('--backend', choices=['claude', 'openrouter'], required=True)
ap.add_argument('--model', default=None, help='claude: opus/sonnet; openrouter: deepseek/deepseek-v4-pro')
ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--chunks', default=None, help='comma-separated chunk numbers (pass b)')
ap.add_argument('--from-month', default='2025-09', help='pass b: chunks from this month on')
ap.add_argument('--tag', default=None, help='pass b: output folder name under out/')
```

## src/reasoning_summary.py

Build a private, offline review page. Prints counts only; never document contents.

Entry points: `build`.

## src/review_reasoning_page.py

Local review page: N random reasonings (default from 2026) next to their conversation and the real reply.
Alex marks right / partly / wrong; votes download as reasoning_votes.json. Prints counts only.
  python3 -I src/review_reasoning_page.py --n 30 --from 2026-01

Entry points: `main`.

```python
ap.add_argument('--n', type=int, default=30)
ap.add_argument('--from', dest='frm', default='2026-01')
ap.add_argument('--seed', type=int, default=99)
```

## src/test_jarvis_planner.py

Synthetic-only tests for remote privacy boundaries and planner failures.

Entry points: `PlannerTests`.

## src/train.py

Module 5/6: LoRA fine-tune with mlx-lm's trainer, but our own data encoding (src/encode.py).

Usage:
  uv run python src/train.py --iters 20  --limit 400  --adapter-path adapters/smoke   # quick smoke test
  uv run python src/train.py                                                         # full run from config.yaml
The adapter folder works directly with:  mlx_lm.chat --model <model> --adapter-path <folder>

Entry points: `reply_only_loss`, `load_split`, `main`.

```python
ap.add_argument('--model', default=T['model'], help='base model repo (default: config.yaml train.model)')
ap.add_argument('--iters', type=int, default=T['iters'])
ap.add_argument('--limit', type=int, default=0, help='use a random subset of train (smoke tests)')
ap.add_argument('--adapter-path', default=T['adapter_path'])
ap.add_argument('--data', default='data/final', help='folder with train/valid jsonl')
ap.add_argument('--resume', default=None, help='adapters.safetensors to continue training from')
ap.add_argument('--lr', type=float, default=T['learning_rate'])
ap.add_argument('--val-examples', type=int, default=max(200, T['batch_size'] * T['val_batches']))
ap.add_argument('--steps-per-eval', type=int, default=T['steps_per_eval'])
ap.add_argument('--save-every', type=int, default=T['save_every'])
ap.add_argument('--cosine', action='store_true', help='100-step linear warmup, then cosine decay to lr/10 (for long runs)')
ap.add_argument('--thinking', action='store_true', help='train WITH thinking: enable_thinking=True, loss covers <think>reasoning</think> + reply')
ap.add_argument('--max-seq', type=int, default=T['max_seq_length'], help='truncate longer sequences (v3 max is 2,412)')
```

## src/watch.sh

Live dashboard for baseline / training runs. Ctrl+C to quit (does NOT stop the run).

## essay/build.py

Build the local reading page: uv run --with markdown python essay/build.py.

## essay/research/capture_examples.py

Bounded offline synthetic chats. No private data, memory cards or API calls.

## essay/research/capture_hinglish.py

Bounded offline synthetic chats. No private data, memory cards or API calls.

## essay/research/draw_architecture.py

Entry points: `box`, `arrow`.

## essay/research/generate_figures.py

## essay/research/render_examples.py

Entry points: `render`.

## essay/research/render_hinglish.py

Render unedited replies from the new, synthetic Hinglish probe.

## chat_B.command

Double-click in Finder, or run ./chat_B.command [chat_v3.py options].

## chat_B_C.command

B generates a thought; C writes the reply using that thought.

## chat_B_targeted.command



## chat_B_weight_edit.command



## chat_D_C.command

D generates a thought; C writes the reply using that thought.

## chat_DeepSeek_C.command

Jarvis DeepSeek plans; local model C writes the reply.

## chat_vanilla.command

Original LFM3 weights, with no adapter or personal memory.

## experiments/gpu_bundle/encode.py

Turn one chat example into (tokens, offset) for training, matching chat time exactly.

Why not mlx-lm's built-in ChatDataset?  It tokenizes the *finished* conversation but
computes the mask offset from the *generation prompt*. For templates that add extra
tokens to the generation prompt (Gemma-4's empty thought channel, Qwen's <think> block),
the offset lands inside the reply and the reply's first tokens are never trained.

Here the training sequence is built as:
    generation prompt (exactly what the model sees at chat time)   -> masked
  + reply + end-of-turn marker (taken from the finished rendering)  -> trained

Entry points: `encode_example`.

## experiments/gpu_bundle/eval_pipeline.py

Thinker + texter pipeline test: the THINK model writes a thought (greedy, <=60 tokens), the TEXTER gets it in its
system prompt and the real reply is scored. Same 500 test examples as before. Prints numbers only.

## experiments/gpu_bundle/eval_think.py

Fair test score for the THINK model: it writes its own thought (greedy, <=60 tokens), then the real reply is
scored given that thought. Same 500 test examples as gpu_train.test_bpc. Prints numbers only.

## experiments/gpu_bundle/gen_eval.py

Runs ON THE GPU VM. For N fixed test conversations, every setup writes its reply (batched generation):
  A  plain (no thinking)          B  thinks, then replies
  BC B writes the thought, C texts   DC D writes the thought, C texts
Output: out_eval/generations.jsonl  ({id, real, A, B, B_thought, BC, BC_thought, DC, DC_thought}). Prints counts only.
  python gen_eval.py --n 300

Entry points: `load`, `gen`, `prompt`, `main`.

```python
ap.add_argument('--n', type=int, default=300)
```

## experiments/gpu_bundle/gpu_train.py

Runs ON THE GPU VM. LoRA fine-tune of LiquidAI/LFM3-1.7B (Hugging Face weights) with the SAME masking as the Mac
pipeline (encode.py): the prompt is masked, only "<thought></think>reply" (think) or "reply" (nothink) is trained.

  python gpu_train.py --variant think   --out runs/think   [--steps 20 --limit 400]   # smoke test
  python gpu_train.py --variant nothink --out runs/nothink
Batch 16 with length grouping, LR 4e-5 cosine (warmup 50), LoRA r16 scale 20 on all linear layers, 1 epoch.
At the end: validation loss, test BPC (reply only, no-think format, 500 fixed test examples), merged bf16 model in
<out>/merged. Prints counts/metrics only, never chat text.

Entry points: `load_rows`, `encode_rows`, `batches`, `collate`, `loss_fn`, `evaluate`, `test_bpc`, `main`.

```python
ap.add_argument('--variant', choices=['think', 'nothink', 'texter', 'thinker'], required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--lr', type=float, default=4e-05)
ap.add_argument('--rank', type=int, default=16)
ap.add_argument('--scale', type=float, default=20.0)
ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--steps', type=int, default=0, help='stop after N steps (smoke test)')
ap.add_argument('--limit', type=int, default=0, help='use N random train examples (smoke test)')
ap.add_argument('--max-seq', type=int, default=2560)
ap.add_argument('--eval-every', type=int, default=200)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--base-only', action='store_true', help="only report the untrained base model's test BPC")
```

## experiments/vm_bundle/vm_reason.py

Runs ON THE GPU VM. Writes Alex's reasoning with an open model via vLLM (offline batch).

  python3 vm_reason.py cards                 # monthly memory cards from notes (sequential, small)
  python3 vm_reason.py reason --limit 600    # pilot: first N selected replies
  python3 vm_reason.py reason                # all selected replies (train/valid, from 2025-11-08)
Each reason run = generate -> verify (same model) -> regenerate failures once -> verify.
Outputs: out/cards.json, out/reasoning.jsonl ({id, reasoning, verdict, why}).
Prints counts only.

Entry points: `render`, `load`, `prev_month`, `cards`, `fewshot`, `reason`, `main`.

```python
ap.add_argument('step', choices=['cards', 'reason'])
ap.add_argument('--limit', type=int, default=0)
```

## experiments/weight_edit_audit/b_holdout.py

## experiments/weight_edit_audit/b_weight_trial.py

## experiments/weight_edit_audit/inspect_model_b.py

## experiments/weight_edit_audit/probe_model_b.py

## experiments/weight_edit_audit/targeted/experiment.py

Entry points: `chosen`, `setvariant`, `score`.

## experiments/v2/reasoning/tools/view.py

Entry points: `short`.

