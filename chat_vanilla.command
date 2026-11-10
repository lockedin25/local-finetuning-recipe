#!/bin/zsh
# Original LFM3 weights, with no adapter or personal memory.
cd "${0:A:h}" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
model_dir="$HOME/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-MLX-bf16/snapshots/52041852d5b5c59bfe120c075d2744ab97959d91"
if [[ ! -s "$model_dir/model.safetensors" ]]; then
  print -u2 "The cached vanilla model is missing: $model_dir"
  exit 1
fi
exec zsh src/lfm3.sh -m mlx_lm chat --model "$model_dir" --trust-remote-code --max-tokens 600 "$@"
