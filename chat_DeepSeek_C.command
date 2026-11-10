#!/bin/zsh
# Jarvis DeepSeek plans; local model C writes the reply.
cd "${0:A:h}" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
model_dir="${LOCAL_FINETUNING_HOME:-$HOME/LocalData/hinglish-clone}/models/C_texter/mlx"
if [[ ! -s "$model_dir/model.safetensors" || ! -s "$model_dir/config.json" ]]; then
  print -u2 "Model C is not ready in MLX format: $model_dir"
  exit 1
fi
if [[ ! -s "${LOCAL_FINETUNING_HOME:-$HOME/LocalData/hinglish-clone}/.jarvis_key" ]]; then
  print -u2 "The Jarvis API key file is missing or empty."
  exit 1
fi
exec zsh src/lfm3.sh src/chat_v3.py --model C --mode jarvis "$@"
