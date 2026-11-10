#!/bin/zsh
# Double-click in Finder, or run ./chat_B.command [chat_v3.py options].
cd "${0:A:h}" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
model_dir="${LOCAL_FINETUNING_HOME:-$HOME/LocalData/hinglish-clone}/models/B_think/mlx"
if [[ ! -s "$model_dir/model.safetensors" || ! -s "$model_dir/config.json" ]]; then
  print -u2 "Model B's MLX files are missing: $model_dir"
  exit 1
fi
exec zsh src/lfm3.sh src/chat_v3.py --model B --mode think "$@"
