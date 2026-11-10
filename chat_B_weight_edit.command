#!/bin/zsh
cd "${0:A:h}" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
model_dir="${LOCAL_FINETUNING_HOME:-$HOME/LocalData/hinglish-clone}/models/B_weight_edit_025/mlx"
if [[ ! -s "$model_dir/model.safetensors" ]]; then
  print -u2 "Experimental edited checkpoint is not available yet."
  exit 1
fi
print "Experimental weight edit: 25% B + 75% original LFM. Conversation quality needs review."
exec zsh src/lfm3.sh src/chat_v3.py --model "$model_dir" --mode think --card none --personal-style "$@"
