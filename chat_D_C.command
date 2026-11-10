#!/bin/zsh
# D generates a thought; C writes the reply using that thought.
cd "${0:A:h}" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
for name in D_thinker C_texter; do
  model_dir="${LOCAL_FINETUNING_HOME:-$HOME/LocalData/hinglish-clone}/models/$name/mlx"
  if [[ ! -s "$model_dir/model.safetensors" || ! -s "$model_dir/config.json" ]]; then
    print -u2 "$name is not ready in MLX format: $model_dir"
    exit 1
  fi
done
exec zsh src/lfm3.sh src/chat_v3.py --model C --mode pipeline --thinker D "$@"
