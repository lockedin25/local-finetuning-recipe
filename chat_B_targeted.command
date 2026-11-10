#!/bin/zsh
cd "${0:A:h}" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
model_dir="${LOCAL_FINETUNING_HOME:-$HOME/LocalData/hinglish-clone}/models/B_targeted_conv/mlx"
[[ -s "$model_dir/model.safetensors" ]] || exit 1
print "Experimental convolution-weight reversion. Coherence improvement is not established."
exec zsh src/lfm3.sh src/chat_v3.py --model "$model_dir" --mode think --card none --personal-style "$@"
