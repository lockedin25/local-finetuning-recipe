#!/bin/zsh
# Module 4: download bake-off candidates one after another (Xet backend off: it hangs on this network).
# Usage: zsh src/download_models.sh            (all, in order)
#        zsh src/download_models.sh <repo> ...  (only these)
cd "${0:A:h}/.."
export HF_HUB_DISABLE_XET=1

MODELS=(
  mlx-community/gemma-4-12B-it-4bit
  mlx-community/Qwen3.5-9B-4bit
  mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
  mlx-community/sarvam-1-4bit
)
(( $# )) && MODELS=("$@")

for repo in $MODELS; do
  echo "[$(date +%H:%M:%S)] START $repo"
  if uv run hf download "$repo" --max-workers 4 > /dev/null 2>> downloads.err; then
    echo "[$(date +%H:%M:%S)] DONE  $repo"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $repo (see downloads.err)"
  fi
done
echo "[$(date +%H:%M:%S)] ALL FINISHED"
