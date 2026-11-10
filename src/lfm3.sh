#!/bin/zsh
# Run any project script with an mlx-lm that can load LFM3.
# No mlx-lm release supports LFM3 yet; this pins the open PR ml-explore/mlx-lm#2016
# (one new file, mlx_lm/models/lfm3.py, reviewed 2026-11-08) as a temporary overlay.
# The project's own environment (used for Gemma) is untouched.
# Usage: zsh src/lfm3.sh src/train.py --model LiquidAI/LFM3-1.7B-MLX-bf16 ...
cd "${0:A:h}/.."
export HF_HUB_DISABLE_XET=1 TRANSFORMERS_VERBOSITY=error
exec uv run --with "mlx-lm @ git+https://github.com/techwritescode/mlx-lm@a1de292b740bd8297e173f2c73047f5b6bdc43f1" python "$@"
