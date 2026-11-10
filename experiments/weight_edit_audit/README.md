# Weight-edit experiments

Set LOCAL_FINETUNING_HOME to an external workspace with compatible MLX checkpoints. The scripts inspect weights, interpolate retained fine-tuning fractions, restore module families, and compare fictional development/holdout prompts. Tensor names and shapes must agree before editing.

The public prompts replace the original fixtures; historical scores do not apply to them. Likelihood margins, keyword matches, and exact matches do not establish conversation quality or privacy removal. Keep generated text, score artifacts, and edited checkpoints outside Git.
