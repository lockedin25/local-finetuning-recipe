# Historical experiment branches

These scripts preserve GPU training, VM reasoning, and weight-edit experiments. They are research entry points, not a single unattended deployment system. Their datasets, checkpoints, annotation files, and outputs are absent.

Set `LOCAL_FINETUNING_HOME` for model and analysis artifacts, `LOCAL_FINETUNING_GPU_HOME` for the GPU data bundle, and `LOCAL_FINETUNING_VM_HOME` for the VM reasoning bundle. Keep all three outside the repository. Defaults and required input shapes are visible in the source. Install PyTorch/PEFT for GPU training and vLLM separately for offline VM reasoning. The local inspection and weight-edit scripts require MLX and compatible base/fine-tuned tensor layouts.

See the full recipe and script reference before running a branch. Publication verification compiles these scripts but does not train a model, rent compute, execute model remote code, or process private examples.
