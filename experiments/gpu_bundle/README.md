# GPU training and evaluation

Set LOCAL_FINETUNING_GPU_HOME to an external bundle containing `data/think/{train,valid,test}.jsonl` and `data/nothink/{train,valid,test}.jsonl`. Records use chat-format messages; thinking records may carry reasoning_content. The trainer derives texter and thinker objectives from the thinking inputs and saves merged bf16 weights.

The evaluation helpers score self-generated thoughts, thinker/writer pipelines, and fixed-context generations. They require compatible CUDA/PyTorch/PEFT and model support. Inputs, runs, weights, and generations remain external. Publication checks compile this branch without running a training job.
