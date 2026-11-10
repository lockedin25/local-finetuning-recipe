# Local fine-tuning recipe

A complete source collection for conversational LoRA fine-tuning on Apple Silicon, with GPU training helpers, memory and rationale experiments, local inference, evaluation, weight-edit experiments, and blog tooling.

This release contains code and technical documentation. It contains no private conversations, training datasets, media, credentials, trained adapters, model checkpoints, generated memories, or historical model responses. Alex and Sam are fictional example identities.

## Start here

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --locked
uv run python tools/recipe.py demo --workspace "$HOME/LocalData/finetuning-demo"
uv run python -m unittest discover -s tests -v
uv run python -m unittest discover -s src -p 'test_*.py' -v
```

Use a new demo workspace. The demonstration parses eight fictional messages, strips an edited marker after multiline merging, and redacts an example email address. It downloads no model weights and calls no model API. It refuses to overwrite an existing export.

The parser, recipe, and tests work without MLX. Training and local inference require Apple Silicon and compatible model support. The separate GPU scripts require their own CUDA environment.

## Train on your own data

Follow the [full recipe](docs/recipe.md). Create an external workspace, configure sender names locally, parse and clean your own authorized export, build session-separated examples, run a small training smoke test, then evaluate before running a full job.

```sh
uv run python tools/recipe.py init --workspace "$HOME/LocalData/my-finetune"
# Edit the new workspace's config.local.yaml and supply your own authorized export.
uv run python tools/recipe.py parse --workspace "$HOME/LocalData/my-finetune"
uv run python tools/recipe.py clean --workspace "$HOME/LocalData/my-finetune"
uv run python tools/recipe.py build --workspace "$HOME/LocalData/my-finetune"
```

## Repository map

| Path | Contents |
|---|---|
| `src/` | Parsing, redaction, media tagging, dataset building, memory/rationale passes, MLX training, evaluation, chat, and monitoring |
| `experiments/gpu_bundle/` | PyTorch/PEFT training, four target variants, merging, and GPU evaluation |
| `experiments/vm_bundle/` | Offline vLLM memory and rationale generation |
| `experiments/weight_edit_audit/` | Interpolation, module restoration, model inspection, synthetic development and holdout checks |
| `essay/` | Public technical article and its build, capture, rendering, and plotting code |
| `prompts/` | Generic prompt templates for the two-pass memory/rationale workflow |
| `tools/` | External-workspace recipe and Git-index publication audit |
| `docs/` | Recipe, architecture, evaluation, privacy, release process, and complete script reference |
| `examples/` | Hand-authored fictional parser fixture |

The [source manifest](SOURCE_MANIFEST.json) lists all 59 imported scripts, including helpers recovered from external experiment directories. Technical comments and docstrings remain in the code; identity-dependent passages and embedded fixtures have public replacements. [Publication scope](docs/publication-scope.md) explains the treatment of private source documents.

## Models and reproducibility

Base-model identifiers and historical settings remain in the code. The LFM3 MLX wrapper pins a specific implementation commit used by the experiments. These identifiers are historical dependencies, not a promise that every upstream model or branch remains accessible. Review upstream code and licenses before enabling `trust_remote_code` or redistributing weights.

The public recipe can reproduce the method on your own data. It cannot reproduce private checkpoints or their exact historical scores. No quality ranking follows from comparing losses across different training targets.

See [architecture](docs/architecture.md), [evaluation](docs/evaluation.md), [script reference](docs/scripts.md), [privacy](docs/privacy.md), and [release checks](docs/releasing.md). No license has been selected for this source release; upstream packages and models retain their own licenses.
