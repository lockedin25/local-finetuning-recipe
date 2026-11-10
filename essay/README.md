# Public technical article and tooling

`draft.md` and `final.md` contain the public technical article. They do not reproduce the private personal essay, author annotations, or internal thesis notes.

```sh
uv sync --locked --extra essay
uv run python essay/build.py
uv run python essay/research/generate_figures.py
uv run python essay/research/draw_architecture.py
```

The HTML and figures are local build products, ignored by Git. `figure_data.json` contains only numeric aggregate metrics. Historical losses from different targets do not establish a quality ranking.

The capture and rendering scripts are preserved for local experiments. They require your own compatible checkpoints. Their generated outputs remain private even for fictional prompts and are excluded from publication.
