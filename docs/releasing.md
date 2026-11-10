# Release source safely

Work in this clean source repository. Do not add a remote to the original private working directory or transplant its history.

1. Review `git status --short`, `git diff`, and the files listed in `SOURCE_MANIFEST.json`. Use named staging paths. Inspect all changes to comments, docs, prompts, examples, and config.
2. Run the tests and syntax checks below. Test hardware-dependent code separately when changing training behavior.
3. Stage only reviewed files, then run `python tools/audit.py --write-manifest`. Review and stage `release-manifest.json` and run `python tools/audit.py` again. It reads index blobs, so cleaning a working copy without updating the index cannot bypass the check.
4. Run `gitleaks dir . --redact --no-banner` and `gitleaks git . --redact --no-banner`. Never paste an unredacted scanner report into an issue or chat.
5. Commit with a public GitHub identity, push normally, verify the remote commit SHA, and check Actions. Never force-push private history into this repository.

```sh
uv run python -m unittest discover -s tests -v
uv run python -m unittest discover -s src -p 'test_*.py' -v
uv run python -m compileall -q src tools tests experiments essay
```

CI repeats the index audit, syntax checks, and synthetic tests without secrets, model downloads, private inputs, or GPUs. Gitleaks is an additional local release check. No automated check can replace a semantic review of newly added prose.

The reviewed manifest excludes its own hash. Changing a staged source file makes the audit fail until the change is reviewed and the manifest is regenerated. Generating a new manifest does not itself approve the content.
