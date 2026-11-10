# Recipe and publication tools

`recipe.py` initializes an external workspace, installs generic prompts without overwriting local edits, and wraps parse/clean/build. Its fictional demo refuses to overwrite an existing export.

`audit.py` checks Git index blobs, allowed source paths, credential patterns, and reviewed hashes. It rejects unexpected artifacts, binary content, large files, symlinks, submodules, and stale manifests. Regenerating a manifest is not a substitute for reviewing the diff.
