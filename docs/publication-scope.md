# Publication scope and source coverage

The public import includes every Python, shell, and launcher script found in the main fine-tuning project and the related GPU, VM, reasoning-tool, and weight-edit directories. `SOURCE_MANIFEST.json` lists the 59 imported files. New recipe, audit, and test code is separate.

Technical comments and docstrings stay with their implementations. Public changes replace participant names, personal filesystem locations, a memory-cleaning prompt containing private facts, identity-dependent style instructions, and embedded evaluation fixtures. Tests and fixtures use new fictional content. Public fixtures have the shapes expected by their consumers; they are not renamed copies of private chats.

The source documentation also contained personal facts, consent history, operational access details, and private thesis notes. Those documents are not copied verbatim. This release provides technical publication versions of the handoff, progress notes, V3 plan, variant testing guides, essay, and research notes, plus a consolidated recipe, architecture, evaluation guide, and generated script reference. Personal anecdotes, private annotations, internal strategy material, and generated conversations remain local. The public article makes no claim to be the author's final personal essay.

Original working files remain unchanged. The public export has no inherited commits, remotes, logs, or assistant-session history. No private weights or datasets were read into a commit or sent to GitHub. The base-model identifiers point to upstream projects; no upstream weights are mirrored here.

Historical aggregate figures can document an experiment without exposing conversations. The included plotting input contains only training-loss curves and numeric style/BPC summaries, with no prompts, text outputs, paths, identities, or sample IDs.
