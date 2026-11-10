# Privacy boundary

This repository publishes implementation, generic prompts, technical comments, configuration templates, fictional fixtures, and reviewed aggregate metrics. It starts with fresh Git history.

Do not add datasets, exports, media, descriptions, memories, generated rationales, embeddings, caches, credentials, account access details, model outputs, screenshots, adapters, checkpoints, or historic assistant sessions. Renaming people or redacting direct identifiers does not make a conversation anonymous.

The recipe keeps generated files outside the checkout. Historical low-level defaults may still name relative data or output paths; use external arguments and environment settings. `.gitignore` helps prevent accidental staging, but an ignored file can still be force-added. `tools/audit.py` checks Git's actual index, allows only reviewed source paths and artifact types, checks common credential signatures, and compares file hashes with a reviewed manifest. Run Gitleaks independently before pushing.

Credentials are read at runtime from local key files. Optional hosted paths expect `.jarvis_key` or `.openrouter_key` beneath `LOCAL_FINETUNING_HOME`; create them locally with permissions limited to your user. Do not paste their values into commands, configs, issues, documentation, or screenshots.

Hosted reasoning, judging, and planning send text to external services. Local media description, local reasoning, and local chat do not require a hosted backend. Read each command before running it; the release process does not execute a paid model call.

Fictional prompts do not make a private model's outputs public. A fine-tune can recall private information. Adapters, merged weights, weight edits, and generations remain private regardless of packaging. Weight interpolation is not a privacy guarantee.

Automated scanning catches known patterns and unexpected files; it cannot prove that arbitrary prose is free of private information. Review the staged diff and source manifest before every release. Report an accidentally published secret through GitHub's private security reporting where enabled; rotate the credential and remove it from all reachable history before treating the incident as resolved.
