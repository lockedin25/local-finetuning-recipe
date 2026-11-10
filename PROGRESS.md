# Technical experiment record

The implementation progressed from tokenizer and template inspection to export parsing, redaction, session building, base-model scoring, custom reply masking, local LoRA training, media-aware context, memory/rationale annotation, four-objective GPU training, and weight-edit probes.

The first recorded LFM training curve improved validation loss from 4.870 to 3.293 nats/token over 9,455 steps in approximately 373 minutes. Recorded base reply BPC values were 2.114 for the Gemma baseline and 1.858 for the LFM baseline; an early LFM adapter recorded 1.413. These are historical aggregate observations on a private evaluation set, not a benchmark independently reproducible from this repository or a claim about the new fictional fixtures.

Important fixes include stripping edited markers after multiline merge, preserving captions, keeping long plain-text messages rather than filtering solely by length, measuring the completed-template/generation-prompt divergence, and excluding padding from target loss. Later work added previous-month memory, rationale coverage/resume checks, matched training variants, and self-generated thought evaluation.

The public release does not retrain models or rerun private evaluations. Its fresh checks cover compilation, the fictional parsing/redaction flow, overwrite prevention, publication boundaries, reply-encoding behavior, and mocked planner error/redaction contracts. See `docs/verification.md` for the release results.
