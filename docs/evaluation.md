# Evaluate the behavior you need

Use a fixed held-out session set for comparisons. Reply bits per character allow comparisons across tokenizers; token loss alone does not. The local evaluator also checks reply length, multiple-text turns, emoji, formal wording, script choice, empty/overlong replies, and verbatim overlap with training replies of at least 25 characters.

These metrics measure separate properties. Style can improve while speaker ownership, memory, and conversation coherence remain weak. Exact-copy checks are limited tests, not a certification that a model has forgotten private data.

For thinking models, distinguish gold-rationale conditioning from self-generated-rationale evaluation. `eval_think.py` first generates its own thought; `eval_pipeline.py` generates a thought and evaluates the writer conditioned on it. `gen_eval.py` generates fixed test contexts across A, B, B-to-C, and D-to-C. All resulting texts stay local.

The optional hosted judge randomizes candidate order and scores sense, intent, and style from 1 to 5. It sends context, reference replies, and candidates to a provider; no hosted judge is invoked during publication checks. Keep a local human review alongside any model judge.

Use synthetic development and held-out prompts for weight experiments. The public fixtures differ from the private experiment fixtures, so historical outcomes cannot be attached to these replacements. Keyword and exact-match scores assess constrained tasks. Positive-minus-negative continuation log probability is a narrow ranking proxy. Neither proves general conversation quality.

Track unclosed thought blocks, repetition, wrong-speaker attribution, unsupported facts, callbacks, unanswered questions, and response drift over multiple turns. Never compare A/B/C/D validation losses as a quality leaderboard: the variants predict different targets with different conditioning.
