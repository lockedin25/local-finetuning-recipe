# Technical research notes

The experiment separates style adaptation from conversation grounding. Reply BPC and formatting proxies can improve without resolving speaker confusion, context drift, unsupported activities, repetition, or unfinished thought blocks. Evaluate these behaviors directly.

V1 uses 12 turns and a 512-token context budget. V2 increases context to 40 turns and 2,048 tokens and preserves captions through typed media descriptions. Memory should be dated and past-only; a monthly card does not provide a complete model of a person's memory.

Four objectives isolate reply imitation, joint rationale/reply modeling, rationale-conditioned writing, and rationale generation. Target lengths and conditioning differ, so validation losses cannot rank their conversation quality. Fixed-context generation, self-generated thought scoring, and local human review provide complementary evidence.

Weight interpolation retains a fraction of the fine-tuning delta. Module restoration selects tensor families to restore from the base. Shape checks are mandatory. Development likelihood margins, holdout exact matches, and conversational review measure different things. These experiments do not prove unlearning or privacy.

The public plotting inputs retain numeric aggregates only. Original conversation examples, generated demonstration outputs, personal essay annotations, and private thesis connections are not included. Public synthetic fixtures were replaced, so old experimental outcomes must not be attached to the new fixtures.
