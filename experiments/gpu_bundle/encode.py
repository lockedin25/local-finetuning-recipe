"""Turn one chat example into (tokens, offset) for training, matching chat time exactly.

Why not mlx-lm's built-in ChatDataset?  It tokenizes the *finished* conversation but
computes the mask offset from the *generation prompt*. For templates that add extra
tokens to the generation prompt (Gemma-4's empty thought channel, Qwen's <think> block),
the offset lands inside the reply and the reply's first tokens are never trained.

Here the training sequence is built as:
    generation prompt (exactly what the model sees at chat time)   -> masked
  + reply + end-of-turn marker (taken from the finished rendering)  -> trained
"""


def encode_example(tok, messages, template_kwargs=None):
    kw = template_kwargs or {}
    full = tok.apply_chat_template(messages, return_dict=False, **kw)
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, return_dict=False, **kw)
    # Where the finished rendering and the generation prompt part ways = start of the reply
    d = 0
    for a, b in zip(full, prompt):
        if a != b:
            break
        d += 1
    target = full[d:]
    # Drop template whitespace AFTER the end-of-turn marker (e.g. Gemma's "<turn|>\n"):
    # it is never generated, so it should not be trained.
    while len(target) > 1 and tok.decode(target[-1:]).strip() == "":
        target = target[:-1]
    return prompt + target, len(prompt)
