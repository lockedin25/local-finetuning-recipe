# Test B-to-C

With compatible local checkpoints under the external workspace, run:

```sh
zsh src/lfm3.sh src/chat_v3.py --model C --mode pipeline --thinker B
```

Inspect the generated plan and the writer separately. A plausible plan does not guarantee a grounded reply.

Use fictional prompts first. Test references to earlier turns, corrected plans, short replies, ambiguity, unanswered questions, and missing media. `/reset` starts a new conversation; `/quit` exits. Keep outputs private because the model may recall its training material. Checkpoint weights and historical test responses are intentionally absent.
