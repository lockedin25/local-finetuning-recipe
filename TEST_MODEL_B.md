# Test B

With compatible local checkpoints under the external workspace, run:

```sh
zsh src/lfm3.sh src/chat_v3.py --model B --mode think
```

Check that the thought block closes, the reply answers the latest turn, and no speaker swap or unsupported fact appears.

Use fictional prompts first. Test references to earlier turns, corrected plans, short replies, ambiguity, unanswered questions, and missing media. `/reset` starts a new conversation; `/quit` exits. Keep outputs private because the model may recall its training material. Checkpoint weights and historical test responses are intentionally absent.
