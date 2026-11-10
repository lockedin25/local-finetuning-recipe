# Test hosted planner plus C

With compatible local checkpoints under the external workspace, run:

```sh
zsh src/lfm3.sh src/chat_v3.py --model C --mode jarvis
```

This mode sends recent text and a memory card to the configured provider. Use only authorized inputs. Never log API bodies or provider-private reasoning.

Use fictional prompts first. Test references to earlier turns, corrected plans, short replies, ambiguity, unanswered questions, and missing media. `/reset` starts a new conversation; `/quit` exits. Keep outputs private because the model may recall its training material. Checkpoint weights and historical test responses are intentionally absent.
