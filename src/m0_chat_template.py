"""Module 0.2: what a chat conversation actually looks like to the model."""
from transformers import AutoTokenizer

CONVO = [{'role': 'system', 'content': 'Reply briefly and naturally in Hinglish.'}, {'role': 'user', 'content': 'kal library chalein?'}, {'role': 'assistant', 'content': 'haan, subah chalte hain'}]





for name, repo in {"Gemma-4-12B": "mlx-community/gemma-4-12B-it-4bit",
                   "Qwen3.5-9B": "mlx-community/Qwen3.5-9B-4bit",
                   "Llama-3.1-8B": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"}.items():
    tok = AutoTokenizer.from_pretrained(repo)
    text = tok.apply_chat_template(CONVO, tokenize=False, add_generation_prompt=True)
    print(f"\n===== {name} =====\n{text}")
