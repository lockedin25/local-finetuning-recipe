"""Module 0: how do different model families tokenize Hinglish vs English?

Downloads only tokenizer files (a few MB each), not model weights.
Sentences are generic examples, not from the real chats.
"""
from transformers import AutoTokenizer

TOKENIZERS = {
    "Qwen2.5": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "Llama-3.2": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "Gemma-2": "mlx-community/gemma-2-2b-it-4bit",
}

SENTENCES = [('hi', 'aaj library kab khulegi'), ('hi', 'kal hum chai banayenge'), ('en', 'Please put the blue notebook on the desk.'), ('en', 'The workshop starts at ten tomorrow.'), ('mix', 'meeting ke baad coffee lete hain'), ('mix', 'ye example entirely fictional hai')]











def main():
    for name, repo in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        print(f"\n=== {name}  (vocab size {len(tok):,}) ===")
        for kind, s in SENTENCES:
            ids = tok.encode(s, add_special_tokens=False)
            pieces = tok.convert_ids_to_tokens(ids)
            words = len(s.split())
            print(f"[{kind:3}] {len(ids):2} tok / {words} words = {len(ids)/words:.2f}  {pieces}")


if __name__ == "__main__":
    main()
