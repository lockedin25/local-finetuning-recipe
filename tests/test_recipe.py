"""No models, network calls, private files or real credentials are needed."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from audit import allowed_path, findings
from encode import encode_example
from recipe import outside_repo


class RecipeTests(unittest.TestCase):
    def test_publication_blocks_credentials_and_private_artifacts(self):
        token = ("gh" + "p_" + "x" * 36).encode()
        self.assertTrue(findings("src/example.py", token))
        for path in [".env", "data/train.jsonl", "models/model.safetensors", "outputs/review.html", "src/credential.key"]:
            self.assertFalse(allowed_path(path), path)
        self.assertTrue(allowed_path("src/train.py"))
        self.assertTrue(allowed_path("experiments/gpu_bundle/gpu_train.py"))

    def test_index_audit_reads_staged_content_even_if_working_copy_is_clean(self):
        from audit import index_files
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            source = repo / "src/example.py"
            source.parent.mkdir()
            token = "gh" + "p_" + "x" * 36
            source.write_text(token)
            subprocess.run(["git", "add", "src/example.py"], cwd=repo, check=True)
            source.write_text("clean working copy")
            with patch("audit.ROOT", repo):
                files, errors = index_files()
            self.assertFalse(errors)
            self.assertTrue(findings("src/example.py", files["src/example.py"]))
            self.assertNotEqual(files["src/example.py"], source.read_bytes())

    def test_workspace_rejects_repository_and_children(self):
        for path in [ROOT, ROOT / "private"]:
            with self.assertRaises(ValueError):
                outside_repo(path)

    def test_fictional_demo_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "demo"
            args = [sys.executable, str(ROOT / "tools/recipe.py"), "demo", "--workspace", str(workspace)]
            first = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            parsed = [json.loads(s) for s in (workspace / "data/interim/messages.jsonl").read_text().splitlines()]
            self.assertEqual(len(parsed), 8)
            self.assertNotIn("<This message was edited>", parsed[-2]["text"])
            cleaned = (workspace / "data/interim/clean.jsonl").read_text()
            self.assertNotIn("example@example.com", cleaned)
            self.assertIn("<EMAIL>", cleaned)
            original = (workspace / "data/raw/chat.txt").read_bytes()
            second = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual((workspace / "data/raw/chat.txt").read_bytes(), original)

    def test_encoding_includes_reply_prefix_and_excludes_trailing_whitespace(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kw):
                return [1, 2, 7] if kw.get("add_generation_prompt") else [1, 2, 3, 4, 0]
            def decode(self, tokens):
                return " " if tokens == [0] else "text"
        ids, offset = encode_example(Tokenizer(), [{"role": "user", "content": "fictional"}, {"role": "assistant", "content": "reply"}])
        self.assertEqual(ids, [1, 2, 7, 3, 4])
        self.assertEqual(offset, 3)


if __name__ == "__main__":
    unittest.main()
