"""Safe workspace setup and a small, fictional parser/redactor demonstration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def outside_repo(value):
    path = Path(value).expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("Choose a workspace outside the repository.")
    return path


def setup(workspace):
    workspace = outside_repo(workspace)
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = workspace / "config.local.yaml"
    if not config.exists():
        text = (REPO / "config.yaml").read_text()
        text = text.replace("raw_chat: data/raw/chat.txt", "raw_chat: " + json.dumps(str(workspace / "data/raw/chat.txt")))
        text = text.replace("messages_out: data/interim/messages.jsonl", "messages_out: " + json.dumps(str(workspace / "data/interim/messages.jsonl")))
        config.write_text(text)
        config.chmod(0o600)
    prompt_dir = workspace / "v2/reasoning"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for template in (REPO / "prompts").glob("*.md"):
        dest = prompt_dir / template.name
        if not dest.exists():
            dest.write_text(template.read_text())
            dest.chmod(0o600)
    env = dict(os.environ, LOCAL_FINETUNING_HOME=str(workspace),
               LOCAL_FINETUNING_CONFIG=str(config), LOCAL_FINETUNING_DATA=str(workspace / "data"))
    return workspace, env


def run_script(name, args, env):
    subprocess.run([sys.executable, str(REPO / "src" / name), *args], cwd=REPO, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["init", "demo", "parse", "clean", "build"])
    parser.add_argument("--workspace", required=True, help="Private directory outside this repository")
    args, extra = parser.parse_known_args()
    workspace, env = setup(args.workspace)
    if args.stage == "init":
        print("Workspace initialized. Edit its config.local.yaml before using your own data.")
        return
    if args.stage == "demo":
        # Never overwrite an existing export, including a real conversation.
        raw = workspace / "data/raw/chat.txt"
        raw.parent.mkdir(parents=True, exist_ok=True)
        with raw.open("x", encoding="utf-8") as f:
            f.write((REPO / "examples/fictional_chat.txt").read_text())
        raw.chmod(0o600)
        run_script("parse_whatsapp.py", [], env)
        run_script("clean_redact.py", [], env)
        print("Fictional demo completed; all generated files remain in the external workspace.")
        return
    if args.stage == "parse":
        run_script("parse_whatsapp.py", extra, env)
    elif args.stage == "clean":
        run_script("clean_redact.py", extra, env)
    else:
        run_script("build_dataset.py", ["--out", str(workspace / "data/final"), *extra], env)


if __name__ == "__main__":
    main()
