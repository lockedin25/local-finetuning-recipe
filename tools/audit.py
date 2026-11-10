"""Audit Git's index, not just working files. Never print matched secret values."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "release-manifest.json"
SECRET_RULES = {
    "provider token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{20,})"),
    "AWS access key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_-]{30,}"),
    "private key": re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
}
ROOT_FILES = {".gitignore", ".env.example", ".python-version", "pyproject.toml", "uv.lock",
              "config.yaml", "SOURCE_MANIFEST.json", MANIFEST, "VERSION", "CHANGELOG.md"}
SOURCE_DIRS = {"src", "tools", "tests", "experiments"}


def allowed_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        return False
    if name in ROOT_FILES or (len(p.parts) == 1 and p.suffix == ".md"):
        return True
    if p.parts[0] in SOURCE_DIRS and p.suffix in {".py", ".sh", ".md"}:
        return True
    if len(p.parts) == 1 and p.suffix == ".command":
        return True
    if p.parts[0] in {"docs", "prompts"} and p.suffix == ".md":
        return True
    if name in {"examples/README.md", "examples/fictional_chat.txt", ".github/workflows/check.yml"}:
        return True
    if p.parts[0] == "essay" and p.suffix in {".py", ".md"}:
        return True
    return name == "essay/research/figure_data.json"


def findings(name, data):
    result = []
    if not allowed_path(name):
        result.append("unapproved path or artifact type")
    if len(data) > 1_000_000:
        result.append("unexpected large file")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        result.append("binary content")
    for label, regex in SECRET_RULES.items():
        if regex.search(data):
            result.append(label)
    return result


def index_files():
    output = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT)
    files = {}
    errors = []
    for entry in output.split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, oid, stage = header.split()
        name = path.decode()
        if mode not in {b"100644", b"100755"} or stage != b"0":
            errors.append((name, "symlink, submodule or unresolved index entry"))
            continue
        files[name] = subprocess.check_output(["git", "cat-file", "blob", oid.decode()], cwd=ROOT)
    return files, errors


def main():
    files, errors = index_files()
    if not files:
        errors.append(("index", "no staged or tracked files"))
    for name, data in files.items():
        errors.extend((name, f) for f in findings(name, data))
    expected = [{"path": name, "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(files.items()) if name != MANIFEST]
    if "--write-manifest" in sys.argv and not errors:
        # This is a review step, not a substitute for reviewing the diff or running Gitleaks.
        (ROOT / MANIFEST).write_text(json.dumps({"files": expected}, indent=2) + "\n")
        print(f"Wrote hashes for {len(expected)} indexed files. Stage the manifest, then rerun the audit.")
        return
    try:
        actual = json.loads(files.get(MANIFEST, b"{}"))["files"]
    except (ValueError, KeyError):
        actual = None
    if actual != expected:
        errors.append((MANIFEST, "missing or stale reviewed-file hashes"))
    if errors:
        for name, reason in errors:
            print(f"BLOCKED: {name}: {reason}")
        raise SystemExit(1)
    print(f"PASS: {len(files)} indexed text files; manifest matches; no blocked paths or credential patterns.")


if __name__ == "__main__":
    main()
