#!/usr/bin/env python3
"""Translate Chinese characters to English, file-by-file, via a LiteLLM proxy.

Usage:
    python zh_to_en.py /path/to/repo [--apply] [--no-ignore] [--rename]
        [--model MODEL] [--base-url URL] [--workers N] [--ext EXT ...]

Default is a dry run that reports what would change.
Use --rename to also translate Chinese characters in filenames.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Chinese character ranges (CJK unified ideographs + full-width punctuation).
_CHINESE_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\uff00-\uffef\u3000-\u303f]+"
)

# Directories / file patterns we never touch.
_IGNORED_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    "vendor", "dist", "build", "target", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox",
}
_IGNORED_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dll", ".dylib", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2",
    ".ttf", ".eot", ".zip", ".gz", ".tar", ".bz2", ".7z", ".pdf", ".doc",
    ".docx", ".xls", ".xlsx", ".po", ".mo", ".lock", ".min.js", ".min.css",
}
class Translator:
    """Translates a batch of Chinese strings to English single strings."""

    def __init__(self, base_url, api_key, model, timeout=120, max_batch=20):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_batch = max_batch
        self._cache = {}
        self._lock = threading.Lock()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def _system_prompt(self):
        return (
            "You are a careful code translation assistant. Translate the Chinese "
            "text in the user message into natural, idiomatic English. The message "
            "is a JSON object mapping an id to a Chinese string. Reply with ONLY a "
            "JSON object mapping the same ids to their English translations. "
            "Preserve the original meaning exactly; do not add explanations, do not "
            "wrap in markdown code fences, do not translate anything that is already "
            "English, and keep any variable/code-style wording the user specifies."
        )

    def _translate_one(self, text: str) -> str:
        with self._lock:
            if text in self._cache:
                return self._cache[text]

        import json

        user_payload = json.dumps({"zh0": text}, ensure_ascii=False)
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            content = _strip_code_fence(content)
            parsed = json.loads(content)
            result = str(parsed.get("zh0", "")).strip()
            if not result:
                raise ValueError("empty translation")
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] translation failed, using placeholder: {exc}",
                  file=sys.stderr)
            result = f"[zh:{text}]"

        with self._lock:
            self._cache[text] = result
        return result


    def _translate_file(self, text: str, spans: list[str]) -> dict[str, str]:
        """Translate all Chinese spans at once using the whole file as context.

        Returns a mapping from the exact original span string to its translation.
        Falls back to per-span translation for any span the model does not cover.
        """
        import json

        unique = list(dict.fromkeys(spans))
        file_prompt = (
            "Below is a source file containing Chinese text marked only in the "
            "'spans' list (verbatim substrings of the file). Using the surrounding "
            "code as context, translate each marker to natural, idiomatic English, "
            "preserving surrounding whitespace/code style. Reply with ONLY a JSON "
            "object mapping each EXACT Chinese span (as given) to its English "
            "translation. Do not translate anything outside 'spans'. Do not wrap in "
            "markdown code fences.\n\n"
            "Spans:\n" + json.dumps(unique, ensure_ascii=False) +
            "\n\nFile:\n```\n" + text + "\n```"
        )
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": file_prompt},
                ],
                temperature=0.0,
                max_tokens=6000,
            )
            content = (resp.choices[0].message.content or "").strip()
            content = _strip_code_fence(content)
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("response is not a JSON object")
            result = {}
            for orig in unique:
                translated = parsed.get(orig)
                if isinstance(translated, str) and translated.strip():
                    result[orig] = translated.strip()
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] whole-file translation failed ({exc}), "
                  "falling back to per-span", file=sys.stderr)
            result = {}

        for orig in unique:
            if orig not in result:
                result[orig] = self._translate_one(orig)
        return result


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


DEFAULT_IGNORE = _IGNORED_DIRS | _IGNORED_SUFFIXES


def should_skip(path: Path, ignore: bool, ext_whitelist: set[str] | None) -> bool:
    parts = set(path.parts)
    if ignore and parts & _IGNORED_DIRS:
        return True
    if path.suffix.lower() in _IGNORED_SUFFIXES:
        return True
    if ext_whitelist is not None:
        return path.suffix.lower() not in ext_whitelist
    return False


def collect_files(root: Path, ignore: bool, ext: list[str]) -> list[Path]:
    whitelist = None
    if ext:
        whitelist = {e if e.startswith(".") else "." + e for e in ext}
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        if ignore:
            dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if should_skip(p, ignore, whitelist):
                continue
            files.append(p)
    return files


def read_text(path: Path):
    with open(path, "rb") as fh:
        data = fh.read()
    # Skip clearly-binary files.
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def replace_chinese(text: str, translator: Translator, mode: str):
    """Replace each Chinese run with its translation, preserving structure."""
    found = _CHINESE_RE.findall(text)
    if not found:
        return text, 0

    if mode == "wholefile":
        map_ = translator._translate_file(text, found)
    else:  # "spans"
        unique = list(dict.fromkeys(found))
        map_ = {}
        for run in unique:
            map_[run] = translator._translate_one(run)

    return _CHINESE_RE.sub(lambda m: map_.get(m.group(0), m.group(0)), text), len(found)


def translate_filename(name: str, translator: Translator, mode: str) -> str:
    """Translate Chinese characters in a filename stem, preserving the extension."""
    p = Path(name)
    stem = p.stem
    suffix = p.suffix

    found = _CHINESE_RE.findall(stem)
    if not found:
        return name

    if mode == "wholefile":
        map_ = translator._translate_file(stem, found)
    else:
        unique = list(dict.fromkeys(found))
        map_ = {}
        for run in unique:
            map_[run] = translator._translate_one(run)

    new_stem = _CHINESE_RE.sub(lambda m: map_.get(m.group(0), m.group(0)), stem)
    new_stem = new_stem.strip().replace(" ", "_")
    if not new_stem:
        return name
    return new_stem + suffix


def process_file(path: Path, translator, apply_changes, stats, mode):
    text = read_text(path)
    if text is None:
        return
    new_text, nfound = replace_chinese(text, translator, mode)
    if nfound == 0:
        return
    stats["files"] += 1
    stats["runs"] += nfound
    if text != new_text:
        print(f"[{'apply' if apply_changes else 'dry'}] {path}")
        if apply_changes:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new_text)
    else:
        print(f"[nochange] {path}")


def rename_files(files: list[Path], translator: Translator, apply_changes: bool,
                 stats: dict, mode: str) -> list[tuple[Path, Path]]:
    """Translate Chinese characters in filenames and rename them.

    Processes deepest paths first to avoid parent-dir conflicts.
    Renames are done sequentially after content translation is complete.
    Returns a list of (old_path, new_path) tuples for files that were renamed.
    """
    to_rename = []
    for path in files:
        stem = path.stem
        if not _CHINESE_RE.search(stem):
            continue
        new_name = translate_filename(path.name, translator, mode)
        if new_name == path.name:
            continue
        new_path = path.parent / new_name
        if new_path == path:
            continue
        to_rename.append((path, new_path))

    to_rename.sort(key=lambda pair: pair[0], reverse=True)

    renamed = []
    for old_path, new_path in to_rename:
        if new_path.exists():
            print(f"[rename-skip] {old_path.name} -> {new_path.name} "
                  f"(target exists)")
            continue
        print(f"[{'apply' if apply_changes else 'dry'}] rename "
              f"{old_path.name} -> {new_path.name}")
        if apply_changes:
            old_path.rename(new_path)
            stats["files"] += 1
            renamed.append((old_path, new_path))
    return renamed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Translate Chinese characters in a repo to English via LiteLLM."
    )
    ap.add_argument("target", help="Repo folder (or single file) to process")
    ap.add_argument("--apply", action="store_true",
                    help="Write changes back to files (default is dry-run)")
    ap.add_argument("--no-ignore", action="store_true",
                    help="Do not skip vendored/build/binary dirs")
    ap.add_argument("--mode", choices=["spans", "wholefile"], default="spans",
                    help="spans=translate each Chinese snippet alone (cheap/safe); "
                         "wholefile=translate all snippets with full file context "
                         "(better style, more tokens)")
    ap.add_argument("--model", default=os.environ.get("LITELLM_MODEL", "gpt-4o-mini"),
                    help="Model name registered in the LiteLLM proxy")
    ap.add_argument("--base-url",
                    default=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1"),
                    help="OpenAI-compatible base URL of the LiteLLM proxy")
    ap.add_argument("--api-key", default=os.environ.get("LITELLM_API_KEY", "sk-dummy"),
                    help="API key for the proxy")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel worker threads")
    ap.add_argument("--ext", nargs="+", default=None,
                    help="Only process these extensions, e.g. --ext py js ts")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Per-request timeout in seconds")
    ap.add_argument("--rename", action="store_true",
                    help="Also translate Chinese characters in filenames")
    args = ap.parse_args(argv)

    target = Path(args.target)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = collect_files(target, ignore=not args.no_ignore, ext=args.ext)
    else:
        print(f"error: not a file or directory: {target}", file=sys.stderr)
        return 2

    if not files:
        print("No matching files found.")
        return 0

    translator = Translator(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
    )

    mode = "apply" if args.apply else "dry-run"
    print(f"Processing {len(files)} file(s) with model={args.model} "
          f"translate={args.mode} mode={mode} target={args.target}\n")

    stats = {"files": 0, "runs": 0, "errors": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_file, p, translator, args.apply, stats, args.mode)
                for p in files]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                print(f"[error] {exc}", file=sys.stderr)

    if args.rename:
        print()
        renamed = rename_files(files, translator, args.apply, stats, args.mode)

    print(f"\nDone. {stats['files']} file(s), {stats['runs']} Chinese run(s), "
          f"{stats['errors']} error(s).")
    if not args.apply:
        print("Dry run only — re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
